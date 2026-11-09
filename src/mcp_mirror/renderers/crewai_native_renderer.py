"""CrewAI native MCP renderer (DESIGN.md section 8).

CrewAI ships two MCP routes. The ``crewai`` row measures the older one,
``crewai_tools.MCPServerAdapter`` (mcpadapt underneath). This row measures the one
built into ``crewai`` itself: ``Agent(mcps=[MCPServerStdio | MCPServerHTTP |
MCPServerSSE])``, which ``Agent.get_mcp_tools`` hands to
``crewai.mcp.tool_resolver.MCPToolResolver``. The resolver lists the server through
``crewai.mcp.MCPClient``, rebuilds each input schema as a Pydantic model with
``create_model_from_schema(enrich_descriptions=True)``, prefixes each name with a
server name, and returns ``MCPNativeTool`` objects that open a fresh ``MCPClient`` per
call.

The capture is the function-tool list ``crewai.utilities.agent_utils.
convert_tools_to_openai_schema`` builds from those tools, which is what CrewAI's
native function-calling executor hands the LLM. No LLM is bound and nothing is sent.

Tool results have two boundaries: ``BaseTool.run`` (the value the tool returns,
which is a ``ToolFailure`` when the server set ``isError``), and
``execute_single_native_tool_call``, the executor step that turns a model's tool
call into the string the next model turn receives.

``MCPClient`` also exposes ``list_prompts`` and ``get_prompt`` and a public
``session`` property, but no resource API, and ``Agent(mcps=...)`` never hands the
client to the caller: those routes are reachable only by building an ``MCPClient``
yourself, which is what the surface and utility probes do.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from importlib import util
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version, contain_crewai_storage

contain_crewai_storage()

ID = "crewai_native"
REQUIRES = ("crewai",)

_CALL_ID = "mcp-mirror-call"


def available() -> bool:
    try:
        return util.find_spec("crewai") is not None and util.find_spec("crewai.mcp") is not None
    except ModuleNotFoundError:
        return False


def get_renderer() -> "CrewAINativeRenderer":
    return CrewAINativeRenderer()


@contextlib.contextmanager
def _quiet():
    """CrewAI's event bus prints Rich panels to stdout; keep what it prints in-line off it.

    Best effort: handlers the bus runs after the call returns still reach stdout, so
    callers that parse stdout read the last JSON line.
    """

    with contextlib.redirect_stdout(sys.stderr):
        yield


class CrewAINativeRenderer:
    id = ID

    def versions(self) -> dict:
        return {"framework": dist_version("crewai"), "adapter": dist_version("crewai")}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "crewai.Agent(mcps=[MCPServerStdio|MCPServerHTTP|MCPServerSSE]).get_mcp_tools "
                "(crewai.mcp.tool_resolver.MCPToolResolver), then "
                "crewai.utilities.agent_utils.convert_tools_to_openai_schema"
            ),
            capture_object=(
                "function-tool dicts convert_tools_to_openai_schema builds from each "
                "crewai.tools.mcp_native_tool.MCPNativeTool"
            ),
            capture_stage="provider_format",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "protocolVersion instrumented from ClientSession.initialize inside "
                "crewai.mcp.MCPClient.connect; the adapter does not expose it"
            ),
            limitation=(
                "No LLM was bound and no crew was kicked off; no provider client "
                "serialized a request. The ReAct text path, used when the LLM does not "
                "support function calling, renders tools into the prompt instead and "
                "was not captured. Stdio servers are launched with paths under the "
                "working directory written relative, because the launch command is "
                "part of every tool name."
            ),
        )

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    @staticmethod
    def _launch(handle: ServerHandle) -> tuple[str, list[str]]:
        """The stdio command and args, with paths under the working directory relative.

        MCPToolResolver names every stdio tool f"{command}_{'_'.join(args)}_{tool}", so
        the launch spelling is part of the name the model sees. The instruments start
        fixtures by absolute path, which would bake the operator's home directory into
        each name, and past 64 characters into a machine-specific hash. A relative
        path launches the same process from the same working directory.
        """

        def relative(part: str) -> str:
            if not os.path.isabs(part):
                return part
            rel = os.path.relpath(part)
            return part if rel.startswith("..") else rel

        return relative(handle.command), [relative(arg) for arg in handle.args]

    @classmethod
    def _config(cls, handle: ServerHandle) -> Any:
        from crewai.mcp import MCPServerHTTP, MCPServerSSE, MCPServerStdio

        headers = dict(handle.headers) or None
        if handle.transport == "http":
            return MCPServerHTTP(url=handle.url, headers=headers, streamable=True)
        if handle.transport == "sse":
            return MCPServerSSE(url=handle.url, headers=headers)
        command, args = cls._launch(handle)
        return MCPServerStdio(command=command, args=args)

    @classmethod
    def _transport(cls, handle: ServerHandle) -> Any:
        """The transport ``MCPToolResolver`` builds for the same config."""

        from crewai.mcp.transports.http import HTTPTransport
        from crewai.mcp.transports.sse import SSETransport
        from crewai.mcp.transports.stdio import StdioTransport

        headers = dict(handle.headers) or None
        if handle.transport == "http":
            return HTTPTransport(url=handle.url, headers=headers, streamable=True)
        if handle.transport == "sse":
            return SSETransport(url=handle.url, headers=headers)
        command, args = cls._launch(handle)
        return StdioTransport(command=command, args=args)

    def _mcp_client(self, handle: ServerHandle, **options: Any) -> Any:
        from crewai.mcp.client import MCPClient

        return MCPClient(transport=self._transport(handle), **options)

    @contextlib.contextmanager
    def _versions(self):
        """Record every protocolVersion a crewai.mcp ClientSession negotiates."""

        from unittest.mock import patch

        from mcp.client.session import ClientSession

        seen: list[str] = []
        original_initialize = ClientSession.initialize

        async def capture_initialize(session, *args, **kwargs):
            result = await original_initialize(session, *args, **kwargs)
            seen.append(str(result.protocolVersion))
            return result

        with patch.object(ClientSession, "initialize", capture_initialize), _quiet():
            yield seen
        if seen:
            observed = set(seen)
            if len(observed) != 1:
                raise RenderError(
                    "crewai_native renderer could not establish one negotiated MCP protocol "
                    f"version (observed: {sorted(observed)})"
                )
            self._negotiated_mcp_spec_version = seen[0]

    def _tools(self, handle: ServerHandle) -> list[Any]:
        """What ``Agent(mcps=[config])`` resolves its MCP config into."""

        from crewai import Agent

        agent = Agent(
            role="mcp-mirror",
            goal="list the server's tools",
            backstory="a measurement harness",
            mcps=[self._config(handle)],
            llm="gpt-4o-mini",
        )
        return agent.get_mcp_tools(agent.mcps)

    @staticmethod
    def _tool(tools: list[Any], name: str) -> Any:
        """Find a tool by its MCP name, its CrewAI name or its sanitized name."""

        from crewai.utilities.string_utils import sanitize_tool_name

        for tool in tools:
            if name in {getattr(tool, "original_tool_name", None), tool.name, sanitize_tool_name(tool.name)}:
                return tool
        available_names = sorted(getattr(t, "original_tool_name", t.name) for t in tools)
        raise RenderError(f"crewai_native renderer found no tool named {name!r} (has: {available_names})")

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def render(self, server: ServerHandle) -> list[ToolRep]:
        try:
            with self._versions():
                tools = self._tools(server)
            from crewai.utilities.agent_utils import convert_tools_to_openai_schema

            schemas, _functions, _mapping = convert_tools_to_openai_schema(tools)
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"crewai_native renderer failed: {exc}") from exc
        # MCPToolResolver keeps name, description and inputSchema only: annotations,
        # title, icons and outputSchema are never read off the listed tool.
        return [normalize_function_tool(schema, origin=self.id) for schema in schemas]

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Direct tool invocation",
                capture_api="crewai.tools.BaseTool.run on the MCPNativeTool Agent.get_mcp_tools returns",
                capture_object="the tool return value: the first content block's text, or a ToolFailure when isError is set",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Native tool call result",
                capture_api="crewai.utilities.agent_utils.execute_single_native_tool_call",
                capture_object="NativeToolCallResult.result, the tool message content the next model turn receives",
                agent_path=True,
                limitation=(
                    "The executor step is driven with a scripted OpenAI-style tool call. "
                    "No LLM was bound and no crew was kicked off."
                ),
            ),
        ]

    def render_result(
        self,
        server: ServerHandle,
        tool: str,
        arguments: dict[str, Any],
        boundary: str | None = None,
    ) -> ResultRep:
        boundary = boundary or self.DEFAULT_RESULT_BOUNDARY
        if boundary not in {"direct", "agent"}:
            raise RenderError(f"crewai_native renderer has no result boundary {boundary!r}")
        with self._versions():
            tools = self._tools(server)
            target = self._tool(tools, tool)
            try:
                if boundary == "direct":
                    return self._direct(tool, target, arguments)
                return self._agent(tool, tools, target, arguments)
            except RenderError:
                raise
            except Exception as exc:  # noqa: BLE001 - the failure mode is the observation
                return normalize_framework_result(
                    tool=tool,
                    origin=self.id,
                    is_error=True,
                    text=f"{type(exc).__name__}: {exc}",
                    error_channel="exception",
                    raw=repr(exc)[:2000],
                )

    def _direct(self, tool: str, target: Any, arguments: dict[str, Any]) -> ResultRep:
        from crewai.tools.tool_failure import ToolFailure

        returned = target.run(**arguments)
        if isinstance(returned, ToolFailure):
            # MCPNativeTool turns isError into a typed ToolFailure; only its message
            # (the first content block) is kept.
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=[{"type": "text", "text": returned.message}],
                is_error=True,
                error_channel="error_field",
                raw=returned.model_dump(mode="json"),
            )
        if isinstance(returned, str):
            return normalize_framework_result(
                tool=tool, origin=self.id, blocks=[{"type": "text", "text": returned}], raw=returned[:2000]
            )
        return normalize_framework_result(tool=tool, origin=self.id, text=str(returned), raw=repr(returned)[:2000])

    def _agent(self, tool: str, tools: list[Any], target: Any, arguments: dict[str, Any]) -> ResultRep:
        from crewai.utilities.agent_utils import convert_tools_to_openai_schema, execute_single_native_tool_call
        from crewai.utilities.string_utils import sanitize_tool_name

        _schemas, functions, _mapping = convert_tools_to_openai_schema(tools)
        call = {
            "id": _CALL_ID,
            "type": "function",
            "function": {"name": sanitize_tool_name(target.name), "arguments": json.dumps(arguments)},
        }
        outcome = execute_single_native_tool_call(
            call,
            available_functions=functions,
            original_tools=tools,
            structured_tools=None,
            tools_handler=None,
            agent=None,
            task=None,
            crew=None,
            event_source=self,
        )
        text = outcome.result
        # The tool message is plain text with no error field. A tool that raised reaches
        # the model as "Error executing tool: ..."; an isError result arrives as the
        # ToolFailure message alone, indistinguishable from a success.
        failed = isinstance(text, str) and text.startswith("Error executing tool")
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=[{"type": "text", "text": text}],
            is_error=True if failed else None,
            error_channel="message" if failed else None,
            raw=json.dumps(outcome.tool_message or {"result": text}, default=str)[:4000],
        )

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """Prompts through ``crewai.mcp.MCPClient``; it has no resource API at all."""

        from .. import probe as P

        no_resources = "crewai.mcp.MCPClient has list_tools, call_tool, list_prompts and get_prompt; no resource API"
        out: dict = {
            feature: {"ok": False, "unsupported": no_resources}
            for feature in P.FEATURES
            if feature.startswith("resources.")
        }

        async def _prompts() -> dict:
            async with self._mcp_client(server) as client:
                listed = await client.list_prompts()
                entries = [P.prompt_entry(p) for p in listed]
                out["prompts.list"] = {"ok": True, "entries": entries}
                first = entries[0] if entries else None
                if first is None:
                    out["prompts.get"] = {"ok": False, "error": "no prompt listed"}
                    return out
                arguments = {a["name"]: "x" for a in first["arguments"] if a.get("required")}
                try:
                    got = await client.get_prompt(first["name"], arguments)
                except Exception as exc:  # noqa: BLE001
                    out["prompts.get"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    return out
                # get_prompt returns {"name", "messages", "arguments"}; the
                # GetPromptResult description is dropped.
                out["prompts.get"] = {
                    "ok": True,
                    "description": got.get("description"),
                    "messages": [P.message_entry(m["role"], m["content"]) for m in got["messages"]],
                }
                return out

        with self._versions():
            try:
                asyncio.run(_prompts())
            except Exception as exc:  # noqa: BLE001
                for feature in ("prompts.list", "prompts.get"):
                    out.setdefault(feature, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return out

    # ------------------------------------------------------------------
    # Sampling, elicitation and roots
    # ------------------------------------------------------------------

    def probe_client_features(self, server: ServerHandle) -> dict:
        """MCPClient.connect builds ``ClientSession(read_stream, write_stream)`` with no
        sampling, elicitation, roots or message callback, and neither MCPClient nor the
        MCPServer* configs take one."""
        from .. import client_probe as C

        return C.without_hooks(lambda tool: self.render_result(server, tool, {}, "direct"))

    # ------------------------------------------------------------------
    # Server utilities
    # ------------------------------------------------------------------

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through Agent(mcps=...) and, where it has one, MCPClient.

        The agent route takes no progress or log handler. Logging level and completion
        are reachable only on the ``ClientSession`` behind ``MCPClient.session``,
        reported with ``route: "session"``.
        """
        import mcp.types as types

        from .. import probe as P

        out: dict = {}

        def call(feature: str, tool: str, arguments: dict | None = None) -> str | None:
            with self._versions():
                target = self._tool(self._tools(servers[feature]), tool)
                return _text(target.run(**(arguments or {})))

        if "tools.list" in servers:
            def _names():
                with self._versions():
                    tools = self._tools(servers["tools.list"])
                return {
                    "route": "adapter",
                    "api": "Agent.get_mcp_tools (MCPToolResolver over MCPClient.list_tools)",
                    "names": [getattr(t, "original_tool_name", t.name) for t in tools],
                }

            out["tools.list"] = P.attempt(_names)
        for feature, tool, reason in (
            ("progress", "slow_with_progress", "MCPClient.call_tool sends no progressToken and takes no progress callback"),
            ("logging", "emit_logs", "MCPClient builds its ClientSession without a logging_callback"),
        ):
            if feature in servers:
                answer = P.attempt(lambda feature=feature, tool=tool: {"text": call(feature, tool)})
                out[feature] = {"ok": False, "unsupported": reason, "text": answer.get("text"), "call": answer}

        def session_run(feature: str, body) -> dict:
            async def run():
                async with self._mcp_client(servers[feature]) as client:
                    return await body(client)

            with self._versions():
                return asyncio.run(run())

        if "set_level" in servers:
            async def _set_level(client):
                await client.session.set_logging_level(P.SET_LEVEL)
                text = await client.call_tool("emit_logs", {})
                return {"route": "session", "api": "crewai.mcp.MCPClient.session.set_logging_level",
                        "level": P.SET_LEVEL, "text": text}

            out["set_level"] = P.attempt(lambda: session_run("set_level", _set_level))
        if "completion" in servers:
            async def _completion(client):
                name, argument, value = P.COMPLETION_PROMPT
                prompt = await client.session.complete(
                    types.PromptReference(type="ref/prompt", name=name), {"name": argument, "value": value}
                )
                uri, argument, value = P.COMPLETION_RESOURCE
                resource = await client.session.complete(
                    types.ResourceTemplateReference(type="ref/resource", uri=uri), {"name": argument, "value": value}
                )
                return {"route": "session", "api": "crewai.mcp.MCPClient.session.complete",
                        "prompt": list(prompt.completion.values), "resource": list(resource.completion.values)}

            out["completion"] = P.attempt(lambda: session_run("completion", _completion))
        if "cancellation" in servers:
            async def _cancellation(client):
                loop = asyncio.get_running_loop()
                started = loop.time()
                task = asyncio.create_task(client.call_tool("cancellable_wait", {"seconds": P.CANCEL_WAIT_S}))
                await asyncio.sleep(P.CANCEL_AFTER_S)
                task.cancel()
                try:
                    outcome = f"returned: {await task}"
                except asyncio.CancelledError:
                    outcome = "cancelled"
                except Exception as exc:  # noqa: BLE001
                    outcome = f"threw: {type(exc).__name__}: {exc}"
                abandoned = loop.time() - started
                await asyncio.sleep(0.3)
                # MCPNativeTool.run, the agent's route, blocks in asyncio.run on a fresh
                # client with no cancel route; MCPClient.call_tool is the one awaitable.
                return {"route": "adapter", "mechanism": "asyncio task.cancel() around crewai.mcp.MCPClient.call_tool",
                        "abandoned_after_s": round(abandoned, 3), "outcome": outcome}

            out["cancellation"] = P.attempt(lambda: session_run("cancellation", _cancellation))
        if "tasks" in servers:
            out["tasks"] = P.attempt(lambda: {"route": "adapter", "text": call("tasks", "task_capable")})
        if "caching" in servers:
            def _caching():
                import time

                marks = []
                for step in ("first", "within_ttl", "after_ttl"):
                    if step == "after_ttl":
                        time.sleep(P.CACHE_PAUSE_S)
                    started = P.wall_clock()
                    # Each Agent.get_mcp_tools is a fresh MCPToolResolver listing; the
                    # config's cache_tools_list (default False) is a fixed 300 s client
                    # cache that never reads a server hint.
                    with self._versions():
                        count = len(self._tools(servers["caching"]))
                    marks.append({"call": step, "started": started, "finished": P.wall_clock(), "count": count})
                return {"route": "adapter", "api": "Agent.get_mcp_tools", "marks": marks}

            out["caching"] = P.attempt(_caching)
        return out

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through Agent(mcps=...).

        Neither the configs nor MCPClient take a message handler, so there is no hook to
        register. Tools are relisted through ``Agent.get_mcp_tools`` and called through
        the MCPNativeTools it returns; prompts are relisted through ``MCPClient.list_prompts``.
        Every one of those opens its own connection.
        """
        import anyio

        from .. import notify as N

        recorder = N.Recorder()

        async def tools():
            found = await anyio.to_thread.run_sync(lambda: self._tools(server))
            return [getattr(t, "original_tool_name", t.name) for t in found]

        async def prompts():
            async with self._mcp_client(server) as client:
                return [p["name"] for p in await client.list_prompts()]

        async def call(name):
            def run():
                return self._tool(self._tools(server), name).run()

            return await anyio.to_thread.run_sync(run)

        with self._versions():
            return anyio.run(
                N.run,
                N.Adapter(
                    call=call,
                    recorder=recorder,
                    hook=None,
                    lists={"tools": tools, "resources": None, "prompts": prompts},
                    no_list_reason="crewai.mcp.MCPClient has no resource API",
                    no_subscribe_reason="crewai.mcp.MCPClient has no resource or subscribe API",
                    negotiated=getattr(self, "_negotiated_mcp_spec_version", None),
                ),
            )


def _text(value: Any) -> str | None:
    if value is None:
        return None
    message = getattr(value, "message", None)
    if isinstance(message, str):
        return message
    return value if isinstance(value, str) else str(value)
