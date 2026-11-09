"""AutoGen renderer (DESIGN.md section 8).

Connects through ``autogen_ext.tools.mcp.McpWorkbench`` and runs one turn of a real
``autogen_agentchat.agents.AssistantAgent`` whose model client is AutoGen's own
``ReplayChatCompletionClient``, extended only to record what it is handed. No model is
called.

The capture boundary is the ``tools`` argument ``AssistantAgent`` passes to
``ChatCompletionClient.create``: the ``autogen_core.tools.ToolSchema`` dicts
``McpWorkbench.list_tools`` builds. That is AutoGen's framework tool type, before any
provider client (OpenAI, Anthropic, ...) converts it. ``McpWorkbench`` rebuilds each
schema from the MCP ``inputSchema``'s ``properties``, ``required`` and
``additionalProperties`` alone and keeps nothing else of the MCP tool.

AutoGen ships a second MCP route, ``mcp_server_tools``, which wraps each tool as a
``McpToolAdapter`` (a ``BaseTool``). Its results are captured as their own boundary;
its definitions are not the ones an ``AssistantAgent`` given ``workbench=`` sees.
"""

from __future__ import annotations

import asyncio
import json
from importlib import util
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version

ID = "autogen"
REQUIRES = ("autogen_ext.tools.mcp", "autogen_agentchat")

_CALL_ID = "mcp-mirror-call"
_TIMEOUT_S = 30
_MODEL_INFO = {
    "vision": True,
    "function_calling": True,
    "json_output": False,
    "family": "unknown",
    "structured_output": False,
}


def available() -> bool:
    return (
        util.find_spec("autogen_ext") is not None
        and util.find_spec("autogen_ext.tools.mcp") is not None
        and util.find_spec("autogen_agentchat") is not None
        and util.find_spec("mcp") is not None
    )


def get_renderer() -> "AutoGenRenderer":
    return AutoGenRenderer()


class AutoGenRenderer:
    id = ID

    def versions(self) -> dict:
        return {"framework": dist_version("autogen-agentchat"), "adapter": dist_version("autogen-ext")}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "autogen_ext.tools.mcp.McpWorkbench.list_tools, as called by "
                "autogen_agentchat.agents.AssistantAgent(workbench=...) for one turn"
            ),
            capture_object=(
                "autogen_core.tools.ToolSchema in the tools argument AssistantAgent "
                "passes to ChatCompletionClient.create"
            ),
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence="protocolVersion from McpWorkbench.initialize_result",
            limitation=(
                "The model client is autogen_ext.models.replay.ReplayChatCompletionClient; "
                "no provider client converted or serialized the ToolSchema."
            ),
        )

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def render(self, server: ServerHandle) -> list[ToolRep]:
        try:
            run = asyncio.run(self._run_agent(server, call=None))
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"autogen renderer failed: {_flatten(exc)}") from exc
        if not run["requests"]:
            raise RenderError("autogen agent run never reached the model client")
        return [self._to_rep(schema) for schema in run["requests"][0]["tools"]]

    def _to_rep(self, schema: dict) -> ToolRep:
        # ToolSchema carries name, description, parameters and an optional strict flag.
        # McpWorkbench drops the rest of the MCP tool (annotations, title, icons,
        # outputSchema) before the schema exists and keeps no copy, so nothing is
        # retained elsewhere either.
        payload = {
            "name": schema.get("name"),
            "description": schema.get("description"),
            "parameters": dict(schema.get("parameters") or {}),
        }
        return normalize_function_tool(payload, origin=self.id)

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    # An AssistantAgent given workbench= reaches a tool through McpWorkbench.call_tool
    # and hands the model FunctionExecutionResult(content=ToolResult.to_text()). The
    # older mcp_server_tools route hands back McpToolAdapter objects whose run_json
    # returns the MCP content blocks themselves, or raises when isError is set.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="workbench",
                label="Workbench tool result",
                capture_api="autogen_ext.tools.mcp.McpWorkbench.call_tool",
                capture_object="the autogen_core.tools.ToolResult it returns",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Function result to the model",
                capture_api=(
                    "autogen_agentchat.agents.AssistantAgent(workbench=McpWorkbench), "
                    "next ChatCompletionClient.create messages"
                ),
                capture_object=(
                    "the FunctionExecutionResult in the FunctionExecutionResultMessage "
                    "the model client receives"
                ),
                agent_path=True,
                limitation=(
                    "The model client is ReplayChatCompletionClient; no provider client "
                    "serialized the request."
                ),
            ),
            ResultBoundary(
                id="tool_adapter",
                label="mcp_server_tools adapter result",
                capture_api="autogen_ext.tools.mcp.mcp_server_tools, then McpToolAdapter.run_json",
                capture_object="the list of MCP content blocks run_json returns, or the exception it raises",
                agent_path=False,
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
        if boundary not in {"workbench", "agent", "tool_adapter"}:
            raise RenderError(f"autogen renderer has no result boundary {boundary!r}")
        try:
            if boundary == "workbench":
                return asyncio.run(self._workbench_result(server, tool, arguments))
            if boundary == "tool_adapter":
                return asyncio.run(self._adapter_result(server, tool, arguments))
            run = asyncio.run(self._run_agent(server, call=(tool, arguments)))
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001 - the failure mode is the observation
            return _exception_rep(self.id, tool, exc)
        if tool not in run["tools"]:
            raise RenderError(f"autogen renderer found no tool named {tool!r}")
        if "agent" not in run:
            raise RenderError("autogen agent run never returned a function result to the model")
        return self._agent_rep(tool, run["agent"])

    async def _workbench_result(self, server: ServerHandle, tool: str, arguments: dict) -> ResultRep:
        async with self._workbench(server) as workbench:
            names = [schema["name"] for schema in await workbench.list_tools()]
            if tool not in names:
                raise RenderError(f"autogen renderer found no tool named {tool!r}")
            result = await workbench.call_tool(tool, arguments)
        return self._tool_result_rep(tool, result)

    def _tool_result_rep(self, tool: str, result: Any) -> ResultRep:
        """``ToolResult`` holds only text and image parts.

        McpWorkbench turns an embedded resource into its JSON as text, and raises on
        any other kind (audio, resource link), which call_tool catches and returns as
        an is_error result whose text is the exception message.
        """
        from autogen_core.tools import ImageResultContent, TextResultContent

        blocks = []
        for part in result.result:
            if isinstance(part, TextResultContent):
                blocks.append({"type": "text", "text": part.content})
            elif isinstance(part, ImageResultContent):
                # autogen_core.Image is a decoded PIL image; the MIME type is gone.
                blocks.append({"type": "image"})
            else:
                blocks.append({"type": type(part).__name__})
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            is_error=bool(result.is_error),
            raw=_redact(result.model_dump(mode="json")),
        )

    async def _adapter_result(self, server: ServerHandle, tool: str, arguments: dict) -> ResultRep:
        from autogen_core import CancellationToken
        from autogen_ext.tools.mcp import mcp_server_tools

        tools = await mcp_server_tools(self._params(server))
        target = next((t for t in tools if t.name == tool), None)
        if target is None:
            raise RenderError(f"autogen renderer found no tool named {tool!r}")
        returned = await target.run_json(dict(arguments), CancellationToken())
        blocks = [
            item.model_dump(by_alias=True, exclude_none=True, mode="json") if hasattr(item, "model_dump") else {"type": "text", "text": str(item)}
            for item in (returned if isinstance(returned, list) else [returned])
        ]
        return normalize_framework_result(tool=tool, origin=self.id, blocks=blocks, raw=_redact(blocks))

    def _agent_rep(self, tool: str, result: dict) -> ResultRep:
        """What the model reads: one string, ``ToolResult.to_text()``, and a flag.

        Images arrive inside that string as ``[Image: <base64>]``; nothing else in the
        message tells the model a part was an image.
        """
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            is_error=bool(result.get("is_error")),
            text=str(result.get("content") or ""),
            raw=_redact(result),
        )

    # ------------------------------------------------------------------
    # The agent run the definitions and the agent boundary share
    # ------------------------------------------------------------------

    async def _run_agent(self, server: ServerHandle, call: tuple[str, dict] | None) -> dict:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_core import FunctionCall
        from autogen_core.models import CreateResult, FunctionExecutionResultMessage, RequestUsage

        captured: dict[str, Any] = {"requests": [], "tools": []}
        script: list[Any] = []
        if call is not None:
            script.append(
                CreateResult(
                    finish_reason="function_calls",
                    content=[FunctionCall(id=_CALL_ID, name=call[0], arguments=json.dumps(call[1]))],
                    usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                    cached=False,
                )
            )
        script.append("done")
        client = _recording_client(script, captured)

        async with self._workbench(server) as workbench:
            # reflect_on_tool_use sends the tool result back to the model client, which
            # is where the agent boundary is read.
            agent = AssistantAgent(
                "mcp_mirror", model_client=client, workbench=workbench, reflect_on_tool_use=call is not None
            )
            await agent.run(task="go")
            # McpSessionActor initializes its session on the first request, so the
            # result exists only once the agent has listed the tools.
            self._note_version(workbench)

        captured["tools"] = [t.get("name") for t in captured["requests"][0]["tools"]] if captured["requests"] else []
        for request in captured["requests"][1:]:
            for message in request["messages"]:
                if isinstance(message, FunctionExecutionResultMessage):
                    for result in message.content:
                        if result.call_id == _CALL_ID:
                            captured["agent"] = result.model_dump(mode="json")
        return captured

    def _note_version(self, workbench) -> None:
        version = getattr(workbench.initialize_result, "protocolVersion", None)
        if not version:
            raise RenderError("autogen renderer could not read McpWorkbench.initialize_result.protocolVersion")
        self._negotiated_mcp_spec_version = str(version)

    def _params(self, server: ServerHandle):
        from autogen_ext.tools.mcp import SseServerParams, StdioServerParams, StreamableHttpServerParams

        if server.transport == "http":
            return StreamableHttpServerParams(url=server.url, headers=dict(server.headers) or None, timeout=_TIMEOUT_S)
        if server.transport == "sse":
            return SseServerParams(url=server.url, headers=dict(server.headers) or None, timeout=_TIMEOUT_S)
        return StdioServerParams(command=server.command, args=list(server.args), read_timeout_seconds=_TIMEOUT_S)

    def _workbench(self, server: ServerHandle, **options):
        from autogen_ext.tools.mcp import McpWorkbench

        return McpWorkbench(self._params(server), **options)

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources, templates and prompts through McpWorkbench's own methods."""
        return asyncio.run(self._probe_async(server))

    async def _probe_async(self, server: ServerHandle) -> dict:
        from .. import probe as P

        out: dict = {}
        async with self._workbench(server) as workbench:
            async def _list():
                return {"entries": [P.resource_entry(r) for r in (await workbench.list_resources()).resources]}

            out["resources.list"] = await _attempt(_list)
            entries = out["resources.list"].get("entries") or []
            text_uri = next((e["uri"] for e in entries if (e.get("mimeType") or "").startswith("text/")), None)
            blob_uri = next((e["uri"] for e in entries if not (e.get("mimeType") or "").startswith("text/")), None)

            async def _read(uri):
                contents = (await workbench.read_resource(uri)).contents
                first = contents[0] if contents else None
                mime = getattr(first, "mimeType", None)
                blob = getattr(first, "blob", None)
                if blob is not None:
                    return P.blob_contents(str(first.uri), mime, blob)
                return P.text_contents(str(first.uri) if first else None, mime, getattr(first, "text", None))

            out["resources.read.text"] = (
                await _attempt(lambda: _read(text_uri)) if text_uri else {"ok": False, "error": "no text resource listed"}
            )
            out["resources.read.blob"] = (
                await _attempt(lambda: _read(blob_uri)) if blob_uri else {"ok": False, "error": "no binary resource listed"}
            )

            async def _templates():
                listed = await workbench.list_resource_templates()
                return {"entries": [P.template_entry(t) for t in listed.resourceTemplates]}

            out["resources.templates"] = await _attempt(_templates)

            async def _prompts():
                return {"entries": [P.prompt_entry(p) for p in (await workbench.list_prompts()).prompts]}

            out["prompts.list"] = await _attempt(_prompts)
            prompts = out["prompts.list"].get("entries") or []

            async def _get():
                first = prompts[0]
                arguments = {a["name"]: "x" for a in first["arguments"] if a.get("required")}
                result = await workbench.get_prompt(first["name"], arguments)
                return {
                    "description": result.description,
                    "messages": [P.message_entry(m.role, m.content) for m in result.messages],
                }

            out["prompts.get"] = await _attempt(_get) if prompts else {"ok": False, "error": "no prompt listed"}
        return out

    # ------------------------------------------------------------------
    # Client features
    # ------------------------------------------------------------------

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling through ``McpWorkbench(model_client=...)``; nothing else.

        The workbench answers ``sampling/createMessage`` by calling the model client it
        was given, here a ReplayChatCompletionClient scripted with the sampling
        sentinel. Its ClientSession is opened with no elicitation or roots callback and
        the workbench offers no parameter for either, so those tools are called with no
        hook registered.
        """
        return asyncio.run(self._probe_client_async(server))

    async def _probe_client_async(self, server: ServerHandle) -> dict:
        from .. import client_probe as C

        log = C.HandlerLog()

        async def call(tool: str, sampling: bool) -> ResultRep:
            options = {}
            if sampling:
                options["model_client"] = _sampling_client(log, C)
            async with self._workbench(server, **options) as workbench:
                result = await workbench.call_tool(tool, {})
            return self._tool_result_rep(tool, result)

        out: dict = {}
        for feature in ("sampling", "sampling.tools"):
            tool = C.FEATURES[feature][0]
            out[feature] = await C.attempt_async(
                feature, "McpWorkbench(model_client=...)", "adapter", lambda tool=tool: call(tool, True), log
            )
        for feature in ("elicitation.form", "elicitation.url", "roots.list", "roots.list_changed"):
            tool = C.FEATURES[feature][0]
            out[feature] = await C.attempt_async(feature, None, None, lambda tool=tool: call(tool, False), log)
        try:
            out["capabilities"] = {"text": C.rep_text(await call(C.CAPABILITIES_TOOL, True))[0]}
        except Exception as exc:  # noqa: BLE001
            out["capabilities"] = {"text": f"{type(exc).__name__}: {exc}"}
        return out

    # ------------------------------------------------------------------
    # Server utilities
    # ------------------------------------------------------------------

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through ``McpWorkbench``.

        The workbench takes server parameters, tool overrides and a model client. Its
        session is private to ``McpSessionActor`` and opened with no logging or message
        callback, so progress, logging, setLevel and completion have no route. Tools
        that show something only when called are still called, so the wire records what
        the workbench sends.
        """
        return asyncio.run(self._utilities_async(servers))

    async def _utilities_async(self, servers: dict[str, ServerHandle]) -> dict:
        from autogen_core import CancellationToken

        from .. import probe as P

        async def call(feature: str, tool: str, arguments: dict | None = None) -> str | None:
            async with self._workbench(servers[feature]) as workbench:
                return (await workbench.call_tool(tool, arguments or {})).to_text()

        out: dict = {}
        if "tools.list" in servers:
            async def _names():
                async with self._workbench(servers["tools.list"]) as workbench:
                    return {"route": "adapter", "names": [t["name"] for t in await workbench.list_tools()]}

            out["tools.list"] = await P.attempt_async(_names)
        for feature, tool, reason in (
            ("progress", "slow_with_progress", "McpWorkbench.call_tool takes no progress callback and sends no progressToken"),
            ("logging", "emit_logs", "McpWorkbench opens its ClientSession without a logging_callback"),
        ):
            if feature in servers:
                answer = await P.attempt_async(lambda feature=feature, tool=tool: _text(call(feature, tool)))
                out[feature] = {"ok": False, "unsupported": reason, "text": answer.get("text"), "call": answer}
        for feature, reason in (
            ("set_level", "McpWorkbench has no logging/setLevel API; its session is private to McpSessionActor"),
            ("completion", "McpWorkbench has no completion API; its session is private to McpSessionActor"),
        ):
            if feature in servers:
                out[feature] = {"ok": False, "unsupported": reason}
        if "cancellation" in servers:
            async def _cancellation():
                async with self._workbench(servers["cancellation"]) as workbench:
                    # McpSessionActor opens its session on the first request; open it
                    # first, as an agent's already is, so the cancel lands mid-call.
                    await workbench.list_tools()
                    loop = asyncio.get_running_loop()
                    token = CancellationToken()
                    started = loop.time()
                    task = asyncio.create_task(
                        workbench.call_tool("cancellable_wait", {"seconds": P.CANCEL_WAIT_S}, cancellation_token=token)
                    )
                    await asyncio.sleep(P.CANCEL_AFTER_S)
                    cancel_error = None
                    try:
                        token.cancel()
                    except Exception as exc:  # noqa: BLE001 - the linked callback's failure is the finding
                        cancel_error = f"{type(exc).__name__}: {exc}"
                    try:
                        result = await task
                        outcome = f"returned: {result.to_text()}" + (" (is_error)" if result.is_error else "")
                    except asyncio.CancelledError:
                        outcome = "cancelled"
                    except Exception as exc:  # noqa: BLE001
                        outcome = f"threw: {type(exc).__name__}: {exc}"
                    if cancel_error:
                        outcome += f"; CancellationToken.cancel() raised {cancel_error}"
                    abandoned = loop.time() - started
                    await asyncio.sleep(0.3)
                    return {"route": "adapter", "mechanism": "CancellationToken.cancel() passed to McpWorkbench.call_tool",
                            "abandoned_after_s": round(abandoned, 3), "outcome": outcome}

            out["cancellation"] = await P.attempt_async(_cancellation)
        if "tasks" in servers:
            out["tasks"] = await P.attempt_async(lambda: _routed_text(call("tasks", "task_capable")))
        if "caching" in servers:
            async def _caching():
                async with self._workbench(servers["caching"]) as workbench:
                    marks = []
                    for label in ("first", "within_ttl", "after_ttl"):
                        if label == "after_ttl":
                            await asyncio.sleep(P.CACHE_PAUSE_S)
                        started = P.wall_clock()
                        listed = await workbench.list_tools()
                        marks.append({"call": label, "started": started, "finished": P.wall_clock(), "count": len(listed)})
                    return {"route": "adapter", "api": "McpWorkbench.list_tools (defaults)", "marks": marks}

            out["caching"] = await P.attempt_async(_caching)
        return out

    # ------------------------------------------------------------------
    # Change notifications
    # ------------------------------------------------------------------

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through McpWorkbench.

        The workbench registers no message handler and takes none, and its session is
        private, so no hook can hear a notification and there is no subscribe route.
        Its list methods ask the server every time, which is what the agent reads.
        """
        return asyncio.run(self._notify_async(server))

    async def _notify_async(self, server: ServerHandle) -> dict:
        from .. import notify as N

        async with self._workbench(server) as workbench:

            async def tools():
                return [t["name"] for t in await workbench.list_tools()]

            await tools()
            negotiated = getattr(workbench.initialize_result, "protocolVersion", None)

            async def resources():
                return [str(r.uri) for r in (await workbench.list_resources()).resources]

            async def prompts():
                return [p.name for p in (await workbench.list_prompts()).prompts]

            async def call(name):
                return (await workbench.call_tool(name, {})).to_text()

            return await N.run(
                N.Adapter(
                    call=call,
                    recorder=N.Recorder(),
                    hook=None,
                    lists={"tools": tools, "resources": resources, "prompts": prompts},
                    no_subscribe_reason=(
                        "McpWorkbench has no subscribe API and its ClientSession is private to McpSessionActor"
                    ),
                    negotiated=str(negotiated) if negotiated else None,
                )
            )


def _recording_client(script: list[Any], captured: dict):
    """AutoGen's ReplayChatCompletionClient, recording each create() it is handed."""

    from autogen_ext.models.replay import ReplayChatCompletionClient

    class RecordingClient(ReplayChatCompletionClient):
        async def create(self, messages, *, tools=(), **kwargs):  # noqa: ANN001
            captured["requests"].append({"tools": [dict(t) for t in tools], "messages": list(messages)})
            return await super().create(messages, tools=tools, **kwargs)

    return RecordingClient(script, model_info=_MODEL_INFO)


def _sampling_client(log, C):  # noqa: ANN001, N803
    """A model client answering every sampling request with the sentinel text."""

    from autogen_ext.models.replay import ReplayChatCompletionClient

    class SamplingClient(ReplayChatCompletionClient):
        async def create(self, messages, *, tools=(), **kwargs):  # noqa: ANN001
            # McpSessionActor never forwards the request's tools, so this is always the
            # single-shot answer.
            log.note("model_client.create", tools=bool(tools), replied="endTurn")
            return await super().create(messages, tools=tools, **kwargs)

    return SamplingClient([C.SAMPLED] * 4, model_info=_MODEL_INFO)


async def _text(pending) -> dict:  # noqa: ANN001
    return {"text": await pending}


async def _routed_text(pending) -> dict:  # noqa: ANN001
    return {"route": "adapter", "text": await pending}


def _exception_rep(origin: str, tool: str, exc: BaseException) -> ResultRep:
    return normalize_framework_result(
        tool=tool,
        origin=origin,
        is_error=True,
        text=f"{type(exc).__name__}: {_flatten(exc)}",
        error_channel="exception",
        raw=repr(exc)[:2000],
    )


def _flatten(exc: BaseException) -> str:
    """The leaf messages of an exception group, which anyio wraps adapter errors in."""
    inner = getattr(exc, "exceptions", None)
    if inner:
        return "; ".join(_flatten(e) for e in inner)
    return str(exc)


def _redact(value: Any) -> Any:
    """Base64 payloads recorded by length, not content."""
    if isinstance(value, dict):
        return {
            k: (f"<{len(v)} chars>" if k in ("data", "blob") and isinstance(v, str) and len(v) > 32 else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


async def _attempt(fn):
    try:
        return {"ok": True, **(await fn())}
    except NotImplementedError as exc:
        return {"ok": False, "unsupported": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the failure is the observation
        return {"ok": False, "error": f"{type(exc).__name__}: {_flatten(exc)}"}
