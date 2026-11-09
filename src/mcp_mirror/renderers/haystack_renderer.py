"""Haystack renderer (DESIGN.md section 8).

Connects through ``mcp-haystack``: an ``MCPToolset`` built from a ``StdioServerInfo``,
``StreamableHttpServerInfo`` or ``SSEServerInfo``. ``MCPToolset.warm_up`` (which
``Agent.warm_up`` calls) opens one ``ClientSession`` on the adapter's background event
loop, lists the tools once and wraps each in a ``haystack.tools.Tool`` with the tool's
name, description and ``inputSchema``.

The capture is the ``tools`` a real ``haystack.components.agents.Agent`` run hands its
chat generator, read as ``Tool.tool_spec``: the ``{name, description, parameters}``
dict every Haystack chat generator serializes (``OpenAIChatGenerator`` wraps it as
``{"type": "function", "function": tool_spec}``). The generator is Haystack's own
``MockChatGenerator``, so no request is built or sent.

The adapter calls ``ClientSession.initialize()`` and keeps no record of the result, so
the negotiated version is read from that call.

Tool results have two boundaries. The Tool's function returns the CallToolResult as a
JSON string (``model_dump_json``), and raises ``MCPInvocationError`` for an ``isError``
result; the Agent's tool step then puts that string, or the error text, into the
``ChatMessage`` with a ``ToolCallResult`` the next generator turn receives.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from importlib import util
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version

ID = "haystack"
REQUIRES = ("haystack_integrations.tools.mcp", "haystack.components.agents")

_TIMEOUT_SECONDS = 30


def available() -> bool:
    try:
        return all(util.find_spec(mod) is not None for mod in REQUIRES)
    except ModuleNotFoundError:
        return False


def get_renderer() -> "HaystackRenderer":
    return HaystackRenderer()


class HaystackRenderer:
    id = ID

    def versions(self) -> dict:
        return {"framework": dist_version("haystack-ai"), "adapter": dist_version("mcp-haystack")}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "haystack_integrations.tools.mcp.MCPToolset handed to "
                "haystack.components.agents.Agent; the tools argument of the chat "
                "generator's run on the first turn"
            ),
            capture_object="haystack.tools.Tool.tool_spec ({name, description, parameters}) for each Tool MCPToolset.warm_up builds",
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "protocol version returned by the ClientSession.initialize that "
                "MCPClient._initialize_session_with_transport calls; the adapter keeps "
                "no record of it"
            ),
            limitation=(
                "The chat generator is haystack's MockChatGenerator; no provider chat "
                "generator serialized a request."
            ),
        )

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def _server_info(self, handle: ServerHandle):
        from haystack_integrations.tools.mcp import SSEServerInfo, StdioServerInfo, StreamableHttpServerInfo

        headers = dict(handle.headers) or None
        if handle.transport == "sse":
            return SSEServerInfo(url=handle.url, headers=headers, timeout=_TIMEOUT_SECONDS)
        if handle.transport == "http":
            return StreamableHttpServerInfo(url=handle.url, headers=headers, timeout=_TIMEOUT_SECONDS)
        return StdioServerInfo(command=handle.command, args=list(handle.args))

    def _build_toolset(self, handle: ServerHandle, **options: Any):
        from haystack_integrations.tools.mcp import MCPToolset

        options.setdefault("connection_timeout", _TIMEOUT_SECONDS)
        options.setdefault("invocation_timeout", _TIMEOUT_SECONDS)
        return MCPToolset(server_info=self._server_info(handle), **options)

    @contextmanager
    def _toolset(self, handle: ServerHandle, **options: Any):
        """A warmed-up MCPToolset, with the version its session negotiated recorded."""

        toolset = self._build_toolset(handle, **options)
        try:
            with self._versions_seen() as seen:
                toolset.warm_up()
            observed = set(seen)
            if len(observed) != 1:
                raise RenderError(
                    "haystack renderer could not establish one negotiated MCP protocol "
                    f"version; observed {sorted(observed)}"
                )
            self._negotiated_mcp_spec_version = observed.pop()
            yield toolset
        finally:
            toolset.close()

    @contextmanager
    def _versions_seen(self):
        """Record the version every ``ClientSession.initialize`` returns meanwhile.

        The adapter's sessions run on its ``AsyncExecutor`` thread; patching the class
        reaches them there.
        """

        from mcp.client.session import ClientSession

        seen: list[str] = []
        original = ClientSession.initialize

        async def capture(session, *args, **kwargs):
            result = await original(session, *args, **kwargs)
            version = getattr(result, "protocol_version", None) or getattr(result, "protocolVersion", None)
            if version:
                seen.append(str(version))
            return result

        ClientSession.initialize = capture
        try:
            yield seen
        finally:
            ClientSession.initialize = original

    @staticmethod
    def _tool(toolset: Any, tool: str) -> Any:
        tools = {t.name: t for t in toolset}
        if tool not in tools:
            raise RenderError(f"haystack renderer found no tool named {tool!r} (has: {sorted(tools)})")
        return tools[tool]

    def _run_agent(self, toolset: Any, script) -> list[dict]:
        """One ``Agent.run`` with MockChatGenerator; returns what each turn was handed."""

        from haystack.components.agents import Agent
        from haystack.components.generators.chat import MockChatGenerator
        from haystack.dataclasses import ChatMessage
        from haystack.tools import flatten_tools_or_toolsets

        turns: list[dict] = []

        def respond(messages, tools):
            turns.append({"messages": list(messages), "tools": flatten_tools_or_toolsets(tools)})
            return script(len(turns))

        agent = Agent(chat_generator=MockChatGenerator(response_fn=respond), tools=toolset)
        agent.warm_up()
        agent.run(messages=[ChatMessage.from_user("go")])
        return turns

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def render(self, server: ServerHandle) -> list[ToolRep]:
        try:
            with self._toolset(server) as toolset:
                turns = self._run_agent(toolset, lambda _turn: "done")
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"haystack renderer failed: {exc}") from exc
        if not turns:
            raise RenderError("haystack agent run never called the chat generator")
        # MCPToolset builds Tool(name, description, parameters=inputSchema, function=...).
        # The MCP tool's annotations, title, icons and outputSchema are not kept on the
        # Tool or the toolset, so nothing is passed as retained metadata.
        return [normalize_function_tool(tool.tool_spec, origin=self.id) for tool in turns[0]["tools"]]

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="haystack.tools.Tool.invoke on a Tool MCPToolset.warm_up builds",
                capture_object=(
                    "the JSON string MCPClient.call_tool returns (CallToolResult.model_dump_json), "
                    "or the ToolInvocationError Tool.invoke raises around the "
                    "MCPInvocationError for an isError result"
                ),
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Tool message to the model",
                capture_api="haystack.components.agents.Agent.run, the messages of the next chat generator run",
                capture_object="the ChatMessage carrying a ToolCallResult that the next generator turn receives",
                agent_path=True,
                limitation=(
                    "The chat generator is haystack's MockChatGenerator; no provider "
                    "chat generator serialized the message."
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
            raise RenderError(f"haystack renderer has no result boundary {boundary!r}")
        runner = self._direct if boundary == "direct" else self._agent
        try:
            return runner(server, tool, arguments)
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

    def _direct(self, server: ServerHandle, tool: str, arguments: dict[str, Any], **options: Any) -> ResultRep:
        with self._toolset(server, **options) as toolset:
            result = self._tool(toolset, tool).invoke(**arguments)
        return self._string_rep(tool, result)

    def _string_rep(self, tool: str, result: Any) -> ResultRep:
        """The string a Tool returns, read the way MCPTool's docstring tells callers to.

        ``MCPClient.call_tool`` returns ``CallToolResult.model_dump_json(by_alias=True)``
        and documents ``json.loads()`` as how to read it, so the blocks and
        ``structuredContent`` are recovered from that JSON. An ``isError`` result never
        gets here: ``call_tool`` raises ``MCPInvocationError`` for it.
        """

        if not isinstance(result, str):
            return normalize_framework_result(tool=tool, origin=self.id, text=repr(result), raw=repr(result)[:2000])
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return normalize_framework_result(tool=tool, origin=self.id, text=result, raw=result[:2000])
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=list(parsed.get("content") or []),
            structured_content=parsed.get("structuredContent"),
            is_error=parsed.get("isError"),
            raw=result[:2000],
        )

    def _agent(self, server: ServerHandle, tool: str, arguments: dict[str, Any]) -> ResultRep:
        from haystack.dataclasses import ChatMessage, ToolCall

        def script(turn: int):
            if turn == 1:
                return ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name=tool, arguments=dict(arguments))])
            return "done"

        with self._toolset(server) as toolset:
            self._tool(toolset, tool)
            turns = self._run_agent(toolset, script)
        messages = [m for turn in turns[1:2] for m in turn["messages"] if m.tool_call_results]
        if not messages:
            raise RenderError("haystack agent run never returned a tool result to the chat generator")
        message = messages[-1]
        results = message.tool_call_results
        # ToolCallResult.result is the Tool's return value passed through
        # _result_to_string: here the CallToolResult JSON as one string. For an isError
        # result the tool step catches MCPInvocationError and sends str(error) with
        # error=True instead.
        blocks = [_result_block(r.result) for r in results]
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            is_error=True if any(r.error for r in results) else None,
            raw=json.dumps(message.to_dict(), default=str)[:4000],
        )

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """MCPToolset exposes tools and nothing else; that is the finding."""
        from .. import probe as P

        return P.unsupported(
            "haystack_integrations.tools.mcp.MCPToolset wraps tools as haystack Tools and "
            "has no resource or prompt API"
        )

    # ------------------------------------------------------------------
    # Sampling, elicitation and roots
    # ------------------------------------------------------------------

    def probe_client_features(self, server: ServerHandle) -> dict:
        """MCPClient builds ``ClientSession(read, write)`` with no sampling, elicitation,
        roots or message callback, and neither the server info nor the toolset takes one."""
        from .. import client_probe as C

        return C.without_hooks(lambda tool: self.render_result(server, tool, {}, "direct"))

    # ------------------------------------------------------------------
    # Server utilities
    # ------------------------------------------------------------------

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through MCPToolset and the Tools it builds.

        The toolset takes no progress or log handler, and its ClientSession is built
        without one. Logging level and completion would need that session, which lives
        on the private ``MCPToolset._worker._client``; the adapter hands no session out.
        Cancellation is the adapter's ``invocation_timeout``, which cancels the call's
        coroutine on its event loop when it expires.
        """
        return asyncio.run(self._utilities_async(servers))

    async def _utilities_async(self, servers: dict[str, ServerHandle]) -> dict:
        from .. import probe as P

        out: dict = {}

        def call(feature: str, tool: str, arguments: dict | None = None) -> str | None:
            with self._toolset(servers[feature]) as toolset:
                return P.result_text(_parsed(self._tool(toolset, tool).invoke(**(arguments or {}))))

        if "tools.list" in servers:
            async def _names():
                with self._toolset(servers["tools.list"]) as toolset:
                    return {"route": "adapter", "api": "MCPToolset.warm_up", "names": [t.name for t in toolset]}

            out["tools.list"] = await P.attempt_async(_names)
        for feature, tool, reason in (
            ("progress", "slow_with_progress", "MCPToolset takes no progress handler and MCPClient.call_tool passes none to the session"),
            ("logging", "emit_logs", "MCPClient builds its ClientSession without a logging_callback, and the toolset takes no log handler"),
        ):
            if feature in servers:
                answer = await P.attempt_async(lambda feature=feature, tool=tool: _wrap(asyncio.to_thread(call, feature, tool)))
                out[feature] = {"ok": False, "unsupported": reason, "text": answer.get("text"), "call": answer}
        for feature, reason in (
            ("set_level", "MCPToolset has no logging/setLevel API; its ClientSession is private to MCPToolset._worker._client"),
            ("completion", "MCPToolset has no completion API; its ClientSession is private to MCPToolset._worker._client"),
        ):
            if feature in servers:
                out[feature] = {"ok": False, "unsupported": reason}
        if "cancellation" in servers:
            async def _cancellation():
                loop = asyncio.get_running_loop()

                def run() -> str:
                    with self._toolset(servers["cancellation"], invocation_timeout=P.CANCEL_AFTER_S) as toolset:
                        target = self._tool(toolset, "cancellable_wait")
                        started = loop.time()
                        try:
                            outcome = f"returned: {P.result_text(_parsed(target.invoke(seconds=P.CANCEL_WAIT_S)))}"
                        except Exception as exc:  # noqa: BLE001
                            outcome = f"threw: {type(exc).__name__}: {exc}"
                        abandoned = loop.time() - started
                        # Leave the session open long enough for a cancel notification to land.
                        import time

                        time.sleep(0.3)
                        return outcome, abandoned

                outcome, abandoned = await asyncio.to_thread(run)
                return {
                    "route": "adapter",
                    "mechanism": "MCPToolset(invocation_timeout=...): AsyncExecutor.run cancels the call_tool future when it expires",
                    "abandoned_after_s": round(abandoned, 3),
                    "outcome": outcome,
                }

            out["cancellation"] = await P.attempt_async(_cancellation)
        if "tasks" in servers:
            out["tasks"] = await P.attempt_async(lambda: _wrap(asyncio.to_thread(call, "tasks", "task_capable"), route="adapter"))
        if "caching" in servers:
            async def _caching():
                def run() -> list[dict]:
                    with self._toolset(servers["caching"]) as toolset:
                        marks = []
                        for step in ("first", "within_ttl", "after_ttl"):
                            if step == "after_ttl":
                                import time

                                time.sleep(P.CACHE_PAUSE_S)
                            started = P.wall_clock()
                            # warm_up is what Agent.warm_up calls before every run; after
                            # the first it returns without listing again.
                            toolset.warm_up()
                            marks.append({"call": step, "started": started, "finished": P.wall_clock(),
                                          "count": len(list(toolset))})
                        return marks

                return {"route": "adapter", "api": "MCPToolset.warm_up", "marks": await asyncio.to_thread(run)}

            out["caching"] = await P.attempt_async(_caching)
        return out

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through MCPToolset.

        MCPClient builds its ClientSession without a message handler, so the probe has
        no hook to register. The toolset lists tools once in ``warm_up`` and has no
        relist, resource, prompt or subscribe API, so the probe can only call the
        fixture's tools through the Tools it built.
        """
        from .. import notify as N

        with self._toolset(server) as toolset:
            negotiated = self._negotiated_mcp_spec_version

            async def run() -> dict:
                async def call(name):
                    return _parsed(await asyncio.to_thread(self._tool(toolset, name).invoke))

                return await N.run(
                    N.Adapter(
                        call=call,
                        recorder=N.Recorder(),
                        hook=None,
                        lists={"tools": None, "resources": None, "prompts": None},
                        no_list_reason="MCPToolset lists tools once in warm_up and has no relist, resource or prompt API",
                        no_subscribe_reason="MCPToolset has no subscribe API and hands out no session",
                        negotiated=negotiated,
                    )
                )

            return asyncio.run(run())


def _parsed(result: Any) -> Any:
    """A Tool's JSON-string return value as the dict ``json.loads`` gives the caller."""

    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


def _result_block(result: Any) -> dict:
    """One ToolCallResult.result as the model is handed it."""

    if isinstance(result, str):
        return {"type": "text", "text": result}
    # outputs_to_string with raw_result can hand ImageContent/TextContent through; the
    # toolset is built without it, so this is only a fallback.
    kind = type(result).__name__
    return {"type": kind, "text": str(result)[:2000]}


async def _wrap(awaitable, route: str | None = None) -> dict:
    text = await awaitable
    return {"route": route, "text": text} if route else {"text": text}
