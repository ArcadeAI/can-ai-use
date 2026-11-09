"""OpenAI Agents SDK renderer (DESIGN.md section 8).

Drives the SDK's real MCP conversion, ``MCPUtil.to_function_tool``, and captures the
resulting SDK ``FunctionTool``. This is the framework's model-tool object, not a
serialized provider request. The SDK keeps the MCP ``title`` (as ``_mcp_title``) on
that object but does not carry the behavioral hints (destructiveHint, readOnlyHint,
etc.).
"""

from __future__ import annotations

from importlib import util
from typing import Any

import anyio

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version

ID = "openai_agents"
# The distribution is ``openai-agents``; it imports as the top-level package ``agents``.
REQUIRES = ("agents", "agents.mcp")


def available() -> bool:
    return all(util.find_spec(mod) is not None for mod in REQUIRES)


def get_renderer() -> "OpenAIAgentsRenderer":
    return OpenAIAgentsRenderer()


class OpenAIAgentsRenderer:
    id = ID

    def versions(self) -> dict:
        version = dist_version("openai-agents")
        return {"framework": version, "adapter": version}

    def evidence(self) -> RendererEvidence:
        negotiated_version = getattr(
            self,
            "_negotiated_mcp_spec_version",
            None,
        )
        return RendererEvidence(
            capture_api="agents.mcp.util.MCPUtil.to_function_tool",
            capture_object="agents.tool.FunctionTool",
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=negotiated_version,
            protocol_version_evidence=(
                "protocol version exposed by the MCPServer client session"
                if negotiated_version and negotiated_version >= "2026-07-28"
                else "protocolVersion exposed by MCPServer.server_initialize_result"
            ),
            limitation=(
                "No Agents SDK model implementation or serialized provider request was captured."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        try:
            return anyio.run(self._render_async, server)
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"openai_agents renderer failed: {exc}") from exc

    async def _render_async(self, server: ServerHandle) -> list[ToolRep]:
        from agents.mcp.util import MCPUtil

        mcp_server = self._build_server(server)
        async with mcp_server:
            initialize_result = mcp_server.server_initialize_result
            session = getattr(mcp_server, "session", None)
            self._negotiated_mcp_spec_version = (
                getattr(initialize_result, "protocolVersion", None)
                or getattr(session, "protocol_version", None)
            )
            if not self._negotiated_mcp_spec_version:
                raise RenderError(
                    "openai_agents renderer could not establish the negotiated "
                    "MCP protocol version"
                )
            mcp_tools = await mcp_server.list_tools()
            reps: list[ToolRep] = []
            for tool in mcp_tools:
                function_tool = MCPUtil.to_function_tool(tool, mcp_server, False)
                reps.append(self._to_rep(function_tool))
            return reps

    # `to_function_tool` wires `invoke_mcp_tool` in as the FunctionTool's
    # `on_invoke_tool`, so the agent loop and the adapter helper run the same code.
    # Both are published so the equality is a stated measurement rather than an
    # assumption. `invoke_mcp_tool` does not raise on `isError`: it reads the flag
    # only to decide whether structuredContent may be preferred, and the error text
    # is emitted as an ordinary text output item.
    DEFAULT_RESULT_BOUNDARY = "direct"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Adapter invocation helper",
                capture_api="agents.mcp.util.MCPUtil.invoke_mcp_tool",
                capture_object="the Agents SDK ToolOutput",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Agent tool invocation",
                capture_api="agents.tool.FunctionTool.on_invoke_tool",
                capture_object="the ToolOutput the agent loop feeds back to the model",
                agent_path=True,
                limitation=(
                    "The FunctionTool callable is driven directly. No model "
                    "implementation was bound and no provider request was captured."
                ),
            ),
            ResultBoundary(
                id="agent_custom_data_extractor",
                label="Agent tool invocation, custom_data_extractor reading is_error",
                capture_api=(
                    "agents.tool.FunctionTool.on_invoke_tool, server built with "
                    "custom_data_extractor=lambda ctx: {'is_error': ctx.is_error}"
                ),
                capture_object="the ToolOutput plus the custom_data the run attaches to its ToolCallOutputItem",
                agent_path=False,
                limitation=(
                    "This is a documented non-default server option, not the "
                    "behaviour a caller gets without configuring it. The flag reaches "
                    "ToolCallOutputItem.custom_data only; the model and on_tool_end "
                    "hooks still receive the unflagged output."
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
        """Invoke one tool through the adapter and capture what the agent receives."""

        return anyio.run(
            self._render_result_async,
            server,
            tool,
            arguments,
            boundary or self.DEFAULT_RESULT_BOUNDARY,
        )

    async def _render_result_async(
        self, server: ServerHandle, tool: str, arguments: dict[str, Any], boundary: str
    ) -> ResultRep:
        import json as _json

        from agents.mcp.util import MCPUtil
        from agents.run_context import RunContextWrapper
        from agents.tool_context import ToolContext

        if boundary not in {"direct", "agent", "agent_custom_data_extractor"}:
            raise RenderError(
                f"openai_agents renderer has no result boundary {boundary!r}"
            )

        opt_in = boundary == "agent_custom_data_extractor"
        mcp_server = self._build_server(
            server,
            custom_data_extractor=(lambda ctx: {"is_error": ctx.is_error}) if opt_in else None,
        )
        async with mcp_server:
            mcp_tool = next(
                (t for t in await mcp_server.list_tools() if t.name == tool), None
            )
            if mcp_tool is None:
                raise RenderError(f"openai_agents renderer could not find tool {tool!r}")
            if opt_in:
                # The extractor's result lands on a ToolContext, which the run loop
                # copies onto the ToolCallOutputItem; a plain RunContextWrapper drops it.
                context = ToolContext(
                    context=None,
                    tool_name=tool,
                    tool_call_id="mcp-mirror-call",
                    tool_arguments=_json.dumps(arguments),
                )
            else:
                context = RunContextWrapper(context=None)
            try:
                if boundary == "direct":
                    output = await MCPUtil.invoke_mcp_tool(
                        mcp_server, mcp_tool, context, _json.dumps(arguments)
                    )
                else:
                    function_tool = MCPUtil.to_function_tool(mcp_tool, mcp_server, False)
                    output = await function_tool.on_invoke_tool(
                        context, _json.dumps(arguments)
                    )
            except Exception as exc:  # noqa: BLE001 - the failure mode is the observation
                # A transport or protocol failure surfaces here as an exception. An
                # `isError` result does not: it returns as ordinary tool output, which
                # is the finding this capture exists to record.
                return normalize_framework_result(
                    tool=tool,
                    origin=self.id,
                    is_error=True,
                    text=f"{type(exc).__name__}: {exc}",
                    error_channel="exception",
                    raw=repr(exc)[:2000],
                )
        if opt_in:
            custom_data = getattr(context, "_custom_data", None) or {}
            return self._result_to_rep(
                tool, output, is_error=bool(custom_data.get("is_error")), error_channel="custom_data"
            )
        return self._result_to_rep(tool, output)

    def _result_to_rep(
        self, tool: str, output: Any, is_error: bool | None = None, error_channel: str | None = None
    ) -> ResultRep:
        """Map the SDK's ToolOutput onto the common result shape.

        Every branch of `invoke_mcp_tool` produces text: structured content is
        `json.dumps`-ed into a string, and each non-text content block is re-serialized
        as JSON inside a `type: "text"` item. Nothing that reaches the model retains the
        kind it had on the wire, so every block is recorded as text.
        """

        flag = {"is_error": is_error, "error_channel": error_channel if is_error else None}
        if isinstance(output, str):
            return normalize_framework_result(
                tool=tool, origin=self.id, text=output, raw=output[:2000], **flag
            )
        items = output if isinstance(output, list) else [output]
        blocks = []
        for item in items:
            entry = item if isinstance(item, dict) else {}
            blocks.append({"type": entry.get("type", "text"), "text": entry.get("text")})
        return normalize_framework_result(
            tool=tool, origin=self.id, blocks=blocks, raw=repr(output)[:2000], **flag
        )

    def probe_client_features(self, server: ServerHandle) -> dict:
        """MCPServerStdio forwards only `message_handler` to its ClientSession, which
        sees notifications, not requests; there is no sampling, elicitation or roots
        hook to register."""
        from .. import client_probe as C

        return C.without_hooks(lambda tool: self.render_result(server, tool, {}, "direct"))

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through ``MCPServerStdio`` and its ``message_handler``.

        What is reachable only on the raw ``ClientSession`` the server object exposes as
        ``.session`` is reported with ``route: "session"``; progress, which needs a
        token the caller builds by hand in ``call_tool(meta=...)``, with ``"manual"``.
        """
        return anyio.run(self._utilities_async, servers)

    async def _utilities_async(self, servers: dict[str, ServerHandle]) -> dict:
        import asyncio

        import mcp.types as types
        from agents.mcp import MCPServerStdio, MCPServerStreamableHttp

        from .. import probe as P

        events: list[dict] = []
        messages: list[dict] = []

        async def on_message(message):
            root = getattr(message, "root", message)
            method = getattr(root, "method", None)
            params = getattr(root, "params", None)
            if method == "notifications/progress":
                events.append({"progress": getattr(params, "progress", None), "total": getattr(params, "total", None),
                               "message": getattr(params, "message", None)})
            elif method == "notifications/message":
                messages.append({"level": getattr(params, "level", None), "data": getattr(params, "data", None)})

        def build(feature: str):
            server = servers[feature]
            if server.transport == "http":
                params: dict[str, Any] = {"url": server.url}
                if server.headers:
                    params["headers"] = dict(server.headers)
                return MCPServerStreamableHttp(params=params, client_session_timeout_seconds=20, message_handler=on_message)
            return MCPServerStdio(params={"command": server.command, "args": list(server.args)},
                                  client_session_timeout_seconds=20, message_handler=on_message)

        async def call(feature: str, tool: str, arguments: dict | None = None, **kwargs):
            async with build(feature) as s:
                return P.result_text(await s.call_tool(tool, arguments or {}, **kwargs))

        out: dict = {}
        if "tools.list" in servers:
            async def _names():
                async with build("tools.list") as s:
                    return {"route": "adapter", "names": [t.name for t in await s.list_tools()]}

            out["tools.list"] = await P.attempt_async(_names)
        if "progress" in servers:
            async def _progress():
                text = await call("progress", "slow_with_progress", meta={"progressToken": "mcp-mirror-progress"})
                return {"route": "manual", "api": "MCPServer.call_tool(meta={'progressToken': ...}) + message_handler",
                        "events": list(events), "text": text}

            out["progress"] = await P.attempt_async(_progress)
        if "logging" in servers:
            async def _logging():
                text = await call("logging", "emit_logs")
                await asyncio.sleep(0.2)
                return {"route": "adapter", "api": "MCPServerStdio(message_handler=...)", "messages": list(messages), "text": text}

            out["logging"] = await P.attempt_async(_logging)
        if "set_level" in servers:
            async def _set_level():
                async with build("set_level") as s:
                    await s.session.set_logging_level(P.SET_LEVEL)
                    text = P.result_text(await s.call_tool("emit_logs", {}))
                return {"route": "session", "api": "MCPServer.session.set_logging_level", "level": P.SET_LEVEL, "text": text}

            out["set_level"] = await P.attempt_async(_set_level)
        if "completion" in servers:
            async def _completion():
                async with build("completion") as s:
                    name, argument, value = P.COMPLETION_PROMPT
                    prompt = await s.session.complete(types.PromptReference(type="ref/prompt", name=name), {"name": argument, "value": value})
                    uri, argument, value = P.COMPLETION_RESOURCE
                    resource = await s.session.complete(types.ResourceTemplateReference(type="ref/resource", uri=uri), {"name": argument, "value": value})
                    return {"route": "session", "api": "MCPServer.session.complete",
                            "prompt": list(prompt.completion.values), "resource": list(resource.completion.values)}

            out["completion"] = await P.attempt_async(_completion)
        if "cancellation" in servers:
            async def _cancellation():
                async with build("cancellation") as s:
                    loop = asyncio.get_running_loop()
                    started = loop.time()
                    task = asyncio.create_task(s.call_tool("cancellable_wait", {"seconds": P.CANCEL_WAIT_S}))
                    await asyncio.sleep(P.CANCEL_AFTER_S)
                    task.cancel()
                    try:
                        outcome = f"returned: {P.result_text(await task)}"
                    except asyncio.CancelledError:
                        outcome = "cancelled"
                    except Exception as exc:  # noqa: BLE001
                        outcome = f"threw: {type(exc).__name__}: {exc}"
                    abandoned = loop.time() - started
                    await asyncio.sleep(0.3)
                    return {"route": "adapter", "mechanism": "asyncio task.cancel() around MCPServer.call_tool",
                            "abandoned_after_s": round(abandoned, 3), "outcome": outcome}

            out["cancellation"] = await P.attempt_async(_cancellation)
        if "tasks" in servers:
            async def _tasks():
                return {"route": "adapter", "text": await call("tasks", "task_capable")}

            out["tasks"] = await P.attempt_async(_tasks)
        if "caching" in servers:
            async def _caching():
                async with build("caching") as s:
                    marks = []
                    for label in ("first", "within_ttl", "after_ttl"):
                        if label == "after_ttl":
                            await asyncio.sleep(P.CACHE_PAUSE_S)
                        started = P.wall_clock()
                        listed = await s.list_tools()
                        marks.append({"call": label, "started": started, "finished": P.wall_clock(), "count": len(listed)})
                    return {"route": "adapter", "api": "MCPServer.list_tools (defaults)", "marks": marks}

            out["caching"] = await P.attempt_async(_caching)
        return out

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources and prompts through the SDK's MCPServer methods, normalized."""
        return anyio.run(self._probe_async, server)

    async def _probe_async(self, server: ServerHandle) -> dict:
        from ..source import _probe_with

        mcp_server = self._build_server(server)
        async with mcp_server:
            return await _probe_with(mcp_server, {})

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through MCPServer.

        ``message_handler`` is the documented hook, handed every session message. The
        server is built with the defaults an agent gets, so ``cache_tools_list`` is off
        and ``list_tools`` asks the server each time. There is no subscribe API; the
        subscribe route is the ClientSession the server object exposes as ``session``.
        """
        return anyio.run(self._notify_async, server)

    async def _notify_async(self, server: ServerHandle) -> dict:
        from .. import notify as N

        recorder = N.Recorder()
        mcp_server = self._build_server(server, message_handler=recorder.message_handler)
        async with mcp_server, anyio.create_task_group() as group:
            session = mcp_server.session
            negotiated = getattr(mcp_server.server_initialize_result, "protocolVersion", None) or getattr(
                session, "protocol_version", None
            )
            subscriptions = N.SessionSubscriptions(session, recorder, negotiated, group)

            async def tools():
                return [t.name for t in await mcp_server.list_tools()]

            async def resources():
                return [str(r.uri) for r in (await mcp_server.list_resources()).resources]

            async def prompts():
                return [p.name for p in (await mcp_server.list_prompts()).prompts]

            out = await N.run(
                N.Adapter(
                    call=lambda name: mcp_server.call_tool(name, {}),
                    recorder=recorder,
                    hook="MCPServerStdio(message_handler=...)",
                    lists={"tools": tools, "resources": resources, "prompts": prompts},
                    subscribe=subscriptions.subscribe,
                    unsubscribe=subscriptions.unsubscribe,
                    subscribe_route="session",
                    subscribe_api=f"MCPServerStdio.session: {subscriptions.api}",
                    listen=subscriptions.listen_all,
                    listen_route="session",
                    listen_api="mcp.client.subscriptions.listen(MCPServerStdio.session, ...)",
                    negotiated=negotiated,
                )
            )
            group.cancel_scope.cancel()
        return out

    def _build_server(self, server: ServerHandle, message_handler=None, custom_data_extractor=None):
        from agents.mcp import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp

        extra: dict[str, Any] = {}
        if message_handler is not None:
            extra["message_handler"] = message_handler
        if custom_data_extractor is not None:
            extra["custom_data_extractor"] = custom_data_extractor

        if server.transport == "http":
            params: dict[str, Any] = {"url": server.url}
            if server.headers:
                params["headers"] = dict(server.headers)
            return MCPServerStreamableHttp(params=params, client_session_timeout_seconds=20, **extra)
        if server.transport == "sse":
            return MCPServerSse(params={"url": server.url}, client_session_timeout_seconds=20, **extra)
        return MCPServerStdio(
            params={"command": server.command, "args": list(server.args)},
            client_session_timeout_seconds=20,
            **extra,
        )

    def _to_rep(self, function_tool: Any) -> ToolRep:
        # The SDK keeps only the MCP title on the captured FunctionTool; the
        # behavioral hints are not carried on that object.
        retained: dict[str, Any] = {}
        title = getattr(function_tool, "_mcp_title", None)
        if title:
            retained["title"] = title
        framework_metadata = {"annotations": retained} if retained else {}
        payload = {
            "name": getattr(function_tool, "name", None),
            "description": getattr(function_tool, "description", None),
            "parameters": getattr(function_tool, "params_json_schema", None) or {},
        }
        return normalize_function_tool(payload, origin=self.id, framework_metadata=framework_metadata)
