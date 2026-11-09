"""Agno renderer (DESIGN.md section 8).

Connects through ``agno.tools.mcp.MCPTools``, the toolkit an Agno ``Agent`` is handed.
Entering it builds a ``fastmcp.Client`` over stdio, Streamable HTTP or HTTP+SSE and
calls ``build_tools``, which turns each MCP tool into an ``agno.tools.Function`` with
the tool's name, description and input schema. The toolkit is built with
``protocol_mode="auto"``, Agno's documented opt-in for the newest era both sides
support; the default, ``"legacy"``, stays on the 2025-11-25 handshake.

The capture is the ``tools`` list a real ``agno.agent.Agent`` run hands its model's
``ainvoke``: ``Model._format_tools`` turns each Function into ``{"type": "function",
"function": Function.to_dict()}``. Agno ships no offline model, so the model is a
scripted ``agno.models.base.Model`` subclass that records what it is handed and never
serializes a request.

Tool results have two boundaries. The Function's entrypoint returns the ``ToolResult``
Agno builds from the CallToolResult; the agent run then puts that result into the
``Message(role="tool")`` the next model turn receives.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from functools import lru_cache
from importlib import util
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version

ID = "agno"
REQUIRES = ("agno.tools.mcp", "agno.agent", "fastmcp")

_CALL_ID = "mcp-mirror-call"
_TIMEOUT_SECONDS = 30


def available() -> bool:
    try:
        return all(util.find_spec(mod) is not None for mod in REQUIRES)
    except ModuleNotFoundError:
        return False


def get_renderer() -> "AgnoRenderer":
    return AgnoRenderer()


@lru_cache(maxsize=1)
def _scripted_model_class():
    """A ``Model`` that answers from a script instead of a provider.

    ``responses(messages, tools)`` is called once per model turn with the messages and
    formatted tools the agent run hands ``ainvoke``, and returns the ``ModelResponse``
    for that turn. No request is built, so nothing is serialized or sent.
    """

    from dataclasses import dataclass
    from typing import Callable, Optional

    from agno.models.base import Model

    @dataclass
    class ScriptedModel(Model):
        id: str = "mcp-mirror-scripted"
        name: str = "ScriptedModel"
        provider: str = "mcp-mirror"
        responses: Optional[Callable[..., Any]] = None

        async def ainvoke(self, messages=None, assistant_message=None, response_format=None, tools=None, **_kwargs):
            return self.responses(list(messages or []), list(tools or []))

        def invoke(self, *args, **kwargs):
            raise NotImplementedError("ScriptedModel runs through Agent.arun only")

        def invoke_stream(self, *args, **kwargs):
            raise NotImplementedError("ScriptedModel runs through Agent.arun only")

        async def ainvoke_stream(self, *args, **kwargs):
            raise NotImplementedError("ScriptedModel runs through Agent.arun only")
            yield  # pragma: no cover - makes this an async generator

        def _parse_provider_response(self, response, **kwargs):
            return response

        def _parse_provider_response_delta(self, response):
            return response

    return ScriptedModel


class AgnoRenderer:
    id = ID

    def versions(self) -> dict:
        return {"framework": dist_version("agno"), "adapter": dist_version("agno")}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "agno.tools.mcp.MCPTools(protocol_mode='auto') handed to agno.agent.Agent; "
                "the tools argument of the model's ainvoke on the first turn"
            ),
            capture_object=(
                "function-tool dicts Model._format_tools builds from each Function: "
                "{'type': 'function', 'function': Function.to_dict()}"
            ),
            capture_stage="provider_format",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "fastmcp.Client.protocol_version on MCPTools.session after the toolkit "
                "connected; protocol_mode='auto' negotiates the newest era, the default "
                "'legacy' keeps 2025-11-25"
            ),
            limitation=(
                "The model is a scripted agno.models.base.Model subclass; no provider "
                "model class serialized a request."
            ),
        )

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def _toolkit(self, handle: ServerHandle, **options: Any):
        from agno.tools.mcp import MCPTools

        options.setdefault("timeout_seconds", _TIMEOUT_SECONDS)
        options.setdefault("protocol_mode", "auto")
        if handle.transport in {"http", "sse"}:
            return MCPTools(
                url=handle.url,
                transport="sse" if handle.transport == "sse" else "streamable-http",
                headers=dict(handle.headers) or None,
                **options,
            )
        from mcp import StdioServerParameters

        return MCPTools(
            transport="stdio",
            server_params=StdioServerParameters(command=handle.command, args=list(handle.args)),
            **options,
        )

    @asynccontextmanager
    async def _connected(self, handle: ServerHandle, **options: Any):
        """An entered MCPTools, with the version its fastmcp Client negotiated recorded."""

        async with self._toolkit(handle, **options) as toolkit:
            version = getattr(toolkit.session, "protocol_version", None)
            if not version:
                raise RenderError("agno renderer could not read the negotiated MCP protocol version")
            self._negotiated_mcp_spec_version = str(version)
            yield toolkit

    async def _run_agent(self, toolkit: Any, script) -> list[dict]:
        """One ``Agent.arun`` with the scripted model; returns what each turn was handed."""

        from agno.agent import Agent

        turns: list[dict] = []

        def responses(messages, tools):
            turns.append({"messages": messages, "tools": tools})
            return script(len(turns))

        model = _scripted_model_class()(responses=responses)
        agent = Agent(model=model, tools=[toolkit], telemetry=False)
        await agent.arun("go")
        return turns

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def render(self, server: ServerHandle) -> list[ToolRep]:
        try:
            return asyncio.run(self._render_async(server))
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"agno renderer failed: {exc}") from exc

    async def _render_async(self, server: ServerHandle) -> list[ToolRep]:
        from agno.models.response import ModelResponse

        async with self._connected(server) as toolkit:
            turns = await self._run_agent(toolkit, lambda _turn: ModelResponse(role="assistant", content="done"))
        if not turns:
            raise RenderError("agno agent run never called the model")
        # build_tools passes Function only name, description and the input schema. The
        # MCP tool's annotations, title, icons and outputSchema are not kept on the
        # Function or the toolkit, so nothing is passed as retained metadata.
        return [normalize_function_tool(tool, origin=self.id) for tool in turns[0]["tools"]]

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="the agno.tools.Function entrypoint MCPTools.build_tools registers",
                capture_object="the agno.tools.function.ToolResult (content, images, audios, metadata) built from the CallToolResult",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Tool message to the model",
                capture_api="agno.agent.Agent.arun, the messages of the next model ainvoke",
                capture_object="the agno.models.message.Message(role='tool') the next model turn receives",
                agent_path=True,
                limitation=(
                    "The model is a scripted agno.models.base.Model subclass; no "
                    "provider model class serialized the message."
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
            raise RenderError(f"agno renderer has no result boundary {boundary!r}")
        runner = self._direct if boundary == "direct" else self._agent
        try:
            return asyncio.run(runner(server, tool, arguments))
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

    @staticmethod
    def _function(toolkit: Any, tool: str) -> Any:
        if tool not in toolkit.functions:
            raise RenderError(f"agno renderer found no tool named {tool!r} (has: {sorted(toolkit.functions)})")
        return toolkit.functions[tool]

    async def _direct(self, server: ServerHandle, tool: str, arguments: dict[str, Any]) -> ResultRep:
        async with self._connected(server) as toolkit:
            result = await self._function(toolkit, tool).entrypoint(**arguments)
        return self._tool_result_rep(tool, result)

    def _tool_result_rep(self, tool: str, result: Any) -> ResultRep:
        """A ToolResult as the caller holds it.

        ``content`` is one string. A CallToolResult with ``isError`` arrives as
        ``"Error from MCP tool '<name>': <repr of the content list>"`` with no flag;
        images and audio become media artifacts plus a placeholder sentence in the text.
        ``metadata`` keeps the result's ``meta`` and ``structuredContent``.
        """

        metadata = getattr(result, "metadata", None) or {}
        content = getattr(result, "content", None)
        blocks = [{"type": "text", "text": content}] if content else []
        blocks += _media_blocks(getattr(result, "images", None), getattr(result, "audios", None))
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            structured_content=metadata.get("structured_content"),
            text=content if isinstance(content, str) else None,
            raw=_raw(result),
        )

    async def _agent(self, server: ServerHandle, tool: str, arguments: dict[str, Any]) -> ResultRep:
        from agno.models.response import ModelResponse

        def script(turn: int):
            if turn == 1:
                return ModelResponse(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": _CALL_ID,
                            "type": "function",
                            "function": {"name": tool, "arguments": json.dumps(arguments)},
                        }
                    ],
                )
            return ModelResponse(role="assistant", content="done")

        async with self._connected(server) as toolkit:
            self._function(toolkit, tool)
            turns = await self._run_agent(toolkit, script)
        history = [m for turn in turns[1:2] for m in turn["messages"]]
        positions = [i for i, m in enumerate(history) if m.role == "tool"]
        if not positions:
            raise RenderError("agno agent run never returned a tool message to the model")
        message = history[positions[-1]]
        # Model._handle_function_call_media moves a result's images and audio off the
        # tool message into a user message right after it ("The tool call above
        # generated the attached media."), which is how the media reaches the model.
        media = [m for m in history[positions[-1] + 1 :] if m.role == "user"]
        content = message.content if isinstance(message.content, str) else (
            None if message.content is None else json.dumps(message.content, default=str)
        )
        blocks = [{"type": "text", "text": content}] if content else []
        blocks += _media_blocks(message.images, message.audio)
        for follow_up in media:
            blocks += _media_blocks(follow_up.images, follow_up.audio)
        # The message carries the ToolResult's content, not its metadata, and
        # tool_call_error is set only when the call raised; an isError result is a
        # successful call whose text starts "Error from MCP tool".
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            is_error=bool(message.tool_call_error),
            text=content,
            raw=json.dumps([_raw(m) for m in [message, *media]])[:4000],
        )

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """MCPTools exposes tools and nothing else; that is the finding."""
        from .. import probe as P

        return P.unsupported(
            "agno.tools.mcp.MCPTools registers tools as Functions and has no resource or prompt API"
        )

    # ------------------------------------------------------------------
    # Sampling, elicitation and roots
    # ------------------------------------------------------------------

    def probe_client_features(self, server: ServerHandle) -> dict:
        """MCPTools builds its fastmcp.Client with a transport, a timeout and a protocol
        mode only: no sampling, elicitation or roots handler, and no option to pass one."""
        from .. import client_probe as C

        return C.without_hooks(lambda tool: self.render_result(server, tool, {}, "direct"))

    # ------------------------------------------------------------------
    # Server utilities
    # ------------------------------------------------------------------

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through MCPTools and the Functions it registers.

        The toolkit takes no progress or log handler and builds its fastmcp.Client
        without one. Logging level and completion are reachable only on that client,
        which the toolkit exposes as ``MCPTools.session``, reported with
        ``route: "session"``.
        """
        return asyncio.run(self._utilities_async(servers))

    async def _utilities_async(self, servers: dict[str, ServerHandle]) -> dict:
        import mcp.types as types

        from .. import probe as P

        out: dict = {}

        async def call(feature: str, tool: str, arguments: dict | None = None) -> str | None:
            async with self._connected(servers[feature]) as toolkit:
                return getattr(await self._function(toolkit, tool).entrypoint(**(arguments or {})), "content", None)

        if "tools.list" in servers:
            async def _names():
                async with self._connected(servers["tools.list"]) as toolkit:
                    return {"route": "adapter", "api": "MCPTools.functions", "names": list(toolkit.functions)}

            out["tools.list"] = await P.attempt_async(_names)
        for feature, tool, reason in (
            ("progress", "slow_with_progress", "MCPTools takes no progress handler and builds its fastmcp.Client without one"),
            ("logging", "emit_logs", "MCPTools takes no log handler; its fastmcp.Client uses fastmcp's default, which writes to Python logging only"),
        ):
            if feature in servers:
                answer = await P.attempt_async(lambda feature=feature, tool=tool: _wrap(call(feature, tool)))
                out[feature] = {"ok": False, "unsupported": reason, "text": answer.get("text"), "call": answer}
        if "set_level" in servers:
            async def _set_level():
                async with self._connected(servers["set_level"]) as toolkit:
                    await toolkit.session.set_logging_level(P.SET_LEVEL)
                    result = await self._function(toolkit, "emit_logs").entrypoint()
                    return {"route": "session", "api": "MCPTools.session.set_logging_level (fastmcp Client)",
                            "level": P.SET_LEVEL, "text": result.content}

            out["set_level"] = await P.attempt_async(_set_level)
        if "completion" in servers:
            async def _completion():
                async with self._connected(servers["completion"]) as toolkit:
                    name, argument, value = P.COMPLETION_PROMPT
                    prompt = await toolkit.session.complete(
                        types.PromptReference(type="ref/prompt", name=name), {"name": argument, "value": value}
                    )
                    uri, argument, value = P.COMPLETION_RESOURCE
                    resource = await toolkit.session.complete(
                        types.ResourceTemplateReference(type="ref/resource", uri=uri), {"name": argument, "value": value}
                    )
                    return {"route": "session", "api": "MCPTools.session.complete (fastmcp Client)",
                            "prompt": list(prompt.values), "resource": list(resource.values)}

            out["completion"] = await P.attempt_async(_completion)
        if "cancellation" in servers:
            async def _cancellation():
                async with self._connected(servers["cancellation"]) as toolkit:
                    function = self._function(toolkit, "cancellable_wait")
                    loop = asyncio.get_running_loop()
                    started = loop.time()
                    task = asyncio.create_task(function.entrypoint(seconds=P.CANCEL_WAIT_S))
                    await asyncio.sleep(P.CANCEL_AFTER_S)
                    task.cancel()
                    try:
                        outcome = f"returned: {(await task).content}"
                    except asyncio.CancelledError:
                        outcome = "cancelled"
                    except Exception as exc:  # noqa: BLE001
                        outcome = f"threw: {type(exc).__name__}: {exc}"
                    abandoned = loop.time() - started
                    await asyncio.sleep(0.3)
                    return {"route": "adapter", "mechanism": "asyncio task.cancel() around the Function entrypoint",
                            "abandoned_after_s": round(abandoned, 3), "outcome": outcome}

            out["cancellation"] = await P.attempt_async(_cancellation)
        if "tasks" in servers:
            out["tasks"] = await P.attempt_async(lambda: _wrap(call("tasks", "task_capable"), route="adapter"))
        if "caching" in servers:
            async def _caching():
                async with self._connected(servers["caching"]) as toolkit:
                    marks = []
                    for step in ("first", "within_ttl", "after_ttl"):
                        if step == "after_ttl":
                            await asyncio.sleep(P.CACHE_PAUSE_S)
                        started = P.wall_clock()
                        # build_tools is the relist an Agent run makes when the toolkit
                        # has refresh_connection=True; connecting lists once.
                        await toolkit.build_tools()
                        marks.append({"call": step, "started": started, "finished": P.wall_clock(),
                                      "count": len(toolkit.functions)})
                    return {"route": "adapter", "api": "MCPTools.build_tools", "marks": marks}

            out["caching"] = await P.attempt_async(_caching)
        return out

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through MCPTools.

        MCPTools takes no message handler and builds its fastmcp.Client without one, so
        the probe has no hook to register. It relists tools through
        ``MCPTools.build_tools``, the relist an Agent run makes when the toolkit has
        ``refresh_connection=True``, and reads the toolkit's Functions. The toolkit has
        no resource, prompt or subscribe API, so subscriptions go through the SDK
        session under the fastmcp Client it exposes as ``MCPTools.session``.
        """
        return asyncio.run(self._notify_async(server))

    async def _notify_async(self, server: ServerHandle) -> dict:
        import anyio

        from .. import notify as N

        recorder = N.Recorder()
        async with self._connected(server) as toolkit, anyio.create_task_group() as group:
            negotiated = self._negotiated_mcp_spec_version
            subscriptions = N.SessionSubscriptions(toolkit.session.session, recorder, negotiated, group)

            async def tools():
                await toolkit.build_tools()
                return list(toolkit.functions)

            async def call(name):
                return await self._function(toolkit, name).entrypoint()

            out = await N.run(
                N.Adapter(
                    call=call,
                    recorder=recorder,
                    hook=None,
                    lists={"tools": tools, "resources": None, "prompts": None},
                    no_list_reason="MCPTools has no resource or prompt API",
                    subscribe=subscriptions.subscribe,
                    unsubscribe=subscriptions.unsubscribe,
                    subscribe_route="session",
                    subscribe_api=f"MCPTools.session.session: {subscriptions.api}",
                    listen=subscriptions.listen_all,
                    listen_route="session",
                    listen_api="mcp.client.subscriptions.listen(MCPTools.session.session, ...)",
                    negotiated=negotiated,
                )
            )
            group.cancel_scope.cancel()
        return out


def _media_blocks(images: Any, audios: Any) -> list[dict]:
    """Agno media artifacts, recorded by kind; the bytes are summarized, not kept."""

    blocks: list[dict] = []
    for kind, items in (("image", images), ("audio", audios)):
        for item in items or []:
            data = getattr(item, "content", None)
            blocks.append(
                {
                    "type": kind,
                    "mimeType": getattr(item, "mime_type", None),
                    "data": f"<{len(data)} bytes>" if isinstance(data, (bytes, bytearray)) else None,
                }
            )
    return blocks


def _raw(value: Any) -> str:
    """A repr with media bytes left out, so the raw capture stays readable."""

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(exclude={"images", "audios", "audio", "videos", "files"}, exclude_none=True)
            return json.dumps(dumped, default=str)[:2000]
        except Exception:  # noqa: BLE001
            pass
    return repr(value)[:2000]


async def _wrap(awaitable, route: str | None = None) -> dict:
    text = await awaitable
    return {"route": route, "text": text} if route else {"text": text}
