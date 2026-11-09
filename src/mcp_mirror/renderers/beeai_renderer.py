"""BeeAI Framework renderer (DESIGN.md section 8).

BeeAI's MCP support is ``beeai_framework.tools.mcp.MCPTool``. Its documented entry
point, ``MCPTool.from_client(client)``, takes an MCP SDK transport context manager
(``stdio_client``, ``streamablehttp_client`` or ``sse_client``) the caller builds.
``MCPSessionProvider`` then opens one ``ClientSession`` on that transport with no
callbacks, calls ``list_tools()`` once and wraps every listed tool in an ``MCPTool``.
``from_client`` also accepts a ``ClientSession`` the caller already holds, in which case
the tools ride on that session.

The capture is the ``tools`` list a real ``RequirementAgent`` run hands LiteLLM.
``LiteLLMChatModel._transform_input`` builds one ``{"type": "function", ...}`` dict per
tool from ``tool.input_schema``, which ``MCPTool`` rebuilds as a Pydantic model from the
MCP input schema (``JSONSchemaModel.create``). With the ChatModel default
``use_strict_tool_schema=True`` the parameters are ``to_strict_json_schema`` of that
model. The model is an ``OpenAIChatModel`` whose ``litellm.acompletion`` is replaced by
a script that records the kwargs and answers with a tool call; nothing is sent.

Tool results have two boundaries. ``direct`` is the ``JSONToolOutput`` that
``MCPTool.run`` returns, or the ``ToolError`` it raises for an ``isError`` result.
``agent`` is the tool message the ``RequirementAgent`` writes into the next LiteLLM call.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager, contextmanager
from importlib import util
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version

ID = "beeai"
REQUIRES = ("beeai_framework", "mcp", "litellm")

_TIMEOUT_SECONDS = 30
_CONNECT_TIMEOUT = 45
_FINAL_ANSWER = "final_answer"
_MODEL_ID = "gpt-4o-mini"
# OpenAIChatModel asserts an api_key setting at construction; the scripted acompletion
# never sends it anywhere.
_UNUSED_API_KEY = "mcp-mirror-unused"
# The opening of RequirementAgentToolErrorPrompt (beeai_framework.agents.requirement.prompts).
_TOOL_ERROR_MARKER = "The tool has failed;"
_SESSION_NOTE = "on a caller-built mcp.ClientSession handed to MCPTool.from_client, which BeeAI uses as given"


def available() -> bool:
    try:
        return all(util.find_spec(mod) is not None for mod in REQUIRES)
    except ModuleNotFoundError:
        return False


def get_renderer() -> "BeeAIRenderer":
    return BeeAIRenderer()


def _tool_call(name: str, arguments: dict[str, Any], turn: int):
    import litellm

    return litellm.ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"mcp-mirror-call-{turn}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
            }
        ],
        model=_MODEL_ID,
    )


class BeeAIRenderer:
    id = ID

    def versions(self) -> dict:
        return {"framework": dist_version("beeai-framework"), "adapter": dist_version("beeai-framework")}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "beeai_framework.tools.mcp.MCPTool.from_client handed to "
                "beeai_framework.agents.requirement.RequirementAgent; the tools "
                "LiteLLMChatModel._transform_input passes litellm.acompletion on the first turn"
            ),
            capture_object=(
                "function-tool dicts {'type': 'function', 'function': {name, description, "
                "parameters, strict}} built from each MCPTool's input_schema, with the "
                "ChatModel default use_strict_tool_schema=True"
            ),
            capture_stage="provider_format",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "protocolVersion instrumented from ClientSession.initialize inside the "
                "MCPSessionProvider that MCPTool.from_client starts; BeeAI does not expose it"
            ),
            limitation=(
                "litellm.acompletion is replaced by a script; no provider request was "
                "serialized. Other agents (ReActAgent, ToolCallingAgent) and other ChatModel "
                "adapters were not captured."
            ),
        )

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def _transport(self, handle: ServerHandle, auth: Any = None):
        """The MCP SDK transport context manager a BeeAI caller hands from_client."""

        headers = dict(handle.headers) or None
        if handle.transport == "sse":
            from mcp.client.sse import sse_client

            return sse_client(handle.url, headers=headers, timeout=_TIMEOUT_SECONDS, auth=auth)
        if handle.transport == "http":
            from mcp.client.streamable_http import streamablehttp_client

            return streamablehttp_client(handle.url, headers=headers, timeout=_TIMEOUT_SECONDS, auth=auth)
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        return stdio_client(StdioServerParameters(command=handle.command, args=list(handle.args)))

    @asynccontextmanager
    async def _tools(self, handle: ServerHandle):
        """``MCPTool.from_client`` over the transport, with its negotiated version recorded.

        On exit the session provider is released the way ``MCPTool.__del__`` does it,
        and the background task that holds the session is given time to close it.
        """

        from unittest.mock import patch

        from beeai_framework.tools.mcp import MCPTool
        from beeai_framework.tools.mcp.utils.session_provider import MCPSessionProvider
        from mcp.client.session import ClientSession

        versions: list[str] = []
        original_initialize = ClientSession.initialize

        async def capture_initialize(session, *args, **kwargs):
            result = await original_initialize(session, *args, **kwargs)
            versions.append(str(result.protocolVersion))
            return result

        with patch.object(ClientSession, "initialize", capture_initialize):
            try:
                # MCPSessionProvider opens ClientSession(read, write) with no read timeout,
                # so an initialize the transport never answers (a failed OAuth flow) waits
                # forever and from_client never returns. The bound turns that hang into an error.
                tools = await asyncio.wait_for(MCPTool.from_client(self._transport(handle)), timeout=_CONNECT_TIMEOUT)
            except asyncio.TimeoutError as exc:
                raise RenderError(
                    f"beeai MCPTool.from_client did not return within {_CONNECT_TIMEOUT}s "
                    "(MCPSessionProvider's ClientSession has no read timeout, so an unanswered initialize never fails)"
                ) from exc
        if len(set(versions)) != 1:
            raise RenderError(f"beeai renderer could not establish one negotiated MCP protocol version (observed: {versions})")
        self._negotiated_mcp_spec_version = versions[0]
        session = tools[0]._session if tools else None
        provider = MCPSessionProvider._instances.get(session) if session is not None else None
        try:
            yield {tool.name: tool for tool in tools}
        finally:
            if provider is not None:
                provider.refs = 1
                provider.destroy()
                for _ in range(100):
                    if provider._session is None:
                        break
                    await asyncio.sleep(0.05)

    @asynccontextmanager
    async def _session(self, handle: ServerHandle, **callbacks: Any):
        """A caller-built ClientSession, entered and initialized, for routes BeeAI takes as given."""

        from datetime import timedelta

        from mcp.client.session import ClientSession

        async with self._transport(handle, auth=callbacks.pop("auth", None)) as (read, write, *_):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=_TIMEOUT_SECONDS), **callbacks) as session:
                result = await session.initialize()
                self._negotiated_mcp_spec_version = str(result.protocolVersion)
                yield session

    @staticmethod
    async def _session_tools(session: Any) -> dict[str, Any]:
        from beeai_framework.tools.mcp import MCPTool

        return {tool.name: tool for tool in await MCPTool.from_client(session)}

    @staticmethod
    def _tool(tools: dict[str, Any], name: str) -> Any:
        if name not in tools:
            raise RenderError(f"beeai renderer found no tool named {name!r} (has: {sorted(tools)})")
        return tools[name]

    @contextmanager
    def _scripted(self, script):
        """Replace the litellm.acompletion BeeAI's LiteLLM adapter calls with ``script``.

        ``script(turn)`` returns the ModelResponse for that turn. Every call's kwargs,
        exactly as ``LiteLLMChatModel._create`` built them, are recorded.
        """

        from unittest.mock import patch

        import beeai_framework.adapters.litellm.chat as litellm_chat

        turns: list[dict] = []

        async def acompletion(**kwargs):
            turns.append(kwargs)
            return script(len(turns))

        with patch.object(litellm_chat, "acompletion", acompletion):
            yield turns

    async def _run_agent(self, tools: list[Any], script) -> list[dict]:
        from beeai_framework.adapters.openai import OpenAIChatModel
        from beeai_framework.agents.requirement import RequirementAgent

        with self._scripted(script) as turns:
            agent = RequirementAgent(llm=OpenAIChatModel(_MODEL_ID, api_key=_UNUSED_API_KEY), tools=tools)
            await agent.run("Call the tool once, then answer.")
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
            raise RenderError(f"beeai renderer failed: {exc}") from exc

    async def _render_async(self, server: ServerHandle) -> list[ToolRep]:
        async with self._tools(server) as tools:
            turns = await self._run_agent(
                list(tools.values()), lambda turn: _tool_call(_FINAL_ANSWER, {"response": "done"}, turn)
            )
        if not turns:
            raise RenderError("beeai agent run never called the model")
        # RequirementAgent adds its own final_answer tool; it is not from MCP. MCPTool
        # exposes the tool's name, description and input schema; the MCP Tool it wraps is
        # kept only on the private MCPTool._tool, so annotations, title, icons and the
        # outputSchema are not recorded as retained metadata.
        return [
            normalize_function_tool(tool, origin=self.id)
            for tool in turns[0].get("tools") or []
            if tool["function"]["name"] != _FINAL_ANSWER
        ]

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="beeai_framework.tools.mcp.MCPTool.run on a tool MCPTool.from_client returned",
                capture_object=(
                    "JSONToolOutput.result (structuredContent when present, else each content "
                    "block with JSON text parsed, one block unwrapped), or the ToolError "
                    "raised for an isError result"
                ),
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Tool message to the model",
                capture_api="beeai_framework.agents.requirement.RequirementAgent.run; the messages of the next litellm.acompletion call",
                capture_object=(
                    "the role 'tool' message LiteLLMChatModel._transform_input builds from "
                    "JSONToolOutput.get_text_content(), or from the tool_error template"
                ),
                agent_path=True,
                limitation="litellm.acompletion is replaced by a script; no provider request was serialized.",
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
        runners = {"direct": self._direct, "agent": self._agent}
        if boundary not in runners:
            raise RenderError(f"beeai renderer has no result boundary {boundary!r}")
        try:
            return asyncio.run(runners[boundary](server, tool, arguments))
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001 - the failure mode is the observation
            return self._raised(tool, exc)

    def _raised(self, tool: str, exc: BaseException) -> ResultRep:
        # MCPTool._run raises ToolError(<the result JSON>) for an isError result; Tool.run
        # wraps whatever is raised in a ToolError. That exception is the only error signal.
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            is_error=True,
            text=f"{type(exc).__name__}: {exc}",
            error_channel="exception",
            raw=repr(exc)[:2000],
        )

    async def _direct(self, server: ServerHandle, tool: str, arguments: dict[str, Any]) -> ResultRep:
        async with self._tools(server) as tools:
            function = self._tool(tools, tool)
            try:
                output = await function.run(arguments)
            except Exception as exc:  # noqa: BLE001
                return self._raised(tool, exc)
        return _output_rep(tool, self.id, output)

    async def _agent(self, server: ServerHandle, tool: str, arguments: dict[str, Any]) -> ResultRep:
        def script(turn: int):
            if turn == 1:
                return _tool_call(tool, arguments, turn)
            return _tool_call(_FINAL_ANSWER, {"response": "done"}, turn)

        async with self._tools(server) as tools:
            self._tool(tools, tool)
            turns = await self._run_agent(list(tools.values()), script)
        if len(turns) < 2:
            raise RenderError("beeai agent run never returned a tool message to the model")
        replies = [m for m in turns[1]["messages"] if m.get("role") == "tool" and m.get("name") == tool]
        if not replies:
            raise RenderError("beeai agent run carried no tool message for the call")
        text = "\n".join(str(m.get("content")) for m in replies)
        # The tool message is one string: get_text_content() is JSON of the output, so
        # every content block (images and audio as base64) reaches the model as text. A
        # raised ToolError is written through the tool_error template instead, which is
        # the only failure signal the model gets; the message carries no error field.
        failed = text.startswith(_TOOL_ERROR_MARKER)
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=[{"type": "text", "text": text}],
            is_error=True if failed else None,
            text=text,
            error_channel="message" if failed else None,
            raw=json.dumps(replies, default=str)[:4000],
        )

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """BeeAI's MCP client converts tools and nothing else; that is the finding."""
        from .. import probe as P

        return P.unsupported(
            "beeai_framework.tools.mcp has MCPTool only: from_client lists and wraps tools, "
            "and BeeAI has no MCP resource or prompt API"
        )

    # ------------------------------------------------------------------
    # Sampling, elicitation and roots
    # ------------------------------------------------------------------

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Hooks on a caller-built ClientSession, exercised through MCPTool.run.

        BeeAI takes no sampling, elicitation or roots handler: the session
        ``MCPSessionProvider`` opens for a transport has no callbacks. ``from_client``
        also takes a ``ClientSession`` as given, so the handlers are registered there, on
        the MCP SDK's session, and every tool is called through ``MCPTool.run``. That is
        recorded as the ``underlying`` route.
        """
        return asyncio.run(self._client_features_async(server))

    async def _client_features_async(self, server: ServerHandle) -> dict:
        import mcp.types as types

        from .. import client_probe as C

        log = C.HandlerLog()
        sdk = C.sdk_callbacks(log)

        async def call(tool: str, callbacks: dict | None = None, before=None) -> ResultRep:
            async with self._session(server, **(callbacks or {})) as session:
                if before is not None:
                    await before(session)
                function = self._tool(await self._session_tools(session), tool)
                try:
                    return _output_rep(tool, self.id, await function.run({}))
                except Exception as exc:  # noqa: BLE001
                    return self._raised(tool, exc)

        sampling = {"sampling_callback": sdk["sampling_callback"]}
        sampling_tools = {
            **sampling,
            "sampling_capabilities": types.SamplingCapability(tools=types.SamplingToolsCapability()),
        }
        elicit = {"elicitation_callback": sdk["elicitation_callback"]}
        roots = {"list_roots_callback": sdk["list_roots_callback"]}

        async def changed_roots(session: Any) -> None:
            await session.send_roots_list_changed()

        hooks = {
            "sampling": ("ask_sampling", "ClientSession(sampling_callback=...)", sampling, None),
            "sampling.tools": (
                "ask_sampling_tools", "ClientSession(sampling_callback=..., sampling_capabilities=...)", sampling_tools, None,
            ),
            "elicitation.form": ("ask_elicitation_form", "ClientSession(elicitation_callback=...)", elicit, None),
            "elicitation.url": ("ask_elicitation_url", "ClientSession(elicitation_callback=...)", elicit, None),
            "roots.list": ("list_client_roots", "ClientSession(list_roots_callback=...)", roots, None),
            "roots.list_changed": (
                "roots_changes_seen", "ClientSession(list_roots_callback=...).send_roots_list_changed()", roots, changed_roots,
            ),
        }
        out: dict = {}
        for feature, (tool, hook, callbacks, before) in hooks.items():
            out[feature] = await C.attempt_async(
                feature, f"{hook} {_SESSION_NOTE}", "underlying",
                lambda tool=tool, callbacks=callbacks, before=before: call(tool, callbacks, before), log,
            )
        try:
            rep = await call(C.CAPABILITIES_TOOL, {**sampling_tools, **elicit, **roots})
            out["capabilities"] = {"text": C.rep_text(rep)[0]}
        except Exception as exc:  # noqa: BLE001
            out["capabilities"] = {"text": f"{type(exc).__name__}: {exc}"}
        return out

    # ------------------------------------------------------------------
    # Server utilities
    # ------------------------------------------------------------------

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through MCPTool and, where only it reaches, the caller's session.

        ``MCPTool._run`` calls ``session.call_tool(name, arguments)`` and nothing else, so
        it passes no progress callback. Log messages, logging/setLevel and completion are
        reachable only on a ClientSession the caller builds and hands ``from_client``,
        reported with ``route: "session"``.
        """
        return asyncio.run(self._utilities_async(servers))

    async def _utilities_async(self, servers: dict[str, ServerHandle]) -> dict:
        import mcp.types as types

        from .. import probe as P

        out: dict = {}

        async def call(feature: str, tool: str, arguments: dict | None = None) -> Any:
            async with self._tools(servers[feature]) as tools:
                return await self._tool(tools, tool).run(arguments or {})

        if "tools.list" in servers:
            async def _names():
                async with self._tools(servers["tools.list"]) as tools:
                    return {"route": "adapter", "api": "MCPTool.from_client", "names": list(tools)}

            out["tools.list"] = await P.attempt_async(_names)
        if "progress" in servers:
            answer = await P.attempt_async(lambda: _wrap(call("progress", "slow_with_progress")))
            out["progress"] = {
                "ok": False,
                "unsupported": "MCPTool._run calls session.call_tool(name, arguments) with no progress_callback, and takes none",
                "text": answer.get("text"),
                "call": answer,
            }
        if "logging" in servers:
            async def _logging():
                messages: list[dict] = []

                async def on_log(params) -> None:  # noqa: ANN001
                    messages.append({"level": str(getattr(params, "level", None)), "data": getattr(params, "data", None)})

                async with self._session(servers["logging"], logging_callback=on_log) as session:
                    await session.set_logging_level("debug")
                    output = await self._tool(await self._session_tools(session), "emit_logs").run({})
                return {"route": "session", "api": f"ClientSession(logging_callback=...) {_SESSION_NOTE}",
                        "messages": messages, "text": _output_text(output)}

            out["logging"] = await P.attempt_async(_logging)
        if "set_level" in servers:
            async def _set_level():
                async with self._session(servers["set_level"]) as session:
                    await session.set_logging_level(P.SET_LEVEL)
                    output = await self._tool(await self._session_tools(session), "emit_logs").run({})
                return {"route": "session", "api": f"ClientSession.set_logging_level {_SESSION_NOTE}",
                        "level": P.SET_LEVEL, "text": _output_text(output)}

            out["set_level"] = await P.attempt_async(_set_level)
        if "completion" in servers:
            async def _completion():
                async with self._session(servers["completion"]) as session:
                    name, argument, value = P.COMPLETION_PROMPT
                    prompt = await session.complete(
                        types.PromptReference(type="ref/prompt", name=name), {"name": argument, "value": value}
                    )
                    uri, argument, value = P.COMPLETION_RESOURCE
                    resource = await session.complete(
                        types.ResourceTemplateReference(type="ref/resource", uri=uri), {"name": argument, "value": value}
                    )
                return {"route": "session", "api": f"ClientSession.complete {_SESSION_NOTE}",
                        "prompt": list(prompt.completion.values), "resource": list(resource.completion.values)}

            out["completion"] = await P.attempt_async(_completion)
        if "cancellation" in servers:
            async def _cancellation():
                async with self._tools(servers["cancellation"]) as tools:
                    function = self._tool(tools, "cancellable_wait")
                    loop = asyncio.get_running_loop()
                    started = loop.time()
                    task = asyncio.ensure_future(function.run({"seconds": P.CANCEL_WAIT_S}))
                    await asyncio.sleep(P.CANCEL_AFTER_S)
                    task.cancel()
                    try:
                        outcome = f"returned: {_output_text(await task)}"
                    except asyncio.CancelledError:
                        outcome = "cancelled"
                    except Exception as exc:  # noqa: BLE001
                        outcome = f"threw: {type(exc).__name__}: {exc}"
                    abandoned = loop.time() - started
                    await asyncio.sleep(0.3)
                return {"route": "adapter", "mechanism": "asyncio task.cancel() around MCPTool.run",
                        "abandoned_after_s": round(abandoned, 3), "outcome": outcome}

            out["cancellation"] = await P.attempt_async(_cancellation)
        if "tasks" in servers:
            out["tasks"] = await P.attempt_async(lambda: _wrap(call("tasks", "task_capable"), route="adapter"))
        if "caching" in servers:
            async def _caching():
                async with self._session(servers["caching"]) as session:
                    marks = []
                    for step in ("first", "within_ttl", "after_ttl"):
                        if step == "after_ttl":
                            await asyncio.sleep(P.CACHE_PAUSE_S)
                        started = P.wall_clock()
                        # MCPTool has no relist; a caller lists again with from_client.
                        count = len(await self._session_tools(session))
                        marks.append({"call": step, "started": started, "finished": P.wall_clock(), "count": count})
                return {"route": "session", "api": f"MCPTool.from_client {_SESSION_NOTE}", "marks": marks}

            out["caching"] = await P.attempt_async(_caching)
        return out

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py.

        BeeAI has no notification hook and no relist: a caller that wants the new tools
        calls ``MCPTool.from_client`` again, which is what the probe does on one
        caller-built session. BeeAI converts tools only, so resources and prompts have no
        listing, and subscriptions go through that session.
        """
        return asyncio.run(self._notify_async(server))

    async def _notify_async(self, server: ServerHandle) -> dict:
        import anyio

        from .. import notify as N

        recorder = N.Recorder()
        async with self._session(server) as session, anyio.create_task_group() as group:
            negotiated = self._negotiated_mcp_spec_version
            subscriptions = N.SessionSubscriptions(session, recorder, negotiated, group)

            async def tools():
                return list(await self._session_tools(session))

            async def call(name):
                return _output_text(await self._tool(await self._session_tools(session), name).run({}))

            out = await N.run(
                N.Adapter(
                    call=call,
                    recorder=recorder,
                    hook=None,
                    lists={"tools": tools, "resources": None, "prompts": None},
                    no_list_reason="beeai_framework.tools.mcp converts tools only; BeeAI has no resource or prompt API",
                    subscribe=subscriptions.subscribe,
                    unsubscribe=subscriptions.unsubscribe,
                    subscribe_route="session",
                    subscribe_api=f"the caller's ClientSession: {subscriptions.api}",
                    negotiated=negotiated,
                )
            )
            group.cancel_scope.cancel()
        return out


def _content(item: Any) -> dict:
    """An MCP content object JSONToolOutput carries, in wire form."""

    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump(mode="json", by_alias=True, exclude_none=True)
    return {"type": "text", "text": item if isinstance(item, str) else json.dumps(item, default=str)}


def _output_rep(tool: str, origin: str, output: Any) -> ResultRep:
    """What ``MCPTool.run`` returned, classified by what the caller can tell.

    ``JSONToolOutput.result`` is the ``structuredContent`` alone when the result has one.
    Otherwise each block is kept as its MCP content object, except that a text block
    whose text parses as JSON becomes the parsed value, and a single block is unwrapped.
    A dict therefore reads as structured content: the caller cannot tell it from JSON
    text.
    """

    value = getattr(output, "result", output)
    raw = _output_text(output)
    if isinstance(value, dict) and "type" not in value:
        return normalize_framework_result(tool=tool, origin=origin, structured_content=value, text=raw, raw=raw[:2000])
    items = value if isinstance(value, list) else [value]
    blocks = [_content(item) for item in items]
    return normalize_framework_result(tool=tool, origin=origin, blocks=blocks, raw=raw[:2000])


def _output_text(output: Any) -> str:
    get_text = getattr(output, "get_text_content", None)
    if callable(get_text):
        return get_text()
    return json.dumps(output, default=str)


async def _wrap(awaitable, route: str | None = None) -> dict:
    text = _output_text(await awaitable)
    return {"route": route, "text": text} if route else {"text": text}
