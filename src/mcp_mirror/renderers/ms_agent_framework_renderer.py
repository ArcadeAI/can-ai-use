"""Microsoft Agent Framework renderer (DESIGN.md section 8).

Connects through ``agent_framework.MCPStdioTool`` or ``MCPStreamableHTTPTool`` and runs
one turn of a real ``agent_framework.Agent`` whose chat client is scripted. No model is
called: the client records the options and messages the agent hands it and answers from
a fixed list.

That hand-off is the capture boundary. ``MCPTool.load_tools`` turns each MCP tool into
an ``agent_framework.FunctionTool``, and ``load_prompts`` turns each MCP prompt into
another ``FunctionTool``; the agent expands the MCP tool into those functions and passes
them to the chat client in ``options["tools"]``. A provider client serializes each with
``FunctionTool.to_json_schema_spec()``, which is what is read here. Only the functions
built from MCP tools are reported as tool definitions; the prompt functions are recorded
by ``probe_surface``.

The framework has no HTTP+SSE client (``MCPWebsocketTool`` is its third transport), so an
``sse`` server raises ``RenderError``.
"""

from __future__ import annotations

import asyncio
import logging
from importlib import util
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version

ID = "ms_agent_framework"
REQUIRES = ("agent_framework",)

_CALL_ID = "mcp-mirror-call"
# Keys the adapter writes on FunctionTool.additional_properties and on a function result.
_IS_TOOL = "_mcp_is_tool"
_REMOTE_NAME = "_mcp_remote_name"
_HOST_PAYLOAD = "_mcp_tool_result_host_payload"
_OPAQUE = ("data", "blob", "uri")


def available() -> bool:
    return util.find_spec("agent_framework") is not None and util.find_spec("mcp") is not None


def get_renderer() -> "MSAgentFrameworkRenderer":
    return MSAgentFrameworkRenderer()


class MSAgentFrameworkRenderer:
    id = ID

    def versions(self) -> dict:
        version = dist_version("agent-framework-core")
        return {"framework": version, "adapter": version}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "agent_framework.MCPStdioTool / MCPStreamableHTTPTool.load_tools, then "
                "FunctionTool.to_json_schema_spec inside an Agent run"
            ),
            capture_object=(
                "agent_framework.FunctionTool in the options['tools'] an Agent hands "
                "its chat client"
            ),
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "protocolVersion returned by ClientSession.initialize, which "
                "MCPTool.connect calls for every session"
            ),
            limitation=(
                "The chat client is scripted; the FunctionTools are captured before any "
                "provider client serializes them, so no provider request body was captured."
            ),
        )

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def render(self, server: ServerHandle) -> list[ToolRep]:
        try:
            return asyncio.run(self._render_async(server))
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"ms_agent_framework renderer failed: {_describe(exc)}") from exc

    async def _render_async(self, server: ServerHandle) -> list[ToolRep]:
        run = await self._run_agent(server, call=None)
        if not run["requests"]:
            raise RenderError("ms_agent_framework agent run never reached the chat client")
        return [
            normalize_function_tool(spec["function"], origin=self.id)
            for spec, is_tool in run["requests"][0]["tools"]
            if is_tool
        ]

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    # MCPTool.call_tool is the adapter's public call; its return value is what the
    # FunctionTool hands the agent. The agent then wraps it in a function_result Content
    # in the next message list. The two disagree on errors: call_tool raises
    # ToolExecutionException, and the agent turns that into a generic error result.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="agent_framework.MCPTool.call_tool",
                capture_object="the list[agent_framework.Content] call_tool returns",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Function result to the model",
                capture_api="agent_framework.Agent.run, next chat-client request messages",
                capture_object="the function_result Content in the tool message the chat client receives",
                agent_path=True,
                limitation=(
                    "The chat client is scripted; no provider client serialized the "
                    "request."
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
            raise RenderError(f"ms_agent_framework renderer has no result boundary {boundary!r}")
        try:
            if boundary == "direct":
                return asyncio.run(self._direct_async(server, tool, arguments))
            run = asyncio.run(self._run_agent(server, call=(tool, arguments)))
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001 - the failure mode is the observation
            return normalize_framework_result(
                tool=tool, origin=self.id, is_error=True,
                text=_describe(exc), error_channel="exception", raw=repr(exc)[:2000],
            )
        if tool not in run["names"]:
            raise RenderError(f"ms_agent_framework renderer found no tool named {tool!r}")
        if "agent" not in run:
            raise RenderError("ms_agent_framework agent run never returned a function result")
        return self._agent_rep(tool, run["agent"])

    async def _direct_async(self, server: ServerHandle, tool: str, arguments: dict[str, Any]) -> ResultRep:
        mcp_tool = self._build_tool(server)
        async with _negotiation(self), mcp_tool:
            names = {_remote(f) for f in mcp_tool.functions if _is_tool(f)}
            if tool not in names:
                raise RenderError(f"ms_agent_framework renderer found no tool named {tool!r}")
            try:
                returned = await mcp_tool.call_tool(tool, **arguments)
            except Exception as exc:  # noqa: BLE001 - an isError result raises here
                return normalize_framework_result(
                    tool=tool, origin=self.id, is_error=True,
                    text=_describe(exc), error_channel="exception", raw=repr(exc)[:2000],
                )
        return self._contents_rep(tool, returned)

    def _contents_rep(self, tool: str, returned: Any) -> ResultRep:
        if isinstance(returned, str):
            return normalize_framework_result(tool=tool, origin=self.id, text=returned, raw=returned[:2000])
        items = list(returned or [])
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=[_block_of(item) for item in items],
            raw=_redact([_dump(item) for item in items]),
        )

    def _agent_rep(self, tool: str, result: dict) -> ResultRep:
        """What the model reads: the function_result's ``items``.

        The complete CallToolResult rides along in ``additional_properties`` for hosts,
        which the chat client does not send, so it is recorded in ``raw`` only.
        """

        items = result.get("items") or []
        error = result.get("exception") is not None
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=[_block_of(item) for item in items],
            is_error=True if error else None,
            text=None if items else result.get("result"),
            error_channel="exception" if error else None,
            raw=_redact(result),
        )

    # ------------------------------------------------------------------
    # The agent run both the definitions and the agent result boundary share
    # ------------------------------------------------------------------

    async def _run_agent(self, server: ServerHandle, call: tuple[str, dict] | None, **options) -> dict:
        from agent_framework import Agent

        captured: dict[str, Any] = {"requests": [], "names": []}
        client = _scripted_client(captured, call)
        mcp_tool = self._build_tool(server, **options)
        async with _negotiation(self), mcp_tool:
            agent = Agent(client=client, tools=[mcp_tool])
            await agent.run("go")
        return captured

    def _build_tool(self, server: ServerHandle, cls: Any = None, **options):
        from agent_framework import MCPStdioTool, MCPStreamableHTTPTool

        if server.transport == "sse":
            raise RenderError(
                "Microsoft Agent Framework has no HTTP+SSE MCP client "
                "(MCPStdioTool, MCPStreamableHTTPTool and MCPWebsocketTool only)"
            )
        options.setdefault("request_timeout", 30)
        if server.transport == "http":
            if server.headers:
                options.setdefault("static_headers", dict(server.headers))
            return (cls or MCPStreamableHTTPTool)(name="src", url=server.url, **options)
        return (cls or MCPStdioTool)(name="src", command=server.command, args=list(server.args), **options)

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """Prompts through the FunctionTools ``load_prompts`` builds; no resource API.

        MCPTool lists prompts only by turning each into a FunctionTool the model can
        call, and expands one through ``MCPTool.get_prompt``, which returns a string.
        MCPTool has no method for resources or resource templates. (``MCPSkillsSource``
        reads resources, but only a ``skill://index.json`` and the skills it names.)
        """
        return asyncio.run(self._probe_async(server))

    async def _probe_async(self, server: ServerHandle) -> dict:
        from .. import probe as P

        out: dict = P.unsupported("MCPTool has no resource or resource-template API")
        mcp_tool = self._build_tool(server)
        async with _negotiation(self), mcp_tool:
            prompts = [f for f in mcp_tool.functions if not _is_tool(f)]

            async def _list():
                return {"entries": [_prompt_entry(f) for f in prompts]}

            out["prompts.list"] = await P.attempt_async(_list)

            async def _get():
                first = prompts[0]
                schema = first.parameters() or {}
                arguments = {name: "x" for name in schema.get("required") or []}
                text = await mcp_tool.get_prompt(_remote(first), **arguments)
                return {"description": None, "messages": [P.message_entry(None, text)]}

            out["prompts.get"] = await P.attempt_async(_get) if prompts else {"ok": False, "error": "no prompt listed"}
        return out

    # ------------------------------------------------------------------
    # Client features
    # ------------------------------------------------------------------

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling through MCPTool's ``client`` hook; nothing for elicitation or roots.

        Sampling is answered by the chat client passed as ``client=``; the adapter
        denies every request unless ``sampling_approval_callback`` approves it, so the
        probe approves all. When a client is set the adapter declares
        ``sampling.tools`` and hands offered tools to the client as FunctionTools.
        The ClientSession it builds gets no elicitation or roots callback, and MCPTool
        has no parameter for either.
        """
        return asyncio.run(self._probe_client_async(server))

    async def _probe_client_async(self, server: ServerHandle) -> dict:
        import warnings

        from .. import client_probe as C

        log = C.HandlerLog()
        sampling = {"client": _sampling_client(log), "sampling_approval_callback": lambda _params: True}

        async def call(tool: str, **options):
            with warnings.catch_warnings():
                # MCPTool warns that MCP sampling is deprecated whenever client= is set.
                warnings.simplefilter("ignore", DeprecationWarning)
                mcp_tool = self._build_tool(server, **options)
            async with _negotiation(self), mcp_tool:
                try:
                    returned = await mcp_tool.call_tool(tool)
                except Exception as exc:  # noqa: BLE001 - an isError result raises here
                    return normalize_framework_result(
                        tool=tool, origin=self.id, is_error=True, text=_describe(exc), error_channel="exception",
                    )
            return self._contents_rep(tool, returned)

        hook = "MCPTool(client=..., sampling_approval_callback=...)"
        out: dict = {}
        out["sampling"] = await C.attempt_async("sampling", hook, "adapter", lambda: call("ask_sampling", **sampling), log)
        out["sampling.tools"] = await C.attempt_async(
            "sampling.tools", hook, "adapter", lambda: call("ask_sampling_tools", **sampling), log,
        )
        for feature, tool in (
            ("elicitation.form", "ask_elicitation_form"),
            ("elicitation.url", "ask_elicitation_url"),
            ("roots.list", "list_client_roots"),
            ("roots.list_changed", "roots_changes_seen"),
        ):
            out[feature] = await C.attempt_async(feature, None, None, lambda tool=tool: call(tool), log)
        try:
            out["capabilities"] = {"text": C.rep_text(await call(C.CAPABILITIES_TOOL, **sampling))[0]}
        except Exception as exc:  # noqa: BLE001
            out["capabilities"] = {"text": f"{type(exc).__name__}: {exc}"}
        return out

    # ------------------------------------------------------------------
    # Server utilities
    # ------------------------------------------------------------------

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through MCPTool.

        Logging is the adapter's ``logging_callback``, which its docstring says to
        override in a subclass. The level it requests at connect is the level of the
        ``agent_framework._mcp`` Python logger. MCPTool passes no progress callback and
        has no completion API; both are reached on ``MCPTool.session``, the raw SDK
        ClientSession, and reported with ``route: "session"``.
        """
        return asyncio.run(self._utilities_async(servers))

    async def _utilities_async(self, servers: dict[str, ServerHandle]) -> dict:
        import mcp.types as types
        from agent_framework import MCPStdioTool, MCPStreamableHTTPTool

        from .. import probe as P

        events: list[dict] = []
        messages: list[dict] = []

        def logged(base):
            class Logged(base):
                async def logging_callback(self, params):  # noqa: ANN001
                    messages.append({"level": getattr(params, "level", None), "data": getattr(params, "data", None)})
                    await super().logging_callback(params)

            return Logged

        def build(feature: str, **options):
            server = servers[feature]
            base = MCPStreamableHTTPTool if server.transport == "http" else MCPStdioTool
            return self._build_tool(server, cls=logged(base), **options)

        async def call(feature: str, tool: str, arguments: dict | None = None):
            async with build(feature) as mcp_tool:
                return P.result_text(_texts(await mcp_tool.call_tool(tool, **(arguments or {}))))

        out: dict = {}
        if "tools.list" in servers:
            async def _names():
                async with build("tools.list") as mcp_tool:
                    return {"route": "adapter", "names": [_remote(f) for f in mcp_tool.functions if _is_tool(f)]}

            out["tools.list"] = await P.attempt_async(_names)
        if "progress" in servers:
            async def _progress():
                async def on_progress(progress, total, message):
                    events.append({"progress": progress, "total": total, "message": message})

                async with build("progress") as mcp_tool:
                    result = await mcp_tool.session.call_tool("slow_with_progress", {}, progress_callback=on_progress)
                return {"route": "session", "api": "MCPTool.session.call_tool(progress_callback=...)",
                        "events": list(events), "text": P.result_text(result)}

            out["progress"] = await P.attempt_async(_progress)
        if "logging" in servers:
            async def _logging():
                text = await call("logging", "emit_logs")
                await asyncio.sleep(0.2)
                return {"route": "adapter", "api": "MCPTool.logging_callback (subclass override)",
                        "messages": list(messages), "text": text}

            out["logging"] = await P.attempt_async(_logging)
        if "set_level" in servers:
            async def _set_level():
                adapter_logger = logging.getLogger("agent_framework._mcp")
                previous = adapter_logger.level
                adapter_logger.setLevel(logging.WARNING)
                try:
                    text = await call("set_level", "emit_logs")
                finally:
                    adapter_logger.setLevel(previous)
                return {"route": "adapter", "api": "logging.getLogger('agent_framework._mcp').setLevel before connect",
                        "level": P.SET_LEVEL, "text": text}

            out["set_level"] = await P.attempt_async(_set_level)
        if "completion" in servers:
            async def _completion():
                async with build("completion") as mcp_tool:
                    session = mcp_tool.session
                    name, argument, value = P.COMPLETION_PROMPT
                    prompt = await session.complete(types.PromptReference(type="ref/prompt", name=name), {"name": argument, "value": value})
                    uri, argument, value = P.COMPLETION_RESOURCE
                    resource = await session.complete(types.ResourceTemplateReference(type="ref/resource", uri=uri), {"name": argument, "value": value})
                return {"route": "session", "api": "MCPTool.session.complete",
                        "prompt": list(prompt.completion.values), "resource": list(resource.completion.values)}

            out["completion"] = await P.attempt_async(_completion)
        if "cancellation" in servers:
            async def _cancellation():
                async with build("cancellation") as mcp_tool:
                    loop = asyncio.get_running_loop()
                    started = loop.time()
                    task = asyncio.create_task(mcp_tool.call_tool("cancellable_wait", seconds=P.CANCEL_WAIT_S))
                    await asyncio.sleep(P.CANCEL_AFTER_S)
                    task.cancel()
                    try:
                        outcome = f"returned: {P.result_text(_texts(await task))}"
                    except asyncio.CancelledError:
                        outcome = "cancelled"
                    except Exception as exc:  # noqa: BLE001
                        outcome = f"threw: {_describe(exc)}"
                    abandoned = loop.time() - started
                    await asyncio.sleep(0.3)
                return {"route": "adapter", "mechanism": "asyncio task.cancel() around MCPTool.call_tool",
                        "abandoned_after_s": round(abandoned, 3), "outcome": outcome}

            out["cancellation"] = await P.attempt_async(_cancellation)
        if "tasks" in servers:
            async def _tasks():
                return {"route": "adapter", "text": await call("tasks", "task_capable")}

            out["tasks"] = await P.attempt_async(_tasks)
        if "caching" in servers:
            async def _caching():
                from agent_framework import Agent

                captured: dict[str, Any] = {"requests": [], "names": []}
                marks = []
                async with build("caching") as mcp_tool:
                    agent = Agent(client=_scripted_client(captured, None), tools=[mcp_tool])
                    for label in ("first", "within_ttl", "after_ttl"):
                        if label == "after_ttl":
                            await asyncio.sleep(P.CACHE_PAUSE_S)
                        started = P.wall_clock()
                        await agent.run("go")
                        tools = [spec for spec, is_tool in captured["requests"][-1]["tools"] if is_tool]
                        marks.append({"call": label, "started": started, "finished": P.wall_clock(), "count": len(tools)})
                return {"route": "adapter", "api": "Agent.run with one connected MCPTool", "marks": marks}

            out["caching"] = await P.attempt_async(_caching)
        return out

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through MCPTool.

        ``message_handler`` is the documented hook: its docstring says to subclass and
        call ``super()``, which keeps the adapter's own reload of tools and prompts on a
        list_changed notification. Tools and prompts are listed from
        ``MCPTool.functions``, which that reload rewrites. There is no resource API, and
        no subscribe API, so subscriptions go through ``MCPTool.session``.
        """
        return asyncio.run(self._notify_async(server))

    async def _notify_async(self, server: ServerHandle) -> dict:
        import anyio
        from agent_framework import MCPStdioTool, MCPStreamableHTTPTool

        from .. import notify as N

        recorder = N.Recorder()
        base = MCPStreamableHTTPTool if server.transport == "http" else MCPStdioTool

        class Recorded(base):
            async def message_handler(self, message):  # noqa: ANN001
                await super().message_handler(message)
                await recorder.message_handler(message)

        mcp_tool = self._build_tool(server, cls=Recorded)
        async with _negotiation(self), mcp_tool, anyio.create_task_group() as group:
            negotiated = self._negotiated_mcp_spec_version
            subscriptions = N.SessionSubscriptions(mcp_tool.session, recorder, negotiated, group)

            async def settled(names):
                # The reload runs as a background task the handler schedules; let it land.
                await asyncio.sleep(0.3)
                while mcp_tool._pending_reload_tasks:  # noqa: SLF001
                    await asyncio.sleep(0.05)
                return names()

            async def tools():
                return await settled(lambda: [_remote(f) for f in mcp_tool.functions if _is_tool(f)])

            async def prompts():
                return await settled(lambda: [_remote(f) for f in mcp_tool.functions if not _is_tool(f)])

            out = await N.run(
                N.Adapter(
                    call=lambda name: mcp_tool.call_tool(name),
                    recorder=recorder,
                    hook="MCPTool.message_handler (subclass override calling super)",
                    lists={"tools": tools, "resources": None, "prompts": prompts},
                    no_list_reason="MCPTool has no resource API",
                    subscribe=subscriptions.subscribe,
                    unsubscribe=subscriptions.unsubscribe,
                    subscribe_route="session",
                    subscribe_api=f"MCPTool.session: {subscriptions.api}",
                    negotiated=negotiated,
                )
            )
            group.cancel_scope.cancel()
        return out


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _negotiation:
    """Record the protocol version every ClientSession.initialize in the block returns."""

    def __init__(self, renderer: MSAgentFrameworkRenderer) -> None:
        self.renderer = renderer
        self.versions: list[str] = []

    async def __aenter__(self):
        from mcp.client.session import ClientSession

        self._cls = ClientSession
        self._original = original = ClientSession.initialize
        versions = self.versions
        renderer = self.renderer

        async def capture_initialize(session, *args, **kwargs):
            result = await original(session, *args, **kwargs)
            version = getattr(result, "protocolVersion", None)
            if version:
                versions.append(str(version))
                renderer._negotiated_mcp_spec_version = str(version)
            return result

        ClientSession.initialize = capture_initialize
        return self

    async def __aexit__(self, *exc_info):
        self._cls.initialize = self._original
        observed = set(self.versions)
        if exc_info[0] is None and len(observed) > 1:
            raise RenderError(
                "ms_agent_framework renderer could not establish one negotiated MCP "
                f"protocol version; observed {sorted(observed)}"
            )
        return False


def _scripted_client(captured: dict, call: tuple[str, dict] | None):
    """A chat client that records each request and answers from a fixed script.

    It carries the framework's ``FunctionInvocationLayer``, as every shipped client
    does, so the agent runs the tool call and sends the result back. Turn one calls
    ``call`` if given, otherwise ends the run. Every later turn ends it.
    """

    from agent_framework import BaseChatClient, ChatResponse, Content, FunctionInvocationLayer, Message

    class ScriptedClient(FunctionInvocationLayer, BaseChatClient):
        async def _respond(self, messages, options):
            tools = [
                (tool.to_json_schema_spec(), bool((tool.additional_properties or {}).get(_IS_TOOL)))
                for tool in (options.get("tools") or [])
                if hasattr(tool, "to_json_schema_spec")
            ]
            captured["names"] = [
                (tool.additional_properties or {}).get(_REMOTE_NAME)
                for tool in (options.get("tools") or [])
                if (getattr(tool, "additional_properties", None) or {}).get(_IS_TOOL)
            ]
            results = [
                c.to_dict()
                for message in messages
                for c in message.contents or []
                if c.type == "function_result"
            ]
            captured["requests"].append({"tools": tools})
            if results:
                captured["agent"] = results[-1]
            if call is not None and len(captured["requests"]) == 1:
                content = Content.from_function_call(call_id=_CALL_ID, name=_local(tools, call[0]), arguments=call[1])
            else:
                content = Content.from_text("done")
            return ChatResponse(messages=[Message(role="assistant", contents=[content])])

        def _inner_get_response(self, *, messages, stream, options, **kwargs):
            if stream:
                raise NotImplementedError("the scripted client does not stream")
            return self._respond(messages, options)

    return ScriptedClient()


def _sampling_client(log):
    """The chat client MCPTool forwards sampling requests to, answering with the sentinels."""

    from agent_framework import BaseChatClient, ChatResponse, Content, Message

    from .. import client_probe as C

    class SamplingClient(BaseChatClient):
        async def _respond(self, messages, options):
            tools = list(options.get("tools") or [])
            after_tool = any(c.type == "function_result" for m in messages for c in m.contents or [])
            log.note("MCPTool.client.get_response", tools=bool(tools), replied="toolUse" if tools and not after_tool else "endTurn")
            if tools and not after_tool:
                content = Content.from_function_call(call_id=C.TOOL_USE_ID, name=tools[0].name, arguments={"word": "mirror"})
            else:
                content = Content.from_text(C.SAMPLED_AFTER_TOOL if after_tool else C.SAMPLED)
            return ChatResponse(messages=[Message(role="assistant", contents=[content])], model="mcp-mirror-stub")

        def _inner_get_response(self, *, messages, stream, options, **kwargs):
            return self._respond(messages, options)

    return SamplingClient()


def _local(tools: list, remote: str) -> str:
    # With no tool_name_prefix the local name is the normalized remote name.
    for spec, is_tool in tools:
        if is_tool and spec["function"]["name"] == remote:
            return remote
    for spec, is_tool in tools:
        if is_tool and spec["function"]["name"].replace("-", "_") == remote.replace("-", "_"):
            return spec["function"]["name"]
    return remote


def _is_tool(function: Any) -> bool:
    return bool((function.additional_properties or {}).get(_IS_TOOL))


def _remote(function: Any) -> str:
    return (function.additional_properties or {}).get(_REMOTE_NAME) or function.name


def _prompt_entry(function: Any) -> dict:
    """A prompt as the adapter lists it: the FunctionTool load_prompts built from it."""

    schema = function.parameters() or {}
    required = set(schema.get("required") or [])
    return {
        "name": _remote(function),
        "title": None,
        "description": function.description or None,
        "arguments": [
            {"name": name, "description": (prop or {}).get("description"), "required": name in required}
            for name, prop in (schema.get("properties") or {}).items()
        ],
    }


def _dump(item: Any) -> Any:
    to_dict = getattr(item, "to_dict", None)
    return to_dict() if callable(to_dict) else item


def _block_of(item: Any) -> dict[str, Any]:
    """Classify one agent_framework Content by what the model can tell it is.

    The adapter maps MCP image and audio blocks to ``data`` Content with a data URI, a
    text resource to plain ``text``, and a resource link to ``uri`` Content. The kind
    recorded is the one the Content carries, not the MCP kind it came from.
    """

    item = _dump(item)
    if isinstance(item, str):
        return {"type": "text", "text": item}
    if not isinstance(item, dict):
        return {"type": type(item).__name__}
    kind = item.get("type")
    media_type = item.get("media_type")
    if kind == "text":
        return {"type": "text", "text": item.get("text")}
    if kind == "data":
        major = (media_type or "").split("/")[0]
        return {"type": major if major in {"image", "audio"} else "data", "mimeType": media_type}
    if kind == "uri":
        return {"type": "uri", "uri": item.get("uri"), "mimeType": media_type}
    return {k: v for k, v in item.items() if k != "additional_properties"}


def _texts(returned: Any) -> Any:
    if isinstance(returned, list):
        return [getattr(item, "text", None) or str(item) for item in returned]
    return returned


def _describe(exc: BaseException) -> str:
    inner = getattr(exc, "inner_exception", None)
    text = f"{type(exc).__name__}: {exc}"
    return f"{text} ({type(inner).__name__}: {inner})" if inner is not None else text


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (f"<{len(v)} chars>" if k in _OPAQUE and isinstance(v, str) and len(v) > 64 else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value
