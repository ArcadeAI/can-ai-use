"""Semantic Kernel renderer (DESIGN.md section 8).

Connects through ``semantic_kernel.connectors.mcp`` (``MCPStdioPlugin``,
``MCPStreamableHttpPlugin``, ``MCPSsePlugin``), adds the connected plugin to a real
``Kernel``, and runs one auto function-calling turn of ``OpenAIChatCompletion``. No
model is called: the service's ``AsyncOpenAI`` client sends through an
``httpx.MockTransport`` that records each serialized ``chat.completions`` request body
and answers from a fixed script. The API key it is given is a placeholder that never
leaves the process.

That request body is the capture boundary. The plugin turns each MCP tool into a
``KernelFunction`` (``load_tools``), and Semantic Kernel's OpenAI connector serializes
every function on the kernel into the request's ``tools`` array. The plugin also
registers each MCP *prompt* as a kernel function (``load_prompts``, on by default), so
prompts reach the same array as callable tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
from importlib import util
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version

ID = "semantic_kernel"
REQUIRES = ("semantic_kernel",)

# The plugin name becomes each function's prefix in the request (``mcp-<tool>``).
_PLUGIN = "mcp"
_MODEL = "mcp-mirror-scripted"
_CALL_ID = "mcp-mirror-call"
_TIMEOUT = 30
_CONNECT_TIMEOUT = 45
# SK's default logging_callback writes server log messages to this logger, and SK sends
# logging/setLevel at connect only when this logger has a level set.
_SK_LOGGER = "semantic_kernel.connectors.mcp"
# Payload keys whose values are base64 bytes; a capture records their length, not them.
_OPAQUE = ("data", "blob")
_INVOKE_ERROR = "An error occurred while invoking the function "


def available() -> bool:
    return (
        util.find_spec("semantic_kernel") is not None
        and util.find_spec("mcp") is not None
        and util.find_spec("openai") is not None
    )


def get_renderer() -> "SemanticKernelRenderer":
    return SemanticKernelRenderer()


class SemanticKernelRenderer:
    id = ID

    def versions(self) -> dict:
        version = dist_version("semantic-kernel")
        return {"framework": version, "adapter": version}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "semantic_kernel.connectors.mcp.MCPStdioPlugin / MCPStreamableHttpPlugin / "
                "MCPSsePlugin added to a Kernel, then OpenAIChatCompletion."
                "get_chat_message_content with FunctionChoiceBehavior.Auto()"
            ),
            capture_object=(
                "the tools array of the serialized chat.completions request body the "
                "AsyncOpenAI client sends (intercepted by an httpx.MockTransport)"
            ),
            capture_stage="provider_request",
            provider_request_captured=True,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "protocolVersion returned by ClientSession.initialize, which the plugin's "
                "_inner_connect calls once per connection"
            ),
            limitation=(
                "The request is intercepted before any network I/O; the scripted "
                "endpoint answers instead of a model. Only the OpenAI chat-completions "
                "connector was exercised."
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
            raise RenderError(f"semantic_kernel renderer failed: {exc}") from exc
        if not run["bodies"]:
            raise RenderError("semantic_kernel agent run never reached the model")
        # The same array carries a function for every MCP prompt (load_prompts is on by
        # default). Those are not tool definitions, so only the functions the plugin
        # registered from tools/list are returned; tests pin that prompts are there too.
        # The function-tool shape has no slot for annotations, icons, title or
        # outputSchema, and KernelFunctionMetadata keeps none of them either, so there
        # is no framework_metadata to record.
        return [
            normalize_function_tool(tool, origin=self.id)
            for tool in run["bodies"][0].get("tools") or []
            if run["origins"].get(tool["function"]["name"]) == "tool"
        ]

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Kernel function return value",
                capture_api="semantic_kernel.Kernel.invoke on the plugin's KernelFunction",
                capture_object=(
                    "FunctionResult.value: the list of TextContent / ImageContent / "
                    "AudioContent / BinaryContent that MCPPluginBase.call_tool returned"
                ),
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Tool message to the model",
                capture_api=(
                    "OpenAIChatCompletion auto function calling, next serialized "
                    "chat.completions request"
                ),
                capture_object="the role=tool message in the request body the model receives",
                agent_path=True,
                limitation="The scripted endpoint answers instead of a model.",
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
            raise RenderError(f"semantic_kernel renderer has no result boundary {boundary!r}")
        try:
            if boundary == "direct":
                return asyncio.run(self._direct(server, tool, arguments))
            run = asyncio.run(self._run_agent(server, call=(tool, arguments)))
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001 - the failure mode is the observation
            return _exception_rep(tool, exc)
        message = run.get("tool_message")
        if message is None:
            raise RenderError("semantic_kernel agent run never sent a tool message")
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(_redact(content), sort_keys=True)
        # Kernel._inner_auto_function_invoke_handler catches a function's exception
        # and sends the model this sentence in its place.
        failed = isinstance(content, str) and content.startswith(_INVOKE_ERROR)
        blocks = []
        if isinstance(content, list):
            # A content-part array would be the only way media reaches the model.
            blocks = [_redact(part) for part in content if isinstance(part, dict)]
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            text=text,
            is_error=True if failed else None,
            error_channel="exception" if failed else None,
            raw=_redact(message),
        )

    async def _direct(self, server: ServerHandle, tool: str, arguments: dict[str, Any]) -> ResultRep:
        from semantic_kernel import Kernel

        async with self._connected(server) as plugin:
            kernel = Kernel()
            kernel.add_plugin(plugin)
            function = _function(kernel, tool)
            try:
                result = await kernel.invoke(function, **arguments)
            except Exception as exc:  # noqa: BLE001 - the failure mode is the observation
                return _exception_rep(tool, exc)
        value = getattr(result, "value", None)
        items = value if isinstance(value, list) else [value]
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=[_block_of(item) for item in items if item is not None],
            raw=repr(value)[:2000],
        )

    # ------------------------------------------------------------------
    # The agent run both the definitions and the agent boundary share
    # ------------------------------------------------------------------

    async def _run_agent(self, server: ServerHandle, call: tuple[str, dict] | None) -> dict:
        from semantic_kernel import Kernel
        from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
        from semantic_kernel.contents import ChatHistory

        endpoint = _ScriptedEndpoint(call)
        async with self._connected(server) as plugin:
            kernel = Kernel()
            service = endpoint.service()
            kernel.add_service(service)
            kernel.add_plugin(plugin)
            history = ChatHistory()
            history.add_user_message("go")
            settings = OpenAIChatPromptExecutionSettings(function_choice_behavior=FunctionChoiceBehavior.Auto())
            await service.get_chat_message_content(history, settings, kernel=kernel)
            # Which MCP list each function came from, keyed by its name in the request.
            origins = {f"{_PLUGIN}-{local}": kind for local, (kind, _remote) in plugin._mcp_registered_names.items()}
        return {**endpoint.captured(), "origins": origins}

    def _connected(self, server: ServerHandle, **options):
        return _Connection(self, server, options)

    def _build_plugin(self, server: ServerHandle, **options):
        from semantic_kernel.connectors.mcp import MCPSsePlugin, MCPStdioPlugin, MCPStreamableHttpPlugin

        options.setdefault("request_timeout", _TIMEOUT)
        if server.transport == "http":
            return MCPStreamableHttpPlugin(
                name=_PLUGIN, url=server.url, headers=dict(server.headers) or None, **options
            )
        if server.transport == "sse":
            return MCPSsePlugin(name=_PLUGIN, url=server.url, **options)
        return MCPStdioPlugin(name=_PLUGIN, command=server.command, args=list(server.args), **options)

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """Prompts through the plugin; it has no resource or template API.

        ``load_prompts`` turns each MCP prompt into a KernelFunction. Building the plugin
        with ``load_tools=False`` is the plugin's own way to get only those, so the
        listing is the kernel plugin's functions and ``get_prompt`` fetches one.
        """
        return asyncio.run(self._probe_async(server))

    async def _probe_async(self, server: ServerHandle) -> dict:
        from semantic_kernel import Kernel

        from .. import probe as P

        reason = "MCPPluginBase has no resource methods; it loads tools and prompts only"
        out: dict = {
            "resources.list": {"ok": False, "unsupported": reason},
            "resources.read.text": {"ok": False, "unsupported": reason},
            "resources.read.blob": {"ok": False, "unsupported": reason},
            "resources.templates": {"ok": False, "unsupported": reason},
        }
        async with self._connected(server, load_tools=False) as plugin:
            kernel = Kernel()
            kernel.add_plugin(plugin)

            async def _list():
                return {"entries": [_prompt_entry(f.metadata) for f in kernel.get_plugin(_PLUGIN).functions.values()]}

            out["prompts.list"] = await P.attempt_async(_list)
            prompts = out["prompts.list"].get("entries") or []

            async def _get():
                first = prompts[0]
                arguments = {a["name"]: "x" for a in first["arguments"] if a.get("required")}
                messages = await plugin.get_prompt(_remote_name(plugin, first["name"]), **arguments)
                return {
                    "description": None,
                    "messages": [P.message_entry(m.role, _first_item(m)) for m in messages],
                }

            out["prompts.get"] = await P.attempt_async(_get) if prompts else {"ok": False, "error": "no prompt listed"}
        return out

    # ------------------------------------------------------------------
    # Client features
    # ------------------------------------------------------------------

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling through the plugin's own handler; no elicitation or roots hook.

        MCPPluginBase opens its ClientSession with ``sampling_callback`` set to its own
        method, which forwards the request to the kernel's chat completion service after
        ``sampling_consent_callback`` approves it. The kernel's service here is the same
        scripted endpoint. The plugin passes no elicitation or roots callback, and takes
        none.
        """
        return asyncio.run(self._client_async(server))

    async def _client_async(self, server: ServerHandle) -> dict:
        from semantic_kernel import Kernel

        from .. import client_probe as C

        log = C.HandlerLog()

        async def consent(_plugin_name, params):
            log.note("sampling_consent_callback", tools=bool(getattr(params, "tools", None)))
            return True

        async def call(tool: str, **options):
            endpoint = _ScriptedEndpoint(None, log=log)
            kernel = Kernel()
            kernel.add_service(endpoint.service())
            async with self._connected(server, kernel=kernel, **options) as plugin:
                kernel.add_plugin(plugin)
                function = _function(kernel, tool)
                try:
                    result = await kernel.invoke(function)
                except Exception as exc:  # noqa: BLE001
                    return _exception_rep(tool, exc)
            value = getattr(result, "value", None)
            items = value if isinstance(value, list) else [value]
            return normalize_framework_result(
                tool=tool, origin=self.id, blocks=[_block_of(i) for i in items if i is not None]
            )

        hook = "MCPStdioPlugin(kernel=..., sampling_consent_callback=...)"
        out: dict = {
            "sampling": await C.attempt_async(
                "sampling", hook, "adapter", lambda: call("ask_sampling", sampling_consent_callback=consent), log
            ),
            "sampling.tools": await C.attempt_async(
                "sampling.tools", hook, "adapter",
                lambda: call("ask_sampling_tools", sampling_consent_callback=consent), log,
            ),
        }
        for feature in ("elicitation.form", "elicitation.url", "roots.list", "roots.list_changed"):
            tool = C.FEATURES[feature][0]
            out[feature] = await C.attempt_async(feature, None, None, lambda tool=tool: call(tool), log)
        try:
            rep = await call(C.CAPABILITIES_TOOL, sampling_consent_callback=consent)
            out["capabilities"] = {"text": C.rep_text(rep)[0]}
        except Exception as exc:  # noqa: BLE001
            out["capabilities"] = {"text": f"{type(exc).__name__}: {exc}"}
        return out

    # ------------------------------------------------------------------
    # Server utilities
    # ------------------------------------------------------------------

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through the plugin and its documented logging route.

        Server log messages go to the ``semantic_kernel.connectors.mcp`` Python logger
        (``MCPPluginBase.logging_callback``), and the level that logger is set to is what
        the plugin sends as ``logging/setLevel`` when it connects. Completion has no
        plugin method; it is reachable only on the public ``plugin.session``.
        """
        return asyncio.run(self._utilities_async(servers))

    async def _utilities_async(self, servers: dict[str, ServerHandle]) -> dict:
        import mcp.types as types
        from semantic_kernel import Kernel

        from .. import probe as P

        async def call(feature: str, tool: str, arguments: dict | None = None):
            async with self._connected(servers[feature]) as plugin:
                kernel = Kernel()
                kernel.add_plugin(plugin)
                result = await kernel.invoke(_function(kernel, tool), **(arguments or {}))
                return P.result_text(getattr(result, "value", None))

        out: dict = {}
        if "tools.list" in servers:
            async def _names():
                async with self._connected(servers["tools.list"], load_prompts=False) as plugin:
                    kernel = Kernel()
                    kernel.add_plugin(plugin)
                    return {"route": "adapter", "names": [_remote_name(plugin, n) for n in kernel.get_plugin(_PLUGIN).functions]}

            out["tools.list"] = await P.attempt_async(_names)
        if "progress" in servers:
            answer = await P.attempt_async(lambda: _text_answer(call("progress", "slow_with_progress")))
            out["progress"] = {
                "ok": False,
                "unsupported": "MCPPluginBase.call_tool passes no progress callback and the plugin takes none",
                "text": answer.get("text"),
                "call": answer,
            }
        if "logging" in servers:
            async def _logging():
                with _SkLog(logging.DEBUG) as captured:
                    text = await call("logging", "emit_logs")
                    await asyncio.sleep(0.2)
                return {
                    "route": "adapter",
                    "api": f"MCPPluginBase.logging_callback -> logging.getLogger({_SK_LOGGER!r})",
                    "messages": captured.messages,
                    "text": text,
                }

            out["logging"] = await P.attempt_async(_logging)
        if "set_level" in servers:
            async def _set_level():
                with _SkLog(logging.WARNING):
                    text = await call("set_level", "emit_logs")
                return {
                    "route": "adapter",
                    "api": f"logging.getLogger({_SK_LOGGER!r}).setLevel(...) before connect",
                    "level": P.SET_LEVEL,
                    "text": text,
                }

            out["set_level"] = await P.attempt_async(_set_level)
        if "completion" in servers:
            async def _completion():
                async with self._connected(servers["completion"]) as plugin:
                    name, argument, value = P.COMPLETION_PROMPT
                    prompt = await plugin.session.complete(
                        types.PromptReference(type="ref/prompt", name=name), {"name": argument, "value": value}
                    )
                    uri, argument, value = P.COMPLETION_RESOURCE
                    resource = await plugin.session.complete(
                        types.ResourceTemplateReference(type="ref/resource", uri=uri), {"name": argument, "value": value}
                    )
                return {
                    "route": "session",
                    "api": "MCPStdioPlugin.session.complete (mcp ClientSession)",
                    "prompt": list(prompt.completion.values),
                    "resource": list(resource.completion.values),
                }

            out["completion"] = await P.attempt_async(_completion)
        if "cancellation" in servers:
            async def _cancellation():
                async with self._connected(servers["cancellation"]) as plugin:
                    kernel = Kernel()
                    kernel.add_plugin(plugin)
                    function = _function(kernel, "cancellable_wait")
                    loop = asyncio.get_running_loop()
                    started = loop.time()
                    outcome = "cancelled"
                    try:
                        result = await asyncio.wait_for(
                            kernel.invoke(function, seconds=P.CANCEL_WAIT_S), timeout=P.CANCEL_AFTER_S
                        )
                        outcome = f"returned: {P.result_text(getattr(result, 'value', None))}"
                    except asyncio.TimeoutError:
                        pass
                    abandoned = loop.time() - started
                    await asyncio.sleep(0.3)
                return {
                    "route": "adapter",
                    "mechanism": "asyncio.wait_for around Kernel.invoke",
                    "abandoned_after_s": round(abandoned, 3),
                    "outcome": outcome,
                }

            out["cancellation"] = await P.attempt_async(_cancellation)
        if "tasks" in servers:
            out["tasks"] = await P.attempt_async(
                lambda: _route_answer(call("tasks", "task_capable"))
            )
        if "caching" in servers:
            async def _caching():
                async with self._connected(servers["caching"], load_prompts=False) as plugin:
                    kernel = Kernel()
                    kernel.add_plugin(plugin)
                    marks = []
                    for label in ("first", "within_ttl", "after_ttl"):
                        if label == "after_ttl":
                            await asyncio.sleep(P.CACHE_PAUSE_S)
                        started = P.wall_clock()
                        # What each agent turn reads: the kernel's function list.
                        listed = kernel.get_full_list_of_function_metadata()
                        marks.append({"call": label, "started": started, "finished": P.wall_clock(), "count": len(listed)})
                return {"route": "adapter", "api": "Kernel.get_full_list_of_function_metadata", "marks": marks}

            out["caching"] = await P.attempt_async(_caching)
        return out

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through the plugin.

        ``message_handler`` is a method the docs say to override in a subclass, calling
        ``super()`` to keep the default, which reloads tools and prompts on their
        list_changed notifications. The probe does exactly that. Tool and prompt lists are
        read from the kernel plugin an agent is handed; there is no resource list and no
        subscribe API, so subscriptions go through the public ``plugin.session``.
        """
        return asyncio.run(self._notify_async(server))

    async def _notify_async(self, server: ServerHandle) -> dict:
        import anyio
        from semantic_kernel import Kernel

        from .. import notify as N

        recorder = N.Recorder()
        versions: list[str] = []
        plugin = _recording_plugin(self._build_plugin(server), recorder)
        async with _Connection(self, server, {}, plugin=plugin, versions=versions) as plugin:
            async with anyio.create_task_group() as group:
                negotiated = versions[0] if versions else None
                subscriptions = N.SessionSubscriptions(plugin.session, recorder, negotiated, group)

                def functions(kind: str) -> list[str]:
                    kernel = Kernel()
                    kernel.add_plugin(plugin)
                    return [
                        remote for local in kernel.get_plugin(_PLUGIN).functions
                        if (entry := plugin._mcp_registered_names.get(local)) and entry[0] == kind
                        for remote in [entry[1]]
                    ]

                async def tools():
                    return functions("tool")

                async def prompts():
                    return functions("prompt")

                async def call(name):
                    kernel = Kernel()
                    kernel.add_plugin(plugin)
                    try:
                        result = await kernel.invoke(_function(kernel, name))
                    except Exception as exc:  # noqa: BLE001 - the failure is the observation
                        # The default message_handler awaits load_tools() inside the
                        # session's receive loop, so the reload's tools/list reply (and
                        # this call's) cannot be read until request_timeout expires.
                        return _exception_rep(name, exc).text
                    return getattr(result, "value", None)

                out = await N.run(
                    N.Adapter(
                        call=call,
                        recorder=recorder,
                        hook="MCPPluginBase.message_handler (subclass override calling super())",
                        lists={"tools": tools, "resources": None, "prompts": prompts},
                        no_list_reason="MCPPluginBase has no resource methods",
                        subscribe=subscriptions.subscribe,
                        unsubscribe=subscriptions.unsubscribe,
                        subscribe_route="session",
                        subscribe_api=f"MCPStdioPlugin.session: {subscriptions.api}",
                        negotiated=negotiated,
                    )
                )
                group.cancel_scope.cancel()
        return out


# ----------------------------------------------------------------------
# Connection, scripted endpoint and helpers
# ----------------------------------------------------------------------


class _Connection:
    """``async with`` a connected plugin, recording the negotiated protocol version."""

    def __init__(self, renderer, server, options, plugin=None, versions=None):
        self.renderer = renderer
        self.server = server
        self.options = options
        self.plugin = plugin
        self.versions = versions if versions is not None else []

    async def __aenter__(self):
        from mcp import ClientSession

        if self.plugin is None:
            self.plugin = self.renderer._build_plugin(self.server, **self.options)
        original = ClientSession.initialize
        versions = self.versions

        async def capture_initialize(session, *args, **kwargs):
            result = await original(session, *args, **kwargs)
            versions.append(str(result.protocolVersion))
            return result

        self._original = original
        ClientSession.initialize = capture_initialize
        try:
            # connect() waits on an event that _inner_connect sets only on success or a
            # transport failure; when session.initialize() raises it is never set and
            # connect() never returns. The bound turns that hang into an error.
            await asyncio.wait_for(self.plugin.connect(), timeout=_CONNECT_TIMEOUT)
        except asyncio.TimeoutError as exc:
            raise RenderError(
                f"semantic_kernel plugin connect() did not return within {_CONNECT_TIMEOUT}s "
                "(MCPPluginBase._inner_connect never signals ready when initialize fails)"
            ) from exc
        finally:
            ClientSession.initialize = original
        if len(set(versions)) != 1:
            await self.plugin.close()
            raise RenderError(
                "semantic_kernel renderer could not establish one negotiated MCP protocol "
                f"version; observed {sorted(set(versions))}"
            )
        self.renderer._negotiated_mcp_spec_version = versions[0]
        return self.plugin

    async def __aexit__(self, *exc):
        await self.plugin.close()


class _ScriptedEndpoint:
    """The chat.completions endpoint an ``AsyncOpenAI`` client reaches, scripted.

    A request that offers tools is an agent turn: the first one calls ``call`` if given,
    and every other one ends the turn. A request without tools is the kernel serving a
    sampling request; it answers with the client probe's sentinel.
    """

    def __init__(self, call: tuple[str, dict] | None, log=None) -> None:
        self.call = call
        self.log = log
        self.bodies: list[dict] = []

    def service(self):
        import httpx
        from openai import AsyncOpenAI
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

        client = AsyncOpenAI(
            api_key="placeholder-not-a-credential",
            base_url="http://scripted.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._handle)),
            max_retries=0,
        )
        return OpenAIChatCompletion(ai_model_id=_MODEL, async_client=client)

    def _handle(self, request):
        import httpx

        from .. import client_probe as C

        body = json.loads(request.content)
        self.bodies.append(body)
        messages = body.get("messages") or []
        after_tool = any(m.get("role") == "tool" for m in messages)
        tools = body.get("tools") or []
        if not tools:
            if self.log is not None:
                self.log.note("OpenAIChatCompletion (kernel service)", tools=False)
            message = {"role": "assistant", "content": C.SAMPLED_AFTER_TOOL if after_tool else C.SAMPLED}
            finish = "stop"
        elif self.call is not None and not after_tool:
            offered = [t["function"]["name"] for t in tools]
            wanted = self.call[0] if self.call[0] in offered else f"{_PLUGIN}-{self.call[0]}"
            name = wanted if wanted in offered else None
            if name is None:
                return httpx.Response(400, json={"error": {"message": f"no tool {self.call[0]!r} offered"}})
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": _CALL_ID, "type": "function",
                     "function": {"name": name, "arguments": json.dumps(self.call[1])}}
                ],
            }
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": "done"}
            finish = "stop"
        return httpx.Response(
            200,
            json={
                "id": f"scripted-{len(self.bodies)}",
                "object": "chat.completion",
                "created": 0,
                "model": _MODEL,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            },
        )

    def captured(self) -> dict:
        out: dict[str, Any] = {"bodies": self.bodies}
        for body in self.bodies:
            for message in body.get("messages") or []:
                if message.get("role") == "tool" and message.get("tool_call_id") == _CALL_ID:
                    out["tool_message"] = message
        return out


class _SkLog(logging.Handler):
    """Collect what SK's default logging_callback writes, at a chosen logger level."""

    def __init__(self, level: int) -> None:
        super().__init__(level=logging.NOTSET)
        self.target_level = level
        self.messages: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.funcName != "logging_callback":
            return  # SK's own diagnostics on the same logger
        self.messages.append({"level": record.levelname.lower(), "data": record.msg})

    def __enter__(self):
        self.logger = logging.getLogger(_SK_LOGGER)
        self.previous = self.logger.level
        self.logger.setLevel(self.target_level)
        self.logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self)
        self.logger.setLevel(self.previous)


def _recording_plugin(plugin, recorder):
    """The documented extension: override message_handler, keep the default via super()."""

    base = type(plugin)

    class Recording(base):  # type: ignore[misc, valid-type]
        async def message_handler(self, message):
            await recorder.message_handler(message)
            await super().message_handler(message)

    plugin.__class__ = Recording
    return plugin


def _local(tool: str, names) -> str:
    """The function name for ``tool``, given either the MCP name or the rendered one."""
    prefix = f"{_PLUGIN}-"
    if tool not in names and tool.startswith(prefix) and tool[len(prefix):] in names:
        return tool[len(prefix):]
    return tool


def _function(kernel, tool: str):
    names = kernel.get_plugin(_PLUGIN).functions
    local = _local(tool, names)
    if local not in names:
        raise RenderError(f"semantic_kernel renderer found no function {tool!r} (has: {sorted(names)})")
    return kernel.get_function(_PLUGIN, local)


def _remote_name(plugin, local: str) -> str:
    # The plugin records which MCP name each normalized function name stands for.
    entry = getattr(plugin, "_mcp_registered_names", {}).get(local)
    return entry[1] if entry else local


def _prompt_entry(metadata) -> dict:
    return {
        "name": metadata.name,
        "title": None,
        "description": metadata.description,
        "arguments": [
            {"name": p.name, "description": p.description, "required": p.is_required}
            for p in metadata.parameters
        ],
    }


def _first_item(message):
    items = getattr(message, "items", None) or []
    if not items:
        return getattr(message, "content", None)
    first = items[0]
    if type(first).__name__ == "TextContent":
        return {"type": "text", "text": first.text}
    return {"type": getattr(first, "content_type", None) or type(first).__name__, "text": None}


def _block_of(item: Any) -> dict:
    """Classify one returned KernelContent by what the caller can tell it is.

    The MCP block the plugin converted is kept on ``inner_content``; its type is
    recorded as ``inner`` so a text-shaped embedded resource can be told apart.
    """

    kind = type(item).__name__
    inner = getattr(item, "inner_content", None)
    block: dict[str, Any] = {"inner": getattr(inner, "type", None)}
    if kind == "TextContent":
        block.update({"type": "text", "text": item.text})
    elif kind == "ImageContent":
        block.update({"type": "image", "mimeType": item.mime_type})
    elif kind == "AudioContent":
        block.update({"type": "audio", "mimeType": item.mime_type})
    elif kind == "BinaryContent":
        block.update({"type": "binary", "mimeType": item.mime_type, "uri": str(item.uri) if item.uri else None})
    elif isinstance(item, str):
        block.update({"type": "text", "text": item})
    else:
        block.update({"type": kind})
    metadata = getattr(item, "metadata", None) or {}
    annotations = {k: v for k, v in metadata.items() if v is not None} if isinstance(metadata, dict) else {}
    if annotations:
        block["annotations"] = annotations
    return block


def _exception_rep(tool: str, exc: BaseException) -> ResultRep:
    # SK wraps the adapter's failure twice (KernelInvokeException around
    # FunctionExecutionException around the real error); keep the whole chain.
    parts, seen = [], exc
    while seen is not None and len(parts) < 4:
        parts.append(f"{type(seen).__name__}: {str(seen)[:500]}")
        seen = seen.__cause__
    text = " <- ".join(parts)
    return normalize_framework_result(
        tool=tool,
        origin=ID,
        is_error=True,
        text=text,
        error_channel="exception",
        raw=repr(exc)[:2000],
    )


async def _text_answer(awaitable) -> dict:
    return {"text": await awaitable}


async def _route_answer(awaitable) -> dict:
    return {"route": "adapter", "text": await awaitable}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (f"<{len(v)} chars>" if k in _OPAQUE and isinstance(v, str) and len(v) > 32 else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value
