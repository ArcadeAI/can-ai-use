"""Pydantic AI renderer (DESIGN.md section 8).

Connects through the supported ``pydantic_ai.mcp.MCPToolset`` API and reads the
tool definitions the agent would expose. Each Pydantic AI
``ToolDefinition`` carries ``name``, ``description`` and
``parameters_json_schema``. This is the framework tool-definition boundary, before
provider-specific request serialization.
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

ID = "pydantic_ai"
REQUIRES = ("pydantic_ai",)


def available() -> bool:
    return util.find_spec("pydantic_ai") is not None and util.find_spec("pydantic_ai.mcp") is not None


def get_renderer() -> "PydanticAIRenderer":
    return PydanticAIRenderer()


class PydanticAIRenderer:
    id = ID

    def versions(self) -> dict:
        version = dist_version("pydantic-ai-slim")
        if version == "unknown":
            version = dist_version("pydantic-ai")
        return {"framework": version, "adapter": version}

    def evidence(self) -> RendererEvidence:
        negotiated_version = getattr(
            self,
            "_negotiated_mcp_spec_version",
            None,
        )
        return RendererEvidence(
            capture_api="pydantic_ai.mcp.MCPToolset.get_tools",
            capture_object="pydantic_ai.tools.ToolDefinition",
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=negotiated_version,
            protocol_version_evidence=(
                "protocol version exposed by MCPToolset.client"
                if negotiated_version and negotiated_version >= "2026-07-28"
                else "protocolVersion exposed by MCPToolset.client.initialize_result"
            ),
            limitation=(
                "No Pydantic AI model adapter or serialized provider request was captured."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        try:
            return anyio.run(self._render_async, server)
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"pydantic_ai renderer failed: {exc}") from exc

    async def _render_async(self, server: ServerHandle) -> list[ToolRep]:
        toolset = self._build_server(server)
        async with toolset:
            tools = await self._list_tools(toolset)
            client = toolset.client
            initialize_result = getattr(client, "initialize_result", None)
            self._negotiated_mcp_spec_version = (
                getattr(initialize_result, "protocolVersion", None)
                or getattr(client, "protocol_version", None)
            )
            if not self._negotiated_mcp_spec_version:
                raise RenderError(
                    "pydantic_ai renderer could not establish the negotiated "
                    "MCP protocol version"
                )
        return [self._to_rep(t) for t in tools]

    # `MCPToolset.call_tool` is the toolset method an agent run calls, and unless a
    # `process_tool_call` hook is installed it delegates straight to
    # `direct_call_tool`. Both are published so that delegation is a stated
    # measurement. The toolset's `tool_error_behavior` (default `'retry'`) decides
    # whether an `isError` result becomes a `ModelRetry` or a `ToolFailed`; the agent
    # run then records the raised `ModelRetry` as a typed `RetryPromptPart`.
    DEFAULT_RESULT_BOUNDARY = "direct"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Direct toolset call",
                capture_api="pydantic_ai.mcp.MCPToolset.direct_call_tool",
                capture_object="the mapped toolset result, or the exception it raises",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Agent toolset call",
                capture_api="pydantic_ai.mcp.MCPToolset.call_tool",
                capture_object="the value the agent's tool executor receives",
                agent_path=True,
                limitation=(
                    "The toolset method is driven directly. The agent's own "
                    "conversion of a raised ModelRetry into a RetryPromptPart "
                    "happens above this boundary and is not captured here."
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
        """Invoke one tool and capture one declared framework-result boundary."""

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
        if boundary not in {"direct", "agent"}:
            raise RenderError(
                f"pydantic_ai renderer has no result boundary {boundary!r}"
            )

        toolset = self._build_server(server)
        try:
            if boundary == "direct":
                returned = await toolset.direct_call_tool(tool, arguments)
            else:
                async with toolset:
                    tools = await self._tools_by_name(toolset)
                    if tool not in tools:
                        raise RenderError(
                            f"pydantic_ai renderer found no tool named {tool!r}"
                        )
                    returned = await toolset.call_tool(
                        tool, arguments, None, tools[tool]
                    )
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001 - the failure mode is the observation
            # A tool that reported failure through `isError` may surface here as a
            # raised exception instead, which is a different contract for the agent
            # rather than a scanner error, so it is recorded rather than propagated.
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                is_error=True,
                text=f"{type(exc).__name__}: {exc}",
                error_channel="exception",
                raw=repr(exc),
            )
        return self._result_to_rep(tool, returned)

    async def _tools_by_name(self, toolset) -> dict[str, Any]:
        """The toolset's own ToolsetTool objects, keyed by name.

        `call_tool` needs the ToolsetTool the agent would have resolved, so it is
        taken from `get_tools` rather than reconstructed.
        """

        result = await toolset.get_tools(None)
        if isinstance(result, dict):
            return dict(result)
        return {getattr(t, "name", getattr(getattr(t, "tool_def", None), "name", "")): t for t in result}

    def _result_to_rep(self, tool: str, returned: Any) -> ResultRep:
        """Map whatever the adapter returned onto the common result shape.

        Pydantic AI returns the structured content when every content part is text and
        the mapped content blocks otherwise, so the same tool can hand the agent either
        a typed value or a list. Both shapes are recorded rather than normalized away.
        """

        if isinstance(returned, list):
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=[self._block_of(item) for item in returned],
                raw=repr(returned)[:2000],
            )
        if isinstance(returned, dict):
            return normalize_framework_result(
                tool=tool, origin=self.id, structured_content=returned, raw=returned
            )
        if getattr(returned, "media_type", None) is not None or type(returned).__name__ in (
            "BinaryImage",
            "BinaryContent",
        ):
            # A result with one media part comes back as that part itself, not a
            # one-item list; it is the same typed object the list case carries.
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=[self._block_of(returned)],
                raw=repr(returned)[:2000],
            )
        return normalize_framework_result(
            tool=tool, origin=self.id, text=str(returned), raw=repr(returned)[:2000]
        )

    @staticmethod
    def _block_of(item: Any) -> dict[str, Any]:
        """Classify one returned item by what the agent can actually tell it is.

        Pydantic AI maps several MCP content kinds onto a bare ``str``. Recording those
        as ``text`` is not a guess about the adapter's intent, it is the agent's view:
        nothing in a plain string distinguishes an embedded resource or a resource link
        from an ordinary text block, so the original kind is genuinely gone.
        """

        if isinstance(item, str):
            return {"type": "text", "text": item}

        media_type = getattr(item, "media_type", None) or ""
        if media_type.startswith("image/") or type(item).__name__ == "BinaryImage":
            return {"type": "image", "mimeType": media_type or None}
        if media_type.startswith("audio/"):
            return {"type": "audio", "mimeType": media_type}
        return {"type": type(item).__name__, "mimeType": media_type or None}

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through ``MCPToolset``'s own options.

        ``log_handler``, ``progress_handler`` and ``log_level`` are constructor options;
        completion is reachable only on the FastMCP client the toolset exposes as
        ``toolset.client``, reported with ``route: "session"``.
        """
        return anyio.run(self._utilities_async, servers)

    async def _utilities_async(self, servers: dict[str, ServerHandle]) -> dict:
        import mcp.types as types
        from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
        from pydantic_ai.mcp import MCPToolset

        from .. import probe as P

        events: list[dict] = []
        messages: list[dict] = []

        async def on_progress(progress, total=None, message=None):
            events.append({"progress": progress, "total": total, "message": message})

        async def on_log(entry):
            messages.append({"level": getattr(entry, "level", None), "data": getattr(entry, "data", None)})

        def toolset(feature: str, **options):
            server = servers[feature]
            if server.transport == "http":
                transport = StreamableHttpTransport(url=server.url, headers=dict(server.headers) or None)
            else:
                transport = StdioTransport(command=server.command, args=list(server.args))
            return MCPToolset(transport, max_retries=0, log_handler=on_log, progress_handler=on_progress, **options)

        async def call(feature: str, tool: str, arguments: dict | None = None, **options):
            async with toolset(feature, **options) as ts:
                return P.result_text(await ts.direct_call_tool(tool, arguments or {}))

        out: dict = {}
        if "tools.list" in servers:
            async def _names():
                async with toolset("tools.list") as ts:
                    return {"route": "adapter", "names": list(await ts.get_tools(None))}

            out["tools.list"] = await P.attempt_async(_names)
        if "progress" in servers:
            async def _progress():
                text = await call("progress", "slow_with_progress")
                return {"route": "adapter", "api": "MCPToolset(progress_handler=...)", "events": list(events), "text": text}

            out["progress"] = await P.attempt_async(_progress)
        if "logging" in servers:
            async def _logging():
                text = await call("logging", "emit_logs")
                await anyio.sleep(0.2)
                return {"route": "adapter", "api": "MCPToolset(log_handler=...)", "messages": list(messages), "text": text}

            out["logging"] = await P.attempt_async(_logging)
        if "set_level" in servers:
            async def _set_level():
                text = await call("set_level", "emit_logs", log_level=P.SET_LEVEL)
                return {"route": "adapter", "api": "MCPToolset(log_level=...)", "level": P.SET_LEVEL, "text": text}

            out["set_level"] = await P.attempt_async(_set_level)
        if "completion" in servers:
            async def _completion():
                async with toolset("completion") as ts:
                    name, argument, value = P.COMPLETION_PROMPT
                    prompt = await ts.client.complete(types.PromptReference(type="ref/prompt", name=name), {"name": argument, "value": value})
                    uri, argument, value = P.COMPLETION_RESOURCE
                    resource = await ts.client.complete(types.ResourceTemplateReference(type="ref/resource", uri=uri), {"name": argument, "value": value})
                    return {"route": "session", "api": "MCPToolset.client.complete (FastMCP client)",
                            "prompt": list(prompt.values), "resource": list(resource.values)}

            out["completion"] = await P.attempt_async(_completion)
        if "cancellation" in servers:
            async def _cancellation():
                async with toolset("cancellation") as ts:
                    started = anyio.current_time()
                    outcome = "cancelled"
                    with anyio.move_on_after(P.CANCEL_AFTER_S):
                        outcome = f"returned: {P.result_text(await ts.direct_call_tool('cancellable_wait', {'seconds': P.CANCEL_WAIT_S}))}"
                    abandoned = anyio.current_time() - started
                    await anyio.sleep(0.3)
                    return {"route": "adapter", "mechanism": "anyio cancel scope around MCPToolset.direct_call_tool",
                            "abandoned_after_s": round(abandoned, 3), "outcome": outcome}

            out["cancellation"] = await P.attempt_async(_cancellation)
        if "tasks" in servers:
            async def _tasks():
                return {"route": "adapter", "text": await call("tasks", "task_capable")}

            out["tasks"] = await P.attempt_async(_tasks)
        if "caching" in servers:
            async def _caching():
                async with toolset("caching") as ts:
                    marks = []
                    for label in ("first", "within_ttl", "after_ttl"):
                        if label == "after_ttl":
                            await anyio.sleep(P.CACHE_PAUSE_S)
                        started = P.wall_clock()
                        listed = await ts.get_tools(None)
                        marks.append({"call": label, "started": started, "finished": P.wall_clock(), "count": len(listed)})
                    return {"route": "adapter", "api": "MCPToolset.get_tools (defaults)", "marks": marks}

            out["caching"] = await P.attempt_async(_caching)
        return out

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources and prompts through MCPToolset's own methods, normalized."""
        return anyio.run(self._probe_async, server)

    async def _probe_async(self, server: ServerHandle) -> dict:
        from .. import probe as P

        toolset = self._build_server(server)
        out: dict = {}
        async with toolset:
            async def _list():
                return {"entries": [P.resource_entry(r) for r in await toolset.list_resources()]}

            out["resources.list"] = await _attempt(_list)
            entries = out["resources.list"].get("entries") or []
            text_uri = next((e["uri"] for e in entries if (e.get("mimeType") or "").startswith("text/")), None)
            blob_uri = next((e["uri"] for e in entries if not (e.get("mimeType") or "").startswith("text/")), None)

            async def _read(uri):
                value = await toolset.read_resource(uri)
                first = value[0] if isinstance(value, list) and value else value
                if isinstance(first, str):
                    # A bare string: the contents arrive, the uri and mimeType do not.
                    return P.text_contents(None, None, first)
                data = getattr(first, "data", None)
                mime = getattr(first, "media_type", None)
                if data is not None:
                    return P.blob_contents(None, mime, data)
                return P.text_contents(None, mime, getattr(first, "text", None))

            out["resources.read.text"] = await _attempt(lambda: _read(text_uri)) if text_uri else {"ok": False, "error": "no text resource listed"}
            out["resources.read.blob"] = await _attempt(lambda: _read(blob_uri)) if blob_uri else {"ok": False, "error": "no binary resource listed"}

            async def _templates():
                return {"entries": [P.template_entry(t) for t in await toolset.list_resource_templates()]}

            out["resources.templates"] = await _attempt(_templates)

            async def _prompts():
                return {"entries": [P.prompt_entry(x) for x in await toolset.list_prompts()]}

            out["prompts.list"] = await _attempt(_prompts)
            prompts = out["prompts.list"].get("entries") or []

            async def _get():
                first = prompts[0]
                arguments = {a["name"]: "x" for a in first["arguments"] if a.get("required")}
                result = await toolset.get_prompt(first["name"], arguments)
                messages = getattr(result, "messages", None) or []
                return {
                    "description": getattr(result, "description", None),
                    "messages": [
                        P.message_entry(getattr(m, "role", None), getattr(m, "content", None))
                        for m in messages
                    ],
                }

            out["prompts.get"] = await _attempt(_get) if prompts else {"ok": False, "error": "no prompt listed"}
        return out

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling, elicitation and roots through MCPToolset's constructor hooks.

        ``sampling_handler``, ``elicitation_handler`` and ``roots`` are MCPToolset
        parameters. Two things are not: declaring ``sampling.tools`` and changing roots
        after connecting. Both exist on the fastmcp ``Client`` the toolset wraps and
        exposes as ``toolset.client``, so they are exercised there and recorded as the
        underlying route.
        """
        return anyio.run(self._probe_client_async, server)

    async def _probe_client_async(self, server: ServerHandle) -> dict:
        import mcp.types as types
        from fastmcp.client.elicitation import ElicitResult

        from .. import client_probe as C

        log = C.HandlerLog()

        def sampling_handler(messages, params, _context):
            reply = C.sampling_reply(messages, getattr(params, "tools", None))
            log.note("sampling_handler", tools=bool(getattr(params, "tools", None)), replied=reply["stopReason"])
            return C.sampling_result(types, reply)

        async def elicitation_handler(_message, response_type, params, _context):
            mode = getattr(params, "mode", None) or ("form" if response_type is not None else "url")
            log.note("elicitation_handler", mode=mode)
            if mode == "url":
                return ElicitResult(action="accept")
            return {"name": C.ELICITED}

        async def call(tool: str, prepare=None, after_connect=None, **hooks):
            toolset = self._build_server(server, **hooks)
            if prepare is not None:
                prepare(toolset)
            async with toolset:
                if after_connect is not None:
                    await after_connect(toolset)
                try:
                    returned = await toolset.direct_call_tool(tool, {})
                except Exception as exc:  # noqa: BLE001 - an isError result raises here
                    return normalize_framework_result(
                        tool=tool, origin=self.id, is_error=True,
                        text=f"{type(exc).__name__}: {exc}", error_channel="exception",
                    )
            return self._result_to_rep(tool, returned)

        def declare_sampling_tools(toolset):
            toolset.client.set_sampling_callback(
                sampling_handler,
                sampling_capabilities=types.SamplingCapability(tools=types.SamplingToolsCapability()),
            )

        async def change_roots(toolset):
            toolset.client.set_roots([C.CHANGED_ROOT_URI])
            await toolset.client.send_roots_list_changed()

        out: dict = {}
        out["sampling"] = await C.attempt_async(
            "sampling", "MCPToolset(sampling_handler=...)", "adapter",
            lambda: call("ask_sampling", sampling_handler=sampling_handler), log,
        )
        out["sampling.tools"] = await C.attempt_async(
            "sampling.tools", "toolset.client.set_sampling_callback(sampling_capabilities=tools)", "underlying",
            lambda: call("ask_sampling_tools", prepare=declare_sampling_tools), log,
        )
        out["elicitation.form"] = await C.attempt_async(
            "elicitation.form", "MCPToolset(elicitation_handler=...)", "adapter",
            lambda: call("ask_elicitation_form", elicitation_handler=elicitation_handler), log,
        )
        out["elicitation.url"] = await C.attempt_async(
            "elicitation.url", "MCPToolset(elicitation_handler=...)", "adapter",
            lambda: call("ask_elicitation_url", elicitation_handler=elicitation_handler), log,
        )
        out["roots.list"] = await C.attempt_async(
            "roots.list", "MCPToolset(roots=[...])", "adapter",
            lambda: call("list_client_roots", roots=[C.ROOT_URI]), log,
        )
        out["roots.list_changed"] = await C.attempt_async(
            "roots.list_changed", "toolset.client.set_roots + send_roots_list_changed", "underlying",
            lambda: call("roots_changes_seen", after_connect=change_roots, roots=[C.ROOT_URI]), log,
        )
        try:
            rep = await call(
                C.CAPABILITIES_TOOL, sampling_handler=sampling_handler,
                elicitation_handler=elicitation_handler, roots=[C.ROOT_URI],
            )
            out["capabilities"] = {"text": C.rep_text(rep)[0]}
        except Exception as exc:  # noqa: BLE001
            out["capabilities"] = {"text": f"{type(exc).__name__}: {exc}"}
        return out

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through MCPToolset.

        ``message_handler`` is the documented hook: MCPToolset runs its own cache
        invalidation first and then the caller's handler. Lists go through the cached
        ``get_tools`` / ``list_resources`` / ``list_prompts`` an agent reads. The toolset
        has no subscribe API, so subscriptions go through the FastMCP client's session.
        """
        return anyio.run(self._notify_async, server)

    async def _notify_async(self, server: ServerHandle) -> dict:
        from .. import notify as N

        recorder = N.Recorder()
        toolset = self._build_server(server, message_handler=recorder.message_handler)
        async with toolset, anyio.create_task_group() as group:
            client = toolset.client
            initialize_result = getattr(client, "initialize_result", None)
            negotiated = getattr(initialize_result, "protocolVersion", None) or getattr(client, "protocol_version", None)
            session = client.session
            subscriptions = N.SessionSubscriptions(session, recorder, negotiated, group)

            async def tools():
                return [t.tool_def.name for t in await self._list_tools(toolset)]

            async def resources():
                return [str(r.uri) for r in await toolset.list_resources()]

            async def prompts():
                return [p.name for p in await toolset.list_prompts()]

            out = await N.run(
                N.Adapter(
                    call=lambda name: toolset.direct_call_tool(name, {}),
                    recorder=recorder,
                    hook="MCPToolset(message_handler=...)",
                    lists={"tools": tools, "resources": resources, "prompts": prompts},
                    subscribe=subscriptions.subscribe,
                    unsubscribe=subscriptions.unsubscribe,
                    subscribe_route="session",
                    subscribe_api=f"MCPToolset.client.session: {subscriptions.api}",
                    listen=subscriptions.listen_all,
                    listen_route="session",
                    listen_api="mcp.client.subscriptions.listen(MCPToolset.client.session, ...)",
                    negotiated=negotiated,
                )
            )
            group.cancel_scope.cancel()
        return out


    def _build_server(self, server: ServerHandle, **hooks):
        from fastmcp.client.transports import SSETransport, StdioTransport, StreamableHttpTransport
        from pydantic_ai.mcp import MCPToolset

        if server.transport == "http":
            transport = StreamableHttpTransport(
                url=server.url,
                headers=dict(server.headers) or None,
            )
        elif server.transport == "sse":
            transport = SSETransport(url=server.url)
        else:
            transport = StdioTransport(
                command=server.command,
                args=list(server.args),
            )
        # Setting max_retries avoids constructing an Agent RunContext solely to list
        # the ToolDefinition objects. No model is created or called.
        if hooks.get("message_handler") is not None:
            # The notification probe starts the fixture cold in a fresh checkout, where
            # the default 5 s initialize budget is not always enough. It only bounds
            # the connection; nothing the probe measures depends on it.
            hooks.setdefault("init_timeout", 30)
        return MCPToolset(transport, max_retries=0, **hooks)

    async def _list_tools(self, mcp_server) -> list[Any]:
        # get_tools() constructs the exact ToolDefinition objects declared as this
        # renderer's capture boundary. max_retries is fixed on the toolset, so the
        # RunContext is not consulted and no Agent or model is needed.
        get_tools = getattr(mcp_server, "get_tools", None)
        if not callable(get_tools):
            raise RenderError("pydantic_ai MCPToolset exposes no get_tools method")
        result = await get_tools(None)
        return list(result.values()) if isinstance(result, dict) else list(result)

    def _to_rep(self, tool: Any) -> ToolRep:
        tool_def = getattr(tool, "tool_def", None)
        if tool_def is not None:
            name = getattr(tool_def, "name", None)
            description = getattr(tool_def, "description", None)
            parameters = getattr(tool_def, "parameters_json_schema", None) or {}
            metadata = getattr(tool_def, "metadata", None) or {}
            retained = metadata.get("annotations") or {}
            # ToolDefinition carries the MCP outputSchema as `return_schema`, but clears
            # it before the definition reaches a model unless `include_return_schema`
            # resolves to True, which it does not by default.
            return_schema = getattr(tool_def, "return_schema", None) or None
            sent = getattr(tool_def, "include_return_schema", None) is True
        else:
            # Fallback: a raw MCP types.Tool.
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", None)
            parameters = getattr(tool, "inputSchema", None) or {}
            retained = self._annotations_of(getattr(tool, "annotations", None))
            return_schema, sent = None, False

        retained = {k: v for k, v in (retained or {}).items() if v is not None}
        framework_metadata = {"annotations": retained} if retained else {}
        payload = {"name": name, "description": description, "parameters": parameters}
        if return_schema and sent:
            payload["outputSchema"] = return_schema
        elif return_schema:
            framework_metadata["outputSchema"] = return_schema
        return normalize_function_tool(payload, origin=self.id, framework_metadata=framework_metadata)

    @staticmethod
    def _annotations_of(annotations: Any) -> dict:
        if annotations is None:
            return {}
        dump = getattr(annotations, "model_dump", None)
        if callable(dump):
            return dump()
        return annotations if isinstance(annotations, dict) else {}


async def _attempt(fn):
    try:
        return {"ok": True, **(await fn())}
    except NotImplementedError as exc:
        return {"ok": False, "unsupported": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the failure is the observation
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
