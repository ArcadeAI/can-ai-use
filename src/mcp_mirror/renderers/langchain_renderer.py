"""LangChain renderer (DESIGN.md section 8).

Loads tools with ``langchain-mcp-adapters`` (the real adapter) and then runs each
through ``langchain_core.utils.function_calling.convert_to_openai_tool``. The result
is an OpenAI-compatible tool dictionary. This renderer does not bind a chat model or
capture a serialized provider request, so it reports that boundary without claiming
stronger provider evidence.
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

ID = "langchain"
REQUIRES = ("langchain_mcp_adapters", "langchain_core")


def available() -> bool:
    return all(util.find_spec(mod) is not None for mod in REQUIRES)


def get_renderer() -> "LangChainRenderer":
    return LangChainRenderer()


def _block_of(item: Any) -> dict[str, Any]:
    """Classify one returned item by what the agent can actually tell it is."""

    if isinstance(item, str):
        return {"type": "text", "text": item}
    if isinstance(item, dict):
        return item
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:  # noqa: BLE001
            pass
    return {"type": type(item).__name__}


def _connection(handle: ServerHandle) -> dict:
    if handle.transport == "http":
        conn = {"url": handle.url, "transport": "streamable_http"}
        if handle.headers:
            conn["headers"] = dict(handle.headers)
        return conn
    if handle.transport == "sse":
        return {"url": handle.url, "transport": "sse"}
    return {"command": handle.command, "args": list(handle.args), "transport": "stdio"}


class LangChainRenderer:
    id = ID

    def versions(self) -> dict:
        return {
            "framework": dist_version("langchain-mcp-adapters"),
            "adapter": dist_version("langchain-core"),
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api="langchain_core.utils.function_calling.convert_to_openai_tool",
            capture_object="OpenAI-compatible function-tool dictionary",
            capture_stage="provider_format",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(
                self,
                "_negotiated_mcp_spec_version",
                None,
            ),
            protocol_version_evidence=(
                "protocolVersion returned by ClientSession.initialize inside the "
                "langchain-mcp-adapters session"
            ),
            limitation=(
                "No chat model was bound and no serialized provider request was captured."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        try:
            return anyio.run(self._render_async, server)
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"langchain renderer failed: {exc}") from exc

    async def _render_async(self, server: ServerHandle) -> list[ToolRep]:
        from langchain_core.utils.function_calling import convert_to_openai_tool
        from langchain_mcp_adapters.client import create_session
        from langchain_mcp_adapters.tools import load_mcp_tools

        connection = _connection(server)
        async with create_session(connection) as session:
            initialize_result = await session.initialize()
            self._negotiated_mcp_spec_version = initialize_result.protocolVersion
            tools = await load_mcp_tools(
                session,
                connection=connection,
                server_name="src",
            )

        return self._to_reps(tools)

    # LangChain is the one adapter measured here whose result boundaries disagree.
    # `load_mcp_tools` builds the StructuredTool with
    # `handle_tool_error=_handle_mcp_tool_error`, and an `isError` result raises the
    # internal `_MCPToolExecutionError`. Invoked with a plain argument dict the tool
    # returns the handler's content blocks, so the failure reads as ordinary text.
    # Invoked with a ToolCall, which is the shape an agent and ToolNode use, the same
    # call returns a ToolMessage carrying `status="error"`. Both are published.
    DEFAULT_RESULT_BOUNDARY = "direct"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Direct tool invocation",
                capture_api="langchain_core.tools.StructuredTool.ainvoke(dict)",
                capture_object="the tool return value, before any ToolMessage is built",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Agent tool invocation",
                capture_api="langchain_core.tools.StructuredTool.ainvoke(ToolCall)",
                capture_object="the ToolMessage an agent or ToolNode puts in the message list",
                agent_path=True,
                limitation=(
                    "The ToolMessage is built by langchain_core from the same call a "
                    "ToolNode makes. No chat model was bound and no provider request "
                    "was captured."
                ),
            ),
        ]

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through ``MultiServerMCPClient`` and its ``Callbacks``.

        ``MultiServerMCPClient.session()`` hands out the raw mcp ``ClientSession``; what
        is reachable only there is reported with ``route: "session"``.
        """
        return anyio.run(self._utilities_async, servers)

    async def _utilities_async(self, servers: dict[str, ServerHandle]) -> dict:
        import asyncio

        import mcp.types as types
        from langchain_mcp_adapters.callbacks import Callbacks
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        from .. import probe as P

        events: list[dict] = []
        messages: list[dict] = []

        async def on_progress(progress, total, message, context):
            events.append({"progress": progress, "total": total, "message": message})

        async def on_log(params, context):
            messages.append({"level": getattr(params, "level", None), "data": getattr(params, "data", None)})

        callbacks = Callbacks(on_progress=on_progress, on_logging_message=on_log)

        def client(feature: str) -> MultiServerMCPClient:
            return MultiServerMCPClient({"src": _connection(servers[feature])}, callbacks=callbacks)

        async def with_tools(feature: str, body):
            c = client(feature)
            async with c.session("src") as session:
                tools = {t.name: t for t in await load_mcp_tools(session, callbacks=callbacks, server_name="src")}
                return await body(session, tools)

        out: dict = {}
        if "tools.list" in servers:
            async def _names():
                return {"route": "adapter", "names": [t.name for t in await client("tools.list").get_tools()]}

            out["tools.list"] = await _attempt(_names)
        if "progress" in servers:
            async def _progress():
                text = await with_tools("progress", lambda s, t: t["slow_with_progress"].ainvoke({}))
                return {"route": "adapter", "api": "Callbacks(on_progress=...)", "events": list(events), "text": P.result_text(text)}

            out["progress"] = await _attempt(_progress)
        if "logging" in servers:
            async def _logging():
                text = await with_tools("logging", lambda s, t: t["emit_logs"].ainvoke({}))
                await asyncio.sleep(0.2)
                return {"route": "adapter", "api": "Callbacks(on_logging_message=...)", "messages": list(messages), "text": P.result_text(text)}

            out["logging"] = await _attempt(_logging)
        if "set_level" in servers:
            async def _set_level():
                async def body(session, tools):
                    await session.set_logging_level(P.SET_LEVEL)
                    return await tools["emit_logs"].ainvoke({})

                text = await with_tools("set_level", body)
                return {"route": "session", "api": "MultiServerMCPClient.session().set_logging_level", "level": P.SET_LEVEL, "text": P.result_text(text)}

            out["set_level"] = await _attempt(_set_level)
        if "completion" in servers:
            async def _completion():
                async def body(session, tools):
                    name, argument, value = P.COMPLETION_PROMPT
                    prompt = await session.complete(types.PromptReference(type="ref/prompt", name=name), {"name": argument, "value": value})
                    uri, argument, value = P.COMPLETION_RESOURCE
                    resource = await session.complete(types.ResourceTemplateReference(type="ref/resource", uri=uri), {"name": argument, "value": value})
                    return {"prompt": prompt.completion.values, "resource": resource.completion.values}

                return {"route": "session", "api": "MultiServerMCPClient.session().complete", **await with_tools("completion", body)}

            out["completion"] = await _attempt(_completion)
        if "cancellation" in servers:
            async def _cancellation():
                async def body(session, tools):
                    loop = asyncio.get_running_loop()
                    started = loop.time()
                    task = asyncio.create_task(tools["cancellable_wait"].ainvoke({"seconds": P.CANCEL_WAIT_S}))
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
                    return {"route": "adapter", "mechanism": "asyncio task.cancel() around StructuredTool.ainvoke", "abandoned_after_s": round(abandoned, 3), "outcome": outcome}

                return await with_tools("cancellation", body)

            out["cancellation"] = await _attempt(_cancellation)
        if "tasks" in servers:
            async def _tasks():
                text = await with_tools("tasks", lambda s, t: t["task_capable"].ainvoke({}))
                return {"route": "adapter", "text": P.result_text(text)}

            out["tasks"] = await _attempt(_tasks)
        if "caching" in servers:
            async def _caching():
                c = client("caching")
                marks = []
                for call in ("first", "within_ttl", "after_ttl"):
                    if call == "after_ttl":
                        await asyncio.sleep(P.CACHE_PAUSE_S)
                    started = P.wall_clock()
                    listed = await c.get_tools()
                    marks.append({"call": call, "started": started, "finished": P.wall_clock(), "count": len(listed)})
                return {"route": "adapter", "api": "MultiServerMCPClient.get_tools", "marks": marks}

            out["caching"] = await _attempt(_caching)
        return out

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources and prompts through langchain-mcp-adapters' loaders, normalized.

        The adapter has no listing API for prompts or templates and no dedicated read
        by URI: ``load_mcp_resources`` lists and reads in one call and returns Blobs,
        ``load_mcp_prompt`` expands by name. Those are the routes an agent has.
        """
        return anyio.run(self._probe_async, server)

    async def _probe_async(self, server: ServerHandle) -> dict:
        from langchain_mcp_adapters.client import create_session
        from langchain_mcp_adapters.prompts import load_mcp_prompt
        from langchain_mcp_adapters.resources import load_mcp_resources

        from .. import probe as P

        out: dict = P.unsupported("langchain-mcp-adapters exposes no listing API for this")
        connection = _connection(server)
        async with create_session(connection) as session:
            await session.initialize()

            async def _resources():
                blobs = await load_mcp_resources(session)
                entries = []
                for blob in blobs:
                    meta = getattr(blob, "metadata", None) or {}
                    entries.append(
                        {
                            "uri": P._s(meta.get("uri") or getattr(blob, "path", None) or getattr(blob, "source", None)),
                            "name": None,
                            "title": None,
                            "description": None,
                            "mimeType": getattr(blob, "mimetype", None),
                            "_blob": blob,
                        }
                    )
                return {"entries": entries}

            listed = await _attempt(_resources)
            blobs = listed.pop("entries", []) if listed.get("ok") else []
            out["resources.list"] = {**listed, "entries": [{k: v for k, v in e.items() if k != "_blob"} for e in blobs]}
            text = next((e for e in blobs if (e.get("mimeType") or "").startswith("text/")), None)
            binary = next((e for e in blobs if e.get("mimeType") and not e["mimeType"].startswith("text/")), None)
            if text is not None:
                out["resources.read.text"] = {"ok": True, **P.text_contents(text["uri"], text["mimeType"], text["_blob"].as_string())}
            else:
                out["resources.read.text"] = {"ok": False, "error": "no text resource loaded"}
            if binary is not None:
                out["resources.read.blob"] = {"ok": True, **P.blob_contents(binary["uri"], binary["mimeType"], binary["_blob"].as_bytes())}
            else:
                out["resources.read.blob"] = {"ok": False, "error": "no binary resource loaded"}

            # Prompt names are not discoverable through the adapter; the fixture's
            # prompt is looked up through the raw session so expansion can be tested.
            raw = await session.list_prompts()
            first = raw.prompts[0] if raw.prompts else None

            async def _get():
                arguments = {a.name: "x" for a in (first.arguments or []) if a.required}
                messages = await load_mcp_prompt(session, first.name, arguments=arguments)
                return {
                    "description": None,
                    "messages": [
                        P.message_entry(getattr(m, "type", None), getattr(m, "content", None))
                        for m in messages
                    ],
                }

            out["prompts.get"] = await _attempt(_get) if first else {"ok": False, "error": "no prompt listed"}
        return out

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling, elicitation and roots through MultiServerMCPClient.

        The adapter's own hook is ``Callbacks(on_elicitation=...)``. Sampling and roots
        have no adapter API; the connection's documented ``session_kwargs`` forwards
        keyword arguments unread to the MCP SDK's ``ClientSession``, so those are
        registered there and recorded as the passthrough route. Nothing in the adapter
        can change roots after connecting, so list_changed is only observed.
        """
        return anyio.run(self._probe_client_async, server)

    async def _probe_client_async(self, server: ServerHandle) -> dict:
        import mcp.types as types
        from langchain_mcp_adapters.callbacks import Callbacks
        from langchain_mcp_adapters.client import MultiServerMCPClient

        from .. import client_probe as C

        log = C.HandlerLog()
        sdk = C.sdk_callbacks(log)

        async def on_elicitation(mcp_context, params, _context):
            return await sdk["elicitation_callback"](mcp_context, params)

        async def call(tool: str, session_kwargs: dict | None = None, callbacks=None):
            connection = _connection(server)
            if session_kwargs:
                connection["session_kwargs"] = dict(session_kwargs)
            client = MultiServerMCPClient({"src": connection}, callbacks=callbacks)
            tools = {t.name: t for t in await client.get_tools()}
            try:
                returned = await tools[tool].ainvoke({})
            except Exception as exc:  # noqa: BLE001 - an isError result raises here
                return normalize_framework_result(
                    tool=tool, origin=self.id, is_error=True,
                    text=f"{type(exc).__name__}: {exc}", error_channel="exception",
                )
            return self._result_to_rep(tool, returned)

        sampling = {"sampling_callback": sdk["sampling_callback"]}
        sampling_tools = {
            **sampling,
            "sampling_capabilities": types.SamplingCapability(tools=types.SamplingToolsCapability()),
        }
        roots = {"list_roots_callback": sdk["list_roots_callback"]}
        elicit = Callbacks(on_elicitation=on_elicitation)
        out: dict = {}
        out["sampling"] = await C.attempt_async(
            "sampling", "session_kwargs.sampling_callback", "passthrough",
            lambda: call("ask_sampling", sampling), log,
        )
        out["sampling.tools"] = await C.attempt_async(
            "sampling.tools", "session_kwargs.sampling_callback + sampling_capabilities", "passthrough",
            lambda: call("ask_sampling_tools", sampling_tools), log,
        )
        out["elicitation.form"] = await C.attempt_async(
            "elicitation.form", "Callbacks.on_elicitation", "adapter",
            lambda: call("ask_elicitation_form", callbacks=elicit), log,
        )
        out["elicitation.url"] = await C.attempt_async(
            "elicitation.url", "Callbacks.on_elicitation", "adapter",
            lambda: call("ask_elicitation_url", callbacks=elicit), log,
        )
        out["roots.list"] = await C.attempt_async(
            "roots.list", "session_kwargs.list_roots_callback", "passthrough",
            lambda: call("list_client_roots", roots), log,
        )
        out["roots.list_changed"] = await C.attempt_async(
            "roots.list_changed", None, None, lambda: call("roots_changes_seen", roots), log,
        )

        async def _capabilities():
            return await call(C.CAPABILITIES_TOOL, {**sampling_tools, **roots}, elicit)

        try:
            out["capabilities"] = {"text": C.rep_text(await _capabilities())[0]}
        except Exception as exc:  # noqa: BLE001
            out["capabilities"] = {"text": f"{type(exc).__name__}: {exc}"}
        return out

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through the adapter.

        The adapter's only notification hook is ``session_kwargs``, passed straight to
        the ClientSession, so a ``message_handler`` there is what an application gets.
        Tools and resources are relisted through ``load_mcp_tools`` and
        ``load_mcp_resources``. There is no prompt listing API. The session it creates is
        the raw SDK ClientSession, so subscribing goes through that.
        """
        return anyio.run(self._notify_async, server)

    async def _notify_async(self, server: ServerHandle) -> dict:
        from langchain_mcp_adapters.client import create_session
        from langchain_mcp_adapters.resources import load_mcp_resources
        from langchain_mcp_adapters.tools import load_mcp_tools

        from .. import notify as N

        recorder = N.Recorder()
        connection = {**_connection(server), "session_kwargs": {"message_handler": recorder.message_handler}}
        async with create_session(connection) as session, anyio.create_task_group() as group:
            negotiated = str((await session.initialize()).protocolVersion)
            subscriptions = N.SessionSubscriptions(session, recorder, negotiated, group)

            async def tools():
                return [t.name for t in await load_mcp_tools(session, server_name="src")]

            async def resources():
                blobs = await load_mcp_resources(session)
                return [str((getattr(b, "metadata", None) or {}).get("uri") or getattr(b, "source", "")) for b in blobs]

            async def call(name):
                tool = next(t for t in await load_mcp_tools(session, server_name="src") if t.name == name)
                return await tool.ainvoke({})

            out = await N.run(
                N.Adapter(
                    call=call,
                    recorder=recorder,
                    hook="create_session(connection['session_kwargs']['message_handler'])",
                    lists={"tools": tools, "resources": resources, "prompts": None},
                    no_list_reason="langchain-mcp-adapters has no prompt listing API",
                    subscribe=subscriptions.subscribe,
                    unsubscribe=subscriptions.unsubscribe,
                    subscribe_route="session",
                    subscribe_api=f"create_session() ClientSession: {subscriptions.api}",
                    negotiated=negotiated,
                )
            )
            group.cancel_scope.cancel()
        return out

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
        from langchain_mcp_adapters.client import create_session
        from langchain_mcp_adapters.tools import load_mcp_tools

        if boundary not in {"direct", "agent"}:
            raise RenderError(f"langchain renderer has no result boundary {boundary!r}")

        connection = _connection(server)
        async with create_session(connection) as session:
            await session.initialize()
            tools = await load_mcp_tools(session, connection=connection, server_name="src")
            target = next((t for t in tools if getattr(t, "name", None) == tool), None)
            if target is None:
                available = sorted(getattr(t, "name", "?") for t in tools)
                raise RenderError(
                    f"langchain renderer found no tool named {tool!r} (has: {available})"
                )
            # A ToolCall is what makes BaseTool.ainvoke return a ToolMessage, which is
            # the object an agent reads. A plain dict returns the raw tool value.
            payload: Any = (
                arguments
                if boundary == "direct"
                else {
                    "name": tool,
                    "args": dict(arguments),
                    "id": "mcp_mirror_result_capture",
                    "type": "tool_call",
                }
            )
            try:
                returned = await target.ainvoke(payload)
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

    def _result_to_rep(self, tool: str, returned: Any) -> ResultRep:
        """Map whatever the adapter returned onto the common result shape.

        langchain-mcp-adapters declares non-text tools as ``content_and_artifact``, so
        a call can come back as a ``(content, artifact)`` tuple where the artifact
        carries the blocks that did not fit into the content string. Both halves are
        recorded, because which half a block landed in is the finding.
        """

        # At the agent boundary the adapter's content and artifact arrive already
        # assembled into a ToolMessage. `status` is the field that carries MCP's
        # isError across that boundary, so it is read before the shape is unpacked.
        status = getattr(returned, "status", None)
        if status is not None and hasattr(returned, "content"):
            message = returned
            blocks = [_block_of(item) for item in message.content] if isinstance(
                message.content, (list, tuple)
            ) else []
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=blocks,
                text=message.content if isinstance(message.content, str) else None,
                is_error=status == "error",
                raw=repr(message)[:2000],
            )

        artifact = None
        if isinstance(returned, tuple) and len(returned) == 2:
            returned, artifact = returned

        blocks = []
        if isinstance(artifact, (list, tuple)):
            blocks = [_block_of(item) for item in artifact]
        elif artifact is not None:
            blocks = [_block_of(artifact)]

        if isinstance(returned, str):
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=blocks,
                text=returned,
                raw=repr((returned, artifact))[:2000],
            )
        if isinstance(returned, dict):
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=blocks,
                structured_content=returned,
                raw=repr((returned, artifact))[:2000],
            )
        if isinstance(returned, (list, tuple)):
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=blocks or [_block_of(item) for item in returned],
                raw=repr((returned, artifact))[:2000],
            )
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            text=str(returned),
            raw=repr((returned, artifact))[:2000],
        )

    def _to_reps(self, tools) -> list[ToolRep]:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        reps: list[ToolRep] = []
        for tool in tools:
            # Capture the provider-shaped conversion output, not a provider request.
            oai_tool = convert_to_openai_tool(tool)
            # langchain-mcp-adapters keeps the MCP annotations on StructuredTool.metadata,
            # separate from the provider-shaped dictionary captured above.
            retained = {k: v for k, v in (getattr(tool, "metadata", None) or {}).items() if v is not None}
            metadata = {"annotations": retained} if retained else {}
            reps.append(normalize_function_tool(oai_tool, origin=self.id, framework_metadata=metadata))
        return reps


async def _attempt(fn):
    try:
        return {"ok": True, **(await fn())}
    except NotImplementedError as exc:
        return {"ok": False, "unsupported": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the failure is the observation
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
