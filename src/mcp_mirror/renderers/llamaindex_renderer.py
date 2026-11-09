"""LlamaIndex renderer (DESIGN.md section 8).

Connects through ``llama-index-tools-mcp``: a ``BasicMCPClient`` handed to
``McpToolSpec``, whose ``to_tool_list_async`` turns each MCP tool into a LlamaIndex
``FunctionTool``. The spec rebuilds every input schema as a Pydantic model
(``create_model_from_json_schema``), and the ``ToolMetadata`` it attaches is what an
LLM integration serializes. The capture is ``ToolMetadata.to_openai_tool(
skip_length_check=True)``, the exact call ``llama-index-llms-openai`` makes for
Chat Completions and Responses; it is a provider-shaped dict, not a request body.

``BasicMCPClient`` opens a fresh ``ClientSession`` (and, over stdio, a fresh server
process) for every method call and calls ``initialize()`` on each, so the negotiated
version is read from those handshakes.

Tool results have two boundaries. ``FunctionTool.acall`` returns a ``ToolOutput``;
a ``FunctionAgent`` driven by core's ``MockFunctionCallingLLM`` then puts that output's
blocks into the ``ChatMessage(role="tool")`` the next model turn receives.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from importlib import util
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version

ID = "llamaindex"
REQUIRES = ("llama_index.tools.mcp", "llama_index.core")

_CALL_ID = "mcp-mirror-call"


def available() -> bool:
    try:
        return all(util.find_spec(mod) is not None for mod in REQUIRES)
    except ModuleNotFoundError:
        return False


def get_renderer() -> "LlamaIndexRenderer":
    return LlamaIndexRenderer()


def _url(handle: ServerHandle) -> str:
    """The ``command_or_url`` BasicMCPClient takes.

    The client has no transport argument: it picks HTTP+SSE only when the URL path ends
    in ``/sse`` or carries ``?transport=sse`` (``enable_sse``), and streamable HTTP for
    any other URL. An SSE endpoint at another path is reached by adding the query flag,
    which is how the adapter documents selecting it.
    """

    from llama_index.tools.mcp.client import enable_sse

    url = handle.url or ""
    if handle.transport == "sse" and not enable_sse(url):
        url += ("&" if "?" in url else "?") + "transport=sse"
    return url


class LlamaIndexRenderer:
    id = ID

    def versions(self) -> dict:
        return {
            "framework": dist_version("llama-index-core"),
            "adapter": dist_version("llama-index-tools-mcp"),
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "llama_index.tools.mcp.McpToolSpec.to_tool_list_async, then "
                "ToolMetadata.to_openai_tool(skip_length_check=True)"
            ),
            capture_object=(
                "OpenAI-compatible function-tool dict built from the FunctionTool's "
                "ToolMetadata (the call llama-index-llms-openai makes)"
            ),
            capture_stage="provider_format",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "protocol version returned by ClientSession.initialize, which "
                "BasicMCPClient._run_session calls for every method call"
            ),
            limitation=(
                "No LLM integration was installed and no serialized provider request "
                "was captured."
            ),
        )

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def _client(self, handle: ServerHandle, **kwargs: Any):
        from llama_index.tools.mcp import BasicMCPClient

        if handle.transport in {"http", "sse"}:
            headers = dict(handle.headers) or None
            return BasicMCPClient(_url(handle), headers=headers, timeout=30, **kwargs)
        return BasicMCPClient(handle.command, args=list(handle.args), timeout=30, **kwargs)

    @contextmanager
    def _versions_seen(self):
        """Record the version every ``ClientSession.initialize`` returns meanwhile."""

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

    def _settle_version(self, seen: list[str]) -> str:
        observed = set(seen)
        if len(observed) != 1:
            raise RenderError(
                "llamaindex renderer could not establish one negotiated MCP protocol "
                f"version; observed {sorted(observed)}"
            )
        self._negotiated_mcp_spec_version = observed.pop()
        return self._negotiated_mcp_spec_version

    async def _tools(self, handle: ServerHandle, **client_kwargs: Any) -> dict[str, Any]:
        from llama_index.tools.mcp import McpToolSpec

        spec = McpToolSpec(client=self._client(handle, **client_kwargs))
        return {t.metadata.name: t for t in await spec.to_tool_list_async()}

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def render(self, server: ServerHandle) -> list[ToolRep]:
        try:
            with self._versions_seen() as seen:
                tools = asyncio.run(self._tools(server))
            self._settle_version(seen)
            return [self._to_rep(tool) for tool in tools.values()]
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"llamaindex renderer failed: {exc}") from exc

    def _to_rep(self, tool: Any) -> ToolRep:
        # FunctionTool keeps only name, description and the rebuilt fn_schema. The MCP
        # tool's annotations, title, icons and outputSchema are not on it anywhere, so
        # nothing is passed as retained metadata.
        return normalize_function_tool(tool.metadata.to_openai_tool(skip_length_check=True), origin=self.id)

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="llama_index.core.tools.FunctionTool.acall",
                capture_object="the ToolOutput (blocks and is_error) FunctionTool builds from the CallToolResult",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Tool message to the model",
                capture_api="llama_index.core.agent.workflow.FunctionAgent.handle_tool_call_results, next LLM turn",
                capture_object="the ChatMessage(role='tool') in the chat history the next model call receives",
                agent_path=True,
                limitation=(
                    "The model is core's MockFunctionCallingLLM with a scripted "
                    "response; no LLM integration serialized the message."
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
            raise RenderError(f"llamaindex renderer has no result boundary {boundary!r}")
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

    async def _target(self, server: ServerHandle, tool: str, **client_kwargs: Any):
        tools = await self._tools(server, **client_kwargs)
        if tool not in tools:
            raise RenderError(f"llamaindex renderer found no tool named {tool!r} (has: {sorted(tools)})")
        return tools[tool]

    async def _direct(self, server: ServerHandle, tool: str, arguments: dict[str, Any], **client_kwargs: Any) -> ResultRep:
        target = await self._target(server, tool, **client_kwargs)
        return self._output_rep(tool, await target.acall(**arguments))

    def _output_rep(self, tool: str, output: Any) -> ResultRep:
        """A ToolOutput as the caller holds it.

        ``FunctionTool._parse_tool_output`` has no case for ``CallToolResult``, so it
        falls through to ``TextBlock(text=str(raw_output))``: one text block holding the
        result's Python repr. ``raw_output`` keeps the CallToolResult itself.
        """

        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=[_block(b) for b in output.blocks],
            is_error=bool(output.is_error),
            raw=repr(output.raw_output)[:2000],
        )

    async def _agent(self, server: ServerHandle, tool: str, arguments: dict[str, Any]) -> ResultRep:
        from llama_index.core.agent.workflow import FunctionAgent
        from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
        from llama_index.core.llms.mock import MockFunctionCallingLLM

        target = await self._target(server, tool)
        seen: list[list[Any]] = []

        def respond(messages, **_kwargs):
            seen.append(list(messages))
            if len(seen) == 1:
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    blocks=[ToolCallBlock(tool_call_id=_CALL_ID, tool_name=tool, tool_kwargs=dict(arguments))],
                )
            return ChatMessage(role=MessageRole.ASSISTANT, content="done")

        agent = FunctionAgent(
            tools=[target],
            llm=MockFunctionCallingLLM(response_generator=respond, is_chat_model=True),
            streaming=False,
        )
        await agent.run(user_msg="go")
        messages = [m for turn in seen[1:2] for m in turn if m.role == MessageRole.TOOL]
        if not messages:
            raise RenderError("llamaindex agent run never returned a tool message to the model")
        message = messages[-1]
        # FunctionAgent builds the message from ToolOutput.blocks alone; ToolOutput.is_error
        # (set only when the call raised) does not travel with it.
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=[_block(b) for b in message.blocks],
            raw=repr(message)[:2000],
        )

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources through ``McpToolSpec.fetch_resources`` and ``BasicMCPClient``.

        The spec lists static resources and templates together; the client reads a
        resource by URI and lists and expands prompts. Every route is the adapter's.
        """
        return asyncio.run(self._probe_async(server))

    async def _probe_async(self, server: ServerHandle) -> dict:
        from llama_index.tools.mcp import McpToolSpec

        from .. import probe as P

        client = self._client(server)
        spec = McpToolSpec(client=client)
        out: dict = {}

        async def _resources():
            listed = await spec.fetch_resources()
            return {"entries": [P.resource_entry(r) for r in listed if hasattr(r, "uri")]}

        async def _templates():
            listed = await spec.fetch_resources()
            return {"entries": [P.template_entry(r) for r in listed if not hasattr(r, "uri")]}

        out["resources.list"] = await _attempt(_resources)
        entries = out["resources.list"].get("entries") or []
        text_uri = next((e["uri"] for e in entries if (e.get("mimeType") or "").startswith("text/")), None)
        blob_uri = next((e["uri"] for e in entries if e.get("mimeType") and not e["mimeType"].startswith("text/")), None)

        async def _read(uri):
            result = await client.read_resource(uri)
            first = (result.contents or [None])[0]
            mime = _field(first, "mime_type", "mimeType")
            blob = _field(first, "blob")
            if blob is not None:
                return P.blob_contents(_field(first, "uri"), mime, blob)
            return P.text_contents(_field(first, "uri"), mime, _field(first, "text"))

        out["resources.read.text"] = (
            await _attempt(lambda: _read(text_uri)) if text_uri else {"ok": False, "error": "no text resource listed"}
        )
        out["resources.read.blob"] = (
            await _attempt(lambda: _read(blob_uri)) if blob_uri else {"ok": False, "error": "no binary resource listed"}
        )
        out["resources.templates"] = await _attempt(_templates)

        async def _prompts():
            return {"entries": [P.prompt_entry(p) for p in (await client.list_prompts()).prompts]}

        out["prompts.list"] = await _attempt(_prompts)
        prompts = out["prompts.list"].get("entries") or []
        first = prompts[0] if prompts else None

        async def _get():
            arguments = {a["name"]: "x" for a in first["arguments"] if a.get("required")}
            messages = await client.get_prompt(first["name"], arguments)
            return {
                # get_prompt returns ChatMessages; the GetPromptResult description is dropped.
                "description": None,
                "messages": [P.message_entry(m.role, m.blocks) for m in messages],
            }

        out["prompts.get"] = await _attempt(_get) if first else {"ok": False, "error": "no prompt listed"}
        return out

    # ------------------------------------------------------------------
    # Sampling, elicitation and roots
    # ------------------------------------------------------------------

    def probe_client_features(self, server: ServerHandle) -> dict:
        """``BasicMCPClient(sampling_callback=...)`` is the adapter's only client hook.

        It is handed to every ClientSession the client opens. There is no elicitation or
        roots parameter, and the sessions are private to ``_run_session``, so those
        tools are called with no handler registered. ``sampling_capabilities`` is not
        forwarded, so a tools-enabled sampling request is not advertised.
        """
        return asyncio.run(self._client_async(server))

    async def _client_async(self, server: ServerHandle) -> dict:
        from .. import client_probe as C

        log = C.HandlerLog()
        sdk = C.sdk_callbacks(log)
        hooked = {"sampling_callback": sdk["sampling_callback"]}

        def call(tool: str, **kwargs):
            return self._direct(server, tool, {}, **kwargs)

        hook = "BasicMCPClient(sampling_callback=...)"
        out: dict = {}
        out["sampling"] = await C.attempt_async("sampling", hook, "adapter", lambda: call("ask_sampling", **hooked), log)
        out["sampling.tools"] = await C.attempt_async(
            "sampling.tools", hook, "adapter", lambda: call("ask_sampling_tools", **hooked), log
        )
        for feature, (tool, _expect) in C.FEATURES.items():
            if feature not in out:
                out[feature] = await C.attempt_async(feature, None, None, lambda tool=tool: call(tool), log)
        try:
            out["capabilities"] = {"text": C.rep_text(await call(C.CAPABILITIES_TOOL, **hooked))[0]}
        except Exception as exc:  # noqa: BLE001
            out["capabilities"] = {"text": f"{type(exc).__name__}: {exc}"[: C.TEXT_LIMIT]}
        return out

    # ------------------------------------------------------------------
    # Server utilities
    # ------------------------------------------------------------------

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through BasicMCPClient and the FunctionTools McpToolSpec builds.

        ``BasicMCPClient.call_tool`` takes a ``progress_callback``, but the FunctionTool
        McpToolSpec builds calls it with the tool arguments only, so progress is asked
        through the client directly. The client has no logging, set-level or completion
        API, and its sessions exist only inside ``_run_session``.
        """
        return asyncio.run(self._utilities_async(servers))

    async def _utilities_async(self, servers: dict[str, ServerHandle]) -> dict:
        from .. import probe as P

        out: dict = {}

        async def call(feature: str, tool: str, arguments: dict | None = None):
            target = await self._target(servers[feature], tool)
            return (await target.acall(**(arguments or {}))).raw_output

        if "tools.list" in servers:
            async def _names():
                return {"route": "adapter", "api": "McpToolSpec.to_tool_list_async", "names": list(await self._tools(servers["tools.list"]))}

            out["tools.list"] = await P.attempt_async(_names)
        if "progress" in servers:
            async def _progress():
                events: list[dict] = []

                async def on_progress(progress, total, message):
                    events.append({"progress": progress, "total": total, "message": message})

                result = await self._client(servers["progress"]).call_tool(
                    "slow_with_progress", {}, progress_callback=on_progress
                )
                return {"route": "adapter", "api": "BasicMCPClient.call_tool(progress_callback=...)", "events": events, "text": P.result_text(result)}

            out["progress"] = await P.attempt_async(_progress)
        if "logging" in servers:
            answer = await P.attempt_async(lambda: _text(call("logging", "emit_logs")))
            out["logging"] = {
                "ok": False,
                "unsupported": "BasicMCPClient builds its ClientSession without a logging_callback; tool_call_logs_callback collects Python log records, not MCP notifications",
                "text": answer.get("text"),
                "call": answer,
            }
        for feature, reason in (
            ("set_level", "BasicMCPClient has no logging/setLevel API; its sessions are private to _run_session"),
            ("completion", "BasicMCPClient has no completion API; its sessions are private to _run_session"),
        ):
            if feature in servers:
                out[feature] = {"ok": False, "unsupported": reason}
        if "cancellation" in servers:
            async def _cancellation():
                target = await self._target(servers["cancellation"], "cancellable_wait")
                loop = asyncio.get_running_loop()
                started = loop.time()
                task = asyncio.create_task(target.acall(seconds=P.CANCEL_WAIT_S))
                await asyncio.sleep(P.CANCEL_AFTER_S)
                task.cancel()
                try:
                    outcome = f"returned: {P.result_text((await task).raw_output)}"
                except asyncio.CancelledError:
                    outcome = "cancelled"
                except Exception as exc:  # noqa: BLE001
                    outcome = f"threw: {type(exc).__name__}: {exc}"
                abandoned = loop.time() - started
                await asyncio.sleep(0.3)
                return {"route": "adapter", "mechanism": "asyncio task.cancel() around FunctionTool.acall", "abandoned_after_s": round(abandoned, 3), "outcome": outcome}

            out["cancellation"] = await P.attempt_async(_cancellation)
        if "tasks" in servers:
            out["tasks"] = await P.attempt_async(lambda: _routed(call("tasks", "task_capable")))
        if "caching" in servers:
            async def _caching():
                marks = []
                for step in ("first", "within_ttl", "after_ttl"):
                    if step == "after_ttl":
                        await asyncio.sleep(P.CACHE_PAUSE_S)
                    started = P.wall_clock()
                    listed = await self._tools(servers["caching"])
                    marks.append({"call": step, "started": started, "finished": P.wall_clock(), "count": len(listed)})
                return {"route": "adapter", "api": "McpToolSpec.to_tool_list_async", "marks": marks}

            out["caching"] = await P.attempt_async(_caching)
        return out

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through the adapter.

        BasicMCPClient takes no message handler, and it opens and closes a session per
        call, so no session outlives a notification. The probe has no hook to register.
        It relists tools through McpToolSpec, resources and prompts through
        ``BasicMCPClient.list_resources`` and ``list_prompts``, and calls the fixture's
        tools as FunctionTools.
        """
        from .. import notify as N

        async def run() -> dict:
            from llama_index.tools.mcp import McpToolSpec

            client = self._client(server)
            spec = McpToolSpec(client=client)
            recorder = N.Recorder()

            async def tools():
                return [t.name for t in await spec.fetch_tools()]

            async def resources():
                # McpToolSpec.fetch_resources also calls resources/templates/list, which
                # this fixture does not serve, so the relist goes through the client.
                return [str(r.uri) for r in (await client.list_resources()).resources]

            async def prompts():
                return [p.name for p in (await client.list_prompts()).prompts]

            async def call(name):
                target = await self._target(server, name)
                return (await target.acall()).raw_output

            return await N.run(
                N.Adapter(
                    call=call,
                    recorder=recorder,
                    hook=None,
                    lists={"tools": tools, "resources": resources, "prompts": prompts},
                    no_subscribe_reason="BasicMCPClient has no subscribe API and its sessions close after each call",
                    negotiated=self._negotiated_mcp_spec_version,
                )
            )

        with self._versions_seen() as seen:
            asyncio.run(self._tools(server))
        self._settle_version(seen)
        return asyncio.run(run())


def _block(block: Any) -> dict:
    """One llama_index content block, recorded by the kind the model is handed."""

    dump = block.model_dump(mode="json", exclude_none=True) if hasattr(block, "model_dump") else dict(block)
    kind = dump.pop("block_type", None) or type(block).__name__
    return {"type": kind, **dump}


def _field(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


async def _text(awaitable) -> dict:
    from .. import probe as P

    return {"text": P.result_text(await awaitable)}


async def _routed(awaitable) -> dict:
    from .. import probe as P

    return {"route": "adapter", "text": P.result_text(await awaitable)}


async def _attempt(fn):
    try:
        return {"ok": True, **(await fn())}
    except NotImplementedError as exc:
        return {"ok": False, "unsupported": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the failure is the observation
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
