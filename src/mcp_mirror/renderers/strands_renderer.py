"""AWS Strands Agents renderer (DESIGN.md section 8).

Connects through ``strands.tools.mcp.MCPClient`` and runs one turn of a real
``strands.Agent`` whose model is a scripted ``strands.models.Model``. No model is called:
the script records what the agent hands ``Model.stream`` and answers from a fixed list.

That call is the capture boundary. ``MCPAgentTool.tool_spec`` converts each MCP tool to a
Strands ``ToolSpec``, and the agent's ``ToolRegistry.get_all_tool_specs`` then normalizes
every spec (``normalize_tool_spec``) and validates it (``validate_tool_spec``), which fills
in a ``type`` and a ``Property <name>`` description on any top-level property that lacks
one. Only the second is what a model provider receives, so it is the one captured. The
object is a Strands ``ToolSpec`` dict: the framework's own tool type, which each model
provider class formats into its request later.

``MCPClient`` runs its ``ClientSession`` on a private background thread and exposes no
protocol version, so the negotiated revision is read from that session
(``MCPClient._background_thread_session``), or from ``ClientSession.initialize`` on the
1.x SDK line, which has no ``protocol_version`` attribute.
"""

from __future__ import annotations

import json
from importlib import util
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version

ID = "strands"
REQUIRES = ("strands", "strands.tools.mcp")

_TOOL_USE_ID = "mcp-mirror-call"
# Tool result content keys Strands uses, mapped to the MCP block kind they stand for.
_KINDS = {"text": "text", "image": "image", "document": "document", "json": "json", "video": "video"}


def available() -> bool:
    return (
        util.find_spec("strands") is not None
        and util.find_spec("strands.tools.mcp") is not None
        and util.find_spec("mcp") is not None
    )


def get_renderer() -> "StrandsRenderer":
    return StrandsRenderer()


class StrandsRenderer:
    id = ID

    def versions(self) -> dict:
        version = dist_version("strands-agents")
        return {"framework": version, "adapter": version}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "strands.tools.mcp.MCPClient.load_tools, then "
                "ToolRegistry.get_all_tool_specs inside a strands.Agent run"
            ),
            capture_object="strands.types.tools.ToolSpec in the tool_specs handed to Model.stream",
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "protocol_version of the ClientSession MCPClient keeps on its background "
                "thread (_background_thread_session), or the ClientSession.initialize "
                "result on the mcp 1.x line; MCPClient exposes no public protocol version"
            ),
            limitation=(
                "The model is a scripted strands.models.Model; the ToolSpec list is "
                "captured before any provider class formats it, so no provider request "
                "body was captured."
            ),
        )

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def render(self, server: ServerHandle) -> list[ToolRep]:
        try:
            run = self._run_agent(server, call=None)
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"strands renderer failed: {exc}") from exc
        if not run["requests"]:
            raise RenderError("strands agent run never reached the model")
        return [self._to_rep(spec, run["retained"].get(spec.get("name"), {})) for spec in run["requests"][0]["tool_specs"]]

    def _to_rep(self, spec: dict, retained: dict) -> ToolRep:
        payload: dict[str, Any] = {
            "name": spec.get("name"),
            "description": spec.get("description"),
            "parameters": (spec.get("inputSchema") or {}).get("json") or {},
        }
        output = (spec.get("outputSchema") or {}).get("json")
        if output is not None:
            payload["outputSchema"] = output
        rep = normalize_function_tool(payload, origin=self.id, framework_metadata=retained)
        # MCPAgentTool passes the MCP annotations through on the ToolSpec itself, so they
        # are on the captured object, not retained beside it.
        annotations = spec.get("annotations") or {}
        if annotations:
            rep = rep.model_copy(update={"annotations": dict(annotations), "raw": spec})
        return rep

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    # One agent run yields both boundaries. The scripted model emits a toolUse; the agent
    # runs MCPAgentTool.stream, fires AfterToolCallEvent with the ToolResult it got back,
    # and appends that result to the conversation the next Model.stream call receives.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="strands.hooks.AfterToolCallEvent.result",
                capture_object="the ToolResult MCPAgentTool yielded (MCPClient.call_tool_async's return)",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Tool result to the model",
                capture_api="strands.models.Model.stream, next call's messages",
                capture_object="the toolResult content block in the Messages the model receives",
                agent_path=True,
                limitation=(
                    "The model is a scripted strands.models.Model; no provider class "
                    "formatted or serialized the request."
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
            raise RenderError(f"strands renderer has no result boundary {boundary!r}")
        try:
            run = self._run_agent(server, call=(tool, arguments))
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
        if tool not in run["tools"]:
            raise RenderError(f"strands renderer found no tool named {tool!r}")
        if boundary == "direct":
            if "direct" not in run:
                raise RenderError("strands agent run never invoked the tool")
            return self._result_rep(tool, run["direct"])
        if "agent" not in run:
            raise RenderError("strands agent run never returned a tool result to the model")
        return self._result_rep(tool, run["agent"])

    def _result_rep(self, tool: str, result: dict) -> ResultRep:
        """A Strands ToolResult: ``status``, a content list, and what MCPClient added.

        Each content item is a one-key dict (``text``, ``image``, ...). ``MCPClient``
        turns an embedded text resource into a ``text`` item and drops audio and
        resource links, so the kinds recorded are the ones Strands kept.
        """

        blocks = []
        for item in result.get("content") or []:
            if not isinstance(item, dict) or not item:
                continue
            key = next(iter(item))
            value = item[key]
            if key == "text":
                blocks.append({"type": "text", "text": value})
            elif key == "image" and isinstance(value, dict):
                data = (value.get("source") or {}).get("bytes")
                blocks.append({"type": "image", "mimeType": f"image/{value.get('format')}", "data": _size(data)})
            else:
                blocks.append({"type": _KINDS.get(key, key)})
        is_error = result.get("isError")
        if is_error is None and result.get("status") == "error":
            is_error = True
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            structured_content=result.get("structuredContent"),
            is_error=is_error,
            raw=_redact(result),
        )

    # ------------------------------------------------------------------
    # The agent run both boundaries share
    # ------------------------------------------------------------------

    def _run_agent(self, server: ServerHandle, call: tuple[str, dict] | None) -> dict:
        from strands import Agent
        from strands.hooks import AfterToolCallEvent

        captured: dict[str, Any] = {"requests": [], "tools": [], "retained": {}}
        client = self._build_client(server)
        with _VersionProbe() as versions:
            agent = Agent(model=_scripted_model(captured, call, client), tools=[client], callback_handler=None)

            def after_tool(event) -> None:  # noqa: ANN001
                captured["direct"] = _plain(event.result)

            agent.add_hook(after_tool, AfterToolCallEvent)
            for tool in agent.tool_registry.registry.values():
                kept = _retained(getattr(tool, "mcp_tool", None))
                if kept:
                    captured["retained"][tool.tool_name] = kept
            try:
                agent("go")
            finally:
                versions.note_session(getattr(client, "_background_thread_session", None))
                agent.cleanup()
                client.stop(None, None, None)
        self._negotiated_mcp_spec_version = versions.one()
        return captured

    def _build_client(self, server: ServerHandle, **options: Any):
        """An ``MCPClient`` for ``server`` the way Strands documents each transport.

        stdio and HTTP+SSE take a transport callable wrapping the MCP SDK's own client;
        Streamable HTTP takes ``url=``, for which MCPClient builds the transport itself.
        """
        from strands.tools.mcp import MCPClient

        if server.transport == "http":
            return MCPClient(url=server.url, headers=dict(server.headers) or None, **options)
        if server.transport == "sse":
            from mcp.client.sse import sse_client

            url = server.url
            return MCPClient(lambda: sse_client(url=url), **options)
        from mcp import StdioServerParameters, stdio_client

        params = StdioServerParameters(command=server.command, args=list(server.args))
        return MCPClient(lambda: stdio_client(params), **options)

    def _started(self, server: ServerHandle, **options: Any):
        """A started client and the revision it negotiated, for the probes."""
        client = self._build_client(server, **options)
        with _VersionProbe() as versions:
            client.start()
            versions.note_session(getattr(client, "_background_thread_session", None))
        return client, versions.one()

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources, templates and prompts through MCPClient's own *_sync methods."""
        import anyio

        from ..source import _probe_with

        client, _version = self._started(server)
        try:
            return anyio.run(_probe_with, _SyncSurface(client), {})
        finally:
            client.stop(None, None, None)

    # ------------------------------------------------------------------
    # Client features
    # ------------------------------------------------------------------

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Elicitation through ``MCPClient(elicitation_callback=...)``; nothing else.

        MCPClient builds its ``ClientSession`` itself and passes it only
        ``elicitation_callback``: there is no sampling or roots parameter and the session
        is private, so sampling and roots are called with no hook registered.
        """
        from .. import client_probe as C

        log = C.HandlerLog()
        callbacks = C.sdk_callbacks(log)

        def call(tool: str, **options: Any) -> ResultRep:
            client, _version = self._started(server, **options)
            try:
                return self._result_rep(tool, _plain(client.call_tool_sync(_TOOL_USE_ID, tool, {})))
            finally:
                client.stop(None, None, None)

        elicit = {"elicitation_callback": callbacks["elicitation_callback"]}
        out: dict = {}
        for feature, (tool, _expect) in C.FEATURES.items():
            if feature.startswith("elicitation."):
                out[feature] = C.attempt(
                    feature, "MCPClient(elicitation_callback=...)", "adapter",
                    lambda tool=tool: call(tool, **elicit), log,
                )
            else:
                out[feature] = C.attempt(feature, None, None, lambda tool=tool: call(tool), log)
        out["capabilities"] = C.capabilities(lambda: call(C.CAPABILITIES_TOOL, **elicit))
        return out

    # ------------------------------------------------------------------
    # Server utilities
    # ------------------------------------------------------------------

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through MCPClient's public methods and constructor hooks.

        ``progress_callback`` and ``call_tool_async(cancel_signal=...)`` are MCPClient's
        own. There is no logging callback, logging/setLevel or completion API, and the
        ``ClientSession`` they would need is private, so those are unsupported.
        """
        import threading
        import time

        from .. import probe as P

        events: list[dict] = []

        async def on_progress(progress, total, message) -> None:  # noqa: ANN001
            events.append({"progress": progress, "total": total, "message": message})

        def with_client(feature: str, fn, **options: Any):
            client, _version = self._started(servers[feature], **options)
            try:
                return fn(client)
            finally:
                client.stop(None, None, None)

        def text(result: Any) -> str | None:
            return P.result_text(_plain(result))

        def sync_attempt(fn) -> dict:
            try:
                return {"ok": True, **fn()}
            except Exception as exc:  # noqa: BLE001 - the failure is the observation
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        out: dict = {}
        if "tools.list" in servers:
            # load_tools is what an agent calls; it walks every page. list_tools_sync
            # returns a single page and hands the cursor back to the caller.
            def _names() -> dict:
                import asyncio

                client = self._build_client(servers["tools.list"])
                try:
                    return {"route": "adapter", "api": "MCPClient.load_tools",
                            "names": [t.tool_name for t in asyncio.run(client.load_tools())]}
                finally:
                    client.stop(None, None, None)

            out["tools.list"] = sync_attempt(_names)
        if "progress" in servers:
            out["progress"] = sync_attempt(lambda: with_client(
                "progress",
                lambda c: {
                    "route": "adapter",
                    "api": "MCPClient(progress_callback=...)",
                    "text": text(c.call_tool_sync(_TOOL_USE_ID, "slow_with_progress", {})),
                    "events": list(events),
                },
                progress_callback=on_progress,
            ))
        if "logging" in servers:
            answer = sync_attempt(lambda: with_client(
                "logging", lambda c: {"text": text(c.call_tool_sync(_TOOL_USE_ID, "emit_logs", {}))}
            ))
            out["logging"] = {
                "ok": False,
                "unsupported": "MCPClient builds its ClientSession without a logging_callback and takes no log hook",
                "text": answer.get("text"),
                "call": answer,
            }
        for feature, reason in (
            ("set_level", "MCPClient has no logging/setLevel API; its ClientSession is private (_background_thread_session)"),
            ("completion", "MCPClient has no completion API; its ClientSession is private (_background_thread_session)"),
        ):
            if feature in servers:
                out[feature] = {"ok": False, "unsupported": reason}
        if "cancellation" in servers:
            def _cancellation(c) -> dict:
                import asyncio

                signal = threading.Event()
                timer = threading.Timer(P.CANCEL_AFTER_S, signal.set)
                started = time.monotonic()
                timer.start()
                try:
                    result = asyncio.run(c.call_tool_async(
                        _TOOL_USE_ID, "cancellable_wait", {"seconds": P.CANCEL_WAIT_S}, cancel_signal=signal
                    ))
                finally:
                    timer.cancel()
                abandoned = time.monotonic() - started
                time.sleep(0.3)
                outcome = "cancelled" if result.get("cancelled") else f"returned: {text(result)}"
                return {
                    "route": "adapter",
                    "mechanism": "MCPClient.call_tool_async(cancel_signal=threading.Event())",
                    "abandoned_after_s": round(abandoned, 3),
                    "outcome": outcome,
                }

            out["cancellation"] = sync_attempt(lambda: with_client("cancellation", _cancellation))
        if "tasks" in servers:
            out["tasks"] = sync_attempt(lambda: with_client(
                "tasks", lambda c: {"route": "adapter", "text": text(c.call_tool_sync(_TOOL_USE_ID, "task_capable", {}))}
            ))
        if "caching" in servers:
            def _caching(c) -> dict:
                marks = []
                for label in ("first", "within_ttl", "after_ttl"):
                    if label == "after_ttl":
                        time.sleep(P.CACHE_PAUSE_S)
                    started = P.wall_clock()
                    listed = c.list_tools_sync()
                    marks.append({"call": label, "started": started, "finished": P.wall_clock(), "count": len(listed)})
                return {"route": "adapter", "api": "MCPClient.list_tools_sync (defaults)", "marks": marks}

            out["caching"] = sync_attempt(lambda: with_client("caching", _caching))
        return out

    # ------------------------------------------------------------------
    # Change notifications
    # ------------------------------------------------------------------

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through MCPClient.

        ``on_tools_changed`` is the documented hook, and only for tools: MCPClient
        re-lists on the notification and updates the cache ``load_tools`` returns, which
        is what an agent reads. Resources and prompts are listed through the *_sync
        methods, which ask the server each time, with no hook. There is no subscribe
        API; at 2026-07-28 MCPClient opens ``subscriptions/listen`` itself, for tools
        only, when ``on_tools_changed`` is set, and hands the stream to no caller.
        """
        import anyio

        from .. import notify as N

        recorder = N.Recorder()

        def on_tools_changed(_previous, _tools) -> None:  # noqa: ANN001
            recorder.note("tools")

        client = self._build_client(server, on_tools_changed=on_tools_changed)
        versions = _VersionProbe()

        async def run() -> dict:
            # load_tools is what starts the client when an agent is handed it.
            with versions:
                await client.load_tools()
                versions.note_session(getattr(client, "_background_thread_session", None))
            negotiated = versions.one()

            async def tools():
                return [t.tool_name for t in await client.load_tools()]

            async def resources():
                listed = await anyio.to_thread.run_sync(client.list_resources_sync)
                return [str(r.uri) for r in listed.resources]

            async def prompts():
                listed = await anyio.to_thread.run_sync(client.list_prompts_sync)
                return [p.name for p in listed.prompts]

            async def call(name: str):
                return await anyio.to_thread.run_sync(lambda: client.call_tool_sync(_TOOL_USE_ID, name, {}))

            return await N.run(
                N.Adapter(
                    call=call,
                    recorder=recorder,
                    hook="MCPClient(on_tools_changed=...)",
                    lists={"tools": tools, "resources": resources, "prompts": prompts},
                    no_subscribe_reason=(
                        "MCPClient has no subscribe API; its ClientSession is private (_background_thread_session)"
                    ),
                    negotiated=negotiated,
                )
            )

        try:
            return anyio.run(run)
        finally:
            client.stop(None, None, None)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _VersionProbe:
    """Collect the revision each ClientSession negotiated while the block runs.

    On the mcp 2.x line the session carries ``protocol_version``; on 1.x the only record
    is the ``initialize`` result, so that call is observed (not altered) for the block.
    """

    def __init__(self) -> None:
        self.seen: list[str] = []
        self._original = None

    def __enter__(self) -> "_VersionProbe":
        from mcp import ClientSession

        original = self._original = ClientSession.initialize
        seen = self.seen

        async def capture_initialize(session, *args, **kwargs):  # noqa: ANN001
            result = await original(session, *args, **kwargs)
            version = getattr(result, "protocol_version", None) or getattr(result, "protocolVersion", None)
            if version:
                seen.append(str(version))
            return result

        ClientSession.initialize = capture_initialize
        return self

    def __exit__(self, *exc: Any) -> None:
        from mcp import ClientSession

        ClientSession.initialize = self._original

    def note_session(self, session: Any) -> None:
        version = getattr(session, "protocol_version", None)
        if version:
            self.seen.append(str(version))

    def one(self) -> str:
        observed = set(self.seen)
        if len(observed) != 1:
            raise RenderError(
                "strands renderer could not establish one negotiated MCP protocol "
                f"version; observed {sorted(observed)}"
            )
        return observed.pop()


class _SyncSurface:
    """MCPClient's blocking *_sync methods under the async names _probe_with calls."""

    def __init__(self, client: Any) -> None:
        self.client = client

    async def list_resources(self):
        return self.client.list_resources_sync()

    async def read_resource(self, uri):  # noqa: ANN001
        return self.client.read_resource_sync(uri)

    async def list_resource_templates(self):
        return self.client.list_resource_templates_sync()

    async def list_prompts(self):
        return self.client.list_prompts_sync()

    async def get_prompt(self, name, arguments):  # noqa: ANN001
        return self.client.get_prompt_sync(name, arguments)


def _scripted_model(captured: dict, call: tuple[str, dict] | None, client: Any):
    """A strands Model that records each ``stream`` call and answers from a fixed script.

    Turn one calls ``call`` if given, otherwise ends the run. Every later turn ends it.
    """

    from strands.models.model import Model

    class ScriptedModel(Model):
        def update_config(self, **model_config: Any) -> None:
            return None

        def get_config(self) -> dict:
            return {}

        async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):  # noqa: ANN001
            raise NotImplementedError("the scripted model has no structured output")
            yield  # pragma: no cover

        async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):  # noqa: ANN001
            specs = json.loads(json.dumps(tool_specs or [], default=str))
            captured["tools"] = [s.get("name") for s in specs]
            captured["requests"].append({"tool_specs": specs})
            results = [
                block["toolResult"]
                for message in messages or []
                for block in message.get("content") or []
                if isinstance(block, dict) and "toolResult" in block
            ]
            if results:
                captured["agent"] = _plain(results[-1])
            yield {"messageStart": {"role": "assistant"}}
            if call is not None and len(captured["requests"]) == 1:
                yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": _TOOL_USE_ID, "name": call[0]}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(call[1])}}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "tool_use"}}
            else:
                yield {"contentBlockDelta": {"delta": {"text": "done"}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "end_turn"}}

    return ScriptedModel()


def _retained(mcp_tool: Any) -> dict:
    """What MCPAgentTool keeps on ``mcp_tool`` but leaves off the ToolSpec."""
    if mcp_tool is None:
        return {}
    kept: dict[str, Any] = {}
    icons = getattr(mcp_tool, "icons", None)
    if icons:
        kept["icons"] = [i.model_dump(by_alias=True, exclude_none=True, mode="json") for i in icons]
    title = getattr(mcp_tool, "title", None)
    if title:
        kept["title"] = title
    return kept


def _plain(value: Any) -> Any:
    """A ToolResult as JSON-able data, with raw bytes reduced to their length."""
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _size(data: Any) -> str | None:
    if data is None:
        return None
    return data if isinstance(data, str) else f"<{len(data)} bytes>"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000]
    return value
