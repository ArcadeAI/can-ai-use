"""Google ADK renderer (DESIGN.md section 8).

Connects through ``google.adk.tools.mcp_tool.McpToolset`` and runs one turn of a real
``LlmAgent`` whose model is a scripted ``BaseLlm``. No model is called: the script
records the ``LlmRequest`` ADK hands the model and answers from a fixed list.

That request is the capture boundary. ADK builds each tool's declaration twice:
``McpTool._get_declaration`` keeps the server's text, and ``McpTool.process_llm_request``
swaps in a *fenced* declaration, wrapping the tool description and every schema
``description`` in untrusted-content markers, before the request goes to the model. Only
the second is what an agent's model sees, so it is the one captured. The object is a
``google.genai.types.FunctionDeclaration`` in ``LlmRequest.config.tools``: Gemini's own
tool type, not yet serialized into a request body.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Iterator
from importlib import util
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version

ID = "google_adk"
REQUIRES = ("google.adk",)

_APP = "mcp-mirror"
_USER = "mcp-mirror"
_CALL_ID = "mcp-mirror-call"
# Payload keys whose values are base64 bytes; a capture records their length, not them.
_OPAQUE = ("data", "blob")


@contextlib.contextmanager
def observe_negotiation(session_class: Any) -> Iterator[list[tuple[str, str]]]:
    """Record the protocol version every session of ``session_class`` settles on.

    Yields a list of ``(version, via)`` pairs. On mcp 2.x both connect paths end in
    ``ClientSession.adopt``: the legacy ``initialize`` handshake calls it with its
    ``InitializeResult``, and the SDK's ``negotiate_auto`` (which ADK 2.11 calls under
    ``ADK_ENABLE_MCP_MODERN_PROTOCOL=1``) calls it with the ``server/discover`` result
    when the probe succeeds, so wrapping ``adopt`` sees every session whichever way it
    connected. mcp 1.x has no ``adopt``; there ``initialize`` is the only path and its
    result carries the version.
    """
    observed: list[tuple[str, str]] = []
    if hasattr(session_class, "adopt"):
        original = session_class.adopt

        def adopt(session: Any, result: Any, *args: Any, **kwargs: Any) -> Any:
            out = original(session, result, *args, **kwargs)
            version = getattr(session, "protocol_version", None) or getattr(result, "protocol_version", None)
            if version:
                via = "initialize" if getattr(session, "initialize_result", None) is not None else "server/discover"
                observed.append((str(version), via))
            return out

        session_class.adopt = adopt
        try:
            yield observed
        finally:
            session_class.adopt = original
        return

    original_initialize = session_class.initialize

    async def initialize(session: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original_initialize(session, *args, **kwargs)
        version = getattr(result, "protocol_version", None) or getattr(result, "protocolVersion", None)
        if version:
            observed.append((str(version), "initialize"))
        return result

    session_class.initialize = initialize
    try:
        yield observed
    finally:
        session_class.initialize = original_initialize


def available() -> bool:
    return (
        util.find_spec("google.adk") is not None
        and util.find_spec("google.adk.tools.mcp_tool") is not None
        and util.find_spec("mcp") is not None
    )


def get_renderer() -> "GoogleADKRenderer":
    return GoogleADKRenderer()


class GoogleADKRenderer:
    id = ID

    def versions(self) -> dict:
        version = dist_version("google-adk")
        return {"framework": version, "adapter": version}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "google.adk.tools.mcp_tool.McpToolset.get_tools, then "
                "McpTool.process_llm_request inside an LlmAgent run"
            ),
            capture_object=(
                "google.genai.types.FunctionDeclaration in the LlmRequest.config.tools "
                "handed to BaseLlm.generate_content_async"
            ),
            capture_stage="provider_format",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "protocol version the session adopted"
                + (f" through {via}" if (via := getattr(self, "_negotiated_via", None)) else "")
                + ": read from ClientSession.adopt (mcp 2.x), which both the initialize "
                "handshake McpToolset's MCPSessionManager runs by default and the SDK's "
                "negotiate_auto (ADK_ENABLE_MCP_MODERN_PROTOCOL=1, server/discover) end in; "
                "from the ClientSession.initialize result on mcp 1.x"
            ),
            limitation=(
                "The model is a scripted BaseLlm; the LlmRequest is captured before "
                "any google-genai client serializes it, so no provider request body "
                "was captured."
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
            raise RenderError(f"google_adk renderer failed: {exc}") from exc

    async def _render_async(self, server: ServerHandle) -> list[ToolRep]:
        run = await self._run_agent(server, call=None)
        if not run["requests"]:
            raise RenderError("google_adk agent run never reached the model")
        declarations = run["requests"][0]["declarations"]
        retained = run["requests"][0]["retained"]
        return [self._to_rep(d, retained.get(d.get("name"), {})) for d in declarations]

    def _to_rep(self, declaration: dict, retained: dict) -> ToolRep:
        parameters = declaration.get("parameters_json_schema")
        if parameters is None:
            # With JSON_SCHEMA_FOR_FUNC_DECL turned off ADK converts the input schema to
            # a genai Schema instead; that is the object the model would get then.
            parameters = declaration.get("parameters") or {}
        payload = {
            "name": declaration.get("name"),
            "description": declaration.get("description"),
            "parameters": parameters,
        }
        # With JSON_SCHEMA_FOR_FUNC_DECL on (the 2.10 default) ADK sends the MCP
        # outputSchema to the model as response_json_schema.
        response_schema = declaration.get("response_json_schema")
        if response_schema is not None:
            payload["outputSchema"] = response_schema
        # McpTool keeps the whole MCP tool as raw_mcp_tool, so what the declaration
        # drops is retained there rather than lost.
        return normalize_function_tool(payload, origin=self.id, framework_metadata=retained)

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    # One agent run yields both boundaries. The stub model emits a function call; ADK
    # invokes McpTool.run_async and passes its return value to the agent's
    # after_tool_callback, then builds the FunctionResponse that the next LlmRequest
    # carries to the model.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="google.adk.agents.LlmAgent after_tool_callback (tool_response)",
                capture_object="the dict McpTool.run_async returned, before ADK wraps it",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Function response to the model",
                capture_api="google.adk.flows.llm_flows BaseLlmFlow, next LlmRequest.contents",
                capture_object="the google.genai.types.FunctionResponse in the LlmRequest the model receives",
                agent_path=True,
                limitation=(
                    "The model is a scripted BaseLlm; no google-genai client serialized "
                    "the request."
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
            raise RenderError(f"google_adk renderer has no result boundary {boundary!r}")
        try:
            run = asyncio.run(self._run_agent(server, call=(tool, arguments)))
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
            raise RenderError(f"google_adk renderer found no tool named {tool!r}")
        if boundary == "direct":
            if "direct" not in run:
                raise RenderError("google_adk agent run never invoked the tool")
            return self._direct_rep(tool, run["direct"])
        if "agent" not in run:
            raise RenderError("google_adk agent run never returned a function response")
        return self._agent_rep(tool, run["agent"])

    def _direct_rep(self, tool: str, returned: Any) -> ResultRep:
        if not isinstance(returned, dict):
            return normalize_framework_result(tool=tool, origin=self.id, text=str(returned), raw=repr(returned)[:2000])
        if "error" in returned and "content" not in returned:
            # _MCP_GRACEFUL_ERROR_HANDLING turns a raised McpError into this dict.
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                is_error=True,
                text=str(returned["error"]),
                error_channel="error_field",
                raw=_redact(returned),
            )
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=list(returned.get("content") or []),
            structured_content=returned.get("structuredContent"),
            is_error=returned.get("isError"),
            raw=_redact(returned),
        )

    def _agent_rep(self, tool: str, function_response: dict) -> ResultRep:
        """What the model reads: a JSON object, plus any media ADK lifted into parts.

        Gemini receives ``FunctionResponse.response`` as a JSON value. MCP content
        blocks survive inside it as JSON, but only ``FunctionResponse.parts`` carries
        media the model can perceive, so only those count as blocks here and the JSON
        is recorded as the text the model reads.
        """

        response = function_response.get("response") or {}
        blocks = []
        for part in function_response.get("parts") or []:
            inline = part.get("inline_data") or part.get("file_data") or {}
            mime = inline.get("mime_type")
            kind = (mime or "").split("/")[0] or "part"
            blocks.append({"type": kind, "mimeType": mime})
        error = response.get("isError")
        if error is None and "error" in response and "content" not in response:
            error = True
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            structured_content=response.get("structuredContent"),
            is_error=error,
            text=json.dumps(_redact(response), sort_keys=True),
            error_channel="error_field" if error and "isError" not in response else None,
            raw=_redact(function_response),
        )

    # ------------------------------------------------------------------
    # The agent run both boundaries share
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        server: ServerHandle,
        call: tuple[str, dict] | None,
        toolset: Any = None,
        **hooks: Any,
    ) -> dict:
        """One agent run. A caller that passes ``toolset`` owns it and keeps its session."""
        from google.adk.agents import LlmAgent
        from google.adk.runners import InMemoryRunner
        from google.genai import types
        from mcp import ClientSession

        captured: dict[str, Any] = {"requests": [], "tools": []}
        model = _scripted_model(captured, call)

        def after_tool(tool, args, tool_context, tool_response):  # noqa: ANN001
            captured["direct"] = tool_response
            return None

        owned = toolset is None
        if owned:
            toolset = self._build_toolset(server, **hooks)
        with observe_negotiation(ClientSession) as negotiations:
            try:
                agent = LlmAgent(
                    name="mcp_mirror",
                    model=model,
                    instruction="",
                    tools=[toolset],
                    after_tool_callback=after_tool,
                )
                runner = InMemoryRunner(agent=agent, app_name=_APP)
                session = await runner.session_service.create_session(app_name=_APP, user_id=_USER)
                message = types.Content(role="user", parts=[types.Part(text="go")])
                async for _event in runner.run_async(user_id=_USER, session_id=session.id, new_message=message):
                    pass
            finally:
                if owned:
                    await toolset.close()

        observed = {version for version, _via in negotiations}
        if not owned and not observed:
            # A reused toolset connected its session on an earlier run.
            return captured
        if len(observed) != 1:
            raise RenderError(
                "google_adk renderer could not establish one negotiated MCP protocol "
                f"version; observed {sorted(observed)}"
            )
        self._negotiated_mcp_spec_version = observed.pop()
        self._negotiated_via = " and ".join(sorted({via for _version, via in negotiations}))
        return captured

    def _build_toolset(self, server: ServerHandle, **hooks: Any):
        from google.adk.tools.mcp_tool import McpToolset

        return McpToolset(connection_params=self._connection_params(server), **hooks)

    def _connection_params(self, server: ServerHandle):
        from google.adk.tools.mcp_tool import (
            SseConnectionParams,
            StdioConnectionParams,
            StreamableHTTPConnectionParams,
        )
        from mcp import StdioServerParameters

        if server.transport == "http":
            params = StreamableHTTPConnectionParams(
                url=server.url,
                headers=dict(server.headers) or None,
                timeout=30,
            )
        elif server.transport == "sse":
            params = SseConnectionParams(url=server.url, timeout=30)
        else:
            params = StdioConnectionParams(
                server_params=StdioServerParameters(command=server.command, args=list(server.args)),
                timeout=30,
            )
        return params

    async def _call(self, server: ServerHandle, tool: str, arguments: dict | None = None, toolset: Any = None, **hooks: Any):
        """The dict McpTool.run_async returned when an agent called ``tool``."""
        run = await self._run_agent(server, call=(tool, arguments or {}), toolset=toolset, **hooks)
        if "direct" not in run:
            raise RenderError(f"google_adk agent run never invoked {tool!r}")
        return run["direct"]

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources through McpToolset's own methods; it has none for templates or prompts."""
        return asyncio.run(self._probe_async(server))

    async def _probe_async(self, server: ServerHandle) -> dict:
        from .. import probe as P

        toolset = self._build_toolset(server)
        out: dict = {}
        try:
            async def _list():
                # list_resources returns names only; get_resource_info is the toolset's
                # route to the rest of each entry.
                names = await toolset.list_resources()
                return {"entries": [P.resource_entry(await toolset.get_resource_info(n)) for n in names]}

            out["resources.list"] = await _attempt(_list)
            entries = out["resources.list"].get("entries") or []
            text_name = next((e["name"] for e in entries if (e.get("mimeType") or "").startswith("text/")), None)
            blob_name = next((e["name"] for e in entries if not (e.get("mimeType") or "").startswith("text/")), None)

            async def _read(name):
                # read_resource takes the resource's name, not its URI.
                contents = await toolset.read_resource(name)
                first = contents[0] if contents else None
                uri = _field(first, "uri")
                mime = _field(first, "mimeType", "mime_type")
                blob = _field(first, "blob")
                if blob is not None:
                    return P.blob_contents(uri, mime, blob)
                return P.text_contents(uri, mime, _field(first, "text"))

            out["resources.read.text"] = (
                await _attempt(lambda: _read(text_name)) if text_name else {"ok": False, "error": "no text resource listed"}
            )
            out["resources.read.blob"] = (
                await _attempt(lambda: _read(blob_name)) if blob_name else {"ok": False, "error": "no binary resource listed"}
            )
        finally:
            await toolset.close()

        for feature, method in (
            ("resources.templates", "list_resource_templates"),
            ("prompts.list", "list_prompts"),
            ("prompts.get", "get_prompt"),
        ):
            if callable(getattr(toolset, method, None)):
                out[feature] = {"ok": False, "error": f"McpToolset.{method} exists but is not probed"}
            else:
                out[feature] = {"ok": False, "unsupported": f"McpToolset has no {method} method"}
        return out

    # ------------------------------------------------------------------
    # Client features, server utilities and notifications
    # ------------------------------------------------------------------
    #
    # Every tool call below is an LlmAgent run whose scripted model calls the fixture
    # tool, so the request goes out through McpTool.run_async as it would for an agent.
    # What McpToolset does not offer is reached, where it can be, on a ClientSession
    # from MCPSessionManager.create_session: the session manager McpToolset and McpTool
    # are built on, which McpTool's constructor takes as a public argument. Those
    # answers carry route "session".

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling and elicitation through McpToolset's constructor hooks.

        ``sampling_callback``, ``sampling_capabilities`` and ``elicitation_callback`` are
        McpToolset parameters, forwarded to the ClientSession its session manager builds.
        Nothing in ADK answers ``roots/list``: that ClientSession is given no
        ``list_roots_callback`` and there is no parameter to pass one, so the roots tools
        are called with no hook.
        """
        return asyncio.run(self._probe_client_async(server))

    async def _probe_client_async(self, server: ServerHandle) -> dict:
        import mcp.types as types

        from .. import client_probe as C

        log = C.HandlerLog()
        sdk = C.sdk_callbacks(log)
        sampling = {"sampling_callback": sdk["sampling_callback"]}
        sampling_tools = {
            **sampling,
            "sampling_capabilities": types.SamplingCapability(tools=types.SamplingToolsCapability()),
        }
        elicit = {"elicitation_callback": sdk["elicitation_callback"]}

        async def call(tool: str, **hooks):
            return self._direct_rep(tool, await self._call(server, tool, **hooks))

        out: dict = {}
        out["sampling"] = await C.attempt_async(
            "sampling", "McpToolset(sampling_callback=...)", "adapter",
            lambda: call("ask_sampling", **sampling), log,
        )
        out["sampling.tools"] = await C.attempt_async(
            "sampling.tools", "McpToolset(sampling_callback=..., sampling_capabilities=tools)", "adapter",
            lambda: call("ask_sampling_tools", **sampling_tools), log,
        )
        out["elicitation.form"] = await C.attempt_async(
            "elicitation.form", "McpToolset(elicitation_callback=...)", "adapter",
            lambda: call("ask_elicitation_form", **elicit), log,
        )
        out["elicitation.url"] = await C.attempt_async(
            "elicitation.url", "McpToolset(elicitation_callback=...)", "adapter",
            lambda: call("ask_elicitation_url", **elicit), log,
        )
        out["roots.list"] = await C.attempt_async(
            "roots.list", None, None, lambda: call("list_client_roots"), log,
        )
        out["roots.list_changed"] = await C.attempt_async(
            "roots.list_changed", None, None, lambda: call("roots_changes_seen"), log,
        )
        try:
            rep = await call(C.CAPABILITIES_TOOL, **sampling_tools, **elicit)
            out["capabilities"] = {"text": C.rep_text(rep)[0]}
        except Exception as exc:  # noqa: BLE001
            out["capabilities"] = {"text": f"{type(exc).__name__}: {exc}"}
        return out

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through McpToolset's own options.

        ``progress_callback`` and ``tool_list_cache_ttl_seconds`` are McpToolset
        parameters. It has no logging hook, no log-level setter and no completion API;
        the level and completion are reachable on the raw session (route "session"), and
        log messages are not reachable at all, since the ClientSession ADK builds is given
        no ``logging_callback``.
        """
        return asyncio.run(self._utilities_async(servers))

    async def _utilities_async(self, servers: dict[str, ServerHandle]) -> dict:
        import anyio
        import mcp.types as types
        from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager

        from .. import probe as P

        events: list[dict] = []

        async def on_progress(progress, total=None, message=None):
            events.append({"progress": progress, "total": total, "message": message})

        async def call(feature: str, tool: str, arguments: dict | None = None):
            return P.result_text(await self._call(servers[feature], tool, arguments, progress_callback=on_progress))

        async def on_session(feature: str, body):
            manager = MCPSessionManager(connection_params=self._connection_params(servers[feature]))
            try:
                return await body(await manager.create_session())
            finally:
                await manager.close()

        out: dict = {}
        if "tools.list" in servers:
            async def _names():
                toolset = self._build_toolset(servers["tools.list"])
                try:
                    return {"route": "adapter", "names": [t.name for t in await toolset.get_tools()]}
                finally:
                    await toolset.close()

            out["tools.list"] = await P.attempt_async(_names)
        if "progress" in servers:
            async def _progress():
                text = await call("progress", "slow_with_progress")
                return {"route": "adapter", "api": "McpToolset(progress_callback=...)", "events": list(events), "text": text}

            out["progress"] = await P.attempt_async(_progress)
        if "logging" in servers:
            # The call still goes out, so the wire shows what the server sent.
            try:
                await call("logging", "emit_logs")
            except Exception:  # noqa: BLE001 - the answer below does not depend on it
                pass
            out["logging"] = {
                "ok": False,
                "unsupported": (
                    "McpToolset has no logging hook, and the ClientSession its "
                    "MCPSessionManager builds is given no logging_callback"
                ),
            }
        if "set_level" in servers:
            async def _set_level():
                async def body(session):
                    await session.set_logging_level(P.SET_LEVEL)
                    return P.result_text(await session.call_tool("emit_logs", {}))

                return {"route": "session", "api": "MCPSessionManager.create_session().set_logging_level",
                        "level": P.SET_LEVEL, "text": await on_session("set_level", body)}

            out["set_level"] = await P.attempt_async(_set_level)
        if "completion" in servers:
            async def _completion():
                async def body(session):
                    name, argument, value = P.COMPLETION_PROMPT
                    prompt = await session.complete(types.PromptReference(type="ref/prompt", name=name), {"name": argument, "value": value})
                    uri, argument, value = P.COMPLETION_RESOURCE
                    resource = await session.complete(types.ResourceTemplateReference(type="ref/resource", uri=uri), {"name": argument, "value": value})
                    return {"prompt": list(prompt.completion.values), "resource": list(resource.completion.values)}

                return {"route": "session", "api": "MCPSessionManager.create_session().complete",
                        **await on_session("completion", body)}

            out["completion"] = await P.attempt_async(_completion)
        if "cancellation" in servers:
            async def _cancellation():
                toolset = self._build_toolset(servers["cancellation"])
                try:
                    started = anyio.current_time()
                    outcome = "cancelled"
                    with anyio.move_on_after(P.CANCEL_AFTER_S):
                        returned = await self._call(
                            servers["cancellation"], "cancellable_wait", {"seconds": P.CANCEL_WAIT_S}, toolset=toolset
                        )
                        outcome = f"returned: {P.result_text(returned)}"
                    abandoned = anyio.current_time() - started
                    await anyio.sleep(0.3)
                finally:
                    await toolset.close()
                return {"route": "adapter", "mechanism": "anyio cancel scope around an LlmAgent run (Runner.run_async)",
                        "abandoned_after_s": round(abandoned, 3), "outcome": outcome}

            out["cancellation"] = await P.attempt_async(_cancellation)
        if "tasks" in servers:
            async def _tasks():
                return {"route": "adapter", "text": await call("tasks", "task_capable")}

            out["tasks"] = await P.attempt_async(_tasks)
        if "caching" in servers:
            async def _caching():
                toolset = self._build_toolset(servers["caching"])
                try:
                    marks = []
                    for label in ("first", "within_ttl", "after_ttl"):
                        if label == "after_ttl":
                            await anyio.sleep(P.CACHE_PAUSE_S)
                        started = P.wall_clock()
                        listed = await toolset.get_tools()
                        marks.append({"call": label, "started": started, "finished": P.wall_clock(), "count": len(listed)})
                finally:
                    await toolset.close()
                return {"route": "adapter", "api": "McpToolset.get_tools (defaults)", "marks": marks}

            out["caching"] = await P.attempt_async(_caching)
        return out

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through McpToolset.

        McpToolset takes no message handler, so nothing an application registers hears a
        notification; the question left is whether its next listing is fresh. Tools are
        relisted with ``get_tools`` and resources with ``list_resources`` plus
        ``get_resource_info`` (for the URI); there is no prompt API. McpToolset has no
        subscribe API either, so the subscription runs on a raw ClientSession from
        ``MCPSessionManager.create_session``, with the update fired on that session.
        """
        return asyncio.run(self._notify_async(server))

    async def _notify_async(self, server: ServerHandle) -> dict:
        import anyio
        from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager

        from .. import notify as N

        no_hook = "McpToolset takes no message handler"
        toolset = self._build_toolset(server)
        try:
            async def tools():
                return [t.name for t in await toolset.get_tools()]

            async def resources():
                return [str((await toolset.get_resource_info(n)).get("uri")) for n in await toolset.list_resources()]

            async def call(name):
                return await self._call(server, name, toolset=toolset)

            # The first run opens the session, so the negotiated version is known after it.
            await tools()
            out = await N.run(
                N.Adapter(
                    call=call,
                    recorder=N.Recorder(),
                    hook=None,
                    lists={"tools": tools, "resources": resources, "prompts": None},
                    no_list_reason="McpToolset has no prompt listing API",
                    no_subscribe_reason="McpToolset has no subscribe API",
                    negotiated=getattr(self, "_negotiated_mcp_spec_version", None),
                )
            )
        finally:
            await toolset.close()

        manager = MCPSessionManager(connection_params=self._connection_params(server))
        try:
            session = await manager.create_session()
            negotiated = out.get("negotiated")
            async with anyio.create_task_group() as group:
                recorder = N.Recorder()
                subscriptions = N.SessionSubscriptions(session, recorder, negotiated, group)
                raw = await N.run(
                    N.Adapter(
                        call=lambda name: session.call_tool(name, {}),
                        recorder=recorder,
                        hook=None,
                        lists={},
                        subscribe=subscriptions.subscribe,
                        unsubscribe=subscriptions.unsubscribe,
                        subscribe_route="session",
                        subscribe_api=f"MCPSessionManager.create_session(): {subscriptions.api} ({no_hook})",
                        negotiated=negotiated,
                    )
                )
                group.cancel_scope.cancel()
            out["resources.subscribe"] = raw["resources.subscribe"]
        finally:
            await manager.close()
        return out


def _scripted_model(captured: dict, call: tuple[str, dict] | None):
    """A BaseLlm that records each LlmRequest and answers from a fixed script.

    Turn one calls ``call`` if given, otherwise ends the run. Every later turn ends it.
    Declarations, and the annotations and icons each McpTool keeps on raw_mcp_tool, are
    read inside the call, because ADK reuses the request's tool objects across turns.
    """

    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class ScriptedModel(BaseLlm):
        model: str = "mcp-mirror-scripted"

        async def generate_content_async(self, llm_request, stream: bool = False):
            declarations = [
                d.model_dump(exclude_none=True, mode="json")
                for t in (llm_request.config.tools or [])
                for d in (getattr(t, "function_declarations", None) or [])
            ]
            retained = {}
            for name, tool in (llm_request.tools_dict or {}).items():
                raw_tool = getattr(tool, "raw_mcp_tool", None)
                kept = {}
                annotations = getattr(raw_tool, "annotations", None)
                if annotations is not None:
                    dumped = annotations.model_dump(by_alias=True, exclude_none=True, mode="json")
                    if dumped:
                        kept["annotations"] = dumped
                icons = getattr(raw_tool, "icons", None)
                if icons:
                    kept["icons"] = [i.model_dump(by_alias=True, exclude_none=True, mode="json") for i in icons]
                if kept:
                    retained[name] = kept
            captured["tools"] = [d.get("name") for d in declarations]
            responses = [
                p.function_response.model_dump(exclude_none=True, mode="json")
                for content in llm_request.contents or []
                for p in content.parts or []
                if p.function_response is not None
            ]
            captured["requests"].append({"declarations": declarations, "retained": retained})
            if responses:
                captured["agent"] = responses[-1]
            if call is not None and len(captured["requests"]) == 1:
                part = types.Part(function_call=types.FunctionCall(name=call[0], args=call[1], id=_CALL_ID))
            else:
                part = types.Part(text="done")
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    return ScriptedModel()


def _field(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (f"<{len(v)} chars>" if k in _OPAQUE and isinstance(v, str) and len(v) > 32 else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


async def _attempt(fn):
    try:
        return {"ok": True, **(await fn())}
    except NotImplementedError as exc:
        return {"ok": False, "unsupported": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the failure is the observation
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
