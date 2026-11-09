"""Google ADK (Go) renderer (DESIGN.md section 8).

ADK for Go (``google.golang.org/adk/v2``) is a Go module, so this renderer follows the
Java ADK bridge: a small command-line worker (``google_adk_go/``) builds ADK's own
``tool/mcptoolset`` Toolset for the server over the official Go MCP SDK
(``github.com/modelcontextprotocol/go-sdk``), hands it to an ``llmagent`` and runs the
agent through ``runner.NewInMemory`` against a scripted ``model.LLM``. ADK lists the MCP
tools itself (``Toolset.Tools`` -> ``mcpTool``), and each ``mcpTool`` adds its
``genai.FunctionDeclaration`` to the ``LLMRequest``. It never reimplements the
conversion (decision D1).

Tool definitions are the ``functionDeclarations`` in the ``LLMRequest`` the model
receives. A tool result is the ``functionResponse`` part in the next ``LLMRequest``, or
the ``map[string]any`` / ``error`` ``mcpTool.Run`` returned, seen by the agent's
``AfterToolCallbacks``. No Gemini model runs and no provider request is serialized.

``mcptoolset.Config`` takes the caller's ``*mcp.Client``, so the probes register the
go-sdk's own ``ClientOptions`` handlers on the client they hand it (route
``passthrough``). The session mcptoolset opens stays private; the worker reads the
negotiated protocol version from it through a sending middleware on that client.

``--runner managed`` installs the pinned Go release and builds the worker against the
module's go.mod and go.sum into a cache directory keyed by the manifest and worker
sources, and points this renderer at it through ``MCP_MIRROR_GO_RUNNER_DIR``. For a
local run, build the bundled module once with
``go -C src/mcp_mirror/renderers/google_adk_go build -o worker .``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..runner_manifests import GO_RUNNER_DIR_ENV, GO_WORKER_BINARY
from ..source import ServerHandle
from . import RenderError

ID = "google_adk_go"
_BUNDLED_WORKER = Path(__file__).parent / "google_adk_go" / GO_WORKER_BINARY
_TIMEOUT_SECONDS = 180
_OPAQUE = ("data", "blob", "content_details")
# The worker starts a third-party binary (the stdio server) and links third-party code;
# neither needs the operator's credentials.
_SECRET_NAME = re.compile(r"TOKEN|SECRET|KEY|PASSWORD|CREDENTIAL|AUTH", re.IGNORECASE)


def _worker() -> Path:
    """The managed build when the runner names one, else the bundled module's build."""

    managed = os.environ.get(GO_RUNNER_DIR_ENV)
    return Path(managed) / GO_WORKER_BINARY if managed else _BUNDLED_WORKER


def available() -> bool:
    return _worker().exists()


def get_renderer() -> "GoogleADKGoRenderer":
    return GoogleADKGoRenderer()


def _environment() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not _SECRET_NAME.search(k)}


@lru_cache(maxsize=4)
def _bundled_versions(worker: str) -> dict[str, str]:
    """Module versions the worker binary was linked with (runtime/debug.ReadBuildInfo)."""

    try:
        proc = subprocess.run(
            [worker, "--versions"], capture_output=True, text=True, timeout=20, env=_environment()
        )
        found = json.loads(proc.stdout)
    except Exception:
        return {}
    return {k: str(v) for k, v in found.items()} if isinstance(found, dict) else {}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (f"<{len(v)} chars>" if k in _OPAQUE and isinstance(v, str) and len(v) > 32 else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


class GoogleADKGoRenderer:
    id = ID

    def versions(self) -> dict:
        found = _bundled_versions(str(_worker()))
        return {
            "framework": found.get("adk", "unknown"),
            "adapter": (
                f"google.golang.org/adk/v2/tool/mcptoolset {found.get('adk', 'unknown')}, "
                f"github.com/modelcontextprotocol/go-sdk {found.get('go-sdk', 'unknown')}, "
                f"google.golang.org/genai {found.get('genai', 'unknown')}, {found.get('go', 'go unknown')}"
            ),
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "google.golang.org/adk/v2 llmagent.New(llmagent.Config{Toolsets: mcptoolset.New(...)}) "
                "run by runner.NewInMemory Runner.Run; a scripted model.LLM.GenerateContent receives "
                "the LLMRequest"
            ),
            capture_object=(
                "genai.FunctionDeclaration in LLMRequest.Config.Tools "
                "(name, description, parametersJsonSchema, responseJsonSchema)"
            ),
            capture_stage="provider_format",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "mcp.ClientSession.InitializeResult().ProtocolVersion, read from the session a "
                "sending middleware on the *mcp.Client passed as mcptoolset.Config.Client sees"
            ),
            limitation="The model is a scripted model.LLM; no genai client serialized the request.",
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        payload = self._run_worker({**self._config(server), "mode": "tools"})
        reps: list[ToolRep] = []
        for declaration in payload["tools"]:
            # mcptoolset's convertTool builds the FunctionDeclaration from name,
            # description, InputSchema (ParametersJsonSchema) and OutputSchema
            # (ResponseJsonSchema). It keeps no annotations, title, icons or _meta.
            tool_payload: dict[str, Any] = {
                "name": declaration.get("name"),
                "description": declaration.get("description"),
                "parameters": declaration.get("parametersJsonSchema") or declaration.get("parameters") or {},
            }
            if declaration.get("responseJsonSchema") is not None:
                tool_payload["outputSchema"] = declaration["responseJsonSchema"]
            rep = normalize_function_tool(tool_payload, origin=self.id, framework_metadata={})
            reps.append(rep.model_copy(update={"raw": declaration}))
        return reps

    # One agent run yields both boundaries. The scripted model emits a function call;
    # ADK invokes mcpTool.Run and passes what it returned to the agent's
    # AfterToolCallbacks, then builds the functionResponse the next LLMRequest carries.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="google.golang.org/adk/v2 llmagent.Config.AfterToolCallbacks (result, err)",
                capture_object="the map[string]any or error mcptoolset's mcpTool.Run returned",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Function response to the model",
                capture_api="google.golang.org/adk/v2 Runner.Run, next LLMRequest.Contents",
                capture_object="the genai.FunctionResponse part in the LLMRequest the model receives",
                agent_path=True,
                limitation="The model is a scripted model.LLM; no genai client serialized the request.",
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
            raise RenderError(f"google_adk_go renderer has no result boundary {boundary!r}")
        config = {**self._config(server), "mode": "tools", "calls": [{"tool": tool, "arguments": arguments}]}
        payload = self._run_worker(config)
        if tool not in {t.get("name") for t in payload["tools"]}:
            raise RenderError(f"google_adk_go renderer found no tool named {tool!r}")
        entry = next((r for r in payload.get("results") or [] if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"google_adk_go worker returned no result for {tool!r}")
        return self._entry_to_rep(tool, entry, boundary)

    def _entry_to_rep(self, tool: str, entry: dict, boundary: str) -> ResultRep:
        if not entry.get("ok"):
            if entry.get("threw"):
                return normalize_framework_result(
                    tool=tool,
                    origin=self.id,
                    is_error=True,
                    text=str(entry.get("error")),
                    error_channel="exception",
                    raw=entry,
                )
            raise RenderError(f"google_adk_go result capture failed: {entry.get('error')}")
        if boundary == "direct":
            if entry.get("direct") is None:
                raise RenderError("google_adk_go agent run never invoked the tool")
            return self._direct_rep(tool, entry["direct"])
        return self._agent_rep(tool, entry.get("agent") or {})

    def _direct_rep(self, tool: str, returned: dict) -> ResultRep:
        """What mcpTool.Run returned: ``{"output": structuredContent}`` when the result has
        structured content, else ``{"output": text}`` with every block flattened to text
        (formatMCPContent); an isError result becomes the Go error ``Tool execution
        failed. Details: ...``."""

        if returned.get("error") is not None:
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                is_error=True,
                text=str(returned["error"]),
                error_channel="exception",
                raw=_redact(returned),
            )
        result = returned.get("result")
        output = result.get("output") if isinstance(result, dict) else result
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            structured_content=output if isinstance(output, dict) else None,
            text=output if isinstance(output, str) else json.dumps(_redact(result), sort_keys=True),
            raw=_redact(returned),
        )

    def _agent_rep(self, tool: str, function_response: dict) -> ResultRep:
        """What the model reads: the functionResponse JSON, plus any media in its parts.

        Gemini receives ``functionResponse.response`` as a JSON value, so it is recorded
        as the text the model reads; only ``functionResponse.parts`` would carry media the
        model can perceive, so only those count as blocks. A structured ``output`` reaches
        the model as a JSON object, so it is the structured content. ADK turns a tool error
        into ``{"error": err.Error()}``.
        """

        response = function_response.get("response") or {}
        blocks = []
        for part in function_response.get("parts") or []:
            inline = part.get("inlineData") or part.get("fileData") or {}
            mime = inline.get("mimeType")
            kind = (mime or "").split("/")[0] or "part"
            blocks.append({"type": kind, "mimeType": mime})
        error = isinstance(response, dict) and "error" in response
        output = response.get("output") if isinstance(response, dict) else None
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            structured_content=output if isinstance(output, dict) else None,
            is_error=True if error else None,
            text=json.dumps(_redact(response), sort_keys=True),
            error_channel="error_field" if error else None,
            raw=_redact(function_response),
        )

    # ------------------------------------------------------------------
    # Probes. Every tool call is an llmagent run whose scripted model calls the
    # fixture tool. Handlers sit on the *mcp.Client handed to mcptoolset.Config.Client.
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """mcptoolset wraps only tools and keeps the session it opens private, so the
        caller has nothing to list or read resources and prompts with."""

        payload = self._run_worker({**self._config(server), "mode": "probe"})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling, elicitation and roots through the go-sdk ClientOptions and roots of
        the *mcp.Client passed as mcptoolset.Config.Client."""
        from .. import client_probe as C

        on_client = "on the *mcp.Client passed as mcptoolset.Config.Client"
        hooks = {
            "sampling": (f"mcp.ClientOptions.CreateMessageWithToolsHandler {on_client}", "passthrough"),
            "sampling.tools": (f"mcp.ClientOptions.CreateMessageWithToolsHandler {on_client}", "passthrough"),
            "elicitation.form": (f"mcp.ClientOptions.ElicitationHandler {on_client}", "passthrough"),
            "elicitation.url": (
                f"mcp.ClientOptions.ElicitationHandler with Capabilities.Elicitation.URL {on_client}",
                "passthrough",
            ),
            "roots.list": (f"mcp.Client.AddRoots {on_client}", "passthrough"),
            "roots.list_changed": (f"mcp.Client.RemoveRoots / AddRoots {on_client}", "passthrough"),
        }
        tools = [tool for tool, _ in C.FEATURES.values()] + [C.CAPABILITIES_TOOL]
        try:
            payload = self._run_worker({**self._config(server), "mode": "client", "clientCalls": tools})
        except RenderError as exc:
            out = {feature: C.record(feature, *hooks[feature], error=str(exc)[:500]) for feature in C.FEATURES}
            out["capabilities"] = {"text": str(exc)[: C.TEXT_LIMIT]}
            return out
        answers = payload.get("client") or {}
        out: dict = {}
        for feature, (tool, _expect) in C.FEATURES.items():
            entry = answers.get(tool) or {"ok": False, "error": "worker returned no answer"}
            hook, route = hooks[feature]
            saw = entry.get("handler_saw") or []
            if entry.get("root_change_error"):
                saw = saw + [{"hook": hook, "error": entry["root_change_error"]}]
            if not entry.get("ok"):
                out[feature] = C.record(feature, hook, route, error=str(entry.get("error")), handler_saw=saw)
                continue
            out[feature] = C.record(feature, hook, route, self._entry_to_rep(tool, entry, "agent"), handler_saw=saw)
        capabilities = answers.get(C.CAPABILITIES_TOOL) or {}
        out["capabilities"] = C.capabilities(lambda: self._entry_to_rep(C.CAPABILITIES_TOOL, capabilities, "agent"))
        return out

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities, one worker run each.

        Progress and logging through the go-sdk ClientOptions handlers on the client
        passed to mcptoolset; mcptoolset has no level setter or completion API and keeps
        its session private.
        """
        from .. import probe as P

        spec = {
            "set_level": P.SET_LEVEL,
            "completion_prompt": list(P.COMPLETION_PROMPT),
            "completion_resource": list(P.COMPLETION_RESOURCE),
            "cancel_after_s": P.CANCEL_AFTER_S,
            "cancel_wait_s": P.CANCEL_WAIT_S,
            "cache_pause_s": P.CACHE_PAUSE_S,
        }
        out: dict = {}
        for feature, server in servers.items():
            config = {**self._config(server), "mode": "utility", "utility": feature, "utilitySpec": spec}
            try:
                payload = self._run_worker(config)
                out[feature] = payload.get("utility") or {"ok": False, "error": "worker returned no utility answer"}
            except RenderError as exc:
                out[feature] = {"ok": False, "error": str(exc)[:300]}
        return out

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes heard by the go-sdk ClientOptions list-changed handlers on
        the client passed to mcptoolset, with tools re-read from Toolset.Tools."""

        payload = self._run_worker({**self._config(server), "mode": "notify"})
        return payload.get("notify") or {}

    def _config(self, server: ServerHandle) -> dict:
        config: dict
        if server.transport in ("http", "sse"):
            config = {"transport": server.transport, "url": server.url}
            if server.headers:
                config["headers"] = dict(server.headers)
        else:
            config = {"transport": "stdio", "command": server.command, "args": list(server.args)}
        # scripts/conformance_client.py sets the OAuth client the go-sdk handler uses.
        auth = getattr(self, "_auth", None)
        if auth:
            config["auth"] = dict(auth)
        return config

    def _run_worker(self, config: dict) -> dict:
        worker = _worker()
        if not worker.exists():
            raise RenderError(
                "google_adk_go worker is not built: run "
                "`go -C src/mcp_mirror/renderers/google_adk_go build -o worker .` or use --runner managed"
            )
        with tempfile.TemporaryDirectory(prefix="mcp-mirror-adk-go-") as scratch:
            config_path = Path(scratch) / "config.json"
            out_path = Path(scratch) / "out.json"
            config_path.write_text(json.dumps(config))
            try:
                # Run from the caller's cwd so a stdio server given by a relative command
                # resolves.
                proc = subprocess.run(
                    [str(worker), str(config_path), str(out_path)],
                    capture_output=True,
                    text=True,
                    timeout=_TIMEOUT_SECONDS,
                    env=_environment(),
                )
            except subprocess.TimeoutExpired as exc:
                raise RenderError(f"google_adk_go worker timed out after {_TIMEOUT_SECONDS}s") from exc
            except Exception as exc:  # noqa: BLE001
                raise RenderError(f"google_adk_go worker failed to start: {exc}") from exc
            try:
                payload = json.loads(out_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
                raise RenderError(f"google_adk_go worker wrote no result ({exc}): {detail[-2000:]}") from exc
        if payload.get("fatal"):
            raise RenderError(f"google_adk_go worker error: {payload['fatal'][-2000:]}")
        if not isinstance(payload.get("tools"), list):
            raise RenderError("google_adk_go worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("google_adk_go worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload
