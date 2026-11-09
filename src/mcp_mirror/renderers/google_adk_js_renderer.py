"""Google ADK (TypeScript) renderer (DESIGN.md section 8).

ADK for TypeScript (``@google/adk``) is a Node library, so this renderer follows the
Mastra bridge: a small Node worker (``google_adk_js_node/_google_adk_js_worker.mjs``)
drives ADK's real ``MCPToolset`` and ``MCPSessionManager`` (adk-js
``core/src/tools/mcp``) and reports what came back. It never reimplements the conversion
(decision D1).

Every capture is taken inside a real agent turn. An ``LlmAgent`` gets the toolset in
``tools`` and an ``InMemoryRunner`` drives it against a scripted ``BaseLlm``. ADK loads
the MCP tools itself (``MCPToolset.getTools``), and each ``MCPTool`` adds its
``FunctionDeclaration`` to the ``LlmRequest``. The tool definitions are the
``functionDeclarations`` in ``LlmRequest.config.tools``; a tool result is the
``functionResponse`` part in the next request's contents. No Gemini model runs and no
provider request is serialized.

``MCPTool`` builds the declaration's ``parameters`` (and ``response``, from the MCP
``outputSchema``) with ``toGeminiSchema``, which writes the genai Schema dialect: the
type is a ``Type`` enum value (``"OBJECT"``, ``"STRING"``, ``"TYPE_UNSPECIFIED"``) and a
``["string", "null"]`` type becomes ``nullable: true``. The comparison is in JSON Schema,
so :func:`_from_gemini` reads only those spellings back (``"OBJECT"`` is ``"object"``,
``TYPE_UNSPECIFIED`` declares no type, ``nullable: true`` adds ``"null"``). Everything
``toGeminiSchema`` dropped stays dropped, and ``ToolRep.raw`` keeps the declaration
exactly as the model received it.

``--runner managed`` installs the manifest's packages and pinned Node into a cache
directory keyed by the manifest digest, and points this renderer at it through
``MCP_MIRROR_NODE_RUNNER_DIR``. For a local run, install the bundled project once with
``npm ci --prefix src/mcp_mirror/renderers/google_adk_js_node``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..runner_manifests import NODE_RUNNER_DIR_ENV
from ..source import ServerHandle
from . import RenderError

ID = "google_adk_js"
_BUNDLED_NODE_DIR = Path(__file__).parent / "google_adk_js_node"
_WORKER_NAME = "_google_adk_js_worker.mjs"
_TIMEOUT_SECONDS = 120
_OPAQUE = ("data", "blob")


def _node_dir() -> Path:
    """The managed install when the runner names one, else the bundled project."""

    managed = os.environ.get(NODE_RUNNER_DIR_ENV)
    return Path(managed) if managed else _BUNDLED_NODE_DIR


def _worker() -> Path:
    return _node_dir() / _WORKER_NAME


def _package_dir(*parts: str) -> Path:
    return _node_dir() / "node_modules" / Path(*parts)


def _node() -> str | None:
    """A managed install carries its own pinned Node; a local run uses PATH."""

    pinned = _node_dir() / "node_modules" / ".bin" / "node"
    if pinned.exists():
        return str(pinned)
    return shutil.which("node")


def available() -> bool:
    return _node() is not None and _worker().exists() and _package_dir("@google", "adk").exists()


def get_renderer() -> "GoogleADKJsRenderer":
    return GoogleADKJsRenderer()


def _package_version(path: Path) -> str:
    try:
        return json.loads((path / "package.json").read_text()).get("version", "unknown")
    except Exception:
        return "unknown"


def _from_gemini(schema: Any) -> Any:
    """Read the genai Schema spellings back as JSON Schema; change nothing else."""

    if isinstance(schema, list):
        return [_from_gemini(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: _from_gemini(sub) for name, sub in value.items()}
        elif key in ("items", "anyOf"):
            out[key] = _from_gemini(value)
        elif key in ("type", "nullable"):
            continue
        else:
            out[key] = value
    kind = schema.get("type")
    if isinstance(kind, str) and kind != "TYPE_UNSPECIFIED":
        kind = kind.lower()
        out["type"] = [kind, "null"] if schema.get("nullable") is True else kind
    elif schema.get("nullable") is True:
        out["nullable"] = True
    if schema.get("nullable") is False:
        out["nullable"] = False
    return out


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (f"<{len(v)} chars>" if k in _OPAQUE and isinstance(v, str) and len(v) > 32 else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


class GoogleADKJsRenderer:
    id = ID

    def versions(self) -> dict:
        return {
            "framework": _package_version(_package_dir("@google", "adk")),
            "adapter": (
                f"@google/adk/tools/mcp {_package_version(_package_dir('@google', 'adk'))}, "
                f"@modelcontextprotocol/sdk {_package_version(_package_dir('@modelcontextprotocol', 'sdk'))}, "
                f"node {_node_version()}"
            ),
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "@google/adk InMemoryRunner.runAsync with LlmAgent({ tools: [MCPToolset] }), "
                "tools loaded by MCPToolset.getTools (@google/adk/tools/mcp)"
            ),
            capture_object=(
                "FunctionDeclaration in the LlmRequest.config.tools handed to "
                "BaseLlm.generateContentAsync"
            ),
            capture_stage="provider_format",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "protocolVersion of the initialize result each @modelcontextprotocol/sdk "
                "Client.connect receives, observed on Client.prototype.request in the node "
                "worker; MCPSessionManager opens one Client per operation and neither ADK "
                "nor the Client exposes the version"
            ),
            limitation=(
                "The model is a scripted BaseLlm; the LlmRequest is captured before any "
                "@google/genai client serializes it, so no provider request body was "
                "captured. Parameter schemas are read back from the genai Schema dialect "
                "(type enum spelling and nullable) for comparison."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        payload = self._run_worker(self._config(server))
        reps: list[ToolRep] = []
        for declaration in payload["tools"]:
            # MCPTool._getDeclaration builds the declaration from name, description,
            # inputSchema and outputSchema alone. The MCP tool is kept only in the
            # TypeScript-private mcpTool field, so annotations, title and icons are not
            # retained anywhere a caller can reach.
            tool_payload: dict[str, Any] = {
                "name": declaration.get("name"),
                "description": declaration.get("description"),
                "parameters": _from_gemini(declaration.get("parameters") or {}),
            }
            if declaration.get("response") is not None:
                tool_payload["outputSchema"] = _from_gemini(declaration["response"])
            rep = normalize_function_tool(tool_payload, origin=self.id)
            reps.append(rep.model_copy(update={"raw": declaration}))
        return reps

    # One agent run yields both boundaries. The scripted model emits a function call;
    # ADK invokes MCPTool.runAsync and passes its return value to the agent's
    # afterToolCallback, then builds the functionResponse the next LlmRequest carries.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="@google/adk LlmAgent afterToolCallback ({ response })",
                capture_object="the value MCPTool.runAsync returned, before ADK wraps it",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Function response to the model",
                capture_api="@google/adk InMemoryRunner.runAsync, next LlmRequest.contents",
                capture_object="the functionResponse part in the LlmRequest the model receives",
                agent_path=True,
                limitation=(
                    "The model is a scripted BaseLlm; no @google/genai client serialized "
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
            raise RenderError(f"google_adk_js renderer has no result boundary {boundary!r}")
        config = self._config(server)
        config["calls"] = [{"tool": tool, "arguments": arguments}]
        payload = self._run_worker(config)
        if tool not in {t.get("name") for t in payload["tools"]}:
            raise RenderError(f"google_adk_js renderer found no tool named {tool!r}")
        entry = next((r for r in payload.get("results") or [] if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"google_adk_js worker returned no result for {tool!r}")
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
            raise RenderError(f"google_adk_js result capture failed: {entry.get('error')}")
        if boundary == "direct":
            if entry.get("direct") is None:
                raise RenderError("google_adk_js agent run never invoked the tool")
            return self._direct_rep(tool, entry["direct"])
        return self._agent_rep(tool, entry.get("agent") or {})

    def _direct_rep(self, tool: str, returned: Any) -> ResultRep:
        if not isinstance(returned, dict):
            return normalize_framework_result(tool=tool, origin=self.id, text=str(returned), raw=returned)
        if "error" in returned and "content" not in returned:
            # ADK's tool-call flow turns a thrown error into { error: message }.
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

        Gemini receives ``functionResponse.response`` as a JSON value. ADK (TypeScript)
        puts the whole CallToolResult there, so MCP content blocks survive only as JSON;
        only ``functionResponse.parts`` carries media the model can perceive, so only
        those count as blocks here and the JSON is recorded as the text the model reads.
        """

        response = function_response.get("response") or {}
        blocks = []
        for part in function_response.get("parts") or []:
            inline = part.get("inlineData") or part.get("fileData") or {}
            mime = inline.get("mimeType")
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
    # Probes. Every tool call is an LlmAgent run whose scripted model calls the
    # fixture tool. What MCPToolset does not offer is reached, where it can be, on the
    # Client from MCPSessionManager.createSession: the session manager MCPToolset and
    # MCPTool are built on, exported beside them. Those answers carry route "session".
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources through MCPToolset.listResources / getResourceInfo / readResource."""

        payload = self._run_worker({**self._config(server), "probe": True})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """MCPToolset takes only a tool filter and a prefix, and the Client its
        MCPSessionManager builds declares no capabilities and registers no request
        handler, so every tool is called through the agent path with nothing to answer
        sampling, elicitation or roots requests."""
        from .. import client_probe as C

        return C.without_hooks(lambda tool: self.render_result(server, tool, {}, "agent"))

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities, one worker run each.

        MCPToolset has no progress, logging or message hook, no log-level setter and no
        completion API, and getTools takes no cache option. Progress, log messages, the
        level and completion are reached on the session Client (route "session").
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
            try:
                payload = self._run_worker({**self._config(server), "utility": feature, "utilitySpec": spec})
                out[feature] = payload.get("utility") or {"ok": False, "error": "worker returned no utility answer"}
            except RenderError as exc:
                out[feature] = {"ok": False, "error": str(exc)[:300]}
        return out

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes read from the next MCPToolset listing; subscriptions on the
        session Client. MCPToolset takes no message handler and has no prompt API."""

        payload = self._run_worker({**self._config(server), "notify": True})
        return payload.get("notify") or {}

    def _config(self, server: ServerHandle) -> dict:
        if server.transport == "sse":
            raise RenderError(
                "google_adk_js: MCPConnectionParams has no SSE variant; ADK (TypeScript) "
                "offers StdioConnectionParams and StreamableHTTPConnectionParams only"
            )
        if server.transport == "http":
            config: dict = {"transport": "http", "url": server.url}
            if server.headers:
                config["headers"] = dict(server.headers)
            return config
        return {"transport": "stdio", "command": server.command, "args": list(server.args)}

    def _run_worker(self, config: dict) -> dict:
        try:
            # Run from the caller's cwd so a stdio server given by a relative command
            # resolves; ESM finds @google/adk from the worker file's own directory.
            environment = os.environ.copy()
            environment["NODE_PATH"] = str(_node_dir() / "node_modules")
            proc = subprocess.run(
                [_node() or "node", str(_worker()), json.dumps(config)],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RenderError(f"google_adk_js worker timed out after {_TIMEOUT_SECONDS}s") from exc
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"google_adk_js worker failed to start: {exc}") from exc

        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            raise RenderError(f"google_adk_js worker error: {detail[-2000:]}")
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RenderError(
                f"google_adk_js worker returned invalid JSON ({exc}): {proc.stdout[:200]!r}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
            raise RenderError("google_adk_js worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("google_adk_js worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload


def _node_version() -> str:
    try:
        return subprocess.run(
            [_node() or "node", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"
