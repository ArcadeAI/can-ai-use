"""Google ADK (Java) renderer (DESIGN.md section 8).

ADK for Java (``com.google.adk:google-adk``) runs on the JVM, so this renderer follows
the Spring AI bridge: a small command-line worker (``google_adk_java/``) builds ADK's own
``com.google.adk.tools.mcp.McpToolset`` for the server, hands it to an ``LlmAgent`` and
runs the agent through ``InMemoryRunner`` against a scripted ``BaseLlm``. ADK loads the
MCP tools itself (``McpToolset.getTools`` -> ``McpTool``), and each ``McpTool`` adds its
``FunctionDeclaration`` to the ``LlmRequest`` in ``BaseTool.processLlmRequest``. It never
reimplements the conversion (decision D1).

Tool definitions are the ``functionDeclarations`` in the ``LlmRequest`` the model
receives. A tool result is the ``functionResponse`` part in the next ``LlmRequest``, or
the ``Map`` ``McpTool.runAsync`` returned, seen by the agent's ``afterToolCallback``.
No Gemini model runs and no provider request is serialized.

``--runner managed`` installs a pinned JDK and Maven, builds the worker jar against the
manifest's packages into a cache directory keyed by the manifest and worker sources,
and points this renderer at it through ``MCP_MIRROR_JAVA_RUNNER_DIR`` and
``MCP_MIRROR_JAVA_HOME``. For a local run, build the bundled project once with
``mvn -f src/mcp_mirror/renderers/google_adk_java/pom.xml package`` and put a JDK 21 on
``PATH``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..runner_manifests import JAVA_HOME_ENV, JAVA_RUNNER_DIR_ENV, JAVA_WORKER_JAR
from ..source import ServerHandle
from . import RenderError

ID = "google_adk_java"
_BUNDLED_JAR = Path(__file__).parent / "google_adk_java" / "target" / "google-adk-java-worker-exec.jar"
_TIMEOUT_SECONDS = 180
_OPAQUE = ("data", "blob", "content_details")


def _jar() -> Path:
    """The managed build when the runner names one, else the bundled project's build."""

    managed = os.environ.get(JAVA_RUNNER_DIR_ENV)
    return Path(managed) / JAVA_WORKER_JAR if managed else _BUNDLED_JAR


def _java() -> str | None:
    """A managed install carries its own pinned JDK; a local run uses JAVA_HOME or PATH."""

    for home in (os.environ.get(JAVA_HOME_ENV), os.environ.get("JAVA_HOME")):
        if home and (Path(home) / "bin" / "java").exists():
            return str(Path(home) / "bin" / "java")
    found = shutil.which("java")
    if found and _runs(found):
        return found
    return None


def _runs(java: str) -> bool:
    # macOS ships /usr/bin/java as a stub that only prints an install prompt.
    try:
        return subprocess.run([java, "-version"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def available() -> bool:
    return _java() is not None and _jar().exists()


def get_renderer() -> "GoogleADKJavaRenderer":
    return GoogleADKJavaRenderer()


@lru_cache(maxsize=4)
def _bundled_versions(jar: str) -> dict[str, str]:
    """Artifact versions from the pom.properties the shade plugin keeps in the worker."""

    found: dict[str, str] = {}
    try:
        with zipfile.ZipFile(jar) as archive:
            for name in archive.namelist():
                match = re.fullmatch(r"META-INF/maven/[^/]+/([^/]+)/pom\.properties", name)
                if not match:
                    continue
                version = re.search(r"^version=(.+)$", archive.read(name).decode(), re.MULTILINE)
                if version:
                    found[match.group(1)] = version.group(1).strip()
    except Exception:
        return {}
    return found


def _java_version() -> str:
    try:
        proc = subprocess.run([_java() or "java", "-version"], capture_output=True, text=True, timeout=20)
    except Exception:
        return "unknown"
    match = re.search(r'version "([^"]+)"', proc.stderr or proc.stdout or "")
    return match.group(1) if match else "unknown"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (f"<{len(v)} chars>" if k in _OPAQUE and isinstance(v, str) and len(v) > 32 else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _present(value: Any) -> Any:
    """Drop the nulls Jackson writes for the record fields a tool left unset."""

    if isinstance(value, dict):
        return {k: _present(v) for k, v in value.items() if v is not None}
    return value


class GoogleADKJavaRenderer:
    id = ID

    def versions(self) -> dict:
        found = _bundled_versions(str(_jar()))
        return {
            "framework": found.get("google-adk", "unknown"),
            "adapter": (
                f"com.google.adk.tools.mcp {found.get('google-adk', 'unknown')}, "
                f"io.modelcontextprotocol.sdk:mcp-core {found.get('mcp-core', 'unknown')}, "
                f"google-genai {found.get('google-genai', 'unknown')}, java {_java_version()}"
            ),
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "com.google.adk LlmAgent.builder().tools(McpToolset) run by InMemoryRunner.runAsync; "
                "a scripted BaseLlm.generateContent receives the LlmRequest"
            ),
            capture_object=(
                "com.google.genai.types.FunctionDeclaration in LlmRequest.config().tools() "
                "(name, description, parametersJsonSchema, responseJsonSchema)"
            ),
            capture_stage="provider_format",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "McpTool.getMcpSession().getCurrentInitializationResult().protocolVersion()"
            ),
            limitation=(
                "The model is a scripted BaseLlm; no google-genai client serialized the request."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        payload = self._run_worker({**self._config(server), "mode": "tools"})
        retained = payload.get("retained") or {}
        reps: list[ToolRep] = []
        for declaration in payload["tools"]:
            # AbstractMcpTool.declaration builds the FunctionDeclaration from name,
            # description, inputSchema (parametersJsonSchema) and outputSchema
            # (responseJsonSchema). McpTool keeps the MCP annotations and _meta beside
            # it, through its public annotations() and meta().
            tool_payload: dict[str, Any] = {
                "name": declaration.get("name"),
                "description": declaration.get("description"),
                "parameters": declaration.get("parametersJsonSchema") or declaration.get("parameters") or {},
            }
            if declaration.get("responseJsonSchema") is not None:
                tool_payload["outputSchema"] = declaration["responseJsonSchema"]
            kept = retained.get(declaration.get("name")) or {}
            metadata: dict[str, Any] = {}
            annotations = _present(kept.get("annotations") or {})
            if annotations:
                metadata["annotations"] = annotations
            if kept.get("meta"):
                metadata["_meta"] = kept["meta"]
            rep = normalize_function_tool(tool_payload, origin=self.id, framework_metadata=metadata)
            reps.append(rep.model_copy(update={"raw": declaration}))
        return reps

    # One agent run yields both boundaries. The scripted model emits a function call;
    # ADK invokes McpTool.runAsync and passes its return value to the agent's
    # afterToolCallback, then builds the functionResponse the next LlmRequest carries.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="com.google.adk LlmAgent.builder().afterToolCallbackSync (response)",
                capture_object="the Map McpTool.runAsync returned (AbstractMcpTool.wrapCallResult)",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Function response to the model",
                capture_api="com.google.adk InMemoryRunner.runAsync, next LlmRequest.contents()",
                capture_object="the functionResponse part in the LlmRequest the model receives",
                agent_path=True,
                limitation=(
                    "The model is a scripted BaseLlm; no google-genai client serialized the request."
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
            raise RenderError(f"google_adk_java renderer has no result boundary {boundary!r}")
        config = {**self._config(server), "mode": "tools", "calls": [{"tool": tool, "arguments": arguments}]}
        payload = self._run_worker(config)
        if tool not in {t.get("name") for t in payload["tools"]}:
            raise RenderError(f"google_adk_java renderer found no tool named {tool!r}")
        entry = next((r for r in payload.get("results") or [] if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"google_adk_java worker returned no result for {tool!r}")
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
            raise RenderError(f"google_adk_java result capture failed: {entry.get('error')}")
        if boundary == "direct":
            if entry.get("direct") is None:
                raise RenderError("google_adk_java agent run never invoked the tool")
            return self._direct_rep(tool, entry["direct"])
        return self._agent_rep(tool, entry.get("agent") or {})

    def _direct_rep(self, tool: str, returned: Any) -> ResultRep:
        """The Map AbstractMcpTool.wrapCallResult built from the CallToolResult.

        It carries ``{"text_output": [...]}`` for text content (each block parsed as JSON
        when it parses, else ``{"text": ...}``), ``{"error": ...}`` for an isError result
        or for content with no text block, and ``{}`` for empty content. Non-text blocks
        and structuredContent are not carried.
        """

        if not isinstance(returned, dict):
            return normalize_framework_result(tool=tool, origin=self.id, text=str(returned), raw=returned)
        error = "error" in returned
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            is_error=True if error else None,
            text=json.dumps(_redact(returned), sort_keys=True),
            error_channel="error_field" if error else None,
            raw=_redact(returned),
        )

    def _agent_rep(self, tool: str, function_response: dict) -> ResultRep:
        """What the model reads: the functionResponse JSON, plus any media in its parts.

        Gemini receives ``functionResponse.response`` as a JSON value, so it is recorded
        as the text the model reads; only ``functionResponse.parts`` would carry media the
        model can perceive, so only those count as blocks.
        """

        response = function_response.get("response") or {}
        blocks = []
        for part in function_response.get("parts") or []:
            inline = part.get("inlineData") or part.get("fileData") or {}
            mime = inline.get("mimeType")
            kind = (mime or "").split("/")[0] or "part"
            blocks.append({"type": kind, "mimeType": mime})
        error = "error" in response
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            is_error=True if error else None,
            text=json.dumps(_redact(response), sort_keys=True),
            error_channel="error_field" if error else None,
            raw=_redact(function_response),
        )

    # ------------------------------------------------------------------
    # Probes. Every tool call is an LlmAgent run whose scripted model calls the
    # fixture tool. What McpToolset does not offer is reached, where it can be, on the
    # McpSyncClient that McpTool.getMcpSession() returns publicly. Those answers carry
    # route "session".
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources and prompts on McpTool.getMcpSession(); McpToolset wraps only tools."""

        payload = self._run_worker({**self._config(server), "mode": "probe"})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """McpToolset takes only connection parameters, an ObjectMapper and a tool filter,
        and McpSessionManager builds the McpSyncClient with empty ClientCapabilities and no
        sampling, elicitation or roots handler, so every tool is called through the agent
        path with nothing to answer those requests."""
        from .. import client_probe as C

        return C.without_hooks(lambda tool: self.render_result(server, tool, {}, "agent"))

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities, one worker run each.

        McpSessionManager installs McpServerLogConsumer, which logs server log messages to
        SLF4J, and nothing else: no progress consumer, no level setter and no completion
        API on McpToolset. The level and completion are reached on the session.
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
        """Catalogue changes read from the next McpToolset.getTools listing; resources,
        prompts and subscriptions on the session. McpSessionManager registers no
        list-changed or resource-update consumer."""

        payload = self._run_worker({**self._config(server), "mode": "notify"})
        return payload.get("notify") or {}

    def _config(self, server: ServerHandle) -> dict:
        if server.transport in ("http", "sse"):
            config: dict = {"transport": server.transport, "url": server.url}
            if server.headers:
                config["headers"] = dict(server.headers)
            return config
        return {"transport": "stdio", "command": server.command, "args": list(server.args)}

    def _run_worker(self, config: dict) -> dict:
        java = _java()
        if java is None:
            raise RenderError("google_adk_java worker needs a JDK 21: set MCP_MIRROR_JAVA_HOME or put java on PATH")
        with tempfile.TemporaryDirectory(prefix="mcp-mirror-adk-java-") as scratch:
            config_path = Path(scratch) / "config.json"
            out_path = Path(scratch) / "out.json"
            config_path.write_text(json.dumps(config))
            try:
                # Run from the caller's cwd so a stdio server given by a relative command
                # resolves.
                proc = subprocess.run(
                    [java, "-jar", str(_jar()), str(config_path), str(out_path)],
                    capture_output=True,
                    text=True,
                    timeout=_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise RenderError(f"google_adk_java worker timed out after {_TIMEOUT_SECONDS}s") from exc
            except Exception as exc:  # noqa: BLE001
                raise RenderError(f"google_adk_java worker failed to start: {exc}") from exc
            try:
                payload = json.loads(out_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
                raise RenderError(f"google_adk_java worker wrote no result ({exc}): {detail[-2000:]}") from exc
        if payload.get("fatal"):
            raise RenderError(f"google_adk_java worker error: {payload['fatal'][-2000:]}")
        if not isinstance(payload.get("tools"), list):
            raise RenderError("google_adk_java worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("google_adk_java worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload
