"""LangChain4j renderer (DESIGN.md section 8).

LangChain4j's MCP client (``dev.langchain4j:langchain4j-mcp``) is Java, so this renderer
follows the Spring AI bridge: a small command-line app (``langchain4j_java/``) connects
a ``DefaultMcpClient``, wraps it in an ``McpToolProvider``, and runs an ``AiServices``
agent over a scripted ``ChatModel`` that calls no provider. It reports what the agent
handed that model. It never reimplements the conversion (decision D1).

Tool definitions are the ``ToolSpecification`` list in the first ``ChatRequest``
AiServices sends the ChatModel. A tool result is the ``ToolExecutionResultMessage`` the
next ``ChatRequest`` carries after a scripted tool call. No provider model runs and no
provider request is serialized.

``--runner managed`` installs a pinned JDK and Maven, builds the worker jar against the
manifest's packages into a cache directory keyed by the manifest and worker sources,
and points this renderer at it through ``MCP_MIRROR_JAVA_RUNNER_DIR`` and
``MCP_MIRROR_JAVA_HOME``. For a local run, build the bundled project once with
``mvn -f src/mcp_mirror/renderers/langchain4j_java/pom.xml package`` and put a JDK 21 on
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

ID = "langchain4j"
_BUNDLED_JAR = Path(__file__).parent / "langchain4j_java" / "target" / "langchain4j-worker-0-exec.jar"
_TIMEOUT_SECONDS = 180
_CLIENT_CALL_TIMEOUT_MS = 10_000
_RAW_LIMIT = 2000

# ToolSpecificationHelper copies a tool's annotations into ToolSpecification.metadata()
# under these keys; the annotation's own title is renamed "title-annotation".
_ANNOTATION_KEYS = {
    "readOnlyHint": "readOnlyHint",
    "destructiveHint": "destructiveHint",
    "idempotentHint": "idempotentHint",
    "openWorldHint": "openWorldHint",
    "title-annotation": "title",
}


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


def get_renderer() -> "LangChain4jRenderer":
    return LangChain4jRenderer()


@lru_cache(maxsize=4)
def _bundled_versions(jar: str) -> dict[str, str]:
    """Artifact versions read from the pom.properties the shade plugin keeps in the jar."""

    found: dict[str, str] = {}
    try:
        with zipfile.ZipFile(jar) as archive:
            for name in archive.namelist():
                match = re.fullmatch(r"META-INF/maven/[^/]+/([^/]+)/pom\.properties", name)
                if not match:
                    continue
                version = re.search(rb"^version=(.+)$", archive.read(name), re.MULTILINE)
                if version:
                    found[match.group(1)] = version.group(1).decode().strip()
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


def _without_nulls(value: Any) -> Any:
    # McpIcon serializes an unset field as null; the server never sent one.
    if isinstance(value, list):
        return [_without_nulls(v) for v in value]
    if isinstance(value, dict):
        return {k: _without_nulls(v) for k, v in value.items() if v is not None}
    return value


class LangChain4jRenderer:
    id = ID

    def versions(self) -> dict:
        found = _bundled_versions(str(_jar()))
        return {
            "framework": found.get("langchain4j-mcp", "unknown"),
            "adapter": (
                f"langchain4j {found.get('langchain4j', 'unknown')}, "
                f"langchain4j-core {found.get('langchain4j-core', 'unknown')}, java {_java_version()}"
            ),
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "AiServices with McpToolProvider over DefaultMcpClient: the "
                "ChatRequest.toolSpecifications() AiServices hands ChatModel.doChat"
            ),
            capture_object=(
                "dev.langchain4j.agent.tool.ToolSpecification (name, description, parameters "
                "as JsonObjectSchema, serialized by JsonSchemaElementUtils.toMap)"
            ),
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "DefaultMcpClient.isModernProtocol(); on the modern path, the "
                "io.modelcontextprotocol/protocolVersion the client stamps into each request's "
                "_meta, read through its McpMetaSupplier; on the legacy path, the server's "
                "initialize answer in DefaultMcpClient's private initializeResult field, read "
                "reflectively because the client exposes no getter"
            ),
            limitation=(
                "No provider model ran. The definitions are what AiServices hands a ChatModel; "
                "no serialized provider request was captured."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        payload = self._run_worker({**self._config(server), "mode": "tools"})
        reps: list[ToolRep] = []
        for entry in payload["tools"]:
            # The ToolSpecification a provider serializes is name, description and
            # parameters. Annotations, the tool title, outputSchema and icons ride on its
            # metadata() map, which providers do not send.
            metadata = entry.get("metadata") or {}
            retained = {
                ours: metadata[theirs]
                for theirs, ours in _ANNOTATION_KEYS.items()
                if metadata.get(theirs) is not None
            }
            framework_metadata: dict[str, Any] = {"annotations": retained} if retained else {}
            if metadata.get("icons"):
                framework_metadata["icons"] = _without_nulls(metadata["icons"])
            if metadata.get("outputSchema"):
                framework_metadata["outputSchema"] = metadata["outputSchema"]
            reps.append(
                normalize_function_tool(
                    {
                        "name": entry.get("name"),
                        "description": entry.get("description"),
                        "parameters": entry.get("parameters") or {},
                    },
                    origin=self.id,
                    framework_metadata=framework_metadata,
                )
            )
        return reps

    # The agent path is the ToolExecutionResultMessage AiServices hands the next model
    # request. The ToolExecutor McpToolProvider hands out is published beside it: an
    # isError result or a non-text block throws from it, where AiServices' default
    # ToolExecutionErrorHandler turns the same exception into text for the model.
    DEFAULT_RESULT_BOUNDARY = "agent"
    _BOUNDARIES = ("agent", "direct")

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="agent",
                label="Tool result message to the model",
                capture_api="AiServices tool loop for a scripted ToolExecutionRequest",
                capture_object="ToolExecutionResultMessage (contents, isError) in the next ChatRequest",
                agent_path=True,
                limitation="No provider model ran; the model's tool call is scripted.",
            ),
            ResultBoundary(
                id="direct",
                label="Adapter tool executor",
                capture_api="McpToolProvider.provideTools(...).toolExecutorByName(name).executeWithContext",
                capture_object="the ToolExecutionResult the McpToolExecutor returns, or the exception it throws",
                agent_path=False,
            ),
        ]

    def render_result(
        self,
        server: ServerHandle,
        tool: str,
        arguments: dict[str, Any],
        boundary: str | None = None,
    ) -> ResultRep:
        """Invoke one tool and capture one declared result boundary."""

        boundary = boundary or self.DEFAULT_RESULT_BOUNDARY
        if boundary not in self._BOUNDARIES:
            raise RenderError(f"langchain4j renderer has no result boundary {boundary!r}")
        config = {
            **self._config(server),
            "mode": "tools",
            "calls": [{"tool": tool, "arguments": arguments}],
            "boundary": boundary,
        }
        payload = self._run_worker(config)
        entry = next((r for r in payload.get("results") or [] if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"langchain4j worker returned no result for {tool!r}")
        return self._entry_to_rep(tool, entry, boundary)

    def _entry_to_rep(self, tool: str, entry: dict, boundary: str) -> ResultRep:
        if not entry.get("ok"):
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                is_error=True,
                text=str(entry.get("error")),
                error_channel="exception",
                raw={k: entry.get(k) for k in ("exception", "error")},
            )
        returned = entry.get("returned") or {}
        raw = {k: (v[:_RAW_LIMIT] if isinstance(v, str) else v) for k, v in returned.items()}
        if boundary == "agent":
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=returned.get("contents") or [],
                is_error=bool(returned.get("isError")),
                raw=raw,
            )
        # ToolExecutionResult.result() holds the server's structuredContent when it sent
        # one; resultText is then its JSON and the content blocks are not read.
        result = returned.get("result")
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            structured_content=result if isinstance(result, (dict, list)) else None,
            is_error=bool(returned.get("isError")),
            text=returned.get("resultText"),
            raw=raw,
        )

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources and prompts through the DefaultMcpClient the provider wraps.

        McpToolProvider wraps only tools; resources and prompts are read on the client.
        """

        payload = self._run_worker({**self._config(server), "mode": "probe"})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Roots through DefaultMcpClient.Builder.roots and McpClient.setRoots.

        DefaultMcpClient answers no sampling or elicitation request and has no hook for
        either, so those are called with no hook registered.
        """
        from .. import client_probe as C

        hooks = {
            "sampling": (None, None),
            "sampling.tools": (None, None),
            "elicitation.form": (None, None),
            "elicitation.url": (None, None),
            "roots.list": ("DefaultMcpClient.Builder.roots(List<McpRoot>)", "adapter"),
            "roots.list_changed": ("McpClient.setRoots(List<McpRoot>)", "adapter"),
        }
        tools = [tool for tool, _ in C.FEATURES.values()] + [C.CAPABILITIES_TOOL]
        try:
            payload = self._run_worker(
                {
                    **self._config(server),
                    "mode": "client",
                    "hooks": True,
                    "clientCalls": tools,
                    # The client leaves an unanswered server request pending, so the
                    # tool call that made it would wait out the default 60s timeout;
                    # bound each call so the whole run fits the worker timeout.
                    "toolExecutionTimeoutMs": _CLIENT_CALL_TIMEOUT_MS,
                }
            )
        except RenderError as exc:
            out = {
                feature: C.record(feature, *hooks[feature], error=str(exc)[:500]) for feature in C.FEATURES
            }
            out["capabilities"] = {"text": str(exc)[:C.TEXT_LIMIT]}
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
        """Server utilities through DefaultMcpClient's builder options, one worker run each."""
        from .. import probe as P

        spec = {"cancel_after_s": P.CANCEL_AFTER_S, "cancel_wait_s": P.CANCEL_WAIT_S, "cache_pause_s": P.CACHE_PAUSE_S}
        out: dict = {}
        for feature, server in servers.items():
            config = {
                **self._config(server),
                "mode": "utility",
                "utility": feature,
                "utilitySpec": spec,
                # logHandler and progressHandler are registered only under hooks.
                "hooks": feature in ("progress", "logging"),
            }
            if feature == "cancellation":
                # The client's only cancellation is its tool execution timeout.
                config["toolExecutionTimeoutMs"] = int(P.CANCEL_AFTER_S * 1000)
            try:
                payload = self._run_worker(config)
                out[feature] = payload.get("utility") or {"ok": False, "error": "worker returned no utility answer"}
            except RenderError as exc:
                out[feature] = {"ok": False, "error": str(exc)[:300]}
        return out

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes heard by an McpClientListener on the DefaultMcpClient.

        Tools are re-read through McpToolProvider, whose client drops its cached tool list
        on tools/list_changed; resources and prompts from the client.
        """

        payload = self._run_worker({**self._config(server), "mode": "notify", "hooks": True})
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
            raise RenderError("langchain4j worker needs a JDK 21: set MCP_MIRROR_JAVA_HOME or put java on PATH")
        with tempfile.TemporaryDirectory(prefix="mcp-mirror-langchain4j-") as scratch:
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
                raise RenderError(f"langchain4j worker timed out after {_TIMEOUT_SECONDS}s") from exc
            except Exception as exc:  # noqa: BLE001
                raise RenderError(f"langchain4j worker failed to start: {exc}") from exc
            try:
                payload = json.loads(out_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
                raise RenderError(f"langchain4j worker wrote no result ({exc}): {detail[-2000:]}") from exc
        if payload.get("fatal"):
            raise RenderError(f"langchain4j worker error: {payload['fatal'][-2000:]}")
        if not isinstance(payload.get("tools"), list):
            raise RenderError("langchain4j worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("langchain4j worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload
