"""Spring AI renderer (DESIGN.md section 8).

Spring AI's MCP client (``org.springframework.ai:spring-ai-starter-mcp-client``) is Java
and needs a Spring Boot context, so this renderer follows the Mastra bridge: a small
command-line Boot application (``spring_ai_java/``) lets the starter's autoconfiguration
build the ``McpSyncClient`` and the ``SyncMcpToolCallbackProvider`` from
``spring.ai.mcp.client.*`` properties, and reports what they produced. It never
reimplements the conversion (decision D1).

Tool definitions are what ``DefaultToolCallingManager.resolveToolDefinitions`` hands a
chat model for the provider's callbacks. A tool result is the ``ToolResponse`` that
``DefaultToolCallingManager.executeToolCalls`` builds for a scripted assistant tool
call, which is what the next model request carries. No provider model runs and no
provider request is serialized.

``--runner managed`` installs a pinned JDK and Maven, builds the worker jar against the
manifest's packages into a cache directory keyed by the manifest and worker sources,
and points this renderer at it through ``MCP_MIRROR_JAVA_RUNNER_DIR`` and
``MCP_MIRROR_JAVA_HOME``. For a local run, build the bundled project once with
``mvn -f src/mcp_mirror/renderers/spring_ai_java/pom.xml package`` and put a JDK 21 on
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

ID = "spring_ai"
_BUNDLED_JAR = Path(__file__).parent / "spring_ai_java" / "target" / "spring-ai-worker-exec.jar"
_TIMEOUT_SECONDS = 180
_RAW_LIMIT = 2000


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


def get_renderer() -> "SpringAiRenderer":
    return SpringAiRenderer()


@lru_cache(maxsize=4)
def _bundled_versions(jar: str) -> dict[str, str]:
    """Artifact versions read from the jar names Spring Boot packed into the worker."""

    found: dict[str, str] = {}
    try:
        with zipfile.ZipFile(jar) as archive:
            for name in archive.namelist():
                match = re.fullmatch(r"BOOT-INF/lib/(.+?)-(\d[^/]*)\.jar", name)
                if match:
                    found[match.group(1)] = match.group(2)
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


class SpringAiRenderer:
    id = ID

    def versions(self) -> dict:
        found = _bundled_versions(str(_jar()))
        return {
            "framework": found.get("spring-ai-starter-mcp-client", "unknown"),
            "adapter": (
                f"spring-ai-mcp {found.get('spring-ai-mcp', 'unknown')}, "
                f"io.modelcontextprotocol.sdk:mcp-core {found.get('mcp-core', 'unknown')}, "
                f"spring-boot {found.get('spring-boot', 'unknown')}, java {_java_version()}"
            ),
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "spring-ai-starter-mcp-client autoconfiguration (McpClientAutoConfiguration, "
                "McpToolCallbackAutoConfiguration): SyncMcpToolCallbackProvider.getToolCallbacks "
                "over the McpSyncClient bean, resolved by DefaultToolCallingManager.resolveToolDefinitions"
            ),
            capture_object="org.springframework.ai.tool.definition.ToolDefinition (name, description, inputSchema)",
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence="McpSyncClient.getCurrentInitializationResult().protocolVersion()",
            limitation=(
                "No ChatModel ran. The definitions are what DefaultToolCallingManager hands a "
                "chat model; no serialized provider request was captured."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        payload = self._run_worker({**self._config(server), "mode": "tools"})
        reps: list[ToolRep] = []
        for entry in payload["tools"]:
            # McpToolUtils.createToolDefinition builds the ToolDefinition from name,
            # description and inputSchema alone. Annotations, title, outputSchema and
            # icons are not carried, on the definition or anywhere beside it.
            schema = entry.get("inputSchema")
            try:
                parameters = json.loads(schema) if isinstance(schema, str) else (schema or {})
            except json.JSONDecodeError:
                parameters = {"<unparsed inputSchema>": schema}
            reps.append(
                normalize_function_tool(
                    {"name": entry.get("name"), "description": entry.get("description"), "parameters": parameters},
                    origin=self.id,
                )
            )
        return reps

    # The agent path is the ToolResponse DefaultToolCallingManager hands the next model
    # request. The adapter's own call API, SyncMcpToolCallback.call, is published beside
    # it: it returns the same string, but an isError result or a transport failure
    # throws instead of reaching the model as text.
    DEFAULT_RESULT_BOUNDARY = "agent"
    _BOUNDARIES = ("agent", "direct")

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="agent",
                label="Tool response to the model",
                capture_api="DefaultToolCallingManager.executeToolCalls for a scripted AssistantMessage.ToolCall",
                capture_object="ToolResponseMessage.ToolResponse.responseData in the returned conversation history",
                agent_path=True,
                limitation="No ChatModel ran; the assistant tool call is scripted.",
            ),
            ResultBoundary(
                id="direct",
                label="Adapter call API",
                capture_api="SyncMcpToolCallback.call(String)",
                capture_object="the String the callback returns, or the exception it throws",
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
            raise RenderError(f"spring_ai renderer has no result boundary {boundary!r}")
        config = {
            **self._config(server),
            "mode": "tools",
            "calls": [{"tool": tool, "arguments": arguments}],
            "boundary": boundary,
        }
        payload = self._run_worker(config)
        entry = next((r for r in payload.get("results") or [] if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"spring_ai worker returned no result for {tool!r}")
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
        # Both boundaries hand over one String: JSON of the CallToolResult content list,
        # or, on the agent path, the ToolExecutionExceptionProcessor's message for an
        # isError result. No isError flag and no structuredContent travel beside it.
        text = returned.get("responseData") if boundary == "agent" else returned.get("text")
        raw = {k: (v[:_RAW_LIMIT] if isinstance(v, str) else v) for k, v in returned.items()}
        return normalize_framework_result(tool=tool, origin=self.id, text=text, raw=raw)

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources and prompts through the McpSyncClient bean the starter publishes.

        Spring AI wraps only tools; resources and prompts are reached on the client.
        """

        payload = self._run_worker({**self._config(server), "mode": "probe"})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling and form elicitation through the starter's @McpSampling and
        @McpElicitation annotations; roots and URL elicitation through an
        McpClientCustomizer<McpClient.SyncSpec> bean, which hands over the MCP SDK builder."""
        from .. import client_probe as C

        hooks = {
            "sampling": ('@McpSampling(clients = "src")', "adapter"),
            "sampling.tools": ('@McpSampling(clients = "src")', "adapter"),
            "elicitation.form": ('@McpElicitation(clients = "src")', "adapter"),
            "elicitation.url": ("McpClientCustomizer<McpClient.SyncSpec>: spec.urlElicitation(...)", "passthrough"),
            "roots.list": ("McpClientCustomizer<McpClient.SyncSpec>: spec.roots(...)", "passthrough"),
            "roots.list_changed": ("McpSyncClient.removeRoot / addRoot on the bean the starter publishes", "underlying"),
        }
        tools = [tool for tool, _ in C.FEATURES.values()] + [C.CAPABILITIES_TOOL]
        try:
            payload = self._run_worker(
                {**self._config(server), "mode": "client", "hooks": True, "clientCalls": tools}
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
        """Server utilities through the starter's public surface, one worker run each."""
        from .. import probe as P

        spec = {"cancel_after_s": P.CANCEL_AFTER_S, "cancel_wait_s": P.CANCEL_WAIT_S, "cache_pause_s": P.CACHE_PAUSE_S}
        out: dict = {}
        for feature, server in servers.items():
            config = {
                **self._config(server),
                "mode": "utility",
                "utility": feature,
                "utilitySpec": spec,
                # @McpProgress and @McpLogging beans exist only under the hooks profile.
                "hooks": feature in ("progress", "logging", "set_level"),
            }
            try:
                payload = self._run_worker(config)
                out[feature] = payload.get("utility") or {"ok": False, "error": "worker returned no utility answer"}
            except RenderError as exc:
                out[feature] = {"ok": False, "error": str(exc)[:300]}
        return out

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes heard by the starter's @Mcp*ListChanged annotations.

        Tools are re-read from SyncMcpToolCallbackProvider, whose cache the starter
        invalidates on McpToolsChangedEvent; resources and prompts from the McpSyncClient.
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
            raise RenderError("spring_ai worker needs a JDK 21: set MCP_MIRROR_JAVA_HOME or put java on PATH")
        with tempfile.TemporaryDirectory(prefix="mcp-mirror-spring-ai-") as scratch:
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
                raise RenderError(f"spring_ai worker timed out after {_TIMEOUT_SECONDS}s") from exc
            except Exception as exc:  # noqa: BLE001
                raise RenderError(f"spring_ai worker failed to start: {exc}") from exc
            try:
                payload = json.loads(out_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
                raise RenderError(f"spring_ai worker wrote no result ({exc}): {detail[-2000:]}") from exc
        if payload.get("fatal"):
            raise RenderError(f"spring_ai worker error: {payload['fatal'][-2000:]}")
        if not isinstance(payload.get("tools"), list):
            raise RenderError("spring_ai worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("spring_ai worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload
