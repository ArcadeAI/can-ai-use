"""Koog renderer (DESIGN.md section 8).

Koog (``ai.koog:agents-mcp``) is JetBrains' Kotlin agent framework, so this renderer
follows the Java bridges: a small command-line worker (``koog_kotlin/``) connects the
Kotlin MCP SDK ``Client`` Koog's own entrypoints build, hands it to
``McpToolRegistryProvider.fromClient`` and runs a Koog ``AIAgent`` (``singleRunStrategy``)
over the resulting ``ToolRegistry``. Koog lists the MCP tools itself
(``McpToolRegistryProvider`` -> ``McpTool``) and its ``OpenAILLMClient`` serializes them
into the chat-completions request. It never reimplements the conversion (decision D1).

Tool definitions are the ``tools`` array of the ``POST /v1/chat/completions`` request
Koog's ``OpenAILLMClient`` sends to a local fake provider. A tool result is the
``role: tool`` message in the next request, or the ``toolResult`` the agent's
``EventHandler.onToolCallCompleted`` receives. No model runs.

``--runner managed`` installs a pinned JDK and Maven, builds the worker jar against the
manifest's packages into a cache directory keyed by the manifest and worker sources, and
points this renderer at it through ``MCP_MIRROR_JAVA_RUNNER_DIR`` and
``MCP_MIRROR_JAVA_HOME``. For a local run, build the bundled project once with
``mvn -f src/mcp_mirror/renderers/koog_kotlin/pom.xml package`` and put a JDK 21 on
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

ID = "koog"
_BUNDLED_JAR = Path(__file__).parent / "koog_kotlin" / "target" / "koog-worker-exec.jar"
_TIMEOUT_SECONDS = 180
_VERSIONS_RESOURCE = "mcp-mirror-versions.properties"
_OPAQUE = ("data", "blob")
# The JVM and the stdio servers it starts get no credential-looking variables.
_SECRET_NAME = re.compile(r"(?i)(token|secret|key|passw|credential|auth|cookie|session)")
# McpTool.execute returns "Error: <joined text>" for an isError result; ToolRegistry turns
# a thrown tool into "Tool with name '<name>' failed to execute due to the error: ...".
_ERROR_PREFIX = "Error: "
_FAILED_PREFIX = "Tool with name "


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


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not _SECRET_NAME.search(k)}


def available() -> bool:
    return _java() is not None and _jar().exists()


def get_renderer() -> "KoogRenderer":
    return KoogRenderer()


@lru_cache(maxsize=4)
def _bundled_versions(jar: str) -> dict[str, str]:
    """Artifact versions the worker's pom wrote into the jar at build time.

    Koog and the Kotlin MCP SDK are Gradle builds and ship no pom.properties.
    """

    try:
        with zipfile.ZipFile(jar) as archive:
            lines = archive.read(_VERSIONS_RESOURCE).decode().splitlines()
    except Exception:
        return {}
    found: dict[str, str] = {}
    for line in lines:
        name, sep, version = line.partition("=")
        if sep and version.strip() and not version.startswith("${"):
            found[name.strip()] = version.strip()
    return found


def _java_version() -> str:
    try:
        proc = subprocess.run([_java() or "java", "-version"], capture_output=True, text=True, timeout=20, env=_env())
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


def _redact_text(text: str) -> str:
    """The agent string embeds the CallToolResult JSON; shorten base64 payloads in it."""

    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text
    if not isinstance(parsed, dict):
        return text
    return json.dumps(_redact(parsed), sort_keys=True)


class KoogRenderer:
    id = ID

    def versions(self) -> dict:
        found = _bundled_versions(str(_jar()))
        return {
            "framework": found.get("agents-mcp-jvm", "unknown"),
            "adapter": (
                f"ai.koog:agents-mcp {found.get('agents-mcp-jvm', 'unknown')}, "
                f"io.modelcontextprotocol:kotlin-sdk-client {found.get('kotlin-sdk-client-jvm', 'unknown')}, "
                f"kotlin {found.get('kotlin-stdlib', 'unknown')}, java {_java_version()}"
            ),
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "ai.koog AIAgent(toolRegistry = McpToolRegistryProvider.fromClient(...)) run with "
                "singleRunStrategy; OpenAILLMClient posts to a local fake /v1/chat/completions"
            ),
            capture_object=(
                "the tools array of the chat-completions request body Koog's OpenAILLMClient sends "
                "({type: function, function: {name, description, parameters}})"
            ),
            capture_stage="provider_request",
            provider_request_captured=True,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "initialize result protocolVersion, read on the transport's onMessage chain "
                "(the Kotlin SDK Client keeps no negotiated version)"
            ),
            limitation=" ".join(
                ["The provider is a local fake; no OpenAI model answered the request."]
                + [
                    f"McpToolRegistryProvider.fromClient left out the listed tool {d.get('name')!r}: "
                    f"DefaultMcpToolDescriptorParser.parse threw {d.get('parser_error')}."
                    for d in getattr(self, "_dropped", [])
                ]
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        payload = self._run_worker({**self._config(server), "mode": "tools"})
        retained = payload.get("retained") or {}
        # fromClient catches the parser's Throwable and only logs it, so a tool whose
        # schema Koog cannot parse never reaches the registry.
        self._dropped = payload.get("dropped") or []
        reps: list[ToolRep] = []
        for entry in payload["tools"]:
            # McpToolRegistryProvider's parser keeps name, description and inputSchema
            # (as ToolParameterDescriptors); OpenAILLMClient serializes those. McpTool's
            # metadata map is the only other thing kept beside the descriptor.
            function = entry.get("function") or {}
            tool_payload: dict[str, Any] = {
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters") or {},
            }
            kept = retained.get(function.get("name")) or {}
            metadata: dict[str, Any] = {}
            if kept.get("metadata"):
                metadata["koog_metadata"] = kept["metadata"]
            rep = normalize_function_tool(tool_payload, origin=self.id, framework_metadata=metadata)
            reps.append(rep.model_copy(update={"raw": entry}))
        return reps

    # One agent run yields both boundaries. The fake provider answers with a tool call;
    # Koog's ToolRegistry runs McpTool.execute, the EventHandler sees its result, and
    # McpTool.encodeResultToString becomes the role=tool message of the next request.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="ai.koog EventHandler.onToolCallCompleted (event.toolResult)",
                capture_object="the CallToolResult McpTool.execute returned, as JSON",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Tool message to the model",
                capture_api="ai.koog OpenAILLMClient, the next POST /v1/chat/completions",
                capture_object="the content of the role=tool message (McpTool.encodeResultToString)",
                agent_path=True,
                limitation="The provider is a local fake; no OpenAI model answered the request.",
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
            raise RenderError(f"koog renderer has no result boundary {boundary!r}")
        config = {**self._config(server), "mode": "tools", "calls": [{"tool": tool, "arguments": arguments}]}
        payload = self._run_worker(config)
        if tool not in {(t.get("function") or {}).get("name") for t in payload["tools"]}:
            raise RenderError(f"koog renderer found no tool named {tool!r}")
        entry = next((r for r in payload.get("results") or [] if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"koog worker returned no result for {tool!r}")
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
                    raw={k: entry.get(k) for k in ("exception", "error")},
                )
            raise RenderError(f"koog result capture failed: {entry.get('error')}")
        if boundary == "direct":
            if entry.get("direct") is None:
                if entry.get("failed"):
                    return normalize_framework_result(
                        tool=tool,
                        origin=self.id,
                        is_error=True,
                        text=str(entry["failed"]),
                        error_channel="exception",
                        raw={"failed": entry["failed"]},
                    )
                raise RenderError("koog agent run never invoked the tool")
            return self._direct_rep(tool, entry["direct"])
        return self._agent_rep(tool, entry.get("agent") or {}, entry.get("failed"))

    def _direct_rep(self, tool: str, returned: Any) -> ResultRep:
        """The CallToolResult McpTool.execute returned: content, isError, structuredContent."""

        if not isinstance(returned, dict):
            return normalize_framework_result(tool=tool, origin=self.id, text=str(returned), raw=returned)
        blocks = _redact(returned.get("content") or [])
        error = bool(returned.get("isError"))
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            structured_content=returned.get("structuredContent"),
            is_error=True if error else None,
            error_channel="is_error" if error else None,
            raw=_redact(returned),
        )

    def _agent_rep(self, tool: str, message: dict, failed: str | None) -> ResultRep:
        """What the model reads: one string, the role=tool message content.

        McpTool.encodeResultToString gives ``"Error: <joined text>"`` for an isError result
        and otherwise the JSON of the whole CallToolResult (content blocks, isError,
        structuredContent). The model reads that JSON as text, so no block reaches it as
        media; structuredContent arrives verbatim inside the string.
        """

        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        structured = None
        is_error: bool | None = None
        channel: str | None = None
        if failed or text.startswith(_FAILED_PREFIX):
            is_error, channel = True, "exception"
        elif text.startswith(_ERROR_PREFIX):
            is_error, channel = True, "text_prefix"
        else:
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                structured = parsed.get("structuredContent")
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            structured_content=structured,
            is_error=is_error,
            text=_redact_text(text),
            error_channel=channel,
            raw={"role": message.get("role"), "content": _redact_text(text)},
        )

    # ------------------------------------------------------------------
    # Probes. Every tool call is an AIAgent run whose fake provider calls the fixture
    # tool. Koog wraps only tools; what it does not offer is reached on the Kotlin SDK
    # Client handed to McpToolRegistryProvider.fromClient (route "session"), and client
    # handlers are registered on that Client before Koog sees it (route "underlying").
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources and prompts on the Client handed to fromClient; Koog wraps only tools."""

        payload = self._run_worker({**self._config(server), "mode": "probe"})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Every McpToolRegistryProvider entrypoint builds a Client with no capabilities and
        no handlers; fromClient takes one the application built. Sampling, elicitation and
        roots are registered on that Kotlin SDK Client, so each carries route
        "underlying"."""
        from .. import client_probe as C

        hooks = {
            "sampling": ("Client.setRequestHandler<CreateMessageRequest> before fromClient", "underlying"),
            "sampling.tools": ("Client.setRequestHandler<CreateMessageRequest> before fromClient", "underlying"),
            "elicitation.form": ("Client.setElicitationHandler before fromClient", "underlying"),
            "elicitation.url": ("Client.setElicitationHandler before fromClient", "underlying"),
            "roots.list": ("Client.addRoot before fromClient", "underlying"),
            "roots.list_changed": ("Client.removeRoot / addRoot / sendRootsListChanged", "underlying"),
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
        """Server utilities, one worker run each.

        Koog registers no notification handler: logging is read through a handler on the
        Client handed to fromClient, the level and completion are set on that Client, and
        progress has no consumer (McpTool.execute passes no progress token).
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
        """Catalogue changes read from the agent's ToolRegistry, which fromClient fills once;
        resources, prompts and subscriptions on the Client."""

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
            raise RenderError("koog worker needs a JDK 21: set MCP_MIRROR_JAVA_HOME or put java on PATH")
        with tempfile.TemporaryDirectory(prefix="mcp-mirror-koog-") as scratch:
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
                    env=_env(),
                )
            except subprocess.TimeoutExpired as exc:
                raise RenderError(f"koog worker timed out after {_TIMEOUT_SECONDS}s") from exc
            except Exception as exc:  # noqa: BLE001
                raise RenderError(f"koog worker failed to start: {exc}") from exc
            try:
                payload = json.loads(out_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
                raise RenderError(f"koog worker wrote no result ({exc}): {detail[-2000:]}") from exc
        if payload.get("fatal"):
            raise RenderError(f"koog worker error: {payload['fatal'][-2000:]}")
        if not isinstance(payload.get("tools"), list):
            raise RenderError("koog worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("koog worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload
