"""Genkit (JS) renderer (DESIGN.md section 8).

Genkit's MCP plugin (``@genkit-ai/mcp``) is a Node library, so this renderer follows the
Mastra bridge: a small Node worker (``genkit_js_node/_genkit_js_worker.mjs``) drives the
real ``createMcpHost`` and the ``GenkitMcpClient`` it builds, and reports what came back.
It never reimplements the conversion (decision D1).

Every capture is taken inside a real ``ai.generate`` call. ``GenkitMcpHost.getActiveTools(ai)``
turns each MCP tool into a Genkit ``ToolAction`` (``createDynamicTool``: name
``<server>/<tool>``, ``inputJsonSchema`` set to the MCP ``inputSchema``, ``outputSchema``
set to ``z.any()``), and ``ai.generate({ model, tools })`` hands them to a scripted model
registered with ``ai.defineModel``. The tool definitions are the ``ToolDefinition``
entries in the ``GenerateRequest.tools`` the model action receives (``toToolDefinition``
strips the ``<server>/`` prefix into ``metadata.originalName``); a tool result is the
``toolResponse`` part in the next request's messages. No model provider runs and no
provider request is serialized.

``--runner managed`` installs the manifest's packages and pinned Node into a cache
directory keyed by the manifest digest, and points this renderer at it through
``MCP_MIRROR_NODE_RUNNER_DIR``. For a local run, install the bundled project once with
``npm ci --prefix src/mcp_mirror/renderers/genkit_js_node``.
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

ID = "genkit_js"
_BUNDLED_NODE_DIR = Path(__file__).parent / "genkit_js_node"
_WORKER_NAME = "_genkit_js_worker.mjs"
_TIMEOUT_SECONDS = 120
_OPAQUE = ("data", "blob", "url")


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
    return _node() is not None and _worker().exists() and _package_dir("@genkit-ai", "mcp").exists()


def get_renderer() -> "GenkitJsRenderer":
    return GenkitJsRenderer()


def _package_version(path: Path) -> str:
    try:
        return json.loads((path / "package.json").read_text()).get("version", "unknown")
    except Exception:
        return "unknown"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (f"<{len(v)} chars>" if k in _OPAQUE and isinstance(v, str) and len(v) > 64 else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _media_kind(part: dict) -> str:
    media = part.get("media") or {}
    return (media.get("contentType") or "").split("/")[0] or "media"


class GenkitJsRenderer:
    id = ID

    def versions(self) -> dict:
        return {
            "framework": _package_version(_package_dir("genkit")),
            "adapter": (
                f"@genkit-ai/mcp {_package_version(_package_dir('@genkit-ai', 'mcp'))}, "
                f"@modelcontextprotocol/sdk {_package_version(_package_dir('@modelcontextprotocol', 'sdk'))}, "
                f"node {_node_version()}"
            ),
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "genkit ai.generate({ tools: await createMcpHost(...).getActiveTools(ai) }) "
                "(@genkit-ai/mcp GenkitMcpHost)"
            ),
            capture_object="ToolDefinition in the GenerateRequest.tools handed to the model action",
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "protocolVersion of the initialize result the @modelcontextprotocol/sdk "
                "Client.connect inside GenkitMcpClient receives, observed on "
                "Client.prototype.request in the node worker; neither Genkit nor the Client "
                "exposes the version"
            ),
            limitation=(
                "The model is a scripted ai.defineModel action; the GenerateRequest is "
                "captured before any provider plugin converts or serializes it, so no "
                "provider request body was captured."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        payload = self._run_worker(self._config(server))
        reps: list[ToolRep] = []
        for tool in payload["tools"]:
            definition = tool.get("definition") or {}
            # createDynamicTool builds the ToolAction from name, description, inputSchema
            # and tool._meta alone. Annotations, title, icons and outputSchema are not
            # kept on the action or anywhere else a caller can reach; the definition's
            # outputSchema is the z.any() Genkit sets on every MCP tool.
            tool_payload: dict[str, Any] = {
                "name": definition.get("name"),
                "description": definition.get("description"),
                "parameters": definition.get("inputSchema") or {},
                "outputSchema": definition.get("outputSchema"),
            }
            rep = normalize_function_tool(tool_payload, origin=self.id)
            reps.append(rep.model_copy(update={"raw": tool}))
        return reps

    # The ToolAction getActiveTools returns is callable on its own, and ai.generate puts
    # its return value in the toolResponse part the next model turn reads. A third
    # boundary exists only under createMcpHost's documented `multipart: true` option,
    # where media come back as toolResponse content parts.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool action return value",
                capture_api="@genkit-ai/mcp ToolAction from GenkitMcpHost.getActiveTools(ai), called with the arguments",
                capture_object="the value createDynamicTool's processResult returned",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Tool response to the model",
                capture_api="genkit ai.generate, next GenerateRequest.messages",
                capture_object="the toolResponse part in the GenerateRequest the model action receives",
                agent_path=True,
                limitation=(
                    "The model is a scripted ai.defineModel action; no provider plugin "
                    "serialized the request."
                ),
            ),
            ResultBoundary(
                id="multipart",
                label="Tool response to the model, createMcpHost({ multipart: true })",
                capture_api="genkit ai.generate with createMcpHost({ multipart: true }), next GenerateRequest.messages",
                capture_object="the toolResponse part (output and content) in the GenerateRequest the model action receives",
                agent_path=False,
                limitation=(
                    "This is a documented non-default host option, not the behaviour a "
                    "caller gets without configuring it."
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
        if boundary not in {"direct", "agent", "multipart"}:
            raise RenderError(f"genkit_js renderer has no result boundary {boundary!r}")
        config = self._config(server)
        config["calls"] = [{"tool": tool, "arguments": arguments}]
        config["boundary"] = "direct" if boundary == "direct" else "agent"
        if boundary == "multipart":
            config["multipart"] = True
        return self._call_rep(tool, self._run_worker(config), boundary)

    def _call_rep(self, tool: str, payload: dict, boundary: str) -> ResultRep:
        if tool not in {(t.get("definition") or {}).get("name") for t in payload["tools"]}:
            raise RenderError(f"genkit_js renderer found no tool named {tool!r}")
        entry = next((r for r in payload.get("results") or [] if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"genkit_js worker returned no result for {tool!r}")
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
            raise RenderError(f"genkit_js result capture failed: {entry.get('error')}")
        structured = entry.get("structuredFromServer") is True
        if boundary == "direct":
            return self._output_rep(tool, entry.get("direct"), None, entry.get("direct"), structured)
        response = entry.get("agent") or {}
        if boundary == "multipart":
            return self._output_rep(tool, response.get("output"), response.get("content"), response, structured)
        return self._output_rep(tool, response.get("output"), None, response, structured)

    def _output_rep(
        self, tool: str, output: Any, content: list | None, raw: Any, structured: bool
    ) -> ResultRep:
        """What the caller (or the model) reads from a Genkit tool output.

        ``processResult`` returns ``{ error }`` for ``isError``, ``structuredContent`` when
        the server sent one, the joined text (JSON-parsed when it looks like JSON) when
        every block is text, the lone block when there is one, and otherwise the whole
        CallToolResult. In multipart mode, image, audio and blob resources become
        ``media`` parts and resource links ``resource`` parts beside the ``output``.
        ``structured`` is the worker's observation that the CallToolResult behind this
        call carried ``structuredContent``, which the output alone cannot tell apart
        from JSON-parsed text.
        """

        blocks: list[dict] = []
        for part in content or []:
            if part.get("media"):
                blocks.append({"type": _media_kind(part), "mimeType": (part.get("media") or {}).get("contentType")})
            elif part.get("resource"):
                blocks.append({"type": "resource_link", "uri": (part.get("resource") or {}).get("uri")})
        if isinstance(output, dict) and set(output) == {"error"}:
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=blocks,
                is_error=True,
                text=str(output["error"]),
                error_channel="error_field",
                raw=_redact(raw),
            )
        if isinstance(output, str):
            return normalize_framework_result(tool=tool, origin=self.id, blocks=blocks, text=output, raw=_redact(raw))
        if output is None:
            # Multipart mode with no text: the parts beside the output are all there is.
            return normalize_framework_result(tool=tool, origin=self.id, blocks=blocks, raw=_redact(raw))
        if isinstance(output, dict) and isinstance(output.get("content"), list) and "type" not in output:
            # More than one block of mixed kinds: the whole CallToolResult, unchanged.
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=list(output.get("content") or []) + blocks,
                structured_content=output.get("structuredContent"),
                is_error=output.get("isError"),
                raw=_redact(raw),
            )
        if isinstance(output, dict) and isinstance(output.get("type"), str):
            # A lone non-text block, handed back as the MCP content block itself.
            return normalize_framework_result(tool=tool, origin=self.id, blocks=[output, *blocks], raw=_redact(raw))
        # structuredContent, or text that parsed as JSON: the model reads a JSON value.
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            structured_content=output if structured else None,
            text=json.dumps(_redact(output), sort_keys=True),
            raw=_redact(raw),
        )

    # ------------------------------------------------------------------
    # Probes. Every tool call is an ai.generate run whose scripted model calls the
    # fixture tool. What @genkit-ai/mcp does not offer is reached, where it can be, on
    # GenkitMcpClient.mcpClient, the public getter for the SDK Client underneath. Those
    # answers carry route "session".
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources and templates through getActiveResources(ai); prompts through
        getActivePrompts(ai) and ExecutablePrompt.render."""

        payload = self._run_worker({**self._config(server), "probe": True})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Roots through createMcpHost's ``roots`` option, the one client hook.

        GenkitMcpClient declares ``roots: { listChanged: true }`` and answers
        ``roots/list`` from its ``roots``; ``updateRoots`` sends
        ``notifications/roots/list_changed``. It declares neither sampling nor
        elicitation and registers no handler for them. Each tool runs in its own worker
        so what the server saw is attributable to one request.
        """
        from .. import client_probe as C

        roots = [{"uri": C.ROOT_URI, "name": "mcp-mirror"}]
        out: dict = {}
        tools = {feature: tool for feature, (tool, _expect) in C.FEATURES.items()}
        tools["capabilities"] = C.CAPABILITIES_TOOL
        for feature, tool in tools.items():
            config: dict = {
                **self._config(server),
                "calls": [{"tool": tool, "arguments": {}}],
                "boundary": "agent",
                "roots": roots,
            }
            hook, route = "createMcpHost({ roots })", "adapter"
            if feature == "roots.list_changed":
                config["rootsChange"] = [{"uri": C.CHANGED_ROOT_URI, "name": "mcp-mirror"}]
                hook = "createMcpHost({ roots }) + GenkitMcpClient.updateRoots"
            elif not feature.startswith("roots") and feature != "capabilities":
                hook, route = None, None
            try:
                rep = self._call_rep(tool, self._run_worker(config), "agent")
            except Exception as exc:  # noqa: BLE001 - the failure is the observation
                error = f"{type(exc).__name__}: {exc}"
                out[feature] = {"text": error[: C.TEXT_LIMIT]} if feature == "capabilities" else C.record(
                    feature, hook, route, error=error
                )
                continue
            if feature == "capabilities":
                out[feature] = {"text": C.rep_text(rep)[0]}
                continue
            out[feature] = C.record(feature, hook, route, rep)
        return out

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities, one worker run each.

        @genkit-ai/mcp has no progress, logging or message hook, no log-level setter and
        no completion API, and getActiveTools takes no cache option. Progress, log
        messages, the level and completion are reached on GenkitMcpClient.mcpClient
        (route "session").
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
        """Catalogue changes read from the next getActive* listing on the same session;
        subscriptions on GenkitMcpClient.mcpClient. GenkitMcpClient registers no
        notification handler."""

        payload = self._run_worker({**self._config(server), "notify": True})
        return payload.get("notify") or {}

    def _config(self, server: ServerHandle) -> dict:
        if server.transport == "sse":
            raise RenderError(
                "genkit_js: McpServerConfig has no SSE variant; @genkit-ai/mcp offers "
                "stdio ({ command }) and Streamable HTTP ({ url }) only"
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
            # resolves; ESM finds genkit from the worker file's own directory.
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
            raise RenderError(f"genkit_js worker timed out after {_TIMEOUT_SECONDS}s") from exc
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"genkit_js worker failed to start: {exc}") from exc

        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            raise RenderError(f"genkit_js worker error: {detail[-2000:]}")
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RenderError(
                f"genkit_js worker returned invalid JSON ({exc}): {proc.stdout[:200]!r}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
            raise RenderError("genkit_js worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("genkit_js worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload


def _node_version() -> str:
    try:
        return subprocess.run(
            [_node() or "node", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"
