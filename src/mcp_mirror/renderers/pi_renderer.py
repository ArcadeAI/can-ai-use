"""Pi renderer (DESIGN.md section 8).

``@earendil-works/pi-mcp`` is the MCP client library inside the Pi coding agent. It is
TypeScript with its own JSON-RPC core (it does not wrap the official MCP SDK), so this
renderer follows the Mastra bridge: a small Node worker (``pi_node/_pi_worker.mjs``)
drives the package's real ``McpClient`` over its ``StdioTransport`` or
``StreamableHttpTransport`` and reports what came back. It never reimplements the client
(decision D1).

pi-mcp has no agent loop. Tool definitions are the ``Tool`` objects
``McpClient.listTools`` returns. A tool result is captured at two boundaries: the
``LlmContent[]`` the package's own ``toLlmContent`` makes for a model, and the
``CallToolResult`` ``McpClient.callTool`` returns. No model runs. The Pi CLI, which wraps
this client, is measured separately as the ``pi_cli`` harness.

``--runner managed`` installs the manifest's packages and pinned Node into a cache
directory keyed by the manifest digest, and points this renderer at it through
``MCP_MIRROR_NODE_RUNNER_DIR``. For a local run, install the bundled project once with
``npm ci --prefix src/mcp_mirror/renderers/pi_node``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_source_tool
from ..runner_manifests import NODE_RUNNER_DIR_ENV
from ..source import ServerHandle
from . import RenderError

ID = "pi"
_BUNDLED_NODE_DIR = Path(__file__).parent / "pi_node"
_WORKER_NAME = "_pi_worker.mjs"
_TIMEOUT_SECONDS = 120


def _node_dir() -> Path:
    """The managed install when the runner names one, else the bundled project."""

    managed = os.environ.get(NODE_RUNNER_DIR_ENV)
    return Path(managed) if managed else _BUNDLED_NODE_DIR


def _worker() -> Path:
    return _node_dir() / _WORKER_NAME


def _package_dir() -> Path:
    return _node_dir() / "node_modules" / "@earendil-works" / "pi-mcp"


def _node() -> str | None:
    """A managed install carries its own pinned Node; a local run uses PATH."""

    pinned = _node_dir() / "node_modules" / ".bin" / "node"
    if pinned.exists():
        return str(pinned)
    return shutil.which("node")


def available() -> bool:
    return _node() is not None and _worker().exists() and _package_dir().exists()


def get_renderer() -> "PiRenderer":
    return PiRenderer()


def _package_version(path: Path) -> str:
    try:
        return json.loads((path / "package.json").read_text()).get("version", "unknown")
    except Exception:
        return "unknown"


def _block(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"type": "text", "text": item}
    return {"type": type(item).__name__}


class PiRenderer:
    id = ID

    def versions(self) -> dict:
        return {
            "framework": _package_version(_package_dir()),
            "adapter": f"@earendil-works/pi-mcp {_package_version(_package_dir())}, node {_node_version()}",
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api="@earendil-works/pi-mcp McpClient.listTools over StdioTransport | StreamableHttpTransport",
            capture_object="Tool (the MCP tools/list entry pi-mcp returns unchanged)",
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence="McpClient.protocolVersion, the public getter on the initialize result",
            limitation=(
                "pi-mcp is a client library with no agent loop; no model ran and no "
                "provider request was captured."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        payload = self._run_worker(self._config(server))
        # listTools hands back each tools/list entry as parsed JSON, so every field the
        # server sent (title, annotations, outputSchema, icons, _meta) is retained.
        return [normalize_source_tool(entry, origin=self.id) for entry in payload["tools"]]

    # The agent path is toLlmContent, the package's documented conversion of a tool
    # result into model content. The README's AgentTool example passes isError beside
    # it from the raw result; that wrapper is user code, so the boundary is the
    # package function alone. McpClient.callTool's raw result is published beside it.
    DEFAULT_RESULT_BOUNDARY = "llm"
    _BOUNDARIES = ("llm", "direct")

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="llm",
                label="Model content from toLlmContent",
                capture_api="@earendil-works/pi-mcp toLlmContent(await McpClient.callTool(...))",
                capture_object="the LlmContent[] (text and image) toLlmContent returns",
                agent_path=True,
                limitation=(
                    "pi-mcp has no agent loop; toLlmContent is its documented model "
                    "conversion, and isError is left for the caller to carry."
                ),
            ),
            ResultBoundary(
                id="direct",
                label="Client call API",
                capture_api="@earendil-works/pi-mcp McpClient.callTool",
                capture_object="the CallToolResult, with content, structuredContent, isError and _meta",
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
            raise RenderError(f"pi renderer has no result boundary {boundary!r}")
        return self._call(self._config(server), tool, arguments, boundary)[0]

    def _call(self, config: dict, tool: str, arguments: dict[str, Any], boundary: str) -> tuple[ResultRep, dict]:
        config = {**config, "calls": [{"tool": tool, "arguments": arguments}]}
        payload = self._run_worker(config)
        entry = next((r for r in payload.get("results") or [] if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"pi worker returned no result for {tool!r}")
        if not entry.get("ok"):
            if entry.get("threw"):
                rep = normalize_framework_result(
                    tool=tool,
                    origin=self.id,
                    is_error=True,
                    text=str(entry.get("error")),
                    error_channel="exception",
                    raw=entry,
                )
                return rep, payload
            raise RenderError(f"pi result capture failed: {entry.get('error')}")
        returned = entry.get("returned") or {}
        if boundary == "direct":
            direct = returned.get("direct") or {}
            content = direct.get("content")
            rep = normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=[_block(i) for i in content] if isinstance(content, list) else [],
                structured_content=direct.get("structuredContent"),
                is_error=direct.get("isError"),
                raw=direct,
            )
        else:
            # LlmContent carries no isError and no structuredContent field; whatever
            # the model learns about either arrives inside the blocks.
            llm = returned.get("llm")
            rep = normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=[_block(i) for i in llm] if isinstance(llm, list) else [],
                raw=llm,
            )
        return rep, payload

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources through McpClient.listResources / readResource / listResourceTemplates."""

        payload = self._run_worker({**self._config(server), "probe": True})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Roots through McpClient's ``roots`` option, the one client-feature hook.

        The option registers a roots/list handler and declares the roots capability
        (without listChanged). McpClient has no sampling or elicitation option and
        registers no handler for either, so those tools run with nothing to answer.
        Each tool runs in its own worker so what the handler saw is attributable.
        """
        from .. import client_probe as C

        config = self._config(server)
        roots = [{"uri": C.ROOT_URI, "name": "mcp-mirror"}]
        out: dict = {}
        for feature, (tool, _expect) in C.FEATURES.items():
            hooked = feature == "roots.list"
            hook = "McpClient({ roots })" if hooked else None
            try:
                rep, payload = self._call({**config, "roots": roots} if hooked else config, tool, {}, "direct")
            except Exception as exc:  # noqa: BLE001 - the failure is the observation
                out[feature] = C.record(feature, hook, "adapter" if hooked else None, error=f"{type(exc).__name__}: {exc}")
                continue
            out[feature] = C.record(
                feature, hook, "adapter" if hooked else None, rep, handler_saw=payload.get("seen") or []
            )
        out["capabilities"] = C.capabilities(
            lambda: self._call({**config, "roots": roots}, C.CAPABILITIES_TOOL, {}, "direct")[0]
        )
        return out

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through McpClient, one worker run each."""
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
        """Catalogue changes heard through McpClient.onNotification, then relisted."""

        payload = self._run_worker({**self._config(server), "notify": True})
        return payload.get("notify") or {}

    def _config(self, server: ServerHandle) -> dict:
        if server.transport == "sse":
            raise RenderError("pi-mcp has no HTTP+SSE transport; only StdioTransport and StreamableHttpTransport")
        if server.transport == "http":
            config: dict = {"transport": "http", "url": server.url}
            if server.headers:
                config["headers"] = dict(server.headers)
            return config
        return {"transport": "stdio", "command": server.command, "args": list(server.args)}

    def _run_worker(self, config: dict) -> dict:
        try:
            # Run from the caller's cwd so a stdio server given by a relative command
            # resolves; ESM finds pi-mcp from the worker file's own directory.
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
            raise RenderError(f"pi worker timed out after {_TIMEOUT_SECONDS}s") from exc
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"pi worker failed to start: {exc}") from exc

        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            raise RenderError(f"pi worker error: {detail[-2000:]}")
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RenderError(f"pi worker returned invalid JSON ({exc}): {proc.stdout[:200]!r}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
            raise RenderError("pi worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("pi worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload


def _node_version() -> str:
    try:
        return subprocess.run(
            [_node() or "node", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"
