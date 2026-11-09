"""Mastra renderer (DESIGN.md section 8).

Mastra is a TypeScript framework, so, to honor the cardinal rule of driving the
framework's *real* adapter (decision D1) rather than reimplementing it, this renderer
shells out to a small Node worker (``mastra_node/_mastra_worker.mjs``) that runs
Mastra's real ``@mastra/mcp`` client and reports the tool actions returned by
``MCPClient.listTools``. It does not run a model provider or capture a request.

``--runner managed`` installs the manifest's packages and its pinned Node into a
cache directory keyed by the manifest digest, and points this renderer at it through
``MCP_MIRROR_NODE_RUNNER_DIR``. For a local run, install the bundled project once with
``npm ci --prefix src/mcp_mirror/renderers/mastra_node`` (requires Node on PATH). When
neither is present the renderer reports itself unavailable, exactly like the Python
optional extras.
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
from ..source import ServerHandle
from ..runner_manifests import NODE_RUNNER_DIR_ENV
from . import RenderError

ID = "mastra"
_BUNDLED_NODE_DIR = Path(__file__).parent / "mastra_node"
_WORKER_NAME = "_mastra_worker.mjs"
_TIMEOUT_SECONDS = 90


def _node_dir() -> Path:
    """The managed install when the runner names one, else the bundled project."""

    managed = os.environ.get(NODE_RUNNER_DIR_ENV)
    return Path(managed) if managed else _BUNDLED_NODE_DIR


def _worker() -> Path:
    return _node_dir() / _WORKER_NAME


def _mastra_pkg() -> Path:
    return _node_dir() / "node_modules" / "@mastra" / "mcp"


def _node() -> str | None:
    """A managed install carries its own pinned Node; a local run uses PATH."""

    pinned = _node_dir() / "node_modules" / ".bin" / "node"
    if pinned.exists():
        return str(pinned)
    return shutil.which("node")


def available() -> bool:
    return _node() is not None and _worker().exists() and _mastra_pkg().exists()


def get_renderer() -> "MastraRenderer":
    return MastraRenderer()


def _block_of(item: Any) -> dict[str, Any]:
    """Classify one returned item by what the agent can actually tell it is."""

    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"type": "text", "text": item}
    return {"type": type(item).__name__}


class MastraRenderer:
    id = ID

    def versions(self) -> dict:
        version = "unknown"
        try:
            version = json.loads((_mastra_pkg() / "package.json").read_text()).get("version", "unknown")
        except Exception:
            pass
        return {"framework": version, "adapter": f"node {_node_version()}"}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api="@mastra/mcp MCPClient.listTools",
            capture_object="Mastra Tool action returned by listTools",
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(
                self,
                "_negotiated_mcp_spec_version",
                None,
            ),
            protocol_version_evidence=(
                "protocolVersion from MCPClient.getServerProtocolVersions, public API "
                "since @mastra/mcp 2.x"
            ),
            limitation=(
                "The worker extracts JSON Schema from Mastra's Standard Schema wrapper; "
                "no AI SDK provider adapter or serialized request was captured."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        payload = self._run_worker(self._config(server))
        reps: list[ToolRep] = []
        for entry in payload["tools"]:
            tool_payload = {
                "name": entry.get("name"),
                "description": entry.get("description"),
                "parameters": entry.get("parameters") or {},
                "outputSchema": entry.get("outputSchema"),
            }
            # The declared tool fields carry no annotations. @mastra/mcp 2.x keeps the
            # server's annotations on `tool.mcp.annotations` and the title on
            # `tool.title`; the worker reports those so J5 can tell "retained
            # elsewhere" from "destroyed". 1.x reported nothing here.
            retained = {
                k: v
                for k, v in ((entry.get("retained") or {}).get("annotations") or {}).items()
                if v is not None
            }
            framework_metadata = {"annotations": retained} if retained else {}
            reps.append(
                normalize_function_tool(
                    tool_payload, origin=self.id, framework_metadata=framework_metadata
                )
            )
        return reps

    # `MCPClient.listTools` is what @mastra/mcp documents passing to `new Agent({tools})`,
    # so a tool action's `execute` is the agent boundary; there is no separate direct
    # path to compare it against. What does change the answer is the per-server
    # `onToolError` option, typed `'throw' | 'return'` and defaulting to `'throw'`.
    # Both settings are published because the default decides whether the agent sees a
    # thrown MastraError or the server's CallToolResult with `isError` intact.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="agent",
                label="Agent tool invocation, onToolError: 'throw' (default)",
                capture_api="@mastra/mcp tool action execute",
                capture_object="the tool action return value, or the MastraError it throws",
                agent_path=True,
                limitation=(
                    "The tool action is executed directly. No AI SDK provider was "
                    "bound and no provider request was captured."
                ),
            ),
            ResultBoundary(
                id="agent_on_tool_error_return",
                label="Agent tool invocation, onToolError: 'return'",
                capture_api="@mastra/mcp tool action execute, server configured onToolError: 'return'",
                capture_object="the CallToolResult envelope the action resolves with",
                agent_path=False,
                limitation=(
                    "This is a documented non-default server option, not the "
                    "behaviour a caller gets without configuring it."
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
        """Invoke one tool and capture one declared framework-result boundary."""

        boundary = boundary or self.DEFAULT_RESULT_BOUNDARY
        if boundary not in {"agent", "agent_on_tool_error_return"}:
            raise RenderError(f"mastra renderer has no result boundary {boundary!r}")

        config = self._config(server)
        config["calls"] = [{"tool": tool, "arguments": arguments}]
        if boundary == "agent_on_tool_error_return":
            config["onToolError"] = "return"
        payload = self._run_worker(config)

        entry = next(
            (r for r in (payload.get("results") or []) if r.get("tool") == tool), None
        )
        if entry is None:
            raise RenderError(f"mastra worker returned no result for {tool!r}")
        if not entry.get("ok"):
            if entry.get("threw"):
                # A tool that reported failure through `isError` may surface as a thrown
                # error instead. That is a different contract for the agent, not a
                # scanner failure, so it is recorded rather than propagated.
                return normalize_framework_result(
                    tool=tool,
                    origin=self.id,
                    is_error=True,
                    text=str(entry.get("error")),
                    error_channel="exception",
                    raw=entry,
                )
            raise RenderError(f"mastra result capture failed: {entry.get('error')}")
        return self._result_to_rep(tool, entry.get("returned"))

    def _result_to_rep(self, tool: str, returned: Any) -> ResultRep:
        """Map whatever the worker reported back onto the common result shape.

        Mastra's action returns a plain JSON value, so the worker has already crossed
        the JS/Python boundary by the time this runs. A dict that still carries MCP's
        ``content`` list is unpacked; anything else is recorded as it arrived.
        """

        if isinstance(returned, dict):
            content = returned.get("content")
            if isinstance(content, list):
                return normalize_framework_result(
                    tool=tool,
                    origin=self.id,
                    blocks=[_block_of(item) for item in content],
                    structured_content=returned.get("structuredContent"),
                    is_error=returned.get("isError"),
                    raw=returned,
                )
            return normalize_framework_result(
                tool=tool, origin=self.id, structured_content=returned, raw=returned
            )
        if isinstance(returned, list):
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=[_block_of(item) for item in returned],
                raw=returned,
            )
        return normalize_framework_result(
            tool=tool, origin=self.id, text=str(returned), raw=returned
        )

    def _run_worker(self, config: dict) -> dict:
        try:
            # Run from the caller's cwd (NOT the node dir) so a stdio server given by a
            # relative command resolves correctly. Node still finds @mastra/mcp because
            # ESM resolves bare imports from the worker file's own directory upward;
            # NODE_PATH covers anything the worker loads through require().
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
            raise RenderError(f"mastra worker timed out after {_TIMEOUT_SECONDS}s") from exc
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"mastra worker failed to start: {exc}") from exc

        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            raise RenderError(f"mastra worker error: {detail}")

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RenderError(
                f"mastra worker returned invalid JSON ({exc}): {proc.stdout[:200]!r}"
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
            raise RenderError("mastra worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("mastra worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources and prompts through MCPClient.resources / .prompts in the worker."""
        payload = self._run_worker({**self._config(server), "probe": True})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling, elicitation and roots through MCPClient's ``inputRequests`` hook.

        @mastra/mcp documents ``inputRequests`` as its answer to embedded elicitation
        requests and states the deprecated roots and sampling capabilities are not
        supported, so it is the only hook registered. Each tool runs in its own worker
        so what the handler saw is attributable to one request.
        """
        from .. import client_probe as C

        out: dict = {}
        tools = {feature: tool for feature, (tool, _expect) in C.FEATURES.items()}
        tools["capabilities"] = C.CAPABILITIES_TOOL
        for feature, tool in tools.items():
            config = {
                **self._config(server),
                "calls": [{"tool": tool, "arguments": {}}],
                "onToolError": "return",
                "inputRequests": {"elicited": C.ELICITED},
            }
            try:
                payload = self._run_worker(config)
            except Exception as exc:  # noqa: BLE001 - the failure is the observation
                error = f"{type(exc).__name__}: {exc}"
                out[feature] = {"text": error[:C.TEXT_LIMIT]} if feature == "capabilities" else C.record(
                    feature, "inputRequests", "adapter", error=error
                )
                continue
            entry = next((r for r in payload.get("results") or [] if r.get("tool") == tool), {})
            if entry.get("ok"):
                rep = self._result_to_rep(tool, entry.get("returned"))
            else:
                rep = normalize_framework_result(
                    tool=tool, origin=self.id, is_error=True,
                    text=str(entry.get("error")), error_channel="exception",
                )
            if feature == "capabilities":
                out[feature] = {"text": C.rep_text(rep)[0]}
                continue
            hooked = feature.startswith("elicitation.")
            out[feature] = C.record(
                feature,
                "inputRequests" if hooked else None,
                "adapter" if hooked else None,
                rep,
                handler_saw=payload.get("inputRequestsSeen") or [],
            )
        return out

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through MCPClient's public options, one worker run each."""
        from .. import probe as P

        spec = {"cancel_after_s": P.CANCEL_AFTER_S, "cancel_wait_s": P.CANCEL_WAIT_S, "cache_pause_s": P.CACHE_PAUSE_S}
        out: dict = {}
        for feature, server in servers.items():
            try:
                payload = self._run_worker({**self._config(server), "utility": feature, "utilitySpec": spec})
                out[feature] = payload.get("utility") or {"ok": False, "error": "worker returned no utility answer"}
            except RenderError as exc:
                out[feature] = {"ok": False, "error": str(exc)[:300]}
        return out

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes through MCPClient's onListChanged / onUpdated hooks and
        resources.subscribe in the worker."""
        payload = self._run_worker({**self._config(server), "notify": True})
        return payload.get("notify") or {}

    def _config(self, server: ServerHandle) -> dict:
        if server.transport == "http":
            config: dict = {"transport": "http", "url": server.url}
            if server.headers:
                config["headers"] = dict(server.headers)
            return config
        if server.transport == "sse":
            # MCPClient has no transport switch: given a URL it tries Streamable HTTP and
            # falls back to HTTP+SSE when that is refused, which is the path measured.
            return {"transport": "http", "url": server.url}
        return {"transport": "stdio", "command": server.command, "args": list(server.args)}


def _node_version() -> str:
    try:
        return subprocess.run(
            [_node() or "node", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"
