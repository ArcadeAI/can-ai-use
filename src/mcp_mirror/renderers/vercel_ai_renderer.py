"""Vercel AI SDK renderer (DESIGN.md section 8).

The AI SDK is TypeScript, so, like the Mastra renderer, this one drives the real
adapter from a small Node worker (``vercel_ai_node/_vercel_ai_worker.mjs``). The
worker calls ``createMCPClient`` from ``@ai-sdk/mcp``, over its stdio transport from
``@ai-sdk/mcp/mcp-stdio`` or its built-in HTTP transport, and reports the tools
``MCPClient.tools()`` returns: the objects a caller passes to ``generateText`` /
``streamText``. It does not run a model provider or capture a request.

``@ai-sdk/mcp`` speaks its own JSON-RPC rather than wrapping an MCP SDK, so the
revision it negotiates is an observation per server, read from the client's public
``initializeResult``, not a property of a pinned SDK.

``--runner managed`` installs the manifest's packages and its pinned Node into a cache
directory keyed by the manifest digest and points this renderer at it through
``MCP_MIRROR_NODE_RUNNER_DIR``. For a local run, install the bundled project once with
``npm ci --prefix src/mcp_mirror/renderers/vercel_ai_node`` (requires Node on PATH).
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

ID = "vercel_ai"
_BUNDLED_NODE_DIR = Path(__file__).parent / "vercel_ai_node"
_WORKER_NAME = "_vercel_ai_worker.mjs"
_TIMEOUT_SECONDS = 90
_DISCOVERY_RACE = "the initialize handshake is not accepted"
_DISCOVERY_RETRIES = 2


def _node_dir() -> Path:
    """The managed install when the runner names one for this worker, else the bundle.

    Every Node runner shares one environment variable, so a managed directory only
    counts if it holds this worker.
    """

    managed = os.environ.get(NODE_RUNNER_DIR_ENV)
    if managed and (Path(managed) / _WORKER_NAME).exists():
        return Path(managed)
    return _BUNDLED_NODE_DIR


def _worker() -> Path:
    return _node_dir() / _WORKER_NAME


def _adapter_pkg() -> Path:
    return _node_dir() / "node_modules" / "@ai-sdk" / "mcp"


def _node() -> str | None:
    """A managed install carries its own pinned Node; a local run uses PATH."""

    pinned = _node_dir() / "node_modules" / ".bin" / "node"
    if pinned.exists():
        return str(pinned)
    return shutil.which("node")


def available() -> bool:
    return _node() is not None and _worker().exists() and _adapter_pkg().exists()


def get_renderer() -> "VercelAIRenderer":
    return VercelAIRenderer()


def _content_block(item: Any) -> dict[str, Any]:
    """One CallToolResult content item, as the caller of ``execute`` receives it."""

    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"type": "text", "text": item}
    return {"type": type(item).__name__}


def _model_part(part: Any) -> dict[str, Any]:
    """One part of a ``toModelOutput`` content value, classified by what the model gets.

    The converter turns an MCP image into an AI SDK ``file`` part with an image media
    type, which a provider sends as an image, so it is recorded as one. Every other
    kind it does not map arrives as a ``text`` part holding the block's JSON.
    """

    if not isinstance(part, dict):
        return {"type": type(part).__name__}
    if part.get("type") == "file":
        media_type = part.get("mediaType")
        data = part.get("data")
        payload = data.get("data") if isinstance(data, dict) else data
        kind = "image" if isinstance(media_type, str) and media_type.startswith("image/") else "file"
        return {"type": kind, "mimeType": media_type, "data": payload}
    return part


class VercelAIRenderer:
    id = ID

    def versions(self) -> dict:
        version = "unknown"
        try:
            version = json.loads((_adapter_pkg() / "package.json").read_text()).get("version", "unknown")
        except Exception:
            pass
        return {"framework": version, "adapter": f"node {_node_version()}"}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api="@ai-sdk/mcp createMCPClient().tools",
            capture_object="AI SDK dynamicTool returned by MCPClient.tools",
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "protocolVersion from MCPClient.initializeResult, public API of "
                "@ai-sdk/mcp; the client implements its own JSON-RPC, so the revision "
                "is observed per server rather than implied by an SDK pin"
            ),
            limitation=(
                "The worker reads the JSON Schema from the AI SDK jsonSchema() wrapper "
                "on each tool's inputSchema; no provider adapter ran and no serialized "
                "request was captured."
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
            }
            # The provider-bound fields carry no annotations. @ai-sdk/mcp copies the
            # server's annotations onto the tool's app-side `metadata.annotations` and
            # its title onto `title`; the worker reports those so J5 can tell
            # "retained elsewhere" from "destroyed".
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

    # generateText calls a tool's `execute`, keeps what it resolves with as the step's
    # tool-result `output`, and builds the model's tool-result message from
    # `tool.toModelOutput(output)` (ai's createToolModelOutput). @ai-sdk/mcp resolves
    # with the CallToolResult, `isError` included, and sets `toModelOutput` to its own
    # MCP converter. The model reads the second; a caller reading steps sees the first.
    DEFAULT_RESULT_BOUNDARY = "model_output"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="model_output",
                label="Tool-result output the model receives (toModelOutput)",
                capture_api="@ai-sdk/mcp tool toModelOutput applied to the execute result",
                capture_object="the ToolResultOutput passed to the provider adapter",
                agent_path=True,
                limitation=(
                    "toModelOutput is called the way ai's createToolModelOutput calls it "
                    "for a tool that resolved; no provider adapter ran and no provider "
                    "request was captured."
                ),
            ),
            ResultBoundary(
                id="execute",
                label="Tool execute result (step toolResults output)",
                capture_api="@ai-sdk/mcp tool execute",
                capture_object="the CallToolResult execute resolves with",
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
        if boundary not in {"model_output", "execute"}:
            raise RenderError(f"vercel_ai renderer has no result boundary {boundary!r}")

        config = self._config(server)
        config["calls"] = [{"tool": tool, "arguments": arguments}]
        config["boundary"] = boundary
        payload = self._run_worker(config)

        entry = next((r for r in (payload.get("results") or []) if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"vercel_ai worker returned no result for {tool!r}")
        if not entry.get("ok"):
            if entry.get("threw"):
                # The adapter raising during a call is a contract the agent sees, not a
                # scanner failure, so it is recorded rather than propagated.
                return normalize_framework_result(
                    tool=tool,
                    origin=self.id,
                    is_error=True,
                    text=str(entry.get("error")),
                    error_channel="exception",
                    raw=entry,
                )
            raise RenderError(f"vercel_ai result capture failed: {entry.get('error')}")
        returned = entry.get("returned")
        if boundary == "execute":
            return self._call_result_to_rep(tool, returned)
        return self._model_output_to_rep(tool, returned)

    def _call_result_to_rep(self, tool: str, returned: Any) -> ResultRep:
        if isinstance(returned, dict) and isinstance(returned.get("content"), list):
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=[_content_block(item) for item in returned["content"]],
                structured_content=returned.get("structuredContent"),
                is_error=returned.get("isError"),
                raw=returned,
            )
        return normalize_framework_result(
            tool=tool, origin=self.id, structured_content=returned, raw=returned
        )

    def _model_output_to_rep(self, tool: str, returned: Any) -> ResultRep:
        """Map a ToolResultOutput (``{type, value}``) onto the common result shape."""

        kind = returned.get("type") if isinstance(returned, dict) else None
        value = returned.get("value") if isinstance(returned, dict) else None
        if kind == "content" and isinstance(value, list):
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=[_model_part(part) for part in value],
                raw=returned,
            )
        if kind == "text":
            return normalize_framework_result(tool=tool, origin=self.id, text=str(value), raw=returned)
        if kind in {"error-text", "error-json"}:
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                is_error=True,
                text=value if isinstance(value, str) else json.dumps(value),
                raw=returned,
            )
        # `json` is what the converter emits for a result with no content list: the
        # model reads the serialized value.
        return normalize_framework_result(
            tool=tool, origin=self.id, text=json.dumps(value), raw=returned
        )

    def _run_worker(self, config: dict) -> dict:
        try:
            return self._run_worker_once(config)
        except RenderError as exc:
            # createMCPClient gives `server/discover` one second, then falls back to a
            # legacy `initialize`. A 2.x server that answers the probe late has already
            # committed to 2026-07-28 and refuses that initialize. It is a startup race,
            # not what any cell measures, so two more fresh
            # connections are tried before the failure is reported.
            if _DISCOVERY_RACE not in str(exc):
                raise
        for attempt in range(_DISCOVERY_RETRIES):
            try:
                return self._run_worker_once(config)
            except RenderError as exc:
                if _DISCOVERY_RACE not in str(exc) or attempt == _DISCOVERY_RETRIES - 1:
                    raise
        raise AssertionError("unreachable")

    def _run_worker_once(self, config: dict) -> dict:
        try:
            # Run from the caller's cwd so a stdio server given by a relative command
            # resolves. ESM resolves @ai-sdk/mcp from the worker file's own directory.
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
            raise RenderError(f"vercel_ai worker timed out after {_TIMEOUT_SECONDS}s") from exc
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"vercel_ai worker failed to start: {exc}") from exc

        if proc.returncode != 0:
            # Report the error line, not the last frame of its Node stack trace.
            lines = [
                line for line in proc.stderr.strip().splitlines()
                if line.strip() and not line.lstrip().startswith("at ")
            ]
            detail = lines[-1] if lines else (proc.stdout.strip() or f"exit {proc.returncode}")
            raise RenderError(f"vercel_ai worker error: {detail}")

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RenderError(
                f"vercel_ai worker returned invalid JSON ({exc}): {proc.stdout[:200]!r}"
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
            raise RenderError("vercel_ai worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("vercel_ai worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources and prompts through the client's own resource and prompt methods."""

        payload = self._run_worker({**self._config(server), "probe": True})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling, elicitation and roots through ``MCPClient.onElicitationRequest``.

        Elicitation is the only server request @ai-sdk/mcp answers; the client replies
        -32601 to sampling and roots whatever is registered, so it is the one hook. Each
        tool runs in its own worker so what the handler saw is attributable to one
        request.
        """
        from .. import client_probe as C

        out: dict = {}
        tools = {feature: tool for feature, (tool, _expect) in C.FEATURES.items()}
        tools["capabilities"] = C.CAPABILITIES_TOOL
        for feature, tool in tools.items():
            config = {
                **self._config(server),
                "calls": [{"tool": tool, "arguments": {}}],
                "boundary": "execute",
                "elicitation": {"elicited": C.ELICITED},
            }
            try:
                payload = self._run_worker(config)
            except Exception as exc:  # noqa: BLE001 - the failure is the observation
                error = f"{type(exc).__name__}: {exc}"
                out[feature] = {"text": error[:C.TEXT_LIMIT]} if feature == "capabilities" else C.record(
                    feature, "onElicitationRequest", "adapter", error=error
                )
                continue
            entry = next((r for r in payload.get("results") or [] if r.get("tool") == tool), {})
            if entry.get("ok"):
                rep = self._call_result_to_rep(tool, entry.get("returned"))
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
                "onElicitationRequest" if hooked else None,
                "adapter" if hooked else None,
                rep,
                handler_saw=payload.get("handlerSaw") or [],
            )
        return out

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through MCPClient's public API, one worker run each."""
        from .. import probe as P

        spec = {
            "cancel_after_s": P.CANCEL_AFTER_S,
            "cancel_wait_s": P.CANCEL_WAIT_S,
            "cache_pause_s": P.CACHE_PAUSE_S,
            "completion_prompt": list(P.COMPLETION_PROMPT),
            "completion_resource": list(P.COMPLETION_RESOURCE),
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
        """Catalogue changes, relisted through the client's own list methods."""
        payload = self._run_worker({**self._config(server), "notify": True})
        return payload.get("notify") or {}

    def _config(self, server: ServerHandle) -> dict:
        if server.transport in ("http", "sse"):
            # `type: 'sse'` is the client's own HTTP+SSE transport, chosen explicitly.
            config: dict = {"transport": server.transport, "url": server.url}
            if server.headers:
                config["headers"] = dict(server.headers)
            return config
        return {"transport": "stdio", "command": server.command, "args": list(server.args)}


def _node_version() -> str:
    try:
        return subprocess.run(
            [_node() or "node", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"
