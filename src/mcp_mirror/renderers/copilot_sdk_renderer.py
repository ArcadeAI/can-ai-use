"""GitHub Copilot SDK renderer (DESIGN.md section 8).

The Copilot SDK (``@github/copilot-sdk``) is TypeScript and drives the Copilot runtime
over its own JSON-RPC; the runtime it starts as a child process is the MCP client. So,
like the Mastra and LangChain.js renderers, this one runs the real SDK from a small Node
worker (``copilot_sdk_node/_copilot_sdk_worker.mjs``).

The worker creates a session with the fixture as its only entry in ``mcpServers`` and a
BYOK OpenAI-compatible provider, and passes the client a ``CopilotRequestHandler``
subclass that answers every model request the runtime sends. The captured object is
the serialized ``chat/completions`` request body, so the capture is a real provider
request and no network call is made. The runtime runs with ``HOME`` and
``COPILOT_HOME`` in a fresh temporary directory and config discovery off, so no
operator login, config or MCP server list is read.

The runtime negotiates its own revision: it tries ``server/discover`` at 2026-07-28
and falls back to ``initialize``. The worker reads the frames it exchanged through the
session's diagnostics API (``session.rpc.diagnostics.read`` at trace level) and reports
the revision the runtime settled on.

``--runner managed`` installs the manifest's packages and its pinned Node into a cache
directory keyed by the manifest digest and points this renderer at it through
``MCP_MIRROR_NODE_RUNNER_DIR``. For a local run, install the bundled project once with
``npm ci --prefix src/mcp_mirror/renderers/copilot_sdk_node`` (requires Node on PATH).
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

ID = "copilot_sdk"
_BUNDLED_NODE_DIR = Path(__file__).parent / "copilot_sdk_node"
_WORKER_NAME = "_copilot_sdk_worker.mjs"
_TIMEOUT_SECONDS = 120


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


def _sdk_pkg() -> Path:
    return _node_dir() / "node_modules" / "@github" / "copilot-sdk"


def _node() -> str | None:
    """A managed install carries its own pinned Node; a local run uses PATH."""

    pinned = _node_dir() / "node_modules" / ".bin" / "node"
    if pinned.exists():
        return str(pinned)
    return shutil.which("node")


def available() -> bool:
    return _node() is not None and _worker().exists() and _sdk_pkg().exists()


def get_renderer() -> "CopilotSDKRenderer":
    return CopilotSDKRenderer()


def _package_json(path: Path) -> dict:
    try:
        return json.loads((path / "package.json").read_text())
    except Exception:
        return {}


class CopilotSDKRenderer:
    id = ID
    # OAuth options for an HTTP server, forwarded to the worker, which signs in through
    # the runtime's own flow (session.rpc.mcp.oauth.login). Set by the conformance client.
    _auth: dict | None = None

    def versions(self) -> dict:
        package = _package_json(_sdk_pkg())
        return {
            "framework": package.get("version", "unknown"),
            "adapter": (
                f"Copilot runtime {package.get('copilotCliVersion', 'unknown')}, "
                f"node {_node_version()}"
            ),
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api="CopilotRequestHandler.sendRequest (CopilotClientOptions.requestHandler)",
            capture_object=(
                "the function entries named src-<tool> in the tools array of the "
                "chat/completions request body the Copilot runtime sent the session's "
                "BYOK provider"
            ),
            capture_stage="provider_request",
            provider_request_captured=True,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "the initialize result, or the revision stamped on each request after a "
                "successful server/discover, in the JSON-RPC frames "
                "session.rpc.diagnostics.read returns at trace level"
            ),
            limitation=(
                "The provider is a BYOK OpenAI-compatible endpoint answered in-process by "
                "the request handler; a GitHub-hosted model was not used. The request "
                "also carries the runtime's built-in tools, which are not recorded. "
                "Retained annotations come from a second turn in which the stub model "
                "calls every tool and every approval request is refused."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        payload = self._run_worker({**self._config(server), "retained": True})
        retained_by_name = payload.get("retained") or {}
        reps: list[ToolRep] = []
        for entry in payload["tools"]:
            # The request tool carries no annotations. The runtime's approval step keeps
            # two of them: a permission request names the tool's title and a readOnly
            # flag, and a tool it runs without asking is one it read as readOnlyHint.
            kept = retained_by_name.get(entry.get("name")) or {}
            annotations = {
                key: value
                for key, value in (("title", kept.get("title")), ("readOnlyHint", kept.get("readOnly")))
                if value is not None
            }
            reps.append(
                normalize_function_tool(
                    {
                        "name": entry.get("name"),
                        "description": entry.get("description"),
                        "parameters": entry.get("parameters") or {},
                    },
                    origin=self.id,
                    framework_metadata={"annotations": annotations} if annotations else {},
                )
            )
        return reps

    # The runtime runs every MCP tool call itself, inside its agent loop, and
    # serializes the result into the next model request; that request is what the
    # model reads, so it is the published boundary. session.rpc.tools.execute runs an
    # offered tool outside a turn and hands the caller the runtime's ToolResult.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="agent",
                label="Agent tool loop (next provider request)",
                capture_api="CopilotRequestHandler.sendRequest",
                capture_object=(
                    "the tool message, and any message the runtime adds for the same "
                    "result, in the chat/completions request that follows the model's "
                    "tool call"
                ),
                agent_path=True,
                limitation=(
                    "The stub model asks for one tool call. The chat/completions wire "
                    "format has no tool-error flag, so an error reaches the model as text "
                    "only. The model is declared vision-capable "
                    "(SessionConfig.modelCapabilities), as gpt-4.1 is; images ride a "
                    "user message the runtime appends, whose framing text is not recorded."
                ),
            ),
            ResultBoundary(
                id="direct",
                label="Direct tool invocation",
                capture_api="CopilotSession.rpc.tools.execute",
                capture_object="the ToolResult the runtime returns, outside a model turn",
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
        if boundary not in {"agent", "direct"}:
            raise RenderError(f"copilot_sdk renderer has no result boundary {boundary!r}")

        config = self._config(server)
        config["calls"] = [{"tool": tool, "arguments": arguments}]
        config["boundary"] = boundary
        payload = self._run_worker(config)
        return self._entry_to_rep(tool, payload)

    def _entry_to_rep(self, tool: str, payload: dict) -> ResultRep:
        entry = next((r for r in (payload.get("results") or []) if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"copilot_sdk worker returned no result for {tool!r}")
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
            raise RenderError(f"copilot_sdk result capture failed: {entry.get('error')}")
        returned = entry.get("returned") or {}
        if returned.get("kind") == "tool_result":
            return self._tool_result_to_rep(tool, returned.get("value"))
        return self._request_to_rep(tool, returned)

    def _request_to_rep(self, tool: str, returned: dict) -> ResultRep:
        """What the next model request carried for the tool call.

        The tool message's text is one block, kept when it is empty: the model still
        received a tool message, and an empty one is what is left of a result whose
        content the runtime dropped. A message the runtime appends for the same result
        contributes its media parts; its framing text ("Here is an image:") is the
        runtime's, not the server's, and stays in ``raw``.
        """

        parts = returned.get("parts") or []
        if not returned.get("messages"):
            raise RenderError(
                f"copilot_sdk worker captured no model request after the {tool!r} call"
            )
        blocks: list[dict[str, Any]] = []
        for part in parts:
            if part.get("role") == "tool":
                blocks.append({"type": "text", "text": part.get("text") or ""})
            elif part.get("type") != "text":
                blocks.append({k: v for k, v in part.items() if k != "role"})
        tool_texts = [p.get("text") for p in parts if p.get("role") == "tool"]
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            text="\n".join(t for t in tool_texts if t is not None),
            raw=returned,
        )

    def _tool_result_to_rep(self, tool: str, value: Any) -> ResultRep:
        """The runtime's ToolResult: ``contents`` holds the blocks, ``resultType`` the flag."""

        if isinstance(value, str):
            return normalize_framework_result(tool=tool, origin=self.id, text=value, raw=value)
        value = value or {}
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=list(value.get("contents") or []),
            structured_content=value.get("structuredContent"),
            is_error=value.get("resultType") not in (None, "success"),
            text=value.get("textResultForLlm"),
            raw=value,
        )

    def _run_worker(self, config: dict) -> dict:
        try:
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
            raise RenderError(f"copilot_sdk worker timed out after {_TIMEOUT_SECONDS}s") from exc
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"copilot_sdk worker failed to start: {exc}") from exc

        if proc.returncode != 0:
            lines = [
                line for line in proc.stderr.strip().splitlines()
                if line.strip() and not line.lstrip().startswith("at ")
            ]
            detail = lines[-1] if lines else (proc.stdout.strip() or f"exit {proc.returncode}")
            raise RenderError(f"copilot_sdk worker error: {detail}")

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RenderError(
                f"copilot_sdk worker returned invalid JSON ({exc}): {proc.stdout[:200]!r}"
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
            raise RenderError("copilot_sdk worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("copilot_sdk worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources through session.rpc.mcp.resources; the session has no prompt API."""

        payload = self._run_worker({**self._config(server), "probe": True})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling, elicitation and roots, each tool run through the agent tool loop.

        The SDK's elicitation hook is ``SessionConfig.onElicitationRequest``. Sampling
        has no host hook: the runtime declares the capability and answers a request by
        running the session's own model, which here is the stub model behind the
        request handler. Roots have no API.
        """
        from .. import client_probe as C

        base = self._config(server)
        out: dict = {}
        tools = {feature: tool for feature, (tool, _expect) in C.FEATURES.items()}
        tools["capabilities"] = C.CAPABILITIES_TOOL
        for feature, tool in tools.items():
            config = {
                **base,
                "calls": [{"tool": tool, "arguments": {}}],
                "boundary": "agent",
                "elicitation": {"elicited": C.ELICITED},
                "sampling": {"sampled": C.SAMPLED, "afterTool": C.SAMPLED_AFTER_TOOL},
            }
            if feature.startswith("elicitation."):
                hook, route = "SessionConfig.onElicitationRequest", "adapter"
            elif feature.startswith("sampling"):
                # The SDK's documented path: the sampling.requested event, answered with
                # rpc.mcp.executeSampling on the session model and resolved with
                # rpc.ui.handlePendingSampling. handler_saw records whether it fired.
                hook, route = "the sampling.requested event (CopilotSession.rpc.mcp.executeSampling)", "adapter"
            else:
                hook, route = None, None
            try:
                payload = self._run_worker(config)
            except Exception as exc:  # noqa: BLE001 - the failure is the observation
                error = f"{type(exc).__name__}: {exc}"
                out[feature] = {"text": error[:C.TEXT_LIMIT]} if feature == "capabilities" else C.record(
                    feature, hook, route, error=error
                )
                continue
            rep = self._entry_to_rep(tool, payload)
            if feature == "capabilities":
                out[feature] = {"text": C.rep_text(rep)[0]}
                continue
            saw = list(payload.get("handlerSaw") or [])
            if feature.startswith("sampling"):
                saw += [
                    {"hook": "session model", "messages": len(s.get("messages") or []), "tools": s.get("tools") or []}
                    for s in payload.get("sampling") or []
                ]
            out[feature] = C.record(feature, hook, route, rep, handler_saw=saw)
        return out

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through the session's RPC and events, one worker run each."""
        from .. import probe as P

        spec = {
            "cancel_after_s": P.CANCEL_AFTER_S,
            "cancel_wait_s": P.CANCEL_WAIT_S,
            "cache_pause_s": P.CACHE_PAUSE_S,
            "completion_prompt": list(P.COMPLETION_PROMPT),
            "completion_resource": list(P.COMPLETION_RESOURCE),
            "set_level": P.SET_LEVEL,
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
        """Catalogue changes through the session's mcp.*.list_changed events."""
        payload = self._run_worker({**self._config(server), "notify": True})
        return payload.get("notify") or {}

    def _config(self, server: ServerHandle) -> dict:
        if server.transport in ("http", "sse"):
            # MCPServerConfig `type: "http"` is Streamable HTTP; `type: "sse"` the
            # deprecated HTTP+SSE transport.
            config: dict = {"transport": server.transport, "url": server.url}
            if server.headers:
                config["headers"] = dict(server.headers)
            if self._auth:
                config["auth"] = dict(self._auth)
            return config
        return {"transport": "stdio", "command": server.command, "args": list(server.args)}


def _node_version() -> str:
    try:
        return subprocess.run(
            [_node() or "node", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"
