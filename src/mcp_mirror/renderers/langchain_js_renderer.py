"""LangChain.js renderer (DESIGN.md section 8).

LangChain.js is TypeScript, so, like the Mastra and Vercel AI renderers, this one
drives the real adapter from a small Node worker
(``langchain_js_node/_langchain_js_worker.mjs``). The worker builds
``MCPAdapter`` from ``@langchain/mcp-adapters``, loads its tools with ``listTools()``
and reports each through ``convertToOpenAITool`` from ``@langchain/core``, the
provider-shaped conversion ``bindTools`` uses. No chat model runs and no provider
request is captured.

The adapter wraps ``@modelcontextprotocol/client`` 2.x, which negotiates either a
handshake-era revision or 2026-07-28 (``mode: "auto"``, the default). The worker reads
the result from the SDK ``Client`` the adapter hands out through ``getClient()``.

``--runner managed`` installs the manifest's packages and its pinned Node into a cache
directory keyed by the manifest digest and points this renderer at it through
``MCP_MIRROR_NODE_RUNNER_DIR``. For a local run, install the bundled project once with
``npm ci --prefix src/mcp_mirror/renderers/langchain_js_node`` (requires Node on PATH).
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

ID = "langchain_js"
_BUNDLED_NODE_DIR = Path(__file__).parent / "langchain_js_node"
_WORKER_NAME = "_langchain_js_worker.mjs"
_TIMEOUT_SECONDS = 90


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
    return _node_dir() / "node_modules" / "@langchain" / "mcp-adapters"


def _core_pkg() -> Path:
    return _node_dir() / "node_modules" / "@langchain" / "core"


def _node() -> str | None:
    """A managed install carries its own pinned Node; a local run uses PATH."""

    pinned = _node_dir() / "node_modules" / ".bin" / "node"
    if pinned.exists():
        return str(pinned)
    return shutil.which("node")


def available() -> bool:
    return _node() is not None and _worker().exists() and _adapter_pkg().exists()


def get_renderer() -> "LangChainJSRenderer":
    return LangChainJSRenderer()


def _package_version(path: Path) -> str:
    try:
        return json.loads((path / "package.json").read_text()).get("version", "unknown")
    except Exception:
        return "unknown"


def _block_of(item: Any) -> dict[str, Any]:
    """One LangChain message content block.

    The adapter emits standard content blocks: ``image`` and ``audio`` with ``data`` and
    ``mimeType``, and a resource link as a ``file`` block naming its URL, which is left
    as a ``file`` so the cells judge it by whether its URI survived.
    """

    if isinstance(item, str):
        return {"type": "text", "text": item}
    if not isinstance(item, dict):
        return {"type": type(item).__name__}
    return item


class LangChainJSRenderer:
    id = ID

    def versions(self) -> dict:
        return {
            "framework": _package_version(_adapter_pkg()),
            "adapter": f"@langchain/core {_package_version(_core_pkg())}, node {_node_version()}",
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api="@langchain/core/utils/function_calling convertToOpenAITool",
            capture_object=(
                "OpenAI-compatible function-tool object built from the DynamicStructuredTool "
                "MCPAdapter.listTools returns"
            ),
            capture_stage="provider_format",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "getNegotiatedProtocolVersion() on the @modelcontextprotocol/client Client "
                "that MCPAdapter.getClient returns for the connected server"
            ),
            limitation=(
                "No chat model was bound and no serialized provider request was captured."
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
            # The provider-shaped tool carries no annotations. The adapter keeps the
            # server's annotations on the tool's app-side `metadata.annotations`; the
            # worker reports those so J5 can tell "retained elsewhere" from "destroyed".
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

    # The adapter builds each tool with responseFormat "content_and_artifact". Invoked
    # with a plain argument object the tool returns its content only (the artifact is
    # dropped) and an `isError` result throws ToolException. LangGraph's prebuilt
    # ToolNode, which createReactAgent and createAgent run tools through, invokes it
    # with the model's ToolCall, and the adapter answers an `isError` result with a
    # ToolMessage of status "error". Both are published.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="agent",
                label="Agent tool invocation (LangGraph ToolNode)",
                capture_api="@langchain/langgraph/prebuilt ToolNode.invoke",
                capture_object="the ToolMessage ToolNode appends for the model's tool call",
                agent_path=True,
                limitation=(
                    "ToolNode runs one AIMessage tool call built by the worker. The "
                    "ToolMessage's content is what a chat model reads; its artifact stays "
                    "app-side and is not recorded as rendered. No chat model was bound and "
                    "no provider request was captured."
                ),
            ),
            ResultBoundary(
                id="direct",
                label="Direct tool invocation",
                capture_api="@langchain/core StructuredTool.invoke(args)",
                capture_object="the tool return value, without a ToolCall, so no ToolMessage",
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
            raise RenderError(f"langchain_js renderer has no result boundary {boundary!r}")

        config = self._config(server)
        config["calls"] = [{"tool": tool, "arguments": arguments}]
        config["boundary"] = boundary
        payload = self._run_worker(config)
        return self._entry_to_rep(tool, payload)

    def _entry_to_rep(self, tool: str, payload: dict) -> ResultRep:
        entry = next((r for r in (payload.get("results") or []) if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"langchain_js worker returned no result for {tool!r}")
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
            raise RenderError(f"langchain_js result capture failed: {entry.get('error')}")
        returned = entry.get("returned") or {}
        if returned.get("kind") == "tool_message":
            return self._message_to_rep(tool, returned.get("value") or {})
        return self._value_to_rep(tool, returned.get("value"))

    def _message_to_rep(self, tool: str, message: dict) -> ResultRep:
        """A ToolMessage: ``status`` carries isError, ``content`` is what the model reads.

        The artifact (structuredContent, embedded resources, annotations, ``_meta``)
        stays app-side, so it is kept as raw evidence and not recorded as rendered.
        """

        content = message.get("content")
        is_error = message.get("status") == "error"
        if isinstance(content, list):
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                blocks=[_block_of(item) for item in content],
                is_error=is_error,
                raw=message,
            )
        text = content if isinstance(content, str) else json.dumps(content)
        return normalize_framework_result(
            tool=tool, origin=self.id, is_error=is_error, text=text, raw=message
        )

    def _value_to_rep(self, tool: str, value: Any) -> ResultRep:
        """What ``StructuredTool.invoke(args)`` returned: the content half only."""

        if isinstance(value, str):
            return normalize_framework_result(tool=tool, origin=self.id, text=value, raw=value)
        if isinstance(value, list):
            return normalize_framework_result(
                tool=tool, origin=self.id, blocks=[_block_of(item) for item in value], raw=value
            )
        return normalize_framework_result(
            tool=tool, origin=self.id, text=json.dumps(value), raw=value
        )

    def _run_worker(self, config: dict) -> dict:
        try:
            # Run from the caller's cwd so a stdio server given by a relative command
            # resolves. ESM resolves the adapter from the worker file's own directory.
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
            raise RenderError(f"langchain_js worker timed out after {_TIMEOUT_SECONDS}s") from exc
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"langchain_js worker failed to start: {exc}") from exc

        if proc.returncode != 0:
            # Report the error line, not the last frame of its Node stack trace.
            lines = [
                line for line in proc.stderr.strip().splitlines()
                if line.strip() and not line.lstrip().startswith("at ")
            ]
            detail = lines[-1] if lines else (proc.stdout.strip() or f"exit {proc.returncode}")
            raise RenderError(f"langchain_js worker error: {detail}")

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RenderError(
                f"langchain_js worker returned invalid JSON ({exc}): {proc.stdout[:200]!r}"
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
            raise RenderError("langchain_js worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("langchain_js worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources through MCPAdapter's resource methods; it has no prompt API."""

        payload = self._run_worker({**self._config(server), "probe": True})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling, elicitation and roots through MCPAdapter's elicitation routes.

        Elicitation is the only server request the adapter answers, by era. A
        handshake-era session needs the server option ``mode: "legacy"`` with an
        ``onElicitation`` callback (``auto`` mode rejects that option). At 2026-07-28 the
        adapter turns each request into a LangGraph interrupt, answered by resuming a
        checkpointed graph with ``createMCPElicitationResume``. A first run reads the era
        the server negotiates, and each tool then runs in its own worker, inside such a
        graph, so what was answered is attributable to one request. Sampling and roots
        have no route: no capability is declared on a legacy session, and on a modern
        one the adapter fails the call.
        """
        from .. import client_probe as C

        base = self._config(server)
        try:
            legacy = self._run_worker(base).get("protocolEra") != "modern"
        except RenderError:
            legacy = False
        hook = "server option mode: legacy + onElicitation" if legacy else (
            "LangGraph interrupt + createMCPElicitationResume"
        )
        out: dict = {}
        tools = {feature: tool for feature, (tool, _expect) in C.FEATURES.items()}
        tools["capabilities"] = C.CAPABILITIES_TOOL
        for feature, tool in tools.items():
            config = {
                **base,
                "calls": [{"tool": tool, "arguments": {}}],
                "boundary": "graph",
                "elicitation": {"elicited": C.ELICITED},
                "legacy": legacy,
            }
            try:
                payload = self._run_worker(config)
            except Exception as exc:  # noqa: BLE001 - the failure is the observation
                error = f"{type(exc).__name__}: {exc}"
                out[feature] = {"text": error[:C.TEXT_LIMIT]} if feature == "capabilities" else C.record(
                    feature, hook, "adapter", error=error
                )
                continue
            rep = self._entry_to_rep(tool, payload)
            if feature == "capabilities":
                out[feature] = {"text": C.rep_text(rep)[0]}
                continue
            hooked = feature.startswith("elicitation.")
            out[feature] = C.record(
                feature,
                hook if hooked else None,
                "adapter" if hooked else None,
                rep,
                handler_saw=payload.get("handlerSaw") or [],
            )
        return out

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through MCPAdapter and its server hooks, one worker run each."""
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
        """Catalogue changes through the server's list-changed hooks, relisted through MCPAdapter."""
        payload = self._run_worker({**self._config(server), "notify": True})
        return payload.get("notify") or {}

    def _config(self, server: ServerHandle) -> dict:
        if server.transport in ("http", "sse"):
            # `transport: 'sse'` selects the adapter's HTTP+SSE transport explicitly;
            # `http` is Streamable HTTP (with the adapter's automatic SSE fallback).
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
