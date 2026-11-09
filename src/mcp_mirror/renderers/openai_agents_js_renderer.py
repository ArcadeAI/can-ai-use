"""OpenAI Agents SDK (JS) renderer (DESIGN.md section 8).

The JavaScript Agents SDK (``@openai/agents``) is TypeScript, so this renderer follows
the Mastra bridge: a small Node worker (``openai_agents_js_node/_openai_agents_js_worker.mjs``)
drives the SDK's real ``MCPServerStdio`` / ``MCPServerStreamableHttp`` / ``MCPServerSSE``
(agents-core ``src/mcp.ts``) and reports what came back. It never reimplements the
conversion (decision D1).

Every capture is taken inside a real ``Runner.run`` turn. An ``Agent`` gets the server in
``mcpServers`` and loads the MCP tools itself, through ``getAllMcpTools`` and
``mcpToFunctionTool``. The model is scripted, and it records each ``ModelRequest`` it
receives. The tool definitions are that request's ``tools``. A tool result is the
``function_call_result`` item in the next request's ``input``. No OpenAI provider model
runs and no provider request is serialized.

``--runner managed`` installs the manifest's packages and pinned Node into a cache
directory keyed by the manifest digest, and points this renderer at it through
``MCP_MIRROR_NODE_RUNNER_DIR``. For a local run, install the bundled project once with
``npm ci --prefix src/mcp_mirror/renderers/openai_agents_js_node``.
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

ID = "openai_agents_js"
_BUNDLED_NODE_DIR = Path(__file__).parent / "openai_agents_js_node"
_WORKER_NAME = "_openai_agents_js_worker.mjs"
_TIMEOUT_SECONDS = 120


def _node_dir() -> Path:
    """The managed install when the runner names one, else the bundled project."""

    managed = os.environ.get(NODE_RUNNER_DIR_ENV)
    return Path(managed) if managed else _BUNDLED_NODE_DIR


def _worker() -> Path:
    return _node_dir() / _WORKER_NAME


def _package_dir(name: str) -> Path:
    return _node_dir() / "node_modules" / "@openai" / name


def _node() -> str | None:
    """A managed install carries its own pinned Node; a local run uses PATH."""

    pinned = _node_dir() / "node_modules" / ".bin" / "node"
    if pinned.exists():
        return str(pinned)
    return shutil.which("node")


def available() -> bool:
    return _node() is not None and _worker().exists() and _package_dir("agents").exists()


def get_renderer() -> "OpenAIAgentsJsRenderer":
    return OpenAIAgentsJsRenderer()


def _package_version(path: Path) -> str:
    try:
        return json.loads((path / "package.json").read_text()).get("version", "unknown")
    except Exception:
        return "unknown"


def _model_block(item: Any) -> dict[str, Any]:
    """One item of a function_call_result output, named by what the model receives."""

    if isinstance(item, str):
        return {"type": "text", "text": item}
    if not isinstance(item, dict):
        return {"type": type(item).__name__}
    kind = item.get("type")
    if kind in ("input_text", "text"):
        return {"type": "text", "text": item.get("text")}
    if kind == "input_image":
        image = item.get("image")
        block: dict[str, Any] = {"type": "image", "image": image}
        if isinstance(image, str) and image.startswith("data:"):
            block["mimeType"] = image[5:].split(";", 1)[0]
        return block
    return dict(item)


def _content_block(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"type": "text", "text": item}
    return {"type": type(item).__name__}


class OpenAIAgentsJsRenderer:
    id = ID

    def versions(self) -> dict:
        return {
            "framework": _package_version(_package_dir("agents")),
            "adapter": f"@openai/agents-core {_package_version(_package_dir('agents-core'))}, node {_node_version()}",
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "@openai/agents Runner.run with Agent({ mcpServers: [MCPServerStdio | "
                "MCPServerStreamableHttp | MCPServerSSE] }), tools loaded by getAllMcpTools"
            ),
            capture_object="SerializedFunctionTool in the ModelRequest.tools handed to Model.getResponse",
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "Client.getNegotiatedProtocolVersion on the @modelcontextprotocol/client "
                "Client the node shim keeps in its protected session field; the SDK has "
                "no public accessor"
            ),
            limitation=(
                "The Model is scripted. No OpenAI provider model ran and no serialized "
                "provider request was captured."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        payload = self._run_worker(self._config(server))
        reps: list[ToolRep] = []
        for entry in payload["tools"]:
            # mcpToFunctionTool builds the FunctionTool from name, description and
            # inputSchema alone. Annotations, title, outputSchema and icons are not
            # carried, on the tool or anywhere beside it, so nothing is retained.
            tool_payload = {
                "name": entry.get("name"),
                "description": entry.get("description"),
                "parameters": entry.get("parameters") or {},
            }
            reps.append(normalize_function_tool(tool_payload, origin=self.id))
        return reps

    # The agent path is the function_call_result the next model turn receives. The
    # adapter's own call API, MCPServer.callTool, is published beside it so the two can
    # be compared. `useStructuredContent` is a documented per-server option, default
    # off, that swaps the content blocks for JSON.stringify(structuredContent).
    DEFAULT_RESULT_BOUNDARY = "agent"
    _BOUNDARIES = ("agent", "direct", "agent_use_structured_content")

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="agent",
                label="Function call result to the model",
                capture_api="@openai/agents Runner.run, next ModelRequest.input",
                capture_object="the function_call_result item the scripted Model receives",
                agent_path=True,
                limitation="The Model is scripted; no provider request was serialized.",
            ),
            ResultBoundary(
                id="direct",
                label="Adapter call API",
                capture_api="@openai/agents MCPServerStdio.callTool",
                capture_object="the CallToolResultContent array, with the metadata the SDK attaches to it",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent_use_structured_content",
                label="Function call result to the model, useStructuredContent: true",
                capture_api="@openai/agents Runner.run, server option useStructuredContent: true",
                capture_object="the function_call_result item the scripted Model receives",
                agent_path=False,
                limitation=(
                    "A documented non-default server option, not the behaviour a caller "
                    "gets without configuring it."
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
        """Invoke one tool and capture one declared result boundary."""

        boundary = boundary or self.DEFAULT_RESULT_BOUNDARY
        if boundary not in self._BOUNDARIES:
            raise RenderError(f"openai_agents_js renderer has no result boundary {boundary!r}")
        config = self._config(server)
        config["calls"] = [{"tool": tool, "arguments": arguments}]
        config["boundary"] = "direct" if boundary == "direct" else "agent"
        if boundary == "agent_use_structured_content":
            config["useStructuredContent"] = True
        payload = self._run_worker(config)

        entry = next((r for r in payload.get("results") or [] if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"openai_agents_js worker returned no result for {tool!r}")
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
            raise RenderError(f"openai_agents_js result capture failed: {entry.get('error')}")
        if boundary == "direct":
            return self._direct_to_rep(tool, entry.get("returned") or {})
        return self._agent_to_rep(tool, entry.get("returned") or {})

    def _agent_to_rep(self, tool: str, item: dict) -> ResultRep:
        """The function_call_result the model receives: a string or output items.

        The item carries no isError and no structuredContent field; whatever the model
        learns about either arrives inside the output.
        """

        output = item.get("output")
        if isinstance(output, str):
            return normalize_framework_result(tool=tool, origin=self.id, text=output, raw=item)
        items = output if isinstance(output, list) else [output]
        return normalize_framework_result(
            tool=tool, origin=self.id, blocks=[_model_block(i) for i in items], raw=item
        )

    def _direct_to_rep(self, tool: str, returned: dict) -> ResultRep:
        content = returned.get("content")
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=[_content_block(i) for i in content] if isinstance(content, list) else [],
            structured_content=returned.get("structuredContent"),
            is_error=returned.get("isError"),
            raw=returned,
        )

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources through MCPServer.listResources / readResource / listResourceTemplates."""

        payload = self._run_worker({**self._config(server), "probe": True})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """The MCPServer options take no sampling, elicitation or roots handler, and the
        SDK registers none on the Client it builds, so every tool is called through the
        agent path with nothing to answer the server's requests."""
        from .. import client_probe as C

        return C.without_hooks(lambda tool: self.render_result(server, tool, {}, "agent"))

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through the SDK's public surface, one worker run each."""
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
        """Catalogue changes read from the next MCPServer listing; the SDK has no hook."""

        payload = self._run_worker({**self._config(server), "notify": True})
        return payload.get("notify") or {}

    def _config(self, server: ServerHandle) -> dict:
        if server.transport in ("http", "sse"):
            config: dict = {"transport": server.transport, "url": server.url}
            if server.headers:
                config["headers"] = dict(server.headers)
            return config
        return {"transport": "stdio", "command": server.command, "args": list(server.args)}

    def _run_worker(self, config: dict) -> dict:
        try:
            # Run from the caller's cwd so a stdio server given by a relative command
            # resolves; ESM finds @openai/agents from the worker file's own directory.
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
            raise RenderError(f"openai_agents_js worker timed out after {_TIMEOUT_SECONDS}s") from exc
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"openai_agents_js worker failed to start: {exc}") from exc

        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            raise RenderError(f"openai_agents_js worker error: {detail[-2000:]}")
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RenderError(
                f"openai_agents_js worker returned invalid JSON ({exc}): {proc.stdout[:200]!r}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
            raise RenderError("openai_agents_js worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError("openai_agents_js worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload


def _node_version() -> str:
    try:
        return subprocess.run(
            [_node() or "node", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"
