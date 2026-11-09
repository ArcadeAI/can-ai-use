"""Microsoft Agent Framework (.NET) renderer (DESIGN.md section 8).

The Agent Framework for .NET (``Microsoft.Agents.AI``) runs on the CLR, so this renderer
follows the JVM bridges: a small console worker (``ms_agent_framework_dotnet/``) connects
to the server the way the Agent Framework documents for .NET. It creates an ``McpClient``
from the MCP C# SDK (``ModelContextProtocol``) with ``McpClient.CreateAsync``, lists the
tools with ``McpClient.ListToolsAsync`` (each ``McpClientTool`` is an ``AIFunction``), and
hands them to ``IChatClient.AsAIAgent(tools: ...)``. The resulting ``ChatClientAgent``
runs against a scripted ``IChatClient``. The worker never reimplements the conversion
(decision D1).

The Agent Framework has no MCP client of its own: its .NET MCP guidance is the MCP C#
SDK's ``McpClient`` and ``McpClientTool``, so those are the adapter here. Their methods,
handlers and options carry route ``"adapter"``.

Tool definitions are the ``AIFunctionDeclaration`` entries in ``ChatOptions.Tools``,
which ``FunctionInvokingChatClient`` hands the model. A tool result is the
``FunctionResultContent`` in the next request's messages, or the value
``McpClientTool.InvokeAsync`` returned, as seen by function middleware registered with
``AIAgentBuilder.Use``. No model runs and no provider request is serialized.

``--runner managed`` installs a pinned .NET SDK, publishes the worker against the
manifest's packages into a cache directory keyed by the manifest and worker sources, and
points this renderer at it through ``MCP_MIRROR_DOTNET_RUNNER_DIR`` and
``MCP_MIRROR_DOTNET_ROOT``. For a local run, publish the bundled project once with
``dotnet publish src/mcp_mirror/renderers/ms_agent_framework_dotnet -c Release`` and put
a .NET 10 SDK on ``PATH``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..runner_manifests import DOTNET_ROOT_ENV, DOTNET_RUNNER_DIR_ENV, DOTNET_WORKER_DIR, DOTNET_WORKER_DLL
from ..source import ServerHandle
from . import RenderError

ID = "ms_agent_framework_dotnet"
_PROJECT = Path(__file__).parent / "ms_agent_framework_dotnet"
_BUNDLED_DLL = _PROJECT / "bin" / "Release" / "net10.0" / "publish" / DOTNET_WORKER_DLL
_TIMEOUT_SECONDS = 180
_OPAQUE = ("data", "blob")
_SECRETISH = ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL", "AUTH")

# Every client-feature hook is an McpClientOptions entry handed to McpClient.CreateAsync.
_CLIENT_HOOKS = {
    "sampling": ("McpClientOptions.Handlers.SamplingHandler", "adapter"),
    "sampling.tools": ("McpClientOptions.Handlers.SamplingHandler with Capabilities.Sampling.Tools", "adapter"),
    "elicitation.form": ("McpClientOptions.Handlers.ElicitationHandler", "adapter"),
    "elicitation.url": ("McpClientOptions.Handlers.ElicitationHandler with Capabilities.Elicitation.Url", "adapter"),
    "roots.list": ("McpClientOptions.Handlers.RootsHandler", "adapter"),
    "roots.list_changed": (
        "McpClientOptions.Handlers.RootsHandler with Capabilities.Roots.ListChanged, then "
        "McpClient.SendNotificationAsync(notifications/roots/list_changed)",
        "adapter",
    ),
}


def _dll() -> Path:
    """The managed build when the runner names one, else the bundled project's publish."""

    managed = os.environ.get(DOTNET_RUNNER_DIR_ENV)
    return Path(managed) / DOTNET_WORKER_DIR / DOTNET_WORKER_DLL if managed else _BUNDLED_DLL


def _dotnet() -> str | None:
    """A managed install carries its own pinned SDK; a local run uses DOTNET_ROOT or PATH."""

    for root in (os.environ.get(DOTNET_ROOT_ENV), os.environ.get("DOTNET_ROOT")):
        if root and (Path(root) / "dotnet").exists():
            return str(Path(root) / "dotnet")
    return shutil.which("dotnet")


def available() -> bool:
    return _dotnet() is not None and _dll().exists()


def get_renderer() -> "MicrosoftAgentFrameworkDotnetRenderer":
    return MicrosoftAgentFrameworkDotnetRenderer()


@lru_cache(maxsize=4)
def _bundled_versions(dll: str) -> dict[str, str]:
    """Package versions from the Worker.deps.json that publish writes beside the worker."""

    try:
        deps = json.loads(Path(dll).with_name("Worker.deps.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    found: dict[str, str] = {}
    for key in deps.get("libraries") or {}:
        name, _, version = key.partition("/")
        found[name] = version
    return found


@lru_cache(maxsize=4)
def _runtime_version(dotnet: str) -> str:
    try:
        proc = subprocess.run([dotnet, "--version"], capture_output=True, text=True, timeout=30, env=_environment())
    except Exception:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _environment() -> dict[str, str]:
    """No token-like variables reach the worker or the stdio server it starts."""

    environment = {k: v for k, v in os.environ.items() if not any(word in k.upper() for word in _SECRETISH)}
    dotnet = _dotnet()
    if dotnet:
        environment.setdefault("DOTNET_ROOT", str(Path(dotnet).resolve().parent))
    environment.update({"DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1"})
    return environment


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (f"<{len(v)} chars>" if k in _OPAQUE and isinstance(v, str) and len(v) > 32 else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _block(item: dict) -> dict:
    """One AIContent as the MCP block kind it came from.

    ``AIContentExtensions.ToAIContent`` turns text into ``TextContent`` and image and
    audio into ``DataContent``. An embedded resource becomes ``TextContent`` or
    ``DataContent`` with the resource URI in ``AdditionalProperties["uri"]``.
    """

    extra = item.get("additionalProperties") or {}
    kind = item.get("type")
    if kind == "text":
        block = {"type": "resource" if "uri" in extra else "text", "text": item.get("text")}
    elif kind == "data":
        media = item.get("mediaType") or ""
        major = media.split("/")[0]
        block = {
            "type": "resource" if "uri" in extra else major if major in ("image", "audio") else "data",
            "mimeType": media,
            "bytes": item.get("bytes"),
            "sha256": item.get("sha256"),
        }
    elif kind == "uri":
        block = {"type": "resource_link", "uri": item.get("uri"), "mimeType": item.get("mediaType")}
    else:
        block = {"type": kind, "json": item.get("json")}
    if "uri" in extra:
        block["uri"] = extra["uri"]
    meta = {k: v for k, v in extra.items() if k != "uri"}
    if meta:
        block["_meta"] = meta
    return block


class MicrosoftAgentFrameworkDotnetRenderer:
    id = ID

    def __init__(self) -> None:
        # Set by the conformance client for an auth/ scenario: ClientOAuthOptions fields.
        self._auth: dict | None = None

    def versions(self) -> dict:
        found = _bundled_versions(str(_dll()))
        dotnet = _dotnet()
        return {
            "framework": found.get("Microsoft.Agents.AI", "unknown"),
            "adapter": (
                f"ModelContextProtocol {found.get('ModelContextProtocol', 'unknown')}, "
                f"Microsoft.Extensions.AI {found.get('Microsoft.Extensions.AI', 'unknown')}, "
                f"dotnet {_runtime_version(dotnet) if dotnet else 'unknown'}"
            ),
        }

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "Microsoft.Agents.AI IChatClient.AsAIAgent(tools: McpClient.ListToolsAsync()) run by "
                "AIAgent.RunAsync; a scripted IChatClient.GetResponseAsync receives the ChatOptions"
            ),
            capture_object=(
                "Microsoft.Extensions.AI.AIFunctionDeclaration in ChatOptions.Tools "
                "(Name, Description, JsonSchema, ReturnJsonSchema), built by McpClientTool"
            ),
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence="McpClient.NegotiatedProtocolVersion",
            limitation=(
                "The model is a scripted IChatClient; no provider client (OpenAI, Azure OpenAI, "
                "Anthropic) serialized the request."
            ),
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        payload = self._run_worker({**self._config(server), "mode": "tools"})
        retained = payload.get("retained") or {}
        reps: list[ToolRep] = []
        for declaration in payload["tools"]:
            # McpClientTool reports the MCP inputSchema as JsonSchema and the outputSchema
            # as ReturnJsonSchema. It keeps the MCP Tool it was built from as the public
            # ProtocolTool property (annotations, icons, _meta, title).
            tool_payload: dict[str, Any] = {
                "name": declaration.get("name"),
                "description": declaration.get("description"),
                "parameters": declaration.get("jsonSchema") or {},
            }
            if declaration.get("returnJsonSchema") is not None:
                tool_payload["outputSchema"] = declaration["returnJsonSchema"]
            kept = retained.get(declaration.get("name")) or {}
            metadata: dict[str, Any] = {}
            for key in ("annotations", "icons", "title", "execution"):
                if kept.get(key):
                    metadata[key] = kept[key]
            if kept.get("_meta"):
                metadata["_meta"] = kept["_meta"]
            rep = normalize_function_tool(tool_payload, origin=self.id, framework_metadata=metadata)
            reps.append(rep.model_copy(update={"raw": declaration}))
        return reps

    # One agent run yields both boundaries. The scripted model emits a function call;
    # FunctionInvokingChatClient invokes McpClientTool through the agent's function
    # middleware, which sees the return value, and puts it in a FunctionResultContent in
    # the next request.
    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="Microsoft.Agents.AI AIAgentBuilder.Use(function middleware), the value next() returns",
                capture_object="the object McpClientTool.InvokeAsync returned (AIContent, AIContent[] or JsonElement)",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Function result to the model",
                capture_api="Microsoft.Agents.AI AIAgent.RunAsync, next IChatClient.GetResponseAsync messages",
                capture_object="FunctionResultContent.Result in the ChatMessage the model receives",
                agent_path=True,
                limitation=(
                    "The model is a scripted IChatClient; no provider client serialized the "
                    "FunctionResultContent."
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
        if boundary not in {"direct", "agent"}:
            raise RenderError(f"{ID} renderer has no result boundary {boundary!r}")
        config = {**self._config(server), "mode": "tools", "calls": [{"tool": tool, "arguments": arguments}]}
        payload = self._run_worker(config)
        if tool not in {t.get("name") for t in payload["tools"]}:
            raise RenderError(f"{ID} renderer found no tool named {tool!r}")
        entry = next((r for r in payload.get("results") or [] if r.get("tool") == tool), None)
        if entry is None:
            raise RenderError(f"{ID} worker returned no result for {tool!r}")
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
                    raw=entry,
                )
            raise RenderError(f"{ID} result capture failed: {entry.get('error')}")
        if boundary == "direct":
            if entry.get("directThrew"):
                return normalize_framework_result(
                    tool=tool,
                    origin=self.id,
                    is_error=True,
                    text=str(entry.get("error")),
                    error_channel="exception",
                    raw={k: entry.get(k) for k in ("exception", "error")},
                )
            if not entry.get("invoked"):
                raise RenderError(f"{ID} agent run never invoked the tool")
            return self._value_rep(tool, entry.get("direct"))
        agent = entry.get("agent") or {}
        if agent.get("exception"):
            # FunctionInvokingChatClient caught the exception and handed the model a
            # string in its place.
            result = (agent.get("result") or {}).get("value")
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                is_error=True,
                text=str(result if result is not None else agent["exception"]),
                error_channel="exception",
                raw=_redact(agent),
            )
        return self._value_rep(tool, agent.get("result"))

    def _value_rep(self, tool: str, value: Any) -> ResultRep:
        """A function result as the worker serialized it.

        ``McpClientTool.InvokeCoreAsync`` returns one ``AIContent`` for a single-block
        result, an ``AIContent[]`` when every block converts, and otherwise (an isError
        result, structuredContent, application ``_meta``, or a block with no AIContent
        form such as a resource link) a ``JsonElement`` of the whole CallToolResult. A
        JsonElement reaches the model as JSON text, so of its blocks only text counts.
        """

        if not isinstance(value, dict):
            return normalize_framework_result(tool=tool, origin=self.id, text=None if value is None else str(value), raw=value)
        kind = value.get("kind")
        if kind == "content":
            return normalize_framework_result(tool=tool, origin=self.id, blocks=[_block(value["item"])], raw=_redact(value))
        if kind == "contents":
            blocks = [_block(item) for item in value.get("items") or []]
            return normalize_framework_result(tool=tool, origin=self.id, blocks=blocks, raw=_redact(value))
        if kind == "json":
            result = value.get("value")
            if isinstance(result, dict):
                error = result.get("isError") is True
                # The model reads the text blocks verbatim inside that JSON; image and
                # audio arrive as base64 strings, not media, so they are not blocks.
                texts = [b for b in result.get("content") or [] if isinstance(b, dict) and b.get("type") == "text"]
                return normalize_framework_result(
                    tool=tool,
                    origin=self.id,
                    blocks=texts,
                    structured_content=result.get("structuredContent"),
                    is_error=True if error else None,
                    text=json.dumps(_redact(result), sort_keys=True),
                    raw=_redact(value),
                )
            return normalize_framework_result(tool=tool, origin=self.id, text=json.dumps(result), raw=value)
        return normalize_framework_result(
            tool=tool, origin=self.id, text=json.dumps(_redact(value.get("value")), sort_keys=True), raw=_redact(value)
        )

    # ------------------------------------------------------------------
    # Probes. Every tool call is a ChatClientAgent run whose scripted model calls the
    # fixture tool. Resources, prompts, completion, the log level and subscriptions are
    # methods on the McpClient the application created; route "adapter" throughout.
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """Resources and prompts through McpClient; AsAIAgent takes only the tools."""

        payload = self._run_worker({**self._config(server), "mode": "probe"})
        return payload.get("probe") or {}

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Sampling, elicitation and roots through the McpClientHandlers on the
        McpClientOptions given to McpClient.CreateAsync; every tool is called through the
        agent."""
        from .. import client_probe as C

        tools = [tool for tool, _ in C.FEATURES.values()] + [C.CAPABILITIES_TOOL]
        try:
            payload = self._run_worker({**self._config(server), "mode": "client", "hooks": True, "clientCalls": tools})
        except RenderError as exc:
            out = {feature: C.record(feature, *_CLIENT_HOOKS[feature], error=str(exc)[:500]) for feature in C.FEATURES}
            out["capabilities"] = {"text": str(exc)[: C.TEXT_LIMIT]}
            return out
        answers = payload.get("client") or {}
        out: dict = {}
        for feature, (tool, _expect) in C.FEATURES.items():
            entry = answers.get(tool) or {"ok": False, "error": "worker returned no answer"}
            hook, route = _CLIENT_HOOKS[feature]
            saw = entry.get("handler_saw") or []
            if entry.get("root_change_error"):
                saw = saw + [{"hook": hook, "error": entry["root_change_error"]}]
            try:
                rep = self._entry_to_rep(tool, entry, "agent")
            except RenderError as exc:
                out[feature] = C.record(feature, hook, route, error=str(exc), handler_saw=saw)
                continue
            out[feature] = C.record(feature, hook, route, rep, handler_saw=saw)
        capabilities = answers.get(C.CAPABILITIES_TOOL) or {}
        out["capabilities"] = C.capabilities(lambda: self._entry_to_rep(C.CAPABILITIES_TOOL, capabilities, "agent"))
        return out

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities, one worker run each.

        Progress through McpClientTool.WithProgress, server logs through a
        notifications/message handler in McpClientOptions, cancellation through the
        CancellationToken given to AIAgent.RunAsync; the level, completion and listings
        through McpClient methods.
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
        """List-changed handlers in McpClientOptions.Handlers.NotificationHandlers, and
        McpClient.SubscribeToResourceAsync with an update handler. The SDK client has no
        subscriptions/listen method."""

        payload = self._run_worker({**self._config(server), "mode": "notify"})
        return payload.get("notify") or {}

    def _config(self, server: ServerHandle) -> dict:
        if server.transport in ("http", "sse"):
            config: dict = {"transport": server.transport, "url": server.url}
            if server.headers:
                config["headers"] = dict(server.headers)
            if self._auth:
                config["auth"] = dict(self._auth)
            return config
        return {"transport": "stdio", "command": server.command, "args": list(server.args)}

    def _run_worker(self, config: dict) -> dict:
        dotnet = _dotnet()
        if dotnet is None:
            raise RenderError(f"{ID} worker needs a .NET 10 runtime: set MCP_MIRROR_DOTNET_ROOT or put dotnet on PATH")
        with tempfile.TemporaryDirectory(prefix="mcp-mirror-maf-dotnet-") as scratch:
            config_path = Path(scratch) / "config.json"
            out_path = Path(scratch) / "out.json"
            config_path.write_text(json.dumps(config))
            try:
                # Run from the caller's cwd so a stdio server given by a relative command
                # resolves.
                proc = subprocess.run(
                    [dotnet, str(_dll()), str(config_path), str(out_path)],
                    capture_output=True,
                    text=True,
                    timeout=_TIMEOUT_SECONDS,
                    env=_environment(),
                )
            except subprocess.TimeoutExpired as exc:
                raise RenderError(f"{ID} worker timed out after {_TIMEOUT_SECONDS}s") from exc
            except Exception as exc:  # noqa: BLE001
                raise RenderError(f"{ID} worker failed to start: {exc}") from exc
            try:
                payload = json.loads(out_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
                raise RenderError(f"{ID} worker wrote no result ({exc}): {detail[-2000:]}") from exc
        if payload.get("fatal"):
            raise RenderError(f"{ID} worker error: {payload['fatal'][-2000:]}")
        if not isinstance(payload.get("tools"), list):
            raise RenderError(f"{ID} worker returned no tool list")
        protocol_version = payload.get("negotiatedMcpSpecVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise RenderError(f"{ID} worker returned no negotiated MCP protocol version")
        self._negotiated_mcp_spec_version = protocol_version
        return payload
