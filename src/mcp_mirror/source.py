"""MCP source loader (DESIGN.md section 7).

Connects to one MCP server (stdio or streamable-HTTP), completes the initialize
handshake, **captures the negotiated protocol version** (non-negotiable, section 12),
lists tools, and returns source ``ToolRep`` objects.

This is the only place in the engine that imports the ``mcp`` SDK. The parsed
``ServerHandle`` is reused by framework renderers so each framework opens its own
connection through its own real adapter (DESIGN.md decision D1).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

import anyio

from .models import ResultRep, ToolRep
from .normalize import normalize_source_result, normalize_source_tool


class SourceConnectionError(RuntimeError):
    """Raised when the server cannot be reached or the handshake fails (exit code 2)."""


@dataclass
class ServerHandle:
    """A parsed server specification, transport-agnostic for downstream renderers."""

    spec: str
    transport: str  # "stdio" | "http" | "sse" (the deprecated HTTP+SSE transport)
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)  # HTTP auth headers for a remote server


def parse_server(spec: str) -> ServerHandle:
    """Parse a CLI ``<server>`` argument into a ``ServerHandle``.

    ``<server>`` is either an HTTP(S) URL (streamable-HTTP transport), an HTTP(S) URL
    prefixed ``sse+`` (the deprecated HTTP+SSE transport, e.g. ``sse+http://host/sse``),
    or a command string launched as a subprocess (stdio transport).
    """

    spec = spec.strip()
    if spec.startswith("sse+http://") or spec.startswith("sse+https://"):
        return ServerHandle(spec=spec, transport="sse", url=spec.removeprefix("sse+"))
    if spec.startswith("http://") or spec.startswith("https://"):
        return ServerHandle(spec=spec, transport="http", url=spec)
    parts = shlex.split(spec)
    if not parts:
        raise SourceConnectionError("empty server specification")
    return ServerHandle(spec=spec, transport="stdio", command=parts[0], args=parts[1:])


def load_source(
    handle: ServerHandle,
    expected_spec_version: str | None = None,
) -> tuple[str, list[ToolRep]]:
    """Synchronous entry point: returns ``(spec_version, source_tool_reps)``."""

    try:
        return anyio.run(_load_async, handle, expected_spec_version)
    except SourceConnectionError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any transport/handshake failure uniformly
        raise SourceConnectionError(f"failed to load MCP server {handle.spec!r}: {exc}") from exc


async def _load_async(
    handle: ServerHandle,
    expected_spec_version: str | None,
) -> tuple[str, list[ToolRep]]:
    if expected_spec_version and expected_spec_version >= "2026-07-28":
        return await _load_modern(handle, expected_spec_version)
    return await _load_legacy(handle)


async def _load_modern(
    handle: ServerHandle,
    expected_spec_version: str,
) -> tuple[str, list[ToolRep]]:
    try:
        from mcp import Client
    except ImportError as exc:
        raise SourceConnectionError(
            f"MCP {expected_spec_version} requires the MCP Python SDK 2.x"
        ) from exc

    async def run(target) -> tuple[str, list[ToolRep]]:
        async with Client(target, mode="auto") as client:
            tools_result = await client.list_tools()
            reps = [
                normalize_source_tool(_tool_to_dict(tool))
                for tool in tools_result.tools
            ]
            return client.protocol_version, reps

    if handle.transport == "http":
        if not handle.headers:
            return await run(handle.url)

        import httpx
        from mcp.client.streamable_http import streamable_http_client

        timeout = httpx.Timeout(30, read=300)
        async with httpx.AsyncClient(
            headers=handle.headers,
            follow_redirects=True,
            timeout=timeout,
        ) as http_client:
            return await run(
                streamable_http_client(handle.url, http_client=http_client)
            )

    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=handle.command, args=handle.args)
    return await run(stdio_client(params))


async def _load_legacy(handle: ServerHandle) -> tuple[str, list[ToolRep]]:
    from mcp import ClientSession

    async def run(read, write) -> tuple[str, list[ToolRep]]:
        async with ClientSession(read, write) as session:
            init_result = await session.initialize()
            spec_version = str(getattr(init_result, "protocolVersion", "unknown"))
            tools_result = await session.list_tools()
            reps = [normalize_source_tool(_tool_to_dict(tool)) for tool in tools_result.tools]
            return spec_version, reps

    if handle.transport == "http":
        import httpx
        from mcp.client.streamable_http import streamable_http_client

        timeout = httpx.Timeout(30, read=300)
        async with httpx.AsyncClient(
            headers=handle.headers or None,
            follow_redirects=True,
            timeout=timeout,
        ) as http_client:
            async with streamable_http_client(
                handle.url,
                http_client=http_client,
            ) as streams:
                read, write = streams[0], streams[1]
                return await run(read, write)

    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=handle.command, args=handle.args)
    async with stdio_client(params) as (read, write):
        return await run(read, write)


async def call_source_tool_async(
    handle: ServerHandle, tool: str, arguments: dict
) -> ResultRep:
    """Invoke one tool directly on the server and capture the untouched result.

    This is the reference the framework renderings are diffed against, the same role
    ``load_source`` plays for definitions. It deliberately uses the raw SDK session so
    nothing between the server and the recorded result can reshape it.
    """

    from mcp import ClientSession

    async def run(read, write) -> ResultRep:
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)
            dump = getattr(result, "model_dump", None)
            payload = dump(by_alias=True, mode="json") if callable(dump) else result
            return normalize_source_result(payload, tool=tool)

    if handle.transport == "http":
        import httpx
        from mcp.client.streamable_http import streamable_http_client

        async with httpx.AsyncClient(
            headers=handle.headers or None,
            follow_redirects=True,
            timeout=httpx.Timeout(30, read=300),
        ) as http_client:
            async with streamable_http_client(handle.url, http_client=http_client) as streams:
                return await run(streams[0], streams[1])

    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=handle.command, args=handle.args)
    async with stdio_client(params) as (read, write):
        return await run(read, write)


def call_source_tool(handle: ServerHandle, tool: str, arguments: dict) -> ResultRep:
    """Synchronous wrapper around :func:`call_source_tool_async`."""

    return anyio.run(call_source_tool_async, handle, tool, arguments)


def _tool_to_dict(tool) -> dict:
    """Convert an MCP SDK ``Tool`` into the dict shape the normalizer expects."""

    model_dump = getattr(tool, "model_dump", None)
    if callable(model_dump):
        try:
            payload = model_dump(by_alias=True, mode="json", exclude_none=True)
        except TypeError:
            payload = model_dump(by_alias=True, exclude_none=True)
        if isinstance(payload, dict):
            return {
                "name": payload.get("name", ""),
                "description": payload.get("description"),
                "inputSchema": (
                    payload.get("inputSchema")
                    or payload.get("input_schema")
                    or {}
                ),
                "annotations": payload.get("annotations") or {},
                "icons": payload.get("icons") or [],
                "outputSchema": payload.get("outputSchema") or payload.get("output_schema"),
            }

    annotations = getattr(tool, "annotations", None)
    annotations_dict: dict = {}
    if annotations is not None:
        dump = getattr(annotations, "model_dump", None)
        if callable(dump):
            try:
                annotations_dict = dump(by_alias=True, exclude_none=True)
            except TypeError:
                annotations_dict = dump()
        elif isinstance(annotations, dict):
            annotations_dict = {
                key: value
                for key, value in annotations.items()
                if value is not None
            }
    return {
        "name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", None),
        "inputSchema": (
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None)
            or {}
        ),
        "annotations": annotations_dict,
        "icons": getattr(tool, "icons", None) or [],
        "outputSchema": getattr(tool, "outputSchema", None) or getattr(tool, "output_schema", None),
    }


# ---------------------------------------------------------------------------
# Resources and prompts, straight from the server, for the probe to compare against.
# ---------------------------------------------------------------------------


async def _probe_with(session, features: dict) -> dict:
    """Run the six probe questions over any client exposing the SDK method names."""
    from . import probe as P

    out: dict = {}

    async def _list_resources():
        result = await session.list_resources()
        return {"entries": [P.resource_entry(r) for r in result.resources]}

    out["resources.list"] = await _attempt_async(_list_resources)
    entries = (out["resources.list"].get("entries") or []) if out["resources.list"].get("ok") else []
    text_uri = next((e["uri"] for e in entries if (e.get("mimeType") or "").startswith("text/")), None)
    blob_uri = next((e["uri"] for e in entries if not (e.get("mimeType") or "").startswith("text/")), None)

    async def _read(uri):
        result = await session.read_resource(uri)
        contents = list(result.contents)
        first = contents[0] if contents else None
        mime = getattr(first, "mimeType", None) or getattr(first, "mime_type", None)
        if first is not None and getattr(first, "blob", None) is not None:
            return P.blob_contents(getattr(first, "uri", uri), mime, first.blob)
        return P.text_contents(getattr(first, "uri", uri), mime, getattr(first, "text", None))

    out["resources.read.text"] = (
        await _attempt_async(lambda: _read(text_uri)) if text_uri else {"ok": False, "error": "no text resource listed"}
    )
    out["resources.read.blob"] = (
        await _attempt_async(lambda: _read(blob_uri)) if blob_uri else {"ok": False, "error": "no binary resource listed"}
    )

    async def _templates():
        result = await session.list_resource_templates()
        templates = getattr(result, "resourceTemplates", None)
        if templates is None:
            templates = getattr(result, "resource_templates", None) or []
        return {"entries": [P.template_entry(t) for t in templates]}

    out["resources.templates"] = await _attempt_async(_templates)

    async def _prompts():
        result = await session.list_prompts()
        return {"entries": [P.prompt_entry(p) for p in result.prompts]}

    out["prompts.list"] = await _attempt_async(_prompts)
    prompts = (out["prompts.list"].get("entries") or []) if out["prompts.list"].get("ok") else []
    first_prompt = prompts[0] if prompts else None

    async def _get():
        arguments = {a["name"]: "x" for a in first_prompt["arguments"] if a.get("required")}
        result = await session.get_prompt(first_prompt["name"], arguments)
        return {
            "description": result.description,
            "messages": [P.message_entry(m.role, m.content) for m in result.messages],
        }

    out["prompts.get"] = (
        await _attempt_async(_get) if first_prompt else {"ok": False, "error": "no prompt listed"}
    )
    return out


async def _attempt_async(fn) -> dict:
    try:
        return {"ok": True, **(await fn())}
    except NotImplementedError as exc:
        return {"ok": False, "unsupported": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def probe_source_async(handle: ServerHandle) -> dict:
    """Ask the server itself, through whichever SDK line is installed."""
    try:
        from mcp import Client  # 2.x
    except ImportError:
        Client = None  # noqa: N806
    if Client is not None:
        target = handle.url if handle.transport == "http" else None
        if target is None:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            target = stdio_client(StdioServerParameters(command=handle.command, args=handle.args))
        async with Client(target, mode="auto") as client:
            return await _probe_with(client, {})
    from mcp import ClientSession

    if handle.transport == "http":
        from mcp.client.streamable_http import streamable_http_client

        async with streamable_http_client(handle.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await _probe_with(session, {})
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=handle.command, args=handle.args)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await _probe_with(session, {})


def probe_source(handle: ServerHandle) -> dict:
    import anyio

    return anyio.run(probe_source_async, handle)
