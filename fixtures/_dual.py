"""Build one low-level MCP server for both Python SDK lines.

The 1.x SDK speaks the handshake-era revisions and registers handlers through
decorators that read the request context from a context variable. The 2.x SDK also
speaks 2026-07-28 and hands every handler an explicit ``(ctx, params)`` pair. Each
fixture that needs more than tool definitions (results, notifications, client
features, utilities) writes its handlers once against :class:`Call` and lets
:func:`build` register them on whichever SDK is installed, the way
``definitions_server.py`` does by hand.

Handlers are ``async def handler(call: Call) -> <typed result>``. Keys of the
``handlers`` mapping name the MCP method they serve.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server

SDK2 = not hasattr(Server("probe"), "list_tools")
MODERN_REVISION = "2026-07-28"

Handler = Callable[["Call"], Awaitable[Any]]


@dataclass
class Call:
    """What a fixture handler needs from one request, on either SDK line."""

    session: Any
    params: Any
    meta: Any
    request_id: Any
    protocol_version: str | None

    @property
    def modern(self) -> bool:
        """2026-07-28 or later: server-to-client requests become input requests and
        change notifications ride ``subscriptions/listen`` streams."""
        return bool(self.protocol_version) and self.protocol_version >= MODERN_REVISION

    @property
    def arguments(self) -> dict:
        return dict(getattr(self.params, "arguments", None) or {})

    @property
    def cursor(self) -> str | None:
        return getattr(self.params, "cursor", None) if self.params is not None else None

    @property
    def progress_token(self) -> str | int | None:
        meta = self.meta
        if meta is None:
            return None
        if isinstance(meta, Mapping):
            return meta.get("progressToken", meta.get("progress_token"))
        return getattr(meta, "progressToken", None)

    @property
    def input_responses(self) -> dict:
        """Responses the client sent back for an earlier ``InputRequiredResult``."""
        responses = getattr(self.params, "input_responses", None) or {}
        return {
            key: value.model_dump(mode="json", by_alias=True, exclude_none=True) if hasattr(value, "model_dump") else value
            for key, value in responses.items()
        }


# The 1.x request type each method name dispatches on.
_V1_REQUESTS = {
    "tools/list": "ListToolsRequest",
    "tools/call": "CallToolRequest",
    "resources/list": "ListResourcesRequest",
    "resources/templates/list": "ListResourceTemplatesRequest",
    "resources/read": "ReadResourceRequest",
    "resources/subscribe": "SubscribeRequest",
    "resources/unsubscribe": "UnsubscribeRequest",
    "prompts/list": "ListPromptsRequest",
    "prompts/get": "GetPromptRequest",
    "completion/complete": "CompleteRequest",
    "logging/setLevel": "SetLevelRequest",
}

# The 2.x constructor keyword each method name is passed as.
_V2_KEYWORDS = {
    "tools/list": "on_list_tools",
    "tools/call": "on_call_tool",
    "resources/list": "on_list_resources",
    "resources/templates/list": "on_list_resource_templates",
    "resources/read": "on_read_resource",
    "resources/subscribe": "on_subscribe_resource",
    "resources/unsubscribe": "on_unsubscribe_resource",
    "prompts/list": "on_list_prompts",
    "prompts/get": "on_get_prompt",
    "completion/complete": "on_completion",
    "logging/setLevel": "on_set_logging_level",
    "subscriptions/listen": "on_subscriptions_listen",
}


def build(name: str, handlers: Mapping[str, Handler], *, listen: Any = None) -> Server:
    """Register ``handlers`` on a server for the installed SDK line.

    ``listen`` is a 2.x ``ListenHandler``; 1.x has no ``subscriptions/listen`` and
    ignores it.
    """

    if not SDK2:
        server: Server = Server(name)
        for method, handler in handlers.items():
            request_type = getattr(types, _V1_REQUESTS[method])
            server.request_handlers[request_type] = _v1_handler(server, handler)
        if "resources/subscribe" in handlers:
            # 1.x hard-codes subscribe=False whatever handlers are registered.
            original = server.get_capabilities

            def get_capabilities(*args, **kwargs):
                capabilities = original(*args, **kwargs)
                if capabilities.resources is not None:
                    capabilities.resources.subscribe = True
                return capabilities

            server.get_capabilities = get_capabilities  # type: ignore[method-assign]
        return server

    keywords: dict[str, Any] = {
        _V2_KEYWORDS[method]: _v2_handler(handler) for method, handler in handlers.items()
    }
    if listen is not None:
        keywords["on_subscriptions_listen"] = listen
    return Server(name, **keywords)


def _v1_handler(server: Server, handler: Handler):
    async def wrapped(request):
        context = server.request_context
        call = Call(
            session=context.session,
            params=getattr(request, "params", None),
            meta=context.meta,
            request_id=context.request_id,
            protocol_version=None,
        )
        result = await handler(call)
        return types.ServerResult(result if result is not None else types.EmptyResult())

    return wrapped


def _v2_handler(handler: Handler):
    async def wrapped(ctx, params):
        call = Call(
            session=ctx.session,
            params=params,
            meta=ctx.meta,
            request_id=ctx.request_id,
            protocol_version=ctx.protocol_version,
        )
        result = await handler(call)
        return result if result is not None else types.EmptyResult()

    return wrapped


def page(items: list, cursor: str | None, size: int) -> tuple[list, str | None]:
    """One page of ``items`` and the cursor for the next, or None on the last page."""

    start = int(cursor) if cursor else 0
    chunk = items[start : start + size]
    following = start + size
    return chunk, (str(following) if following < len(items) else None)


def text_result(text: str, *, is_error: bool = False, structured: dict | None = None) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structuredContent=structured,
        isError=is_error,
    )


def run_stdio(build_server: Callable[[], Server], notification_options: NotificationOptions | None = None) -> None:
    async def _main() -> None:
        from mcp.server.stdio import stdio_server

        server = build_server()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(notification_options or NotificationOptions()),
            )

    anyio.run(_main)
