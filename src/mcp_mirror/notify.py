"""The shared shape of a change-notification probe against fixtures/notify_server.py.

The question for every adapter is the same: when the server changes its catalogue and
says so, does the change reach the agent? Each adapter drives the same sequence through
its own public API. It lists, calls a fixture tool that mutates the catalogue and fires
the notification, waits, and lists again. It records two things separately: whether a
hook the adapter exposes heard the notification, and whether the adapter's next listing
shows the change. Subscriptions go through whatever subscribe route the adapter offers.
The renderer says whether that route is its own API or the raw MCP session it hands out.

Keys of the returned mapping::

    tools.list_changed      {"ok", "hook", "notified", "before", "after", "sentinel", "server_said"}
    resources.list_changed  same shape, entries identified by resource URI
    prompts.list_changed    same shape, entries identified by prompt name
    resources.subscribe     {"ok", "route", "api", "subscribed", "subscribe_error", "notified",
                             "unsubscribed", "unsubscribe_error", "notified_after_unsubscribe",
                             "server_said"}
    subscriptions.listen    {"ok", "route", "api", "opened", "open_error", "honored", "events"}
                            (only when the adapter negotiated 2026-07-28)

plus ``"error": str`` or ``"unsupported": str`` instead of ``ok: True``. ``hook`` is None
when the adapter offers nothing that receives the notification, and ``route`` is
``"adapter"`` or ``"session"``. Nothing here decides a support code; src/mcp_mirror/cells.py
reads the capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import anyio

WATCHED_URI = "file:///fixtures/watched.txt"
MODERN = "2026-07-28"

# (probe key, mutating fixture tool, notification kind, sentinel prefix)
LIST_CHANGES = (
    ("tools.list_changed", "add_tool", "tools", "SENTINEL_ADDED_TOOL"),
    ("resources.list_changed", "add_resource", "resources", "fixtures/added-"),
    ("prompts.list_changed", "add_prompt", "prompts", "SENTINEL_ADDED_PROMPT"),
)
KEYS = tuple(key for key, *_ in LIST_CHANGES) + ("resources.subscribe", "subscriptions.listen")

# Wire method, or SDK 2.x event class name, -> notification kind.
KINDS = {
    "notifications/tools/list_changed": "tools",
    "notifications/resources/list_changed": "resources",
    "notifications/prompts/list_changed": "prompts",
    "notifications/resources/updated": "updated",
    "ToolsListChanged": "tools",
    "ResourcesListChanged": "resources",
    "PromptsListChanged": "prompts",
    "ResourceUpdated": "updated",
}
WAIT_SECONDS = 3.0
# After an unsubscribe, silence is the expected answer, so waiting longer only slows
# the run; this is long enough for a notification already in flight to land.
QUIET_SECONDS = 1.5


def unsupported(reason: str) -> dict[str, Any]:
    return {key: {"ok": False, "unsupported": reason} for key in KEYS}


def method_of(message: Any) -> str | None:
    """The notification method carried by whatever a message handler was handed."""
    for candidate in (message, getattr(message, "root", None)):
        method = getattr(candidate, "method", None)
        if isinstance(method, str):
            return method
    if isinstance(message, dict):
        return message.get("method")
    return None


@dataclass
class Recorder:
    """Counts notifications by kind as an adapter hook hands them over."""

    seen: dict[str, int] = field(default_factory=dict)

    def note(self, kind_or_method: str | None) -> None:
        kind = KINDS.get(kind_or_method or "", kind_or_method)
        if kind:
            self.seen[kind] = self.seen.get(kind, 0) + 1

    async def message_handler(self, message: Any) -> None:
        self.note(method_of(message))

    async def wait(self, kind: str, since: int, seconds: float = WAIT_SECONDS) -> bool:
        with anyio.move_on_after(seconds):
            while self.seen.get(kind, 0) <= since:
                await anyio.sleep(0.05)
        return self.seen.get(kind, 0) > since

    def count(self, kind: str) -> int:
        return self.seen.get(kind, 0)


def text_of(result: Any) -> str | None:
    """The first text block of a tool result, in whichever shape the adapter returned it."""
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        for item in result:
            found = text_of(item)
            if found:
                return found
        return None
    if isinstance(result, dict):
        if isinstance(result.get("text"), str):
            return result["text"]
        return text_of(result.get("content"))
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(result, "content", None)
    return text_of(content) if content is not None else None


Lister = Callable[[], Awaitable[list[str]]] | None


@dataclass
class Adapter:
    """What one adapter offers the probe. ``None`` means it has no such route."""

    call: Callable[[str], Awaitable[Any]]
    recorder: Recorder
    hook: str | None
    lists: dict[str, Lister]
    no_list_reason: str = "the adapter has no listing API for this"
    subscribe: Callable[[str], Awaitable[Any]] | None = None
    unsubscribe: Callable[[str], Awaitable[Any]] | None = None
    subscribe_route: str | None = None
    subscribe_api: str | None = None
    no_subscribe_reason: str = "the adapter has no subscribe route"
    listen: Callable[[], Any] | None = None
    listen_route: str | None = None
    listen_api: str | None = None
    negotiated: str | None = None


class SessionSubscriptions:
    """Subscribe through the raw MCP session an adapter hands out.

    Before 2026-07-28 that is ``resources/subscribe``, and the update arrives on the
    session, where the adapter's own message hook (if any) hears it. At 2026-07-28 the
    only route is a ``subscriptions/listen`` stream naming the URI. The SDK opens it on
    the session, and this pumps its events into the recorder until unsubscribe closes it.
    """

    def __init__(self, session: Any, recorder: Recorder, negotiated: str | None, group: Any) -> None:
        self.session = session
        self.recorder = recorder
        self.modern = bool(negotiated and negotiated >= MODERN)
        self.group = group
        self._stop: anyio.Event | None = None
        self._done: anyio.Event | None = None

    @property
    def api(self) -> str:
        if self.modern:
            return "mcp.client.subscriptions.listen(session, resource_subscriptions=[uri])"
        return "ClientSession.subscribe_resource / unsubscribe_resource"

    async def subscribe(self, uri: str) -> None:
        if not self.modern:
            await self.session.subscribe_resource(uri)
            return
        from mcp.client.subscriptions import listen

        acked, failed = anyio.Event(), []
        self._stop, self._done = anyio.Event(), anyio.Event()

        async def pump() -> None:
            try:
                async with listen(self.session, resource_subscriptions=[uri]) as subscription:
                    acked.set()

                    async def read() -> None:
                        async for event in subscription:
                            self.recorder.note(type(event).__name__)

                    async with anyio.create_task_group() as inner:
                        inner.start_soon(read)
                        await self._stop.wait()
                        inner.cancel_scope.cancel()
            except Exception as exc:  # noqa: BLE001
                failed.append(exc)
                acked.set()
            finally:
                self._done.set()

        self.group.start_soon(pump)
        await acked.wait()
        if failed:
            raise failed[0]

    async def unsubscribe(self, uri: str) -> None:
        if not self.modern:
            await self.session.unsubscribe_resource(uri)
            return
        if self._stop is not None and self._done is not None:
            self._stop.set()
            await self._done.wait()

    def listen_all(self) -> Any:
        from mcp.client.subscriptions import listen

        return listen(
            self.session,
            tools_list_changed=True,
            prompts_list_changed=True,
            resources_list_changed=True,
            resource_subscriptions=[WATCHED_URI],
        )


async def _attempt(fn: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
    try:
        return {"ok": True, **(await fn())}
    except NotImplementedError as exc:
        return {"ok": False, "unsupported": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the failure is the observation
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}


async def _list_change(adapter: Adapter, tool: str, kind: str, lister: Lister, sentinel: str) -> dict[str, Any]:
    if lister is None:
        return {"ok": False, "unsupported": adapter.no_list_reason}

    async def run() -> dict[str, Any]:
        before = sorted(await lister())
        since = adapter.recorder.count(kind)
        said = text_of(await adapter.call(tool))
        notified = await adapter.recorder.wait(kind, since)
        after = sorted(await lister())
        return {
            "hook": adapter.hook,
            "notified": notified,
            "before": before,
            "after": after,
            "sentinel": sentinel,
            "server_said": said,
        }

    return await _attempt(run)


async def _subscription(adapter: Adapter) -> dict[str, Any]:
    if adapter.subscribe is None:
        return {"ok": False, "unsupported": adapter.no_subscribe_reason}

    async def run() -> dict[str, Any]:
        out: dict[str, Any] = {
            "route": adapter.subscribe_route,
            "api": adapter.subscribe_api,
            "subscribed": False,
            "subscribe_error": None,
            "notified": False,
            "unsubscribed": False,
            "unsubscribe_error": None,
            "notified_after_unsubscribe": None,
            "server_said": [],
        }
        try:
            await adapter.subscribe(WATCHED_URI)
            out["subscribed"] = True
        except Exception as exc:  # noqa: BLE001 - an adapter refusing is the finding
            out["subscribe_error"] = f"{type(exc).__name__}: {exc}"[:500]
        since = adapter.recorder.count("updated")
        out["server_said"].append(text_of(await adapter.call("update_resource")))
        out["notified"] = await adapter.recorder.wait("updated", since)
        if not out["subscribed"] or adapter.unsubscribe is None:
            return out
        try:
            await adapter.unsubscribe(WATCHED_URI)
            out["unsubscribed"] = True
        except Exception as exc:  # noqa: BLE001
            out["unsubscribe_error"] = f"{type(exc).__name__}: {exc}"[:500]
            return out
        since = adapter.recorder.count("updated")
        out["server_said"].append(text_of(await adapter.call("update_resource")))
        out["notified_after_unsubscribe"] = await adapter.recorder.wait("updated", since, QUIET_SECONDS)
        return out

    return await _attempt(run)


async def _listen(adapter: Adapter) -> dict[str, Any]:
    """Open one stream for every event kind, fire each, and record what it delivers."""
    if adapter.listen is None:
        return {"ok": False, "unsupported": "the adapter has no route to subscriptions/listen"}

    async def run() -> dict[str, Any]:
        out: dict[str, Any] = {
            "route": adapter.listen_route,
            "api": adapter.listen_api,
            "opened": False,
            "open_error": None,
            "honored": None,
            "events": [],
        }
        try:
            stream = adapter.listen()
            subscription = await stream.__aenter__()
        except Exception as exc:  # noqa: BLE001
            out["open_error"] = f"{type(exc).__name__}: {exc}"[:500]
            return out
        out["opened"] = True
        try:
            honored = getattr(subscription, "honored", None)
            dump = getattr(honored, "model_dump", None)
            out["honored"] = dump(mode="json", by_alias=True, exclude_none=True) if callable(dump) else honored

            async def drain() -> None:
                async for event in subscription:
                    out["events"].append(type(event).__name__)

            async with anyio.create_task_group() as group:
                group.start_soon(drain)
                for tool in ("add_tool", "add_resource", "add_prompt", "update_resource"):
                    await adapter.call(tool)
                with anyio.move_on_after(WAIT_SECONDS):
                    while len(set(out["events"])) < 4:
                        await anyio.sleep(0.05)
                group.cancel_scope.cancel()
        finally:
            await stream.__aexit__(None, None, None)
        return out

    return await _attempt(run)


async def run(adapter: Adapter) -> dict[str, Any]:
    """Run the whole sequence in one session, in a fixed order."""
    out: dict[str, Any] = {"negotiated": adapter.negotiated}
    for key, tool, kind, sentinel in LIST_CHANGES:
        out[key] = await _list_change(adapter, tool, kind, adapter.lists.get(kind), sentinel)
    out["resources.subscribe"] = await _subscription(adapter)
    if adapter.negotiated and adapter.negotiated >= MODERN:
        out["subscriptions.listen"] = await _listen(adapter)
    return out


async def probe_source_async(handle: Any) -> dict[str, Any]:
    """The same sequence straight through the protocol-matched SDK: the control.

    It says what the fixture sent on each channel, so an adapter that heard nothing can
    be told apart from a fixture that said nothing.
    """
    recorder = Recorder()

    async def drive(session: Any, negotiated: str | None, lists: dict[str, Lister], call) -> dict[str, Any]:
        async with anyio.create_task_group() as group:
            subscriptions = SessionSubscriptions(session, recorder, negotiated, group)
            out = await run(
                Adapter(
                    call=call,
                    recorder=recorder,
                    hook="message_handler",
                    lists=lists,
                    subscribe=subscriptions.subscribe,
                    unsubscribe=subscriptions.unsubscribe,
                    subscribe_route="session",
                    subscribe_api=subscriptions.api,
                    listen=subscriptions.listen_all,
                    listen_route="session",
                    listen_api="mcp.client.subscriptions.listen",
                    negotiated=negotiated,
                )
            )
            group.cancel_scope.cancel()
        return out

    def lists_for(target: Any) -> dict[str, Lister]:
        async def tools():
            return [t.name for t in (await target.list_tools()).tools]

        async def resources():
            return [str(r.uri) for r in (await target.list_resources()).resources]

        async def prompts():
            return [p.name for p in (await target.list_prompts()).prompts]

        return {"tools": tools, "resources": resources, "prompts": prompts}

    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=handle.command, args=list(handle.args))
    try:
        from mcp import Client  # 2.x
    except ImportError:
        Client = None  # noqa: N806
    if Client is not None:
        # The response cache would answer a relist from memory; the control asks the server.
        async with Client(stdio_client(params), mode="auto", message_handler=recorder.message_handler, cache=None) as client:
            return await drive(
                client.session, client.protocol_version, lists_for(client), lambda name: client.call_tool(name, {})
            )
    from mcp import ClientSession

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, message_handler=recorder.message_handler) as session:
            init = await session.initialize()
            return await drive(
                session, str(init.protocolVersion), lists_for(session), lambda name: session.call_tool(name, {})
            )


def probe_source(handle: Any) -> dict[str, Any]:
    return anyio.run(probe_source_async, handle)
