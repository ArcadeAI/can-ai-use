"""Notification probe captures derive cells the way cells.derive_notify_cells says."""

from __future__ import annotations

import json
from pathlib import Path

import anyio

from mcp_mirror import notify
from mcp_mirror.cells import derive_notify_cells

TOOL = "SENTINEL_ADDED_TOOL"


def _change(*, notified: bool, followed: bool, hook: str | None = "hook", sentinel: str = TOOL, **extra) -> dict:
    return {
        "ok": True,
        "hook": hook,
        "notified": notified,
        "before": ["add_tool"],
        "after": ["add_tool"] + ([f"src_{sentinel}"] if followed else []),
        "sentinel": sentinel,
        "server_said": "SENTINEL_MUTATED",
        **extra,
    }


def _subscribe(route: str = "adapter", **overrides) -> dict:
    return {
        "ok": True,
        "route": route,
        "api": "api",
        "subscribed": True,
        "subscribe_error": None,
        "notified": True,
        "unsubscribed": True,
        "unsubscribe_error": None,
        "notified_after_unsubscribe": False,
        **overrides,
    }


LEGACY_SOURCE = {
    "tools.list_changed": _change(notified=True, followed=True),
    "resources.list_changed": _change(notified=True, followed=True),
    "prompts.list_changed": _change(notified=True, followed=True),
    "resources.subscribe": _subscribe("session"),
}


def _write(snapshot: Path, source: dict, frameworks: dict) -> None:
    path = snapshot / "measurements" / "notify" / "all-frameworks.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "source": source,
        "frameworks": {name: {"ok": True, "notify": payload} for name, payload in frameworks.items()},
    }))


def test_hook_and_relist_decide_the_list_changed_rows(tmp_path: Path) -> None:
    _write(tmp_path, LEGACY_SOURCE, {
        "mastra": {
            "tools.list_changed": _change(notified=True, followed=True),
            "resources.list_changed": _change(notified=True, followed=True, hook_error="needs 2026-07-28"),
            "prompts.list_changed": _change(notified=True, followed=False),
            "resources.subscribe": _subscribe(subscribed=False, subscribe_error="needs 2026-07-28", notified=False),
        },
        "openai_agents": {
            "tools.list_changed": _change(notified=False, followed=True),
            "resources.list_changed": _change(notified=False, followed=False),
            "prompts.list_changed": {"ok": False, "unsupported": "no prompt listing API"},
            "resources.subscribe": _subscribe("session"),
        },
        "crewai": {
            "tools.list_changed": _change(notified=False, followed=False, hook=None),
            "resources.list_changed": {"ok": False, "error": "RuntimeError: boom"},
            "prompts.list_changed": {"ok": False, "unsupported": "no prompt API"},
            "resources.subscribe": {"ok": False, "unsupported": "no subscribe API"},
        },
    })

    cells = derive_notify_cells(tmp_path)

    assert cells["tools-list-changed"]["mastra"].code == "y"
    # The hook fired, but registering it raised: reachable, not on the spec's terms.
    assert cells["resources-list-changed"]["mastra"].code == "a"
    assert cells["prompts-list-changed"]["mastra"].code == "a"
    assert cells["tools-list-changed"]["openai-agents"].code == "a"
    assert cells["resources-list-changed"]["openai-agents"].code == "n"
    assert cells["prompts-list-changed"]["openai-agents"].code == "n"
    assert cells["tools-list-changed"]["crewai"].code == "n"
    assert "offers no hook" in cells["tools-list-changed"]["crewai"].evidence[0]
    assert cells["resources-list-changed"]["crewai"].code == "u"
    assert cells["resource-subscriptions"]["mastra"].code == "n"
    assert cells["resource-subscriptions"]["openai-agents"].code == "a"
    assert cells["resource-subscriptions"]["crewai"].code == "n"
    # No listen stream exists before 2026-07-28, so there is no row to derive.
    assert "subscriptions-listen" not in cells
    assert {cell.instrument for row in cells.values() for cell in row.values()} == {"notify"}


def test_a_silent_fixture_publishes_no_verdict(tmp_path: Path) -> None:
    silent = {**LEGACY_SOURCE, "tools.list_changed": _change(notified=False, followed=True)}
    _write(tmp_path, silent, {"pydantic_ai": {"tools.list_changed": _change(notified=True, followed=True)}})

    assert derive_notify_cells(tmp_path)["tools-list-changed"]["pydantic-ai"].code == "u"


def test_subscriptions_and_listen_at_2026(tmp_path: Path) -> None:
    every = list(notify.KINDS)[4:]
    source = {
        "tools.list_changed": _change(notified=False, followed=True),
        "resources.list_changed": _change(notified=False, followed=True),
        "prompts.list_changed": _change(notified=False, followed=True),
        "resources.subscribe": _subscribe("session"),
        "subscriptions.listen": {"ok": True, "route": "session", "opened": True, "events": every},
    }
    _write(tmp_path, source, {
        "mastra": {
            "tools.list_changed": _change(notified=True, followed=True),
            "resources.subscribe": _subscribe(),
            "subscriptions.listen": {"ok": True, "route": "adapter", "api": "managed", "opened": True, "events": every},
        },
        "pydantic_ai": {
            # At 2026-07-28 the change travels only on a listen stream, which the
            # adapter never opens, and its cached list is never invalidated.
            "tools.list_changed": _change(notified=False, followed=False),
            "resources.subscribe": _subscribe("session", notified_after_unsubscribe=True),
            "subscriptions.listen": {"ok": True, "route": "session", "api": "raw", "opened": True, "events": every},
        },
        "openai_agents": {
            "subscriptions.listen": {"ok": True, "route": "session", "api": "raw", "opened": False, "open_error": "x"},
        },
    })

    cells = derive_notify_cells(tmp_path)

    # The control client heard the tools change on its listen stream.
    assert cells["tools-list-changed"]["mastra"].code == "y"
    assert cells["tools-list-changed"]["pydantic-ai"].code == "n"
    assert cells["resource-subscriptions"]["mastra"].code == "y"
    assert cells["resource-subscriptions"]["pydantic-ai"].code == "a"
    assert "after unsubscribe" in cells["resource-subscriptions"]["pydantic-ai"].evidence[0]
    assert cells["subscriptions-listen"]["mastra"].code == "y"
    assert cells["subscriptions-listen"]["pydantic-ai"].code == "a"
    assert cells["subscriptions-listen"]["openai-agents"].code == "n"


def test_recorder_counts_methods_and_events() -> None:
    class Root:
        method = "notifications/tools/list_changed"

    class Message:
        root = Root()

    async def body() -> tuple[bool, bool]:
        recorder = notify.Recorder()
        await recorder.message_handler(Message())
        recorder.note("ResourceUpdated")
        heard = await recorder.wait("tools", 0, seconds=0.1)
        silent = await recorder.wait("prompts", 0, seconds=0.1)
        return heard, silent

    assert anyio.run(body) == (True, False)
    assert notify.text_of({"content": [{"type": "text", "text": "SENTINEL"}]}) == "SENTINEL"
