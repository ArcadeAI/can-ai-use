"""The utilities instrument's rules, on hand-built captures.

Each rule weighs two halves: what the adapter handed its caller and what the recorder
saw on the wire. These cases pin the boundaries between codes so a change to a rule
shows up here before it shows up as drift in a published row.
"""

from __future__ import annotations

import json

from mcp_mirror import cells

SOURCE = {
    "tools": [f"t{i}" for i in range(60)],
    "page_size": 20,
    "progress_steps": 5,
    "completion": {"prompt": ["crimson", "cyan", "cobalt"], "resource": ["rec_1", "rec_10", "rec_11"]},
    "tasks_extension": "io.modelcontextprotocol/tasks",
}


class Wire:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.capture = 0

    def launch(self) -> "Wire":
        self.capture = len({r["capture"] for r in self.records})
        return self

    def add(self, direction: str, kind: str, method: str | None, message: dict, **extra) -> dict:
        record = {
            "capture": self.capture, "type": "message", "seq": len(self.records),
            "t": extra.pop("t", f"2026-09-30T00:00:{len(self.records):02d}.000000+00:00"),
            "direction": direction, "kind": kind, "method": method, "id": message.get("id"), "message": message, **extra,
        }
        self.records.append(record)
        return record

    def request(self, method: str, id_: int, params: dict | None = None, **extra) -> dict:
        return self.add("client_to_server", "request", method, {"id": id_, "method": method, "params": params or {}}, **extra)

    def reply(self, method: str, id_: int, result: dict) -> dict:
        return self.add("server_to_client", "response", method, {"id": id_, "result": result})

    def notify(self, direction: str, method: str, params: dict) -> dict:
        return self.add(direction, "notification", method, {"method": method, "params": params})


def _pages(wire: Wire, follow: bool) -> None:
    wire.request("tools/list", 1)
    wire.reply("tools/list", 1, {"tools": SOURCE["tools"][:20], "nextCursor": "20"})
    if follow:
        wire.request("tools/list", 2, {"cursor": "20"})
        wire.reply("tools/list", 2, {"tools": SOURCE["tools"][20:40], "nextCursor": "40"})
        wire.request("tools/list", 3, {"cursor": "40"})
        wire.reply("tools/list", 3, {"tools": SOURCE["tools"][40:]})


def test_pagination_needs_every_page():
    wire = Wire()
    _pages(wire, follow=True)
    full = cells._u_pagination(SOURCE, {"ok": True, "route": "adapter", "names": SOURCE["tools"]}, wire.records)
    assert full.code == "y" and "60 of 60" in full.evidence[0]
    first = Wire()
    _pages(first, follow=False)
    stopped = cells._u_pagination(SOURCE, {"ok": True, "route": "adapter", "names": SOURCE["tools"][:20]}, first.records)
    assert stopped.code == "n" and "1 tools/list pages" in stopped.evidence[1]


def test_progress_needs_a_token_and_a_callback():
    wire = Wire()
    wire.request("tools/call", 4, {"name": "slow_with_progress", "_meta": {"progressToken": 4}})
    for step in range(1, 6):
        wire.notify("server_to_client", "notifications/progress", {"progressToken": 4, "progress": step})
    events = [{"progress": s} for s in range(1, 6)]
    assert cells._u_progress(SOURCE, {"ok": True, "route": "adapter", "events": events}, wire.records).code == "y"
    assert cells._u_progress(SOURCE, {"ok": True, "route": "manual", "events": events}, wire.records).code == "a"
    silent = Wire()
    silent.request("tools/call", 4, {"name": "slow_with_progress"})
    assert cells._u_progress(SOURCE, {"ok": False, "unsupported": "none"}, silent.records).code == "n"


def test_logging_at_2026_needs_the_meta_level():
    wire = Wire()
    wire.request("tools/call", 5, {"name": "emit_logs"})
    cell = cells._u_logging(SOURCE, {"ok": True, "route": "adapter", "messages": []}, wire.records)
    assert cell.code == "n" and cells.LOG_LEVEL_META in cell.evidence[0]
    heard = Wire()
    heard.request("tools/call", 5, {"name": "emit_logs", "_meta": {cells.LOG_LEVEL_META: "debug"}})
    heard.notify("server_to_client", "notifications/message", {"level": "info", "data": {"message": "SENTINEL_LOG_INFO"}})
    answer = {"ok": True, "route": "adapter", "messages": [{"level": "info", "data": {"message": "SENTINEL_LOG_INFO"}}]}
    assert cells._u_logging(SOURCE, answer, heard.records).code == "y"


def test_set_level_and_completion_through_a_raw_session_are_partial():
    wire = Wire()
    wire.request("logging/setLevel", 6, {"level": "warning"})
    wire.reply("logging/setLevel", 6, {})
    answer = {"ok": True, "route": "session", "api": "x.session", "text": "SENTINEL_LOGS_SENT 2 at level>=warning"}
    assert cells._u_set_level(SOURCE, answer, wire.records).code == "a"
    assert cells._u_set_level(SOURCE, {**answer, "route": "adapter"}, wire.records).code == "y"
    assert cells._u_set_level(SOURCE, {"ok": False, "unsupported": "none"}, []).code == "n"
    completion = {"ok": True, "route": "session", "prompt": SOURCE["completion"]["prompt"], "resource": SOURCE["completion"]["resource"]}
    assert cells._u_completion(SOURCE, completion, []).code == "a"
    assert cells._u_completion(SOURCE, {**completion, "route": "adapter", "resource": []}, []).code == "a"
    assert cells._u_completion(SOURCE, {**completion, "route": "adapter"}, []).code == "y"


def test_cancellation_is_what_the_server_hears():
    wire = Wire()
    wire.request("tools/call", 7, {"name": "cancellable_wait"})
    answer = {"ok": True, "route": "adapter", "mechanism": "task.cancel()", "abandoned_after_s": 1.0, "outcome": "cancelled"}
    unheard = cells._u_cancellation(SOURCE, answer, wire.records)
    assert unheard.code == "n" and "no notifications/cancelled" in unheard.evidence[1]
    wire.notify("client_to_server", "notifications/cancelled", {"requestId": 7, "reason": "caller cancelled"})
    assert cells._u_cancellation(SOURCE, answer, wire.records).code == "y"
    assert cells._u_cancellation(SOURCE, {"ok": False, "unsupported": "no route"}, []).code == "n"


def test_tasks_probe_only_says_n_or_not_judged():
    wire = Wire()
    wire.records.append({"capture": 0, "type": "negotiation", "via": "initialize", "client_capabilities": {}})
    wire.request("tools/call", 8, {"name": "task_capable"})
    assert cells._u_tasks(SOURCE, {"ok": True}, wire.records, modern=False).code == "n"
    wire.records[0]["client_capabilities"] = {"tasks": {}}
    assert cells._u_tasks(SOURCE, {"ok": True}, wire.records, modern=False).code == "u"
    assert cells._u_tasks(SOURCE, {"ok": True}, wire.records, modern=True).code == "n"


def test_caching_times_refetches_against_the_ttl():
    def run(fetch_within: bool, fetch_after: bool) -> str:
        wire = Wire()
        stamps = {"first": "01", "within_ttl": "02", "after_ttl": "05"}
        marks = []
        for call, second in stamps.items():
            marks.append({"call": call, "started": f"2026-09-30T00:00:{second}.000000+00:00",
                          "finished": f"2026-09-30T00:00:{second}.500000+00:00", "count": 60})
            if call == "first" or (call == "within_ttl" and fetch_within) or (call == "after_ttl" and fetch_after):
                wire.request("tools/list", len(wire.records), t=f"2026-09-30T00:00:{second}.100000+00:00")
                wire.reply("tools/list", len(wire.records) - 1, {"tools": [], "ttlMs": 1500, "cacheScope": "private"})
        return cells._u_caching(SOURCE, {"ok": True, "route": "adapter", "marks": marks}, wire.records).code

    assert run(False, True) == "y"
    assert run(True, True) == "a"  # never uses the cache
    assert run(False, False) == "a"  # serves past the ttl
    assert run(True, False) == "n"


def test_cells_only_for_features_the_revision_defines(tmp_path):
    (tmp_path / "surface.json").write_text(json.dumps({"features": [{"id": "pagination-cursor"}]}))
    capture = tmp_path / "measurements" / "utilities" / "all.json"
    capture.parent.mkdir(parents=True)
    wire = Wire()
    _pages(wire, follow=True)
    capture.write_text(json.dumps({
        "source": SOURCE,
        "frameworks": {"openai_agents": {"ok": True, "utilities": {
            "tools.list": {"ok": True, "route": "adapter", "names": SOURCE["tools"]}}, "wire": {"tools.list": wire.records}}},
    }))
    derived = cells.derive_utility_cells(tmp_path)
    assert set(derived) == {"pagination-cursor"}
    assert derived["pagination-cursor"]["openai-agents"].code == "y"
