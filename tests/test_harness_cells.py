"""Harness wire rules derive cells from recorder captures the way plan/harness_surface.yaml says.

The captures here are written by hand in the recorder's shape (see recorder.py), so each
rule is tested against the exact wire pattern it claims to read, without a harness
installed. The committed claude_code capture is checked separately: it must validate and
agree with the published build.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

from mcp_mirror import cells

ROOT = Path(__file__).resolve().parents[1]
SERVERS = cells.SERVERS
FRESH = {"passed": True, "checks": {name: {"passed": True} for name in cells.FRESH_CHECKS}}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


measure_harness = _load("measure_harness", ROOT / "scripts" / "measure_harness.py")
harnesses = measure_harness.harnesses
validate = _load("validate", ROOT / "data" / "validator" / "validate.py")


class Wire:
    """Builds one capture in the recorder's JSONL shape."""

    def __init__(self) -> None:
        self.records: list[dict] = [{"type": "capture", "seq": 0, "elapsed": 0.0}]
        self.clock = 0.0

    def _add(self, record: dict, after: float = 0.01) -> dict:
        self.clock += after
        record = {"seq": len(self.records), "elapsed": round(self.clock, 3), **record}
        self.records.append(record)
        return record

    def msg(self, direction: str, kind: str, method: str | None, id_=None, message: dict | None = None,
            origin: str | None = None, after: float = 0.01, headers: dict | None = None) -> dict:
        record = {
            "type": "message", "direction": direction, "kind": kind, "method": method, "id": id_,
            "origin": origin or ("client" if direction == "client_to_server" else "server"),
            "message": message or {},
        }
        if headers is not None:
            record["http"] = {"method": "POST", "headers": headers}
        return self._add(record, after)

    def request(self, method: str, id_, params: dict | None = None, **kw) -> dict:
        return self.msg("client_to_server", "request", method, id_, {"method": method, "params": params or {}}, **kw)

    def respond(self, method: str, id_, result: dict | None = None, error: bool = False, **kw) -> dict:
        kind = "error" if error else "response"
        return self.msg("server_to_client", kind, method, id_, {"result": result or {}}, **kw)

    def handshake(self, requested: str = "2025-11-25", capabilities: dict | None = None, headers=None) -> None:
        self.request("initialize", 0, {"protocolVersion": requested}, headers=headers)
        self.respond("initialize", 0, {"protocolVersion": requested})
        self._add({"type": "negotiation", "via": "initialize", "requested_revision": requested,
                   "negotiated_revision": requested, "client_capabilities": capabilities or {"roots": {}}})

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in self.records))


def _startup(wire: Wire, *, server: str, pages: int = 3, headers=None) -> None:
    wire.handshake(headers=headers)
    wire.msg("client_to_server", "notification", "notifications/initialized", headers=headers)
    ping = wire.msg("server_to_client", "request", "ping", "mcp-mirror-ping-1", origin="recorder")
    wire.msg("client_to_server", "response", "ping", ping["id"], headers=headers)
    total = pages if server == "util" else 1
    for page in range(total):
        cursor = str(page * 10) if page else None
        wire.request("tools/list", 10 + page, {"cursor": cursor} if cursor else {}, headers=headers)
        following = str((page + 1) * 10) if page + 1 < (3 if server == "util" else 1) else None
        wire.respond("tools/list", 10 + page, {"tools": [], **({"nextCursor": following} if following else {})})
    if server == "def":
        wire.request("prompts/list", 20, headers=headers)
        wire.request("resources/list", 21, headers=headers)


def _session(root: Path, session_id: str, level: int, *, run: int | None = None, build=None,
             output: list[dict] | None = None, transport: str = "stdio") -> dict:
    folder = f"level{level}-{session_id}" + (f"-{run}" if run else "")
    captures = {}
    for server in SERVERS:
        wire = Wire()
        (build or (lambda w, s: _startup(w, server=s)))(wire, server)
        wire.write(root / folder / f"{server}.jsonl")
        captures[server] = f"{folder}/{server}.jsonl"
    (root / folder / "harness.jsonl").write_text("".join(json.dumps(e) + "\n" for e in output or []))
    (root / folder / cells.HARNESS_FRESH).write_text(json.dumps(FRESH))
    session = {"id": session_id, "level": level, "transport": transport, "captures": captures,
               "output": f"{folder}/harness.jsonl", "fresh": f"{folder}/{cells.HARNESS_FRESH}"}
    if run:
        session["run"] = run
    return session


def _write_run(snapshot: Path, level_1: list[dict], level_2: list[dict] | None = None) -> cells.HarnessRun:
    root = snapshot / "measurements" / "wire" / "demo"
    manifest = {
        "harness": "demo", "version": "1.0.0", "captured_at": "2026-09-29T00:00:00+00:00",
        "command": "uv run python scripts/measure_harness.py demo",
        "level_1": {"status": "ran", "sessions": level_1},
        "level_2": (
            {"status": "ran", "sessions": level_2}
            if level_2 is not None
            else {"status": "skipped", "reason": cells.LEVEL_2_SKIPPED, "sessions": []}
        ),
    }
    (root / "session.json").write_text(json.dumps(manifest))
    return cells.HarnessRun(root / "session.json")


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "measurements" / "wire" / "demo"
    root.mkdir(parents=True)
    return root


def test_level_1_startup_cells(tmp_path: Path) -> None:
    root = _root(tmp_path)
    run = _write_run(tmp_path, [_session(root, "stdio", 1)])
    derived = {cell_id: rule(run) for cell_id, rule in cells.HARNESS_RULES.items()}
    assert derived["h-negotiated-revision"].value == "2025-11-25 (requested 2025-11-25)"
    assert derived["h-declared-capabilities"].value == "roots"
    assert derived["h-lists-tools-at-startup"].code == "y"
    assert derived["h-lists-prompts-at-startup"].code == "y"
    # resources/list but never resources/templates/list: one of the two.
    assert derived["h-lists-resources-at-startup"].code == "a"
    assert derived["h-follows-pagination"].code == "y"
    assert derived["h-tool-count-limit"].value == "3 pages, all of them"
    assert derived["h-answers-ping"].code == "y"
    assert derived["h-transport-stdio"].code == "y"
    # No HTTP or slow-start session ran, and no list_changed ever fired.
    assert derived["h-transport-streamable-http"].code == "u"
    assert derived["h-startup-timeout"].code == "u"
    assert derived["h-reacts-to-list-changed"].code == "u"
    # No model stub, so what the model was shown is not visible.
    assert derived["h-tools-offered-to-model"].code == "u"
    # Level 2 skipped: every level-2 cell is u and says why.
    for cell_id in ("h-calls-read-only-tool", "h-destructive-blocked-by-default", "h-expands-prompt"):
        assert derived[cell_id].code == "u"
        assert derived[cell_id].evidence[0] == cells.LEVEL_2_SKIPPED


def test_pagination_stopping_at_page_one_is_n(tmp_path: Path) -> None:
    root = _root(tmp_path)
    run = _write_run(tmp_path, [_session(root, "stdio", 1, build=lambda w, s: _startup(w, server=s, pages=1))])
    assert cells.HARNESS_RULES["h-follows-pagination"](run).code == "n"
    assert cells.HARNESS_RULES["h-tool-count-limit"](run).value == "1 page, stopped early"


def test_http_without_session_id_is_a(tmp_path: Path) -> None:
    root = _root(tmp_path)
    with_id = {"mcp-session-id": "abc"}
    good = _session(root, "http", 1, transport="http",
                    build=lambda w, s: _startup(w, server=s, headers=with_id))
    run = _write_run(tmp_path, [good])
    assert cells.HARNESS_RULES["h-transport-streamable-http"](run).code == "y"
    bare = _session(root, "http", 1, transport="http", build=lambda w, s: _startup(w, server=s, headers={}))
    run = _write_run(tmp_path, [bare])
    assert cells.HARNESS_RULES["h-transport-streamable-http"](run).code == "a"


def test_slow_start_listed_or_dropped(tmp_path: Path) -> None:
    root = _root(tmp_path)

    def dropped(wire: Wire, server: str) -> None:
        if server == "util":
            wire.request("initialize", 0)
            return
        _startup(wire, server=server)

    run = _write_run(tmp_path, [_session(root, "stdio", 1), _session(root, "slow-start", 1, build=dropped)])
    assert cells.HARNESS_RULES["h-startup-timeout"](run).code == "n"

    def slow(wire: Wire, server: str) -> None:
        wire.request("initialize", 0)
        wire.respond("initialize", 0, after=8.2)
        wire.request("tools/list", 1)

    run = _write_run(tmp_path, [_session(root, "stdio", 1), _session(root, "slow-start", 1, build=slow)])
    cell = cells.HARNESS_RULES["h-startup-timeout"](run)
    assert cell.code == "y" and "after 8.2s" in cell.evidence[0]


def test_destructive_held_reached_and_undecidable(tmp_path: Path) -> None:
    root = _root(tmp_path)

    def calls_delete(wire: Wire, server: str) -> None:
        _startup(wire, server=server)
        if server == "def":
            wire.request("tools/call", 30, {"name": "delete_account", "arguments": {"account_id": "a"}})

    held = [{"event": "permission_denied", "tool": "mcp__def__delete_account"}]
    level_1 = [_session(root, "stdio", 1)]
    runs = [_session(root, "destructive", 2, run=n, output=held) for n in (1, 2, 3)]
    assert cells.HARNESS_RULES["h-destructive-blocked-by-default"](_write_run(tmp_path, level_1, runs)).code == "y"

    runs[1] = _session(root, "destructive", 2, run=2, build=calls_delete)
    cell = cells.HARNESS_RULES["h-destructive-blocked-by-default"](_write_run(tmp_path, level_1, runs))
    assert cell.code == "a" and cell.evidence[0] == "2/3 runs (runs disagreed)"

    silent = [_session(root, "destructive", 2, run=n) for n in (1, 2, 3)]
    assert cells.HARNESS_RULES["h-destructive-blocked-by-default"](_write_run(tmp_path, level_1, silent)).code == "u"


def test_level_2_calls_reads_expands_answers(tmp_path: Path) -> None:
    root = _root(tmp_path)

    def session(wire: Wire, server: str) -> None:
        _startup(wire, server=server)
        if server == "def":
            wire.request("tools/call", 30, {"name": "search_records", "arguments": {"query": "x"}})
            wire.request("resources/read", 31, {"uri": "file:///fixtures/notes.txt"})
            wire.request("prompts/get", 32, {"name": "summarize_records", "arguments": {"status": "active"}})
        if server == "cf":
            ask = wire.msg("server_to_client", "request", "elicitation/create", 7)
            wire.msg("client_to_server", "response", "elicitation/create", ask["id"])
            wire.msg("server_to_client", "request", "sampling/createMessage", 8)
            wire.msg("client_to_server", "error", "sampling/createMessage", 8)
        if server == "ntf":
            wire.msg("server_to_client", "notification", "notifications/tools/list_changed")
            wire.request("tools/list", 40, after=0.5)

    level_1 = [_session(root, "stdio", 1)]
    ids = ("read_only", "read_only-unattended", "resource", "prompt", "elicit", "sampling", "relist")
    level_2 = [_session(root, i, 2, run=n, build=session) for i in ids for n in (1, 2, 3)]
    run = _write_run(tmp_path, level_1, level_2)
    codes = {cell_id: rule(run).code for cell_id, rule in cells.HARNESS_RULES.items()}
    assert codes["h-calls-read-only-tool"] == "y"
    assert codes["h-read-only-runs-without-prompt"] == "y"
    assert codes["h-reads-resource"] == "y"
    assert codes["h-expands-prompt"] == "y"
    assert codes["h-answers-elicitation"] == "y"
    assert codes["h-answers-sampling"] == "n"
    assert codes["h-relist-next-turn"] == "y"
    assert codes["h-reacts-to-list-changed"] == "y"
    # No slow or error session in this run: said so, not guessed.
    assert codes["h-tolerates-progress"] == "u"
    assert codes["h-error-surfaced"] == "u"


def test_claude_stream_json_reducer_keeps_only_what_cells_read() -> None:
    raw = "\n".join(json.dumps(line) for line in [
        {"type": "system", "subtype": "commands_changed", "commands": [{"name": "private-skill"}]},
        {"type": "system", "subtype": "init", "cwd": "/Users/someone/x", "claude_code_version": "2.1.0",
         "permissionMode": "default", "mcp_servers": [{"name": "def", "status": "connected"}],
         "tools": ["Bash", "mcp__def__search_records"]},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "mcp__def__delete_account",
                                                       "input": {"account_id": "a"}}]}},
        {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "result": "held",
         "permission_denials": [{"tool_name": "mcp__def__delete_account", "tool_input": {}}]},
    ])
    events = harnesses.load("claude_code").reduce_output(raw)
    assert [e["event"] for e in events] == ["init", "tool_use", "permission_denied", "result"]
    assert events[0]["mcp_tools"] == ["mcp__def__search_records"]
    assert "private-skill" not in json.dumps(events) and "/Users/" not in json.dumps(events)


def test_committed_surface_is_generated_from_the_plan() -> None:
    committed = json.loads((ROOT / "data" / "harnesses" / "surface.json").read_text())
    assert committed == measure_harness.surface_from_plan()
    assert {f["id"] for f in committed["features"]} == set(cells.HARNESS_RULES)


def test_committed_harness_snapshot_validates_and_matches_the_build() -> None:
    harnesses, rows, errors = validate.validate_harnesses(ROOT / "data" / "harnesses")
    assert errors == []
    assert "claude_code" in harnesses["agents"]
    built = json.loads((ROOT / "data" / "fulldata" / "harnesses.json").read_text())["data"]
    for row in rows:
        assert built[row["id"]]["stats"] == row["stats"]
        assert built[row["id"]]["values"] == row["values"]


# -- fixture revisions ---------------------------------------------------------


def test_level_1_cells_exist_per_fixture_revision(tmp_path: Path) -> None:
    root = _root(tmp_path)

    def modern(wire: Wire, server: str) -> None:
        wire.request("server/discover", 0)
        wire.respond("server/discover", 0)
        wire._add({"type": "negotiation", "via": "server/discover", "requested_revision": "2026-07-28",
                   "negotiated_revision": None, "client_capabilities": {"roots": {}}})
        wire._add({"type": "negotiation", "via": "request-meta", "negotiated_revision": "2026-07-28",
                   "client_capabilities": {"roots": {}}})
        wire.request("tools/list", 1)
        wire.respond("tools/list", 1, {"tools": []})

    old = {**_session(root, "stdio", 1), "revision": "2025-11-25"}
    new_root = root / "new"
    new = _session(new_root, "stdio", 1, build=modern)
    new = {**new, "revision": "2026-07-28",
           "captures": {k: f"new/{v}" for k, v in new["captures"].items()}, "output": f"new/{new['output']}"}
    run = _write_run(tmp_path, [old, new])
    manifest = json.loads(run.path.read_text())
    manifest["level_1"]["revisions"] = ["2025-11-25", "2026-07-28"]
    run.path.write_text(json.dumps(manifest))
    run = cells.HarnessRun(run.path)

    per = cells.harness_revision_cells(run, "h-negotiated-revision")
    assert per["2025-11-25"].value == "2025-11-25 (requested 2025-11-25)"
    assert per["2026-07-28"].value == "2026-07-28 (requested 2026-07-28)"
    cell = cells.HARNESS_RULES["h-negotiated-revision"](run)
    assert cell.value == ("2025-11-25 fixture: 2025-11-25 (requested 2025-11-25); "
                          "2026-07-28 fixture: 2026-07-28 (requested 2026-07-28)")
    # A stateless revision injects no ping; that is u with the reason, and u does not vote.
    ping = cells.harness_revision_cells(run, "h-answers-ping")
    assert ping["2026-07-28"].code == "u" and "2026-07-28" in ping["2026-07-28"].evidence[0]
    assert cells.HARNESS_RULES["h-answers-ping"](run).code == "y"
    # Level-2 cells have no per-revision form.
    assert cells.harness_revision_cells(run, "h-expands-prompt") == {}


# -- what the model was shown --------------------------------------------------

LISTED = [
    {"name": "search_records", "description": "Search. Read-only.", "inputSchema": {"type": "object"},
     "annotations": {"readOnlyHint": True}},
    {"name": "explain_topic", "description": "An ellipsis\u2026 here.", "inputSchema": {"type": "object"}},
]


def _offered_session(root: Path, offered: list[dict], hint_keys: list[str] | None = None) -> dict:
    def build(wire: Wire, server: str) -> None:
        wire.handshake()
        wire.request("tools/list", 1)
        wire.respond("tools/list", 1, {"tools": LISTED if server == "def" else []})

    session = _session(root, "stdio", 1, build=build)
    request = {"request": 1, "api": "anthropic-messages", "offered": ["Bash", *(t["name"] for t in offered)],
               "fixture_tools": offered, "hint_keys_in_request": hint_keys or []}
    (root / "level1-stdio" / "model.jsonl").write_text(json.dumps({"format": "x"}) + "\n" + json.dumps(request) + "\n")
    return {**session, "model": "level1-stdio/model.jsonl"}


def _as_offered(tool: dict, **change) -> dict:
    return {"name": f"mcp__def__{tool['name']}", "description": tool["description"],
            "input_schema": tool["inputSchema"], "server": "def", "fixture_tool": tool["name"], **change}


def test_tools_offered_to_model_diffs_against_tools_list(tmp_path: Path) -> None:
    root = _root(tmp_path)
    faithful = [_as_offered(t) for t in LISTED]
    cell = cells.HARNESS_RULES["h-tools-offered-to-model"](_write_run(tmp_path, [_offered_session(root, faithful)]))
    assert cell.code == "y"
    assert cell.value == ("2/2 tools offered; descriptions 2/2 unchanged; schemas 2/2 unchanged; "
                          "annotations not sent")

    rewritten = [_as_offered(LISTED[0]), _as_offered(LISTED[1], description="An ellipsis... here.")]
    cell = cells.HARNESS_RULES["h-tools-offered-to-model"](_write_run(tmp_path, [_offered_session(root, rewritten)]))
    assert cell.code == "a"
    assert any("def/explain_topic: description changed from character 11" in e for e in cell.evidence)

    hinted = [_as_offered(LISTED[0], extra={"annotations": {"readOnlyHint": True}})]
    cell = cells.HARNESS_RULES["h-tools-offered-to-model"](_write_run(tmp_path, [_offered_session(root, hinted)]))
    assert cell.code == "a"  # one of two tools missing
    assert "annotations sent as fields on 1 of 1 annotated tools" in cell.value
    assert "def/explain_topic: not offered" in cell.evidence

    as_text = _offered_session(root, faithful, hint_keys=["readOnlyHint"])
    cell = cells.HARNESS_RULES["h-tools-offered-to-model"](_write_run(tmp_path, [as_text]))
    assert cell.value.endswith("annotation hint names appear in the request text (readOnlyHint)")


def test_offered_name_matches_each_harness_prefix() -> None:
    names = ["Bash", "mcp__def__search_records", "mcp--cf--ask_sampling", "util_slow_with_progress", "add_tool"]
    assert cells.offered_name(names, "def", "search_records") == "mcp__def__search_records"
    assert cells.offered_name(names, "cf", "ask_sampling") == "mcp--cf--ask_sampling"
    assert cells.offered_name(names, "util", "slow_with_progress") == "util_slow_with_progress"
    assert cells.offered_name(names, "ntf", "add_tool") == "add_tool"
    assert cells.offered_name(names, "def", "delete_account") is None


# -- scripted stubs ------------------------------------------------------------


def _post(url: str, body: dict) -> tuple[str, str]:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.headers.get("content-type", ""), response.read().decode()


SCRIPT = [{"server": "def", "tool": "search_records", "arguments": {"query": "x"}}]


def test_anthropic_stub_scripts_a_tool_use_then_ends_the_turn() -> None:
    tools = [{"name": "Bash", "description": "run", "input_schema": {}},
             {"name": "mcp__def__search_records", "description": "Search.", "input_schema": {"type": "object"}}]
    with measure_harness.ModelStub("anthropic", SCRIPT, text="done") as stub:
        url = stub.url + "/v1/messages"
        _, body = _post(url, {"model": "m", "tools": tools, "messages": [{"role": "user", "content": "go"}]})
        first = json.loads(body)
        assert first["stop_reason"] == "tool_use"
        assert first["content"][0] == {"type": "tool_use", "id": "toolu_mcp_mirror_1",
                                       "name": "mcp__def__search_records", "input": {"query": "x"}}
        answered = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_mcp_mirror_1",
                                                  "content": "SENTINEL", "is_error": True}]}]
        kind, stream = _post(url, {"model": "m", "stream": True, "tools": tools, "messages": answered})
        assert kind == "text/event-stream"
        assert '"text": "done"' in stream and '"stop_reason": "end_turn"' in stream
    first_log, second_log = stub.brain.log
    assert [t["name"] for t in first_log["tools"]] == ["Bash", "mcp__def__search_records"]
    assert first_log["reply"]["tool"] == "mcp__def__search_records"
    assert second_log["last_tool_result"] == {"text": "SENTINEL", "is_error": True}


def test_stub_says_so_when_the_scripted_tool_is_not_offered() -> None:
    with measure_harness.ModelStub("anthropic", SCRIPT, text="done") as stub:
        _, body = _post(stub.url + "/v1/messages", {"model": "m", "messages": []})
    assert json.loads(body)["content"][0]["text"] == "def search_records is not among the tools offered"
    assert stub.brain.log[0]["reply"]["not_offered"] == "def search_records"


def test_openai_stub_chat_completions_and_responses() -> None:
    chat_tools = [{"type": "function", "function": {"name": "def_search_records", "description": "Search.",
                                                    "parameters": {"type": "object"}}}]
    response_tools = [{"type": "function", "name": "def_search_records", "description": "Search.",
                       "parameters": {"type": "object"}}]
    with measure_harness.ModelStub("openai", SCRIPT, text="done") as stub:
        base = harnesses.stub_env("openai", stub.url)["OPENAI_BASE_URL"]
        _, body = _post(f"{base}/chat/completions", {"model": "m", "tools": chat_tools, "messages": []})
        call = json.loads(body)["choices"][0]["message"]["tool_calls"][0]
        assert call["function"] == {"name": "def_search_records", "arguments": '{"query": "x"}'}
        _, stream = _post(f"{base}/chat/completions", {"model": "m", "stream": True, "tools": chat_tools,
                                                       "messages": [{"role": "tool", "content": "ok"}]})
        assert '"content": "done"' in stream and stream.rstrip().endswith("data: [DONE]")
        _, body = _post(f"{base}/responses", {"model": "m", "tools": response_tools, "input": []})
        item = json.loads(body)["output"][0]
        assert item["type"] == "function_call" and item["name"] == "def_search_records"
        _, stream = _post(f"{base}/responses", {"model": "m", "stream": True, "tools": response_tools,
                                                "input": [{"type": "function_call_output", "output": "ok"}]})
        assert "response.completed" in stream and '"delta": "done"' in stream
    assert [entry["api"] for entry in stub.brain.log] == ["openai-chat", "openai-chat", "openai-responses", "openai-responses"]
    assert stub.brain.log[0]["tools"][0]["input_schema"] == {"type": "object"}


def test_gemini_stub_function_call() -> None:
    tools = [{"functionDeclarations": [{"name": "def__search_records", "description": "Search.",
                                        "parametersJsonSchema": {"type": "object"}}]}]
    with measure_harness.ModelStub("gemini", SCRIPT, text="done") as stub:
        base = harnesses.stub_env("gemini", stub.url)["GOOGLE_GEMINI_BASE_URL"]
        _, body = _post(f"{base}/v1beta/models/stub:generateContent", {"tools": tools, "contents": []})
        part = json.loads(body)["candidates"][0]["content"]["parts"][0]
        assert part == {"functionCall": {"name": "def__search_records", "args": {"query": "x"}}}
        answered = [{"role": "user", "parts": [{"functionResponse": {"name": "x", "response": {"ok": 1}}}]}]
        _, stream = _post(f"{base}/v1beta/models/stub:streamGenerateContent?alt=sse", {"tools": tools, "contents": answered})
        assert '"text": "done"' in stream


def test_model_log_keeps_fixture_definitions_once() -> None:
    tool = {"name": "mcp__def__search_records", "description": "Search.", "input_schema": {}}
    log = [
        {"api": "a", "tools": [{"name": "Bash", "description": "long harness text", "input_schema": {}}, tool]},
        {"api": "a", "tools": [{"name": "Bash", "description": "long harness text", "input_schema": {}}, tool]},
    ]
    reduced = measure_harness.reduce_model_log(log, {"def": ["search_records"]})
    assert reduced[0]["offered"] == ["Bash", "mcp__def__search_records"]
    assert reduced[0]["fixture_tools"] == [{**tool, "server": "def", "fixture_tool": "search_records"}]
    assert reduced[1]["fixture_tools"] == "same as the previous request"
    assert "long harness text" not in json.dumps(reduced)


def test_claude_code_prompt_invocation_and_resource_script() -> None:
    plan = measure_harness.session_plan()
    spec = measure_harness.harness_plan(plan, "claude_code")
    module = harnesses.load("claude_code")
    assert measure_harness.session_prompt(plan, module, "prompt") == (
        "/mcp__def__summarize_records active", "prompt_invocation")
    assert measure_harness.session_prompt(plan, module, "read_only")[1] == "prompt"
    assert measure_harness.session_script(plan, spec, "resource")[0]["builtin"] == "ReadMcpResourceTool"
    assert measure_harness.session_script(plan, spec, "destructive") == [
        {"server": "def", "tool": "delete_account", "arguments": {"account_id": "a", "confirm": True}}]
    assert [line["revision"] for line in plan["fixture_lines"]] == ["2025-11-25", "2026-07-28"]


def test_committed_claude_code_run_covers_both_revisions_at_both_levels() -> None:
    run = cells.HarnessRun(ROOT / "data" / "harnesses" / "measurements" / "wire" / "claude_code" / "session.json")
    assert run.revisions() == ["2025-11-25", "2026-07-28"]
    for cell_id in cells.HARNESS_REVISION_RULES:
        assert set(cells.harness_revision_cells(run, cell_id)) == {"2025-11-25", "2026-07-28"}
    assert run.level_status("level_2")[0] == "ran"
    assert run.revisions("level_2") == ["2025-11-25", "2026-07-28"]
    for session_id in ("read_only", "read_only-unattended", "destructive", "destructive-unattended", "prompt"):
        assert len(run.sessions("level_2", session_id)) == 6
        for revision in run.revisions("level_2"):
            assert len(run.of_revision(run.sessions("level_2", session_id), revision)) == 3
    for session in run.sessions("level_1") + run.sessions("level_2"):
        assert (run.root / session["model"]).is_file()


# -- per-harness modules -------------------------------------------------------


def test_every_harness_module_and_the_template_meet_the_interface() -> None:
    assert "claude_code" in harnesses.available()
    assert "_template" not in harnesses.available()
    for harness_id in harnesses.available():
        module = harnesses.load(harness_id)
        assert harness_id in measure_harness.session_plan()["harnesses"], f"{harness_id} has no plan block"
        assert module.STUB_KIND in (*harnesses.STUB_KINDS, None)
    template = _load("harnesses._template", ROOT / "scripts" / "harnesses" / "_template.py")
    assert harnesses.interface_problems(template) == []


def test_measure_harness_has_no_per_harness_tables() -> None:
    for name in ("CONFIG_WRITERS", "OUTPUT_REDUCERS", "_claude_json", "_reduce_claude_stream_json"):
        assert not hasattr(measure_harness, name)
    assert set(measure_harness.STUBS) == set(harnesses.STUB_KINDS)
    source = (ROOT / "scripts" / "measure_harness.py").read_text()
    for harness_detail in ("--permission-mode", "mcp__", "stream-json", "ANTHROPIC_BASE_URL"):
        assert harness_detail not in source
    block = measure_harness.session_plan()["harnesses"]["claude_code"]
    assert set(block) == {"scripts"}


def test_claude_code_argv_is_the_one_the_plan_used_to_spell_out() -> None:
    module = harnesses.load("claude_code")
    # The project-local binary, absolute once installed (the harness runs in a temp cwd).
    assert module.BINARY == ".mcp-mirror/harnesses/claude_code/node_modules/.bin/claude"
    assert module.INSTALL[-1] == "@anthropic-ai/claude-code@2.1.296"
    binary = str(harnesses.resolve_binary(module.BINARY) or module.BINARY)
    base = [binary, "-p", "hi", "--output-format", "stream-json", "--verbose", "--bare", "--max-turns", "6",
            "--strict-mcp-config", "--mcp-config", "/c.json", "--setting-sources", "project",
            "--no-session-persistence"]
    assert module.argv("hi", None, Path("/c.json"), 6) == base
    assert module.argv("hi", "default", Path("/c.json"), 6) == [*base, "--permission-mode", "default"]
    assert module.argv("hi", "unattended", Path("/c.json"), 6) == [
        *base, "--allowedTools", "mcp__def__*", "mcp__res__*", "mcp__ntf__*", "mcp__cf__*", "mcp__util__*"]
    assert module.env("http://stub") == {"ANTHROPIC_BASE_URL": "http://stub", "ANTHROPIC_API_KEY": "mcp-mirror-stub",
                                         "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}


def test_harness_env_drops_real_keys(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "real")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "real")
    monkeypatch.setenv("DEMO_HARNESS_TOKEN", "real")
    module = type("M", (), {"SCRUBBED_ENV": ("DEMO_HARNESS_",)})
    env = measure_harness._scrubbed(module)
    assert not {"ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "DEMO_HARNESS_TOKEN"} & set(env)


# -- level 2 per revision, MRTR at 2026-07-28 ----------------------------------


def _mrtr(answer: bool):
    def build(wire: Wire, server: str) -> None:
        wire._add({"type": "negotiation", "via": "request-meta", "negotiated_revision": "2026-07-28",
                   "client_capabilities": {"elicitation": {}}})
        if server != "cf":
            return
        wire.request("tools/call", 1, {"name": "ask_elicitation_form", "arguments": {}})
        wire.respond("tools/call", 1, {"resultType": "input_required", "inputRequests": {
            "ask_elicitation_form": {"method": "elicitation/create", "params": {"mode": "form"}}}})
        if answer:
            wire.request("tools/call", 2, {"name": "ask_elicitation_form", "arguments": {},
                                           "inputResponses": {"ask_elicitation_form": {"action": "decline"}}})
            wire.respond("tools/call", 2, {"content": []})
    return build


def test_level_2_cells_exist_per_revision_and_read_mrtr_at_2026(tmp_path: Path) -> None:
    root = _root(tmp_path)

    def classic(wire: Wire, server: str) -> None:
        _startup(wire, server=server)
        if server == "cf":
            ask = wire.msg("server_to_client", "request", "elicitation/create", 7)
            wire.msg("client_to_server", "response", "elicitation/create", ask["id"])

    def at(revision: str, session: dict) -> dict:
        return {**session, "revision": revision}

    old = [at("2025-11-25", _session(root / "old", "elicit", 2, run=n, build=classic)) for n in (1, 2, 3)]
    new = [at("2026-07-28", _session(root / "new", "elicit", 2, run=n, build=_mrtr(True))) for n in (1, 2, 3)]
    for session in old + new:
        prefix = "old" if session["revision"] == "2025-11-25" else "new"
        session["captures"] = {k: f"{prefix}/{v}" for k, v in session["captures"].items()}
        session["output"] = f"{prefix}/{session['output']}"
        session["fresh"] = f"{prefix}/{session['fresh']}"
    run = _write_run(tmp_path, [_session(root, "stdio", 1)], old + new)
    manifest = json.loads(run.path.read_text())
    manifest["level_1"]["revisions"] = manifest["level_2"]["revisions"] = ["2025-11-25", "2026-07-28"]
    run.path.write_text(json.dumps(manifest))
    run = cells.HarnessRun(run.path)

    per = cells.harness_revision_cells(run, "h-answers-elicitation")
    assert per["2025-11-25"].code == "y"
    assert per["2026-07-28"].code == "y" and "MRTR" in per["2026-07-28"].evidence[1]
    assert cells.HARNESS_RULES["h-answers-elicitation"](run).code == "y"
    # No read_only-unattended session at either revision: u, and says which.
    assert "against the 2026-07-28 fixture" in cells.harness_revision_cells(run, "h-calls-read-only-tool")["2026-07-28"].evidence[0]

    unanswered = [at("2026-07-28", _session(root / "new", "elicit", 2, run=n, build=_mrtr(False))) for n in (1, 2, 3)]
    for session in unanswered:
        session["captures"] = {k: f"new/{v}" for k, v in session["captures"].items()}
        session["fresh"] = f"new/{session['fresh']}"
    manifest["level_2"]["sessions"] = old + unanswered
    run.path.write_text(json.dumps(manifest))
    run = cells.HarnessRun(run.path)
    cell = cells.harness_revision_cells(run, "h-answers-elicitation")["2026-07-28"]
    assert cell.code == "n" and "never answered" in cell.evidence[1]
    assert cells.HARNESS_RULES["h-answers-elicitation"](run).code == "a"


def test_measuring_keeps_visibility_and_defaults_a_new_row_to_preview(tmp_path: Path, monkeypatch) -> None:
    # "published" carries no key at all, which is how every published row is written.
    agents = {"shown": {"visibility": "published"}, "held": {"visibility": "preview"}, "published": {}}
    (tmp_path / "harnesses.json").write_text(json.dumps({"agents": agents}))
    monkeypatch.setattr(measure_harness, "SNAPSHOT", tmp_path)
    for harness_id in ("shown", "held", "published", "new"):
        measure_harness.update_harness_row(harness_id, {"name": harness_id}, "1.0.0", "2026-09-29")
    written = json.loads((tmp_path / "harnesses.json").read_text())["agents"]
    assert {k: v.get("visibility") for k, v in written.items()} == {
        "held": "preview", "new": "preview", "shown": "published", "published": None,
    }
    source = (ROOT / "scripts" / "measure_harness.py").read_text()
    assert '"visibility": "published"' not in source and "= \"published\"" not in source


# -- per transport, prompt arguments, context notes ----------------------------


def _stdio_and_http(tmp_path: Path, http_build) -> cells.HarnessRun:
    root = _root(tmp_path)
    http = _session(root, "http", 1, build=http_build, transport="http")
    return _write_run(tmp_path, [_session(root, "stdio", 1), http])


def test_transport_cells_name_both_transports_when_they_differ(tmp_path: Path) -> None:
    def http(wire: Wire, server: str) -> None:
        # Asks for a different revision and answers no ping, as Cline's stdio and HTTP
        # clients differ, but the other way round.
        wire.handshake("2026-07-28", {"roots": {}, "elicitation": {}})
        wire.msg("server_to_client", "request", "ping", "mcp-mirror-ping-1", origin="recorder")
        wire.request("tools/list", 10)

    run = _stdio_and_http(tmp_path, http)
    revision = cells.HARNESS_RULES["h-negotiated-revision"](run)
    assert revision.code == "y"
    assert revision.value == "stdio: 2025-11-25 (requested 2025-11-25) | HTTP: 2026-07-28 (requested 2026-07-28)"
    assert any(line.startswith("HTTP: ") for line in revision.evidence)
    assert cells.HARNESS_RULES["h-declared-capabilities"](run).value == "stdio: roots | HTTP: elicitation, roots"
    ping = cells.HARNESS_RULES["h-answers-ping"](run)
    assert ping.code == "a" and ping.value is None
    split = cells.harness_transport_cells(run, "h-answers-ping")
    assert {t: part["cell"].code for t, part in split.items()} == {"stdio": "y", "http": "n"}
    # The other level-1 cells keep reading one transport.
    assert cells.harness_transport_cells(run, "h-lists-tools-at-startup") == {}


def test_transport_cells_that_agree_keep_one_value(tmp_path: Path) -> None:
    run = _stdio_and_http(tmp_path, lambda wire, server: _startup(wire, server=server))
    assert cells.HARNESS_RULES["h-negotiated-revision"](run).value == "2025-11-25 (requested 2025-11-25)"
    assert cells.HARNESS_RULES["h-answers-ping"](run).code == "y"
    assert not cells.transports_differ({t: p["cell"] for t, p in
                                        cells.harness_transport_cells(run, "h-answers-ping").items()})


def test_prompts_get_with_the_wrong_arguments_is_partial(tmp_path: Path) -> None:
    root = _root(tmp_path)

    def placeholders(wire: Wire, server: str) -> None:
        _startup(wire, server=server)
        if server == "def":
            wire.request("prompts/get", 32, {"name": "summarize_records", "arguments": {"status": "$1", "tone": "$2"}})

    level_2 = [_session(root, "prompt", 2, run=n, build=placeholders) for n in (1, 2, 3)]
    cell = cells.HARNESS_RULES["h-expands-prompt"](_write_run(tmp_path, [_session(root, "stdio", 1)], level_2))
    assert cell.code == "a"
    assert '{"status": "$1", "tone": "$2"}' in cell.evidence[1]

    never = [_session(root, "prompt", 2, run=n) for n in (1, 2, 3)]
    assert cells.HARNESS_RULES["h-expands-prompt"](_write_run(tmp_path, [_session(root, "stdio", 1)], never)).code == "n"


def test_a_context_note_is_numbered_even_on_a_y_cell(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _write_run(tmp_path, [_session(root, "stdio", 1)])
    surface = measure_harness.surface_from_plan()
    agent = {"current_version": "1.0.0", "cell_notes": {"h-answers-ping": "measured headless; see #1"}}
    rows = {r["id"]: r for r in cells.publish_harness_rows(tmp_path, surface, {"agents": {"demo": agent}})}
    ping = rows["h-answers-ping"]
    assert ping["stats"]["demo"]["1.0.0"] == "y #1"
    assert ping["notes_by_num"]["1"].endswith(f"{cells.CONTEXT_NOTE}measured headless; see #1")
    assert rows["h-lists-tools-at-startup"]["stats"]["demo"]["1.0.0"] == "y"


def test_committed_cell_notes_name_real_cells() -> None:
    agents = json.loads((ROOT / "data" / "harnesses" / "harnesses.json").read_text())["agents"]
    for agent_id, agent in agents.items():
        for cell_id, note in (agent.get("cell_notes") or {}).items():
            assert cell_id in cells.HARNESS_RULES, f"{agent_id}: {cell_id}"
            assert "http" in note or ".md" in note or ".ts" in note or ".jsonl" in note, f"{agent_id} {cell_id} cites nothing"


def test_committed_transport_splits() -> None:
    snapshot = ROOT / "data" / "harnesses"
    runs = {run.harness: run for run in cells.load_harness_runs(snapshot)}
    claude = cells.harness_transport_cells(runs["claude_code"], "h-negotiated-revision")
    # Since 2.1.292 stdio asks for 2026-07-28 too (server/discover first), so the two
    # transports agree; on 2.1.285 stdio stayed on 2025-11-25.
    assert claude["http"]["by_revision"]["2026-07-28"].value.startswith("2026-07-28")
    assert claude["stdio"]["by_revision"]["2026-07-28"].value.startswith("2026-07-28")
    assert claude["stdio"]["by_revision"]["2025-11-25"].value.startswith("2025-11-25 (requested 2026-07-28, then 2025-11-25")
    cline = cells.harness_transport_cells(runs["cline_cli"], "h-negotiated-revision")
    assert cline["stdio"]["cell"].value.count("2024-11-05") == 4
    assert cline["http"]["by_revision"]["2025-11-25"].value.startswith("2025-11-25")
    ping = cells.harness_transport_cells(runs["cline_cli"], "h-answers-ping")
    assert (ping["stdio"]["cell"].code, ping["http"]["cell"].code) == ("n", "y")


def test_harness_rows_word_their_codes_by_what_the_row_asks() -> None:
    # On the destructive row an n means the call reached the server; "never called" or
    # "blocked by prompt" would say the opposite of what was measured.
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_data", ROOT / "scripts" / "build_data.py")
    build_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_data)
    published = json.loads((ROOT / "data" / "fulldata" / "harnesses.json").read_text())["data"]
    destructive = published["h-destructive-blocked-by-default"]
    assert destructive["labels"] == {"n": "ran without asking"}
    labels = {cell["label"] for cell in destructive["cells"].values() if cell["code"] == "n"}
    assert labels == {"ran without asking"}
    evidence = ["run 1: reached the server", "context: a permission event was auto-approved"]
    assert build_data._harness_label("h-destructive-blocked-by-default", "n", evidence, None,
                                     labels=destructive["labels"])[0] == "ran without asking"
    assert build_data._harness_label("h-reads-resource", "n", [], None)[0] == "never called"
    # Every n label a row names is defined in the page's legend.
    page = "".join(path.read_text() for path in (ROOT / "site" / "src").rglob("*.tsx"))
    for row in published.values():
        for word in (row.get("labels") or {}).values():
            assert f'"{word}"' in page, f"{row['id']}: {word!r} has no legend entry"



def test_no_harness_row_is_left_in_preview() -> None:
    # A re-measure rewrote two published rows as preview, hiding them from the site.
    # Every harness row is published; a new row may be preview only until it is reviewed,
    # and then this test should name it here deliberately.
    agents = json.loads((ROOT / "data" / "harnesses" / "harnesses.json").read_text())["agents"]
    assert [a for a, row in agents.items() if row.get("visibility") == "preview"] == []
