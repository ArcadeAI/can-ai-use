"""scripts/harnesses/goose.py: config, listing, argv, output reduction, and the committed run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from mcp_mirror import cells

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_harness_cells import ROOT, harnesses, measure_harness  # noqa: E402

RUN = ROOT / "data" / "harnesses" / "measurements" / "wire" / "goose" / "session.json"


def test_goose_writes_named_extensions_into_its_isolated_config(tmp_path: Path) -> None:
    module = harnesses.load("goose")
    env = {"GOOSE_PATH_ROOT": str(tmp_path / "root")}
    path = module.write_config({"def": {"command": "sh", "args": ["-c", "x y"]}, "res": {"url": "http://127.0.0.1:9/mcp"}},
                               tmp_path / "work", env)
    assert path == tmp_path / "root" / "config" / "config.yaml"
    extensions = yaml.safe_load(path.read_text())["extensions"]
    assert extensions["def"] == {"enabled": True, "type": "stdio", "name": "def", "cmd": "sh", "args": ["-c", "x y"]}
    assert extensions["res"] == {"enabled": True, "type": "streamable_http", "name": "res", "uri": "http://127.0.0.1:9/mcp"}


def test_goose_tool_names_map_to_their_extension() -> None:
    module = harnesses.load("goose")
    assert module.mcp_server_of("def__search_records") == "def"
    assert module.mcp_server_of("extensionmanager__read_resource") is None
    assert module.mcp_server_of("shell") is None
    assert cells.offered_name(["shell", "def__search_records"], "def", "search_records") == "def__search_records"


def test_goose_argv_and_modes() -> None:
    module = harnesses.load("goose")
    binary = str(harnesses.resolve_binary(module.BINARY) or "")
    if not binary:
        return  # not installed in this checkout; the committed run records the argv
    tail = [binary, "run", "-t", "hi", "--output-format", "stream-json", "--no-session", "--max-turns", "6"]
    assert module.argv("hi", None, Path("/c"), 6) == ["env", *tail]
    assert module.argv("hi", "default", Path("/c"), 6) == ["env", *tail]
    assert module.argv("hi", "unattended", Path("/c"), 6) == ["env", "GOOSE_MODE=auto", *tail]
    assert module.env("http://stub")["OPENAI_HOST"] == "http://stub"


def test_goose_prompt_invocation_and_resource_script() -> None:
    plan = measure_harness.session_plan()
    module = harnesses.load("goose")
    assert measure_harness.session_prompt(plan, module, "prompt") == (
        "/prompt summarize_records status=active", "prompt_invocation")
    spec = measure_harness.harness_plan(plan, "goose")
    assert measure_harness.session_script(plan, spec, "resource")[0]["builtin"] == "extensionmanager__read_resource"


def test_goose_stream_json_reducer_keeps_only_what_cells_read() -> None:
    raw = "\n".join([
        "    __( O)>  banner /Users/someone/x",
        json.dumps({"type": "message", "message": {"role": "assistant", "content": [
            {"type": "toolRequest", "toolCall": {"status": "success",
                                                 "value": {"name": "def__delete_account", "arguments": {"account_id": "a"}}}}]}}),
        json.dumps({"type": "message", "message": {"role": "user", "content": [
            {"type": "toolResponse", "toolResult": {"status": "success", "value": {
                "content": [{"type": "text", "text": "boom"}, {"type": "image", "data": "AAAA"}], "isError": True}}}]}}),
        json.dumps({"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}}),
        json.dumps({"type": "complete", "total_tokens": 4}),
    ])
    events = harnesses.load("goose").reduce_output(raw)
    assert [e["event"] for e in events] == ["tool_use", "tool_result", "text", "result"]
    assert events[1] == {"event": "tool_result", "is_error": True, "text": "boom"}
    assert events[-1]["num_turns"] == 2 and events[-1]["text"] == "done"
    assert "/Users/" not in json.dumps(events) and "AAAA" not in json.dumps(events)


def test_committed_goose_run_is_fresh_and_covers_both_levels() -> None:
    run = cells.HarnessRun(RUN)
    fresh, line = run.freshness()
    assert fresh, line
    assert run.revisions() == ["2025-11-25", "2026-07-28"]
    assert run.level_status("level_2")[0] == "ran"
    for session in run.sessions("level_1") + run.sessions("level_2"):
        record = json.loads((run.root / session["fresh"]).read_text())
        assert record["checks"]["servers_listed"]["listed"] == sorted(cells.SERVERS)
        # 1.53.0+ refreshes its model catalogue from models.dev on every start; the module
        # declares that host, the sink refuses it, and nothing else is ever tried.
        egress = record["checks"]["egress"]
        assert egress["passed"] and egress["not_allowed"] == []
        assert {a["host"] for a in egress["attempts"]} <= {"models.dev"}
        assert "/Users/" not in " ".join(session["argv"])
