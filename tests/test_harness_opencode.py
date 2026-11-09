"""The opencode harness module: its output reducer, argv and config, without running OpenCode."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


measure_harness = sys.modules.get("measure_harness") or _load("measure_harness", ROOT / "scripts" / "measure_harness.py")
opencode = measure_harness.harnesses.load("opencode")


def _events(*items: dict) -> str:
    return "\n".join(json.dumps(item) for item in items)


def test_reducer_keeps_calls_results_and_the_final_text() -> None:
    raw = _events(
        {"type": "step_start", "sessionID": "ses_x", "part": {"type": "step-start"}},
        {"type": "tool_use", "part": {"tool": "def_delete_account", "callID": "c1", "state": {
            "status": "error", "input": {"account_id": "a", "confirm": False},
            "error": "SENTINEL_TOOL_ERROR: confirm was not set", "time": {"start": 1}}}},
        {"type": "step_finish", "part": {"reason": "tool-calls", "tokens": {"total": 2}}},
        {"type": "text", "part": {"text": "done"}},
        {"type": "step_finish", "part": {"reason": "stop"}},
    )
    events = opencode.reduce_output(raw + "\nnot json")
    assert [e["event"] for e in events] == ["tool_use", "tool_result", "text", "result"]
    assert events[1] == {"event": "tool_result", "is_error": True, "text": "SENTINEL_TOOL_ERROR: confirm was not set"}
    assert events[3]["num_turns"] == 2 and events[3]["text"] == "done"
    assert "ses_x" not in json.dumps(events) and "tokens" not in json.dumps(events)


def test_an_auto_rejected_ask_is_a_permission_denied() -> None:
    raw = _events({"type": "tool_use", "part": {"tool": "def_delete_account", "state": {
        "status": "error", "input": {"account_id": "a"},
        "error": "The user rejected permission to use this specific tool call."}}})
    kinds = [e["event"] for e in opencode.reduce_output(raw)]
    assert kinds == ["tool_use", "permission_denied", "tool_result"]


def test_prompt_invocation_runs_as_a_command() -> None:
    start = opencode.prompt_invocation("def", "summarize_records", {"status": "active"})
    assert start == "/def:summarize_records active"
    argv = opencode.argv(start, "default", Path("opencode.json"), 6)
    assert argv[1:5] == ["run", "active", "--command", "def:summarize_records"]
    assert "--auto" in opencode.argv("x", "unattended", Path("opencode.json"), 6)
    assert "--auto" not in argv


def test_config_mounts_the_servers_and_points_the_model_at_the_stub(tmp_path: Path) -> None:
    servers = {"def": {"command": "sh", "args": ["-c", "x"]}, "res": {"url": "http://127.0.0.1:1/mcp"}}
    config = json.loads(opencode.write_config(servers, tmp_path, {}).read_text())
    assert config["mcp"]["def"] == {"type": "local", "command": ["sh", "-c", "x"]}
    assert config["mcp"]["res"]["type"] == "remote"
    assert config["model"] == config["small_model"] == "anthropic/stub"
    assert config["provider"]["anthropic"]["options"]["baseURL"] == "{env:ANTHROPIC_BASE_URL}/v1"
    assert opencode.mcp_server_of("def_search_records") == "def"
    assert opencode.mcp_server_of("read_mcp_resource") is None
