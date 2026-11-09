"""The cline_cli harness module: its NDJSON reducer, config file and approval switches,
checked without Cline installed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("measure_harness", ROOT / "scripts" / "measure_harness.py")
measure_harness = importlib.util.module_from_spec(spec)
sys.modules["measure_harness"] = measure_harness
spec.loader.exec_module(measure_harness)
cline = measure_harness.harnesses.load("cline_cli")


def test_reducer_keeps_tool_blocks_errors_and_the_result() -> None:
    raw = "\n".join(json.dumps(line) for line in [
        {"type": "hook_event", "hookEventName": "agent_start", "agentId": "agent_1"},
        {"type": "agent_event", "event": {"type": "content_start", "contentType": "tool",
                                          "toolName": "def__delete_account", "input": {"confirm": False}}},
        {"type": "agent_event", "event": {"type": "content_end", "contentType": "tool", "toolName": "def__delete_account",
                                          "output": {"content": [{"type": "text", "text": "no"}], "isError": True}}},
        {"type": "agent_event", "event": {"type": "content_end", "contentType": "tool", "toolName": "cf__ask_sampling",
                                          "output": {"error": "timed out"}, "error": "{\"error\":\"timed out\"}"}},
        {"type": "agent_event", "event": {"type": "content_end", "contentType": "text", "text": "done"}},
        {"type": "run_result", "finishReason": "completed", "iterations": 2, "text": "done",
         "usage": {"inputTokens": 2}, "model": {"id": "gpt-5"}},
    ])
    events = cline.reduce_output(raw)
    assert [e["event"] for e in events] == ["tool_use", "tool_result", "tool_result", "text", "result"]
    assert events[1]["is_error"] and events[2]["is_error"]
    assert events[4] == {"event": "result", "subtype": "completed", "is_error": False, "num_turns": 2, "text": "done"}
    assert "agent_1" not in json.dumps(events) and "usage" not in json.dumps(events)


def test_config_lands_in_the_isolated_data_dir(tmp_path: Path) -> None:
    env = {"CLINE_DATA_DIR": str(tmp_path / "data")}
    path = cline.write_config({"def": {"command": "sh", "args": ["-c", "x"]}, "util": {"url": "http://127.0.0.1:1/mcp"}},
                              tmp_path, env)
    assert path == tmp_path / "data" / "settings" / "cline_mcp_settings.json"
    assert json.loads(path.read_text())["mcpServers"]["util"] == {"type": "streamableHttp", "url": "http://127.0.0.1:1/mcp",
                                                                       "timeout": 60}
    assert json.loads((path.parent / "global-settings.json").read_text())["telemetryOptOut"] is True


def test_default_mode_is_the_shipped_default_and_tools_map_to_servers() -> None:
    assert cline.mode_args("default") == [] and cline.mode_args(None) == []
    assert cline.mode_args("unattended") == ["--auto-approve", "true"]
    assert cline.mcp_server_of("def__search_records") == "def"
    assert cline.mcp_server_of("team_spawn_teammate") is None


def test_every_fixture_gets_the_documented_timeout_except_one_started_late(tmp_path: Path) -> None:
    env = {"CLINE_DATA_DIR": str(tmp_path / "data")}
    servers = {"def": {"command": "sh", "args": ["-c", "exec python fixtures/definitions_server.py"]},
               "util": {"command": "sh", "args": ["-c", "exec record \"sh -c 'sleep 8; exec python u.py'\""]}}
    entries = json.loads(cline.write_config(servers, tmp_path, env).read_text())["mcpServers"]
    assert entries["def"]["timeout"] == 60
    assert "timeout" not in entries["util"]


def test_a_held_call_is_a_permission_denial() -> None:
    error = '{"error":"Tool \\"def__delete_account\\" requires approval in a TTY session -- NOT a tool or system failure."}'
    raw = json.dumps({"type": "agent_event", "event": {"type": "content_end", "contentType": "tool",
                                                       "toolName": "def__delete_account", "error": error}})
    events = cline.reduce_output(raw)
    assert [e["event"] for e in events] == ["permission_denied", "tool_result"]
