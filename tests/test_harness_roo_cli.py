"""The roo_cli harness module: its stream-json reducer, config file and approval switch,
checked without Roo installed."""

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
roo = measure_harness.harnesses.load("roo_cli")


def _use(server: str, tool: str, arguments: dict) -> dict:
    raw = json.dumps({"type": "use_mcp_tool", "serverName": server, "toolName": tool, "arguments": json.dumps(arguments)})
    return {"type": "tool_use", "id": 2, "subtype": "mcp", "tool_use": {"name": "mcp_server", "input": {"raw": raw}}, "done": True}


def _reduce(lines: list[dict]) -> list[dict]:
    return roo.reduce_output("\n".join(json.dumps(line) for line in lines))


def test_reducer_names_mcp_calls_as_offered_and_reads_is_error() -> None:
    events = _reduce([
        {"type": "system", "subtype": "init", "content": "Task started"},
        {"type": "user", "id": 1, "content": "prompt", "done": True},
        _use("def", "delete_account", {"account_id": "a", "confirm": False}),
        {"type": "tool_result", "subtype": "mcp", "tool_result": {"name": "mcp_server", "output": "Error:\nno"}},
        _use("def", "search_records", {"query": "x"}),
        {"type": "error", "id": 3, "content": "Error executing MCP tool: invalid_union"},
        {"type": "result", "id": 4, "content": "done", "done": True, "success": True, "cost": {"totalCost": 0}},
    ])
    assert events == [
        {"event": "tool_use", "tool": "mcp--def--delete_account", "input": {"account_id": "a", "confirm": False}},
        {"event": "tool_result", "is_error": True, "text": "Error:\nno"},
        {"event": "tool_use", "tool": "mcp--def--search_records", "input": {"query": "x"}},
        {"event": "tool_result", "is_error": True, "text": "Error executing MCP tool: invalid_union"},
        {"event": "result", "subtype": "success", "is_error": False, "num_turns": 0, "text": "done"},
    ]


def test_an_unanswered_mcp_ask_is_the_permission_event() -> None:
    events = _reduce([{"type": "user", "id": 1, "content": "p", "done": True}, _use("def", "search_records", {})])
    assert events[-1] == {"event": "permission_denied", "tool": "mcp--def--search_records", "input": {}}
    assert roo.mcp_server_of("mcp--def--search_records") == "def" and roo.mcp_server_of("read_file") is None


def test_a_partial_ask_before_the_complete_one_is_dropped() -> None:
    ask = {"type": "access_mcp_resource", "serverName": "def", "uri": "file:///fixtures/notes.txt"}
    events = _reduce([
        {"type": "tool_use", "id": 5, "subtype": "mcp",
         "content": json.dumps({**ask, "serverName": "", "uri": ""}), "tool_use": {"name": "mcp_server"}},
        {"type": "tool_use", "id": 5, "subtype": "mcp", "done": True,
         "tool_use": {"name": "mcp_server", "input": {"raw": json.dumps(ask)}}},
        {"type": "tool_result", "subtype": "mcp", "tool_result": {"name": "mcp_server", "output": "text"}},
    ])
    assert events == [
        {"event": "tool_use", "tool": "access_mcp_resource",
         "input": {"server_name": "def", "uri": "file:///fixtures/notes.txt"}},
        {"event": "tool_result", "is_error": False, "text": "text"},
    ]


def test_unattended_lists_every_fixture_tool_in_always_allow(tmp_path: Path) -> None:
    servers = {"def": {"command": "sh", "args": ["-c", "x"]}, "util": {"url": "http://127.0.0.1:1/mcp"}}
    tools = {"def": ["search_records", "delete_account"], "util": ["slow_with_progress"]}
    default = json.loads(roo.write_config(servers, tmp_path, {}, mode="default", tools=tools).read_text())["mcpServers"]
    assert default == {"def": {"command": "sh", "args": ["-c", "x"]},
                       "util": {"type": "streamable-http", "url": "http://127.0.0.1:1/mcp"}}
    unattended = json.loads(roo.write_config(servers, tmp_path, {}, mode="unattended", tools=tools).read_text())
    assert unattended["mcpServers"]["def"]["alwaysAllow"] == ["search_records", "delete_account"]
    assert unattended["mcpServers"]["util"]["alwaysAllow"] == ["slow_with_progress"]
    assert roo.mode_args("unattended") == roo.mode_args("default") == []
