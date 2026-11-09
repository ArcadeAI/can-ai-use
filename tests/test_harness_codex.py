"""scripts/harnesses/codex.py and the Responses `namespace` tools Codex sends the OpenAI
stub, checked without Codex installed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_harness_cells import SCRIPT, _post, harnesses, measure_harness  # noqa: E402

codex = harnesses.load("codex")


def test_openai_stub_flattens_namespace_tools_and_calls_into_the_namespace() -> None:
    tools = [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
             {"type": "namespace", "name": "mcp__def", "description": "def tools",
              "tools": [{"type": "function", "name": "search_records", "description": "Search.",
                         "parameters": {"type": "object"}}]}]
    script = [{"server": "def", "tool": "search_records", "arguments": {"query": "x"}}]
    assert script == SCRIPT
    with measure_harness.ModelStub("openai", script, text="done") as stub:
        base = harnesses.stub_env("openai", stub.url)["OPENAI_BASE_URL"]
        _, body = _post(f"{base}/responses", {"model": "m", "tools": tools, "input": []})
    item = json.loads(body)["output"][0]
    assert item["type"] == "function_call"
    assert (item["namespace"], item["name"]) == ("mcp__def", "search_records")
    offered = stub.brain.log[0]["tools"]
    assert [t["name"] for t in offered] == ["exec_command", "mcp__def__search_records"]
    assert offered[1]["description"] == "Search."


def test_codex_writes_mcp_servers_into_codex_home(tmp_path: Path) -> None:
    home = tmp_path / "codex"
    home.mkdir()
    path = codex.write_config({"def": {"command": "sh", "args": ["-c", "x y"]}, "res": {"url": "http://127.0.0.1:9/mcp"}},
                              tmp_path / "work", {"CODEX_HOME": str(home)})
    assert path == home / "config.toml"
    import tomllib

    config = tomllib.loads(path.read_text())
    assert config["mcp_servers"]["def"] == {"command": "sh", "args": ["-c", "x y"]}
    assert config["mcp_servers"]["res"] == {"url": "http://127.0.0.1:9/mcp"}
    assert config["features"]["plugins"] is False and config["analytics"]["enabled"] is False


def test_codex_modes() -> None:
    assert codex.mode_args(None) == []
    assert codex.mode_args("default") == ["-c", "mcp_optional_startup_grace_ms=30000"]
    unattended = codex.mode_args("unattended")
    assert 'approval_policy="never"' in unattended
    assert 'mcp_servers.def.default_tools_approval_mode="approve"' in unattended


def test_codex_reducer_marks_a_call_held_for_approval() -> None:
    raw = "\n".join(json.dumps(line) for line in [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "def", "tool": "delete_account",
                                            "arguments": {"confirm": True}, "status": "failed",
                                            "error": {"message": "MCP tool call requires approval, but approval policy is never"}}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "def", "tool": "search_records",
                                            "arguments": {"query": "x"}, "status": "completed",
                                            "result": {"content": [{"type": "text", "text": "ok"}]}}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1}},
    ])
    events = codex.reduce_output(raw)
    assert [e["event"] for e in events] == ["tool_use", "permission_denied", "tool_result", "tool_use", "tool_result",
                                           "text", "result"]
    assert events[0]["tool"] == "mcp__def__delete_account" and events[2]["is_error"] is True
    assert events[4]["is_error"] is False
    assert events[-1] == {"event": "result", "subtype": "turn.completed", "is_error": False, "num_turns": None, "text": ""}
