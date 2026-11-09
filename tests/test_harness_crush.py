"""scripts/harnesses/crush.py, checked without Crush installed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_harness_cells import harnesses  # noqa: E402

crush = harnesses.load("crush")


def test_crush_writes_a_crushrc_with_the_stub_provider_and_every_server(tmp_path: Path) -> None:
    path = crush.write_config({"def": {"command": "/py", "args": ["-c", "x y"]}, "res": {"url": "http://127.0.0.1:9/mcp"}},
                              tmp_path, {})
    assert path == tmp_path / ".crushrc"
    lines = path.read_text().splitlines()
    assert lines[0].startswith("provider add stub --type anthropic --base-url \"$ANTHROPIC_BASE_URL\"")
    assert "model large stub/stub" in lines and "option default-providers false" in lines
    assert "mcp add def --command /py --args -c --args 'x y'" in lines
    assert "mcp add res --type http --url http://127.0.0.1:9/mcp" in lines


def test_crush_runs_both_modes_with_no_flag() -> None:
    assert crush.mode_args("default") == crush.mode_args("unattended") == []
    argv = crush.argv("hi", "unattended", Path(".crushrc"), 6)
    assert argv[1:] == ["run", "-q", "--model", "stub/stub", "hi"]


def test_crush_tool_names_and_text_output() -> None:
    assert crush.mcp_server_of("mcp_def_search_records") == "def"
    assert crush.mcp_server_of("read_mcp_resource") is None
    assert crush.mcp_server_of("bash") is None
    assert crush.reduce_output("  ready\n") == [
        {"event": "result", "subtype": None, "is_error": None, "num_turns": None, "text": "ready"}]
    assert crush.reduce_output("\n") == []
