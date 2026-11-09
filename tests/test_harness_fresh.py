"""Every harness session runs fresh: an isolated home, the five fixtures only, no egress.

The harness here is a small Python script standing in for Codex: it reads its MCP servers
from the user config $CODEX_HOME/config.toml (default ~/.codex) and the project config
./.codex/config.toml, lists them, offers each server's tools to the stub model, and can be
told to reach the internet. The operator's real home is a
temp dir with a planted `gmail` server, so nothing here touches a real config.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from mcp_mirror import cells

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_harness_cells import ROOT, _write_run, _root, _session, measure_harness  # noqa: E402

harnesses = measure_harness.harnesses

FAKE_HARNESS = r'''
import json, os, sys, tomllib, urllib.request
from pathlib import Path

servers = {}
for home in (Path(os.environ.get("CODEX_HOME") or Path(os.environ["HOME"]) / ".codex"), Path(".codex")):
    if (home / "config.toml").exists():
        servers |= tomllib.loads((home / "config.toml").read_text()).get("mcp_servers", {})
if sys.argv[1] == "list":
    for name in servers:
        print(f"{name}: {servers[name].get('command', 'http')} - configured")
    raise SystemExit(0)
if os.environ.get("FAKE_EGRESS"):
    try:
        urllib.request.urlopen(os.environ["FAKE_EGRESS"], timeout=5)
    except Exception as exc:
        print(json.dumps({"egress": str(exc)}), file=sys.stderr)
tools = [{"name": f"mcp__{name}__probe", "input_schema": {"type": "object"}} for name in servers]
tools.append({"name": "shell", "input_schema": {"type": "object"}})
body = json.dumps({"model": "stub", "messages": [{"role": "user", "content": "hi"}], "tools": tools}).encode()
request = urllib.request.Request(os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages", body, {"content-type": "application/json"})
urllib.request.urlopen(request, timeout=10).read()
print(json.dumps({"type": "init", "servers": sorted(servers)}))
'''


def _fake_module(tmp_path: Path, checked: bool = True, **extra) -> types.ModuleType:
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE_HARNESS)
    module = types.ModuleType("fake_codex")
    module.BINARY = sys.executable
    module.STUB_KIND = "anthropic"
    module.install_check = lambda: "0.0.1"

    def write_config(servers, workdir, env):
        path = workdir / ".codex" / "config.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for name, server in servers.items():
            lines += [f"[mcp_servers.{name}]", f"command = {json.dumps(server.get('command', 'http'))}"]
        path.write_text("\n".join(lines) + "\n")
        return path

    module.write_config = write_config
    module.list_servers = lambda env, workdir: harnesses.names_before_colon(
        harnesses.list_command([sys.executable, str(script), "list"], env, workdir))
    module.mode_args = lambda mode: []
    module.argv = lambda prompt, mode, config, max_turns: [sys.executable, str(script), "run"]
    module.env = lambda url: harnesses.stub_env("anthropic", url)
    module.reduce_output = lambda text: []
    module.prompt_invocation = lambda *_: None
    for name, value in extra.items():
        setattr(module, name, value)
    assert not checked or harnesses.interface_problems(module) == []
    return module


@pytest.fixture
def real_home(tmp_path: Path, monkeypatch) -> Path:
    """A stand-in for the operator's home, with a gmail server planted in its Codex config."""
    home = tmp_path / "real-home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "config.toml").write_text('[mcp_servers.gmail]\ncommand = "gmail-mcp"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    return home


def _run(module, tmp_path: Path, name: str, *, isolate: bool = True, env_extra: dict | None = None) -> dict:
    out_root = tmp_path / "out"
    fixtures = {server: "fixtures/definitions_server.py" for server in cells.SERVERS}
    with measure_harness.ModelStub("anthropic", text="done") as stub:
        record = measure_harness.run_session(
            harness=module, fixtures=fixtures, session_dir=out_root / name, out_root=out_root, prompt="hi",
            max_turns=2, timeout=60, env_extra=module.env(stub.url) | (env_extra or {}), secrets={}, mode=None,
            transport="stdio", delays={}, stub=stub, isolate=isolate,
        )
    return {**record, "fresh_record": json.loads((out_root / record["fresh"]).read_text())}


def test_planted_server_never_reaches_an_isolated_run(real_home: Path, tmp_path: Path) -> None:
    module = _fake_module(tmp_path)
    record = _run(module, tmp_path, "isolated")
    fresh = record["fresh_record"]
    assert fresh["passed"], fresh
    assert fresh["checks"]["servers_listed"]["listed"] == sorted(cells.SERVERS)
    model = (tmp_path / "out" / record["model"]).read_text()
    assert "gmail" not in model
    assert sorted(fresh["checks"]["tools_offered"]["servers"]) == sorted(cells.SERVERS)
    assert set(fresh["checks"]["home"]["isolated"]) >= {"HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "GEMINI_CLI_HOME",
                                                        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
                                                        "XDG_STATE_HOME"}
    assert (real_home / ".codex" / "config.toml").read_text() == '[mcp_servers.gmail]\ncommand = "gmail-mcp"\n'


def test_pre_run_list_catches_the_planted_server_without_isolation(real_home: Path, tmp_path: Path) -> None:
    # Isolation off, as the harness would run from the operator's shell: its own list
    # command reads the real home's gmail, check 2 fails, and the harness is never started.
    record = _run(_fake_module(tmp_path), tmp_path, "leaky", isolate=False)
    fresh = record["fresh_record"]
    listed = fresh["checks"]["servers_listed"]
    assert not listed["passed"] and listed["unexpected"] == ["gmail"] and listed["missing"] == []
    assert not fresh["checks"]["home"]["passed"] and not fresh["passed"]
    assert record["exit_code"] is None and "not started" in record["stderr_tail"][0]


def test_post_run_check_fails_a_foreign_mcp_tool(tmp_path: Path) -> None:
    model = tmp_path / "model.jsonl"
    output = tmp_path / "harness.jsonl"
    offered = ["Bash", "mcp__def__search_records", "mcp__gmail__send_email"]
    model.write_text(json.dumps({"format": "x"}) + "\n" + json.dumps({"offered": offered}) + "\n")
    output.write_text(json.dumps({"format": "x"}) + "\n")
    check = measure_harness.check_tools_offered(harnesses, model, output)
    assert not check["passed"] and check["foreign_tools"] == ["mcp__gmail__send_email"]
    output.write_text(json.dumps({"format": "x"}) + "\n" + json.dumps(
        {"event": "init", "mcp_servers": [{"name": "def"}, {"name": "slack"}], "mcp_tools": []}) + "\n")
    model.write_text(json.dumps({"format": "x"}) + "\n" + json.dumps({"offered": offered[:2]}) + "\n")
    check = measure_harness.check_tools_offered(harnesses, model, output)
    assert not check["passed"] and check["foreign_servers"] == ["slack"]


def test_egress_sink_catches_a_planted_request(real_home: Path, tmp_path: Path) -> None:
    module = _fake_module(tmp_path)
    record = _run(module, tmp_path, "egress", env_extra={"FAKE_EGRESS": "http://updates.example.com/v"})
    egress = record["fresh_record"]["checks"]["egress"]
    assert not egress["passed"] and egress["not_allowed"] == ["updates.example.com"]
    assert egress["attempts"][0] | {"count": 1} == {"host": "updates.example.com", "port": 80, "method": "GET",
                                                    "allowed": None, "count": 1}
    # Allowlisted with a reason: still refused and recorded, no longer a failure.
    allowed = _fake_module(tmp_path, EGRESS_ALLOWED={"updates.example.com": "version check"})
    record = _run(allowed, tmp_path, "egress-allowed", env_extra={"FAKE_EGRESS": "https://updates.example.com/v"})
    egress = record["fresh_record"]["checks"]["egress"]
    assert egress["passed"] and egress["attempts"][0]["allowed"] == "version check"
    assert egress["attempts"][0]["method"] == "CONNECT" and egress["attempts"][0]["port"] == 443


@pytest.mark.skipif(not measure_harness.shutil.which("sandbox-exec"), reason="needs macOS sandbox-exec")
def test_withheld_proxy_variables_hold_egress_in_the_sandbox(real_home: Path, tmp_path: Path) -> None:
    # TEST-NET-1 is never routed, so even a sandbox that failed would send nothing.
    module = _fake_module(tmp_path, WITHHOLD_PROXY_ENV={"stdio": "breaks with NO_PROXY"})
    record = _run(module, tmp_path, "withheld", env_extra={"FAKE_EGRESS": "http://192.0.2.1/v"})
    egress = record["fresh_record"]["checks"]["egress"]
    assert egress["passed"] and egress["route"] == "sandbox" and egress["attempts"] == []
    assert egress["proxy_env"] == "withheld on this transport: breaks with NO_PROXY"
    assert "EPERM" in egress["sandbox"]["probe"] and egress["tradeoff"] == measure_harness.SANDBOX_TRADEOFF
    # The harness still reached the stub on loopback, and its own egress was refused by the sandbox.
    assert record["fresh_record"]["passed"]
    assert any("Operation not permitted" in line for line in record["stderr_tail"])
    # Another transport keeps the sink.
    assert measure_harness.withheld_proxy(module, "http") is None


def test_a_home_file_fails_freshness_unless_allowlisted(real_home: Path, tmp_path: Path) -> None:
    (real_home / ".fake-settings").write_text("{}")
    record = _run(_fake_module(tmp_path, HOME_FILES=(".fake-settings",)), tmp_path, "declared")
    home = record["fresh_record"]["checks"]["home"]
    assert not home["passed"] and home["refused"] == [".fake-settings"]
    record = _run(_fake_module(tmp_path, HOME_FILES=(".fake-settings",),
                               HOME_FILES_ALLOWED={".fake-settings": "test only"}), tmp_path, "allowed")
    assert record["fresh_record"]["passed"]
    assert harnesses.interface_problems(_fake_module(tmp_path, False, EGRESS_ALLOWED={"x": ""})) == [
        "EGRESS_ALLOWED must map each entry to a reason"]


def test_fresh_json_gates_publication(tmp_path: Path) -> None:
    root = _root(tmp_path)
    session = _session(root, "stdio", 1)
    run = _write_run(tmp_path, [session])
    cell = cells.HARNESS_RULES["h-lists-tools-at-startup"](run)
    assert cell.code == "y" and cell.evidence[-1].startswith("fresh.json: all four freshness checks")

    failed = {"passed": False, "checks": {**{n: {"passed": True} for n in cells.FRESH_CHECKS}, "egress": {"passed": False}}}
    (root / session["fresh"]).write_text(json.dumps(failed))
    run = cells.HarnessRun(run.path)
    for rule in cells.HARNESS_RULES.values():
        cell = rule(run)
        assert cell.code == "u" and "failed egress" in cell.evidence[0] and session["fresh"] in cell.evidence[0]
    assert cells.harness_revision_cells(run, "h-lists-tools-at-startup") == {}

    # A check the runner added beyond the four (a live session's sandbox, the process
    # group) gates publication the same way.
    extra = {"passed": False, "checks": {**{n: {"passed": True} for n in cells.FRESH_CHECKS}, "sandbox": {"passed": False}}}
    (root / session["fresh"]).write_text(json.dumps(extra))
    run = cells.HarnessRun(run.path)
    cell = cells.HARNESS_RULES["h-lists-tools-at-startup"](run)
    assert cell.code == "u" and "failed sandbox" in cell.evidence[0]

    (root / session["fresh"]).unlink()
    run = cells.HarnessRun(run.path)
    assert "missing" in cells.HARNESS_RULES["h-lists-tools-at-startup"](run).evidence[0]


def test_committed_claude_code_run_is_fresh() -> None:
    run = cells.HarnessRun(ROOT / "data" / "harnesses" / "measurements" / "wire" / "claude_code" / "session.json")
    fresh, line = run.freshness()
    assert fresh, line
    assert run.manifest["fresh"] == {"passed": True, "sessions": len(run.sessions("level_1") + run.sessions("level_2")),
                                     "failed": []}
    for session in run.sessions("level_1") + run.sessions("level_2"):
        record = json.loads((run.root / session["fresh"]).read_text())
        assert record["checks"]["servers_listed"]["listed"] == sorted(cells.SERVERS)
        assert record["checks"]["egress"]["attempts"] == []
        assert record["checks"]["tools_offered"]["foreign_tools"] == []


def test_claude_code_lists_servers_from_its_project_config(tmp_path: Path) -> None:
    module = harnesses.load("claude_code")
    path = module.write_config({"def": {"command": "false", "args": []}, "res": {"url": "http://127.0.0.1:9/"}}, tmp_path, {})
    assert path == tmp_path / ".mcp.json"
    listing = "Checking MCP server health…\n\ndef: false  - ⏸ Pending approval\nres: http://127.0.0.1:9/ (HTTP) - ⏸ Pending\n"
    assert harnesses.names_before_colon(listing) == {"def", "res"}
    assert harnesses.mcp_server_of("mcp__def__search_records") == "def"
    assert harnesses.mcp_server_of("ReadMcpResourceTool") is None


def test_isolated_home_paths_are_scrubbed_in_every_form(tmp_path: Path) -> None:
    home = measure_harness.IsolatedHome(harnesses.load("claude_code"))
    try:
        root = str(home.root)
        slug = "".join(c if c.isalnum() else "-" for c in root)
        text = (f"saved to {root}/.claude/x and {root.removeprefix('/private')}/work, "
                f"see ~/.claude/projects/{slug}-work/abc/tool-results/1.txt")
        scrubbed = home.scrub(text)
        assert "mcp-mirror-home-" not in scrubbed and "/var/folders" not in scrubbed
        assert scrubbed == "saved to ~/.claude/x and ~/work, see ~/.claude/projects/-home-work/abc/tool-results/1.txt"
    finally:
        home.remove()
    assert not home.root.exists()


def test_version_check_never_touches_the_real_home(tmp_path, monkeypatch) -> None:
    import os
    import stat

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import harnesses

    tool = tmp_path / "fake-harness"
    tool.write_text('#!/bin/sh\nmkdir -p "$HOME/.config/fake" "$XDG_DATA_HOME/fake"\necho "fake 1.2.3"\n')
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    monkeypatch.setenv("HOME", str(real_home))
    assert harnesses.version_of([str(tool)]) == "1.2.3"
    assert list(real_home.iterdir()) == []


def test_install_never_touches_the_real_home(tmp_path, monkeypatch) -> None:
    # Kilo's npm postinstall creates ~/.cache/kilo, ~/.config/kilo and two more on install.
    from types import SimpleNamespace

    binary = tmp_path / "installed" / "fake-harness"
    script = f'mkdir -p "$HOME/.cache/fake" "$XDG_CONFIG_HOME/fake" "{binary.parent}"; touch "{binary}"'
    module = SimpleNamespace(BINARY=str(binary), INSTALL=["sh", "-c", script],
                             install_check=lambda: "1.2.3")
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    monkeypatch.setenv("HOME", str(real_home))
    assert measure_harness.ensure_installed(module, {}) == "1.2.3"
    assert binary.exists()
    assert list(real_home.iterdir()) == []


def test_write_config_gets_mode_and_tools_only_when_it_declares_them(tmp_path: Path) -> None:
    legacy = _fake_module(tmp_path)
    assert measure_harness._config_options(legacy, "unattended", {"def": ["search_records"]}) == {}
    seen: list[dict] = []
    plain = legacy.write_config

    def write_config(servers, workdir, env, *, mode=None, tools=None):
        seen.append({"mode": mode, "tools": tools})
        return plain(servers, workdir, env)

    module = _fake_module(tmp_path, write_config=write_config)
    tools = {server: ["probe"] for server in cells.SERVERS}
    out_root = tmp_path / "out"
    with measure_harness.ModelStub("anthropic", text="done") as stub:
        measure_harness.run_session(
            harness=module, fixtures={server: "fixtures/definitions_server.py" for server in cells.SERVERS},
            session_dir=out_root / "opted-in", out_root=out_root, prompt="hi", max_turns=2, timeout=60,
            env_extra=module.env(stub.url), secrets={}, mode="unattended", transport="stdio", delays={}, stub=stub,
            tools=tools,
        )
    # Once with the placeholders for the list check, once for the session.
    assert seen == [{"mode": "unattended", "tools": tools}] * 2


def test_fixture_tool_names_follow_every_page() -> None:
    names = measure_harness.fixture_tool_names({"util": "fixtures/utilities_server.py"})["util"]
    assert "slow_with_progress" in names and "page_filler_15" in names and names[-1] == "SENTINEL_LAST_TOOL"
    assert len(names) == len(set(names))


def test_crewai_storage_stays_out_of_the_real_home(tmp_path) -> None:
    # crewai_core stores data in appdirs.user_data_dir(<cwd name>), which created
    # ~/Library/Application Support/<checkout> on every CrewAI run in a checkout.
    import os
    import subprocess

    pytest.importorskip("crewai_core")
    home = tmp_path / "home"
    checkout = tmp_path / "some-checkout"
    home.mkdir()
    checkout.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "CREWAI_STORAGE_DIR"}
    env["HOME"] = str(home)
    code = (
        "import mcp_mirror.renderers.crewai_renderer\n"
        "from crewai_core.paths import db_storage_path\n"
        "print(db_storage_path())"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=checkout, env=env,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert not out.startswith(str(home)), out
    assert not any(home.rglob("*")), list(home.rglob("*"))
