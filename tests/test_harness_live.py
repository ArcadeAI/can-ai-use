"""A harness with no stub is measured live, and the live path leaks nothing.

The key reaches only the harness's process: one `op run` wraps the whole measurement,
the runner takes the env file's variables out of its own environment once
(live_secrets) and starts each harness directly in the isolated session's variables plus
those, so the harness keeps the throwaway home and sees nothing else of the operator's,
and no other child sees the key at all. The sink
forwards CONNECT only to the hosts the module names, only in a live run, and the
published row says it was measured live, with the service and the model.
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import sys
import threading
import types
from pathlib import Path

import pytest

from mcp_mirror import cells

sys.path.insert(0, str(Path(__file__).parent))
from test_harness_cells import ROOT, _root, _session, _write_run, measure_harness  # noqa: E402
from test_harness_fresh import _fake_module, real_home  # noqa: E402,F401

sys.path.insert(0, str(ROOT / "data" / "validator"))
import validate  # noqa: E402

harnesses = measure_harness.harnesses

# A harness with no stub: it lists its servers, then reports what its environment held.
FAKE_LIVE_HARNESS = r'''
import json, os, sys, tomllib
from pathlib import Path

import pytest

servers = {}
for home in (Path(os.environ.get("CODEX_HOME") or Path(os.environ["HOME"]) / ".codex"), Path(".codex")):
    if (home / "config.toml").exists():
        servers |= tomllib.loads((home / "config.toml").read_text()).get("mcp_servers", {})
if sys.argv[1] == "list":
    # `mcp list` runs before the session and must not see the key.
    print(f"__key__: {'present' if os.environ.get('FAKE_KEY') else 'absent'} - configured", file=sys.stderr)
    for name in servers:
        print(f"{name}: {servers[name].get('command', 'http')} - configured")
    raise SystemExit(0)
# The key itself is never echoed (the secret check would catch it, as it should): only that it
# was the env file's, which no shell variable of this name is.
print(json.dumps({"type": "init", "model": "fake-model-2", "servers": sorted(servers),
                  "key": "the file's" if os.environ.get("FAKE_KEY", "").startswith("fake-secret-for-") else os.environ.get("FAKE_KEY"),
                  "home": os.environ["HOME"], "op_account": os.environ.get("OP_ACCOUNT"),
                  "operator_home_seen": os.path.exists(os.path.join(os.environ["HOME"], ".codex", "config.toml"))}))
'''


def _reduce(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        item = json.loads(line)
        if item.get("type") == "init":
            events.append({"event": "init", "model": item["model"], "mcp_servers": item["servers"],
                           "mcp_tools": [], "key": item["key"], "home": item["home"],
                           "op_account": item["op_account"], "operator_home_seen": item["operator_home_seen"]})
    return events


def _live_module(tmp_path: Path):
    # In its own directory, as an installed harness is: the live sandbox allows only that.
    (tmp_path / "harness-install").mkdir()
    script = tmp_path / "harness-install" / "fake_live.py"
    script.write_text(FAKE_LIVE_HARNESS)
    module = _fake_module(tmp_path, checked=False, STUB_KIND=None, LIVE_SERVICE="Fake",
                          LIVE_EGRESS_ALLOWED={"api.fake.example": "the fake service"})
    module.BINARY = str(script)
    module.list_servers = lambda env, workdir: harnesses.names_before_colon(
        harnesses.list_command([sys.executable, str(script), "list"], env, workdir))
    module.argv = lambda prompt, mode, config, max_turns: [sys.executable, str(script), "run"]
    module.env = lambda url: {}
    module.reduce_output = _reduce
    assert harnesses.interface_problems(module) == []
    return module


def test_live_secrets_come_from_the_runner_environment_once_and_leave_it(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env.harness"
    env_file.write_text("FAKE_KEY=op://Vault/Item/credential\n# a comment\nOTHER=op://Vault/Item/password\n")
    assert measure_harness.env_file_variables(env_file) == ["FAKE_KEY", "OTHER"]
    # Not wrapped in op run: the blocker and live_secrets both name the variable and the wrapper.
    monkeypatch.delenv("FAKE_KEY", raising=False)
    monkeypatch.setenv("OTHER", "fake-secret-for-OTHER")
    plan = {"level_2": {"env_file": str(env_file.relative_to(ROOT)) if env_file.is_relative_to(ROOT) else str(env_file)}}
    monkeypatch.setattr(measure_harness, "ROOT", Path("/"))
    blocker = measure_harness.live_blocker({"level_2": {"env_file": str(env_file).lstrip("/")}})
    assert blocker and "FAKE_KEY is not in the environment" in blocker and "op run --env-file .env.harness" in blocker
    try:
        measure_harness.live_secrets(env_file)
    except harnesses.HarnessError as exc:
        assert "FAKE_KEY is not set" in str(exc) and "measure_locked.py" in str(exc)
    else:
        raise AssertionError("a missing variable must not pass")
    # Wrapped once: the values are taken out of this process's environment, so no other
    # child inherits them, and the blocker is gone.
    monkeypatch.setenv("FAKE_KEY", "fake-secret-for-FAKE_KEY")
    assert measure_harness.live_blocker({"level_2": {"env_file": str(env_file).lstrip("/")}}) is None
    secrets = measure_harness.live_secrets(env_file)
    assert secrets == {"FAKE_KEY": "fake-secret-for-FAKE_KEY", "OTHER": "fake-secret-for-OTHER"}
    assert "FAKE_KEY" not in os.environ and "OTHER" not in os.environ
    # Once taken, the blocker would report them missing: the runner reads the blocker once,
    # before live_secrets, and never again (the first full run was skipped by exactly that).
    assert measure_harness.live_blocker({"level_2": {"env_file": str(env_file).lstrip("/")}})
    assert len(re.findall(r"live_blocker\(plan\)", (ROOT / "scripts" / "measure_harness.py").read_text())) == 1
    del plan


def test_live_launch_starts_the_harness_isolated_with_the_secrets_alone(monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/operator/home")
    monkeypatch.setenv("OP_ACCOUNT", "operator.1password.com")
    isolated = {"HOME": "/tmp/iso", "PATH": "/usr/bin", "XDG_CONFIG_HOME": "/tmp/iso/.config"}
    launch, launch_env = measure_harness.live_launch(isolated, ["cursor-agent", "-p", "hi"], {"CURSOR_API_KEY": "k"})
    assert launch == ["cursor-agent", "-p", "hi"]
    # No op in front of the harness any more: it starts directly, in the isolated session
    # environment plus the env file's values, with nothing of the operator's.
    assert launch_env == {**isolated, "CURSOR_API_KEY": "k"}
    assert "OP_ACCOUNT" not in launch_env and launch_env["HOME"] == "/tmp/iso"


@pytest.mark.skipif(not measure_harness.shutil.which("sandbox-exec"), reason="needs macOS sandbox-exec")
def test_a_fake_live_session_gets_the_secret_in_the_isolated_home(real_home: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OP_ACCOUNT", "operator.1password.com")
    monkeypatch.setenv("FAKE_KEY", "fake-secret-for-FAKE_KEY")
    env_file = tmp_path / ".env.harness"
    env_file.write_text("FAKE_KEY=op://Vault/Item/credential\n")
    secrets = measure_harness.live_secrets(env_file)
    module = _live_module(tmp_path)
    out_root = tmp_path / "out"
    fixtures = {server: "fixtures/definitions_server.py" for server in cells.SERVERS}
    record = measure_harness.run_session(
        harness=module, fixtures=fixtures, session_dir=out_root / "live", out_root=out_root, prompt="hi",
        max_turns=2, timeout=60, env_extra={}, secrets=secrets,
        mode=None, transport="stdio", delays={}, stub=None, live=True,
    )
    fresh = json.loads((out_root / record["fresh"]).read_text())
    assert fresh["passed"], fresh
    # The pre-run `mcp list` ran without the key: only the measured session gets it.
    assert fresh["checks"]["servers_listed"]["passed"]
    init = [json.loads(line) for line in (out_root / record["output"]).read_text().splitlines()[1:]][0]
    # The secret came from the env file's variable; the home stayed the throwaway one (scrubbed to ~).
    assert init["key"] == "the file's"
    assert init["home"] == "~" and init["operator_home_seen"] is False
    assert init["op_account"] is None and init["mcp_servers"] == sorted(cells.SERVERS)
    assert init["model"] == "fake-model-2"
    assert measure_harness.observed_models(out_root, [{"output": record["output"]}]) == ["fake-model-2"]
    # Nothing was forwarded: the fake service was never contacted, and the check says which hosts may be.
    assert fresh["checks"]["egress"]["forwarded"] == [] and fresh["checks"]["egress"]["attempts"] == []
    assert "model" not in record  # no stub, so no model.jsonl
    # The secret is in no file the session wrote, home included.
    checked = fresh["checks"]["secrets"]
    assert checked["passed"] and checked["variables"] == ["FAKE_KEY"] and checked["matches"] == []
    assert checked["paths"][1] == "<session home>"
    assert "fake-secret-for" not in json.dumps(fresh)


def test_the_secret_check_names_the_file_that_holds_the_key_and_never_the_key(tmp_path: Path) -> None:
    secrets = {"FAKE_KEY": "fake-secret-for-FAKE_KEY", "OTHER": "fake-secret-for-OTHER"}
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "a.jsonl").write_text('{"key": "not-the-secret"}\n')
    assert measure_harness.check_secrets_absent(secrets, [clean])["passed"]
    leaky = tmp_path / "leaky"
    (leaky / "deep").mkdir(parents=True)
    (leaky / "deep" / "stdout").write_text("Bearer fake-secret-for-FAKE_KEY\n")
    (leaky / "b64.txt").write_bytes(base64.b64encode(b"fake-secret-for-OTHER"))
    result = measure_harness.check_secrets_absent(secrets, [clean, leaky])
    assert not result["passed"]
    assert result["matches"] == [f"match: {leaky / 'b64.txt'} holds OTHER", f"match: {leaky / 'deep' / 'stdout'} holds FAKE_KEY"]
    assert "fake-secret" not in json.dumps(result)
    # An empty value is a failure to check, not a pass.
    try:
        measure_harness.check_secrets_absent({"MISSING": ""}, [clean])
    except harnesses.HarnessError as exc:
        assert "MISSING is not set" in str(exc)
    else:
        raise AssertionError("an unset variable must not pass")


def _echo_server() -> tuple[socket.socket, int]:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def serve() -> None:
        try:
            conn, _ = server.accept()
        except OSError:
            return  # the listener was closed without a client: the refused case
        with conn:
            data = conn.recv(1024)
            conn.sendall(b"echo:" + data)

    threading.Thread(target=serve, daemon=True).start()
    return server, server.getsockname()[1]


def _connect(sink_url: str, target: str) -> tuple[bytes, socket.socket]:
    host, port = sink_url.removeprefix("http://").split(":")
    client = socket.create_connection((host, int(port)), timeout=10)
    client.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
    return client.recv(1024), client


def test_sink_forwards_only_the_named_hosts_and_records_each(tmp_path: Path) -> None:
    server, port = _echo_server()
    module = _fake_module(tmp_path, LIVE_EGRESS_ALLOWED={"127.0.0.1": "the fake service, for this test"})
    try:
        with measure_harness.EgressSink(measure_harness.live_egress_allowed(module)) as sink:
            status, client = _connect(sink.url, f"127.0.0.1:{port}")
            assert status.startswith(b"HTTP/1.0 200") or status.startswith(b"HTTP/1.1 200"), status
            client.sendall(b"hello")
            assert client.recv(1024) == b"echo:hello"
            client.close()
            refused, other = _connect(sink.url, "updates.example.com:443")
            assert b" 403 " in refused
            other.close()
            check = sink.check(module)
        assert check["forwarded"] == ["127.0.0.1"] and check["not_allowed"] == ["updates.example.com"]
        assert not check["passed"]
        forwarded = next(a for a in check["attempts"] if a["host"] == "127.0.0.1")
        assert forwarded["forwarded"] and forwarded["allowed"] == "the fake service, for this test"
        # A stubbed session's sink forwards nothing, even to that host.
        with measure_harness.EgressSink() as sink:
            server2, port2 = _echo_server()
            refused, client = _connect(sink.url, f"127.0.0.1:{port2}")
            assert b" 403 " in refused
            client.close()
            server2.close()
            assert "forwarded" not in sink.check(module)
    finally:
        server.close()


def _live_manifest(level_1: list[dict], level_2: list[dict]) -> dict:
    return {
        "harness": "demo", "version": "1.0.0", "captured_at": "2026-10-13T00:00:00+00:00",
        "command": "uv run python scripts/measure_harness.py demo --live",
        "level_1": {"status": "ran", "model": "live, under op run --env-file .env.harness", "sessions": level_1},
        "level_2": {"status": "ran", "model": "live, under op run --env-file .env.harness", "runs": 3, "sessions": level_2},
        "live": {"env_file": ".env.harness", "egress_allowed": {"api.demo.example": "the service"},
                 "models": ["demo-model"]},
    }


def test_the_loader_reads_the_live_run_and_the_row_says_so(tmp_path: Path, monkeypatch) -> None:
    stub_root = _root(tmp_path)
    (stub_root / "session.json").write_text(json.dumps({
        "harness": "demo", "version": "0.9.0", "captured_at": "2026-09-01T00:00:00+00:00",
        "level_1": {"status": "skipped", "reason": "no STUB_KIND", "sessions": []},
        "level_2": {"status": "skipped", "reason": "no STUB_KIND", "sessions": []},
    }))
    live_root = tmp_path / "measurements" / "wire-live" / "demo"
    live_root.mkdir(parents=True)
    (live_root / "session.json").write_text(json.dumps(_live_manifest([_session(live_root, "stdio", 1)], [])))
    runs = cells.load_harness_runs(tmp_path)
    assert [run.harness for run in runs] == ["demo"]
    assert runs[0].live == {"env_file": ".env.harness", "egress_allowed": {"api.demo.example": "the service"},
                            "models": ["demo-model"]}
    assert runs[0].manifest["version"] == "1.0.0" and runs[0].blocked() is None
    assert cells.HarnessRun(stub_root / "session.json").live is None

    # The row the runner writes for a live run, against a copy of the committed table.
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "harnesses.json").write_bytes((ROOT / "data" / "harnesses" / "harnesses.json").read_bytes())
    monkeypatch.setattr(measure_harness, "SNAPSHOT", snapshot)
    entry = {"name": "Demo", "install": "demo install"}
    measure_harness.update_harness_row("demo", entry, "1.0.0", "2026-10-13", out_root=snapshot / "measurements" / "wire-live" / "demo",
                                       live=True, live_service="Demo", models=["demo-model"])
    row = json.loads((snapshot / "harnesses.json").read_text())["agents"]["demo"]
    assert row["capture"] == "measurements/wire-live/demo/session.json"
    assert row["measured_live"] is True and row["live_service"] == "Demo" and row["model"] == "demo-model"
    assert row["visibility"] == "preview"
    measure_harness.update_harness_row("demo", entry, "1.1.0", "2026-10-14", out_root=snapshot / "measurements" / "wire" / "demo")
    row = json.loads((snapshot / "harnesses.json").read_text())["agents"]["demo"]
    assert row["capture"] == "measurements/wire/demo/session.json"
    assert "measured_live" not in row and "live_service" not in row and "model" not in row


def _snapshot_with(tmp_path: Path, agent: dict, manifest: dict, wire: str) -> Path:
    source = ROOT / "data" / "harnesses"
    document = json.loads((source / "harnesses.json").read_text())
    document["agents"] = {"demo": {**next(iter(document["agents"].values())), **agent}}
    harness_dir = tmp_path / "harnesses"
    harness_dir.mkdir(exist_ok=True)
    (harness_dir / "surface.json").write_bytes((source / "surface.json").read_bytes())
    (harness_dir / "harnesses.json").write_text(json.dumps(document))
    root = harness_dir / "measurements" / wire / "demo"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {**manifest, "level_1": {**manifest["level_1"], "sessions": [_session(root, "stdio", 1)]}}
    (root / "session.json").write_text(json.dumps(manifest))
    return harness_dir


def test_the_validator_wants_a_live_row_to_say_so_and_only_then(tmp_path: Path) -> None:
    live = {"capture": "measurements/wire-live/demo/session.json", "measured_live": True, "live_service": "Demo",
            "model": "demo-model", "current_version": "1.0.0",
            "version_list": [{"version": "1.0.0", "era": 0, "release_date": None}]}
    manifest = _live_manifest([], [])
    harness_dir = _snapshot_with(tmp_path, live, manifest, "wire-live")
    _, _, errors = validate.validate_harnesses(harness_dir)
    assert [e for e in errors if "demo" in e] == [], errors

    # A live capture without the row's mark, or a row's mark without a live capture, is an error.
    for change in ({"measured_live": False}, {"live_service": ""}):
        harness_dir = _snapshot_with(tmp_path, {**live, **change}, manifest, "wire-live")
        _, _, errors = validate.validate_harnesses(harness_dir)
        assert any("measured_live" in e or "live_service" in e for e in errors), (change, errors)
    harness_dir = _snapshot_with(tmp_path, live, {k: v for k, v in manifest.items() if k != "live"}, "wire-live")
    _, _, errors = validate.validate_harnesses(harness_dir)
    assert any("has no live block" in e for e in errors), errors
    # Both a stubbed and a live run for one harness: the validator says so.
    (harness_dir / "measurements" / "wire" / "demo").mkdir(parents=True)
    (harness_dir / "measurements" / "wire" / "demo" / "session.json").write_text(json.dumps(manifest))
    _, _, errors = validate.validate_harnesses(harness_dir)
    assert any("stubbed or live, not both" in e for e in errors), errors


def test_runs_that_disagree_say_so_in_the_note(tmp_path: Path) -> None:
    root = _root(tmp_path)

    def called(wire, server):
        from test_harness_cells import _startup
        _startup(wire, server=server)
        if server == "def":
            wire.request("tools/call", 30, {"name": "search_records", "arguments": {"query": "x"}})

    sessions = [_session(root, "read_only-unattended", 2, run=1, build=called),
                _session(root, "read_only-unattended", 2, run=2),
                _session(root, "read_only-unattended", 2, run=3, build=called)]
    run = _write_run(tmp_path, [_session(root, "stdio", 1)], sessions)
    cell = cells.HARNESS_RULES["h-calls-read-only-tool"](run)
    assert cell.code == "a" and cell.evidence[0] == "2/3 runs (runs disagreed)"
    unanimous = _write_run(tmp_path, [_session(root, "stdio", 1)], [_session(root, "read_only-unattended", 2, run=1, build=called)])
    assert cells.HARNESS_RULES["h-calls-read-only-tool"](unanimous).evidence[0] == "1/1 runs"


def test_cursor_module_declares_its_live_service() -> None:
    module = harnesses.load("cursor_agent")
    assert module.STUB_KIND is None and module.LIVE_SERVICE == "Cursor"
    assert set(module.LIVE_EGRESS_ALLOWED) == {"api2.cursor.sh", "agentn.us.api5.cursor.sh"}
    # Refused, and still fresh: the update check, npx fetching the TypeScript LSP, the
    # CLI's own GitHub lookups and its Sentry telemetry.
    assert set(module.EGRESS_ALLOWED) == {"downloads.cursor.com", "registry.npmjs.org", "github.com",
                                          "o4504648565915648.ingest.us.sentry.io"}
    # Level 1 runs the default and the documented approval switch (the default starts no server).
    assert "--approve-mcps" in module.LEVEL1_UNATTENDED and "--approve-mcps" in module.mode_args("unattended")
    # Level 2's default approves the servers and nothing else; level 1's default pass has no switch.
    assert module.mode_args("default") == ["--approve-mcps"] and module.mode_args(None) == []
    assert module.mode_args("unattended") == ["--force", "--approve-mcps"]
    assert measure_harness.level_1_unattended(module) == {"unattended": module.LEVEL1_UNATTENDED}
    assert measure_harness.level_1_modes(module, "stdio") == [(None, "stdio"), ("unattended", "stdio-unattended")]
    assert "CURSOR_" in module.SCRUBBED_ENV
    # The key stays out of the keychain in every session: the bundle-only memory store.
    assert module.env("") == {"AGENT_CLI_CREDENTIAL_STORE": "memory"} == module.QUIET
    assert measure_harness.session_env(module, None) == module.QUIET
    init = module.reduce_output(json.dumps({"type": "system", "subtype": "init", "model": "auto", "permissionMode": "default"}))
    assert init == [{"event": "init", "harness_version": None, "model": "auto", "permission_mode": "default",
                     "mcp_servers": None, "mcp_tools": None}]


def test_level_1_modes_default_only_without_the_declaration() -> None:
    goose = harnesses.load("goose")
    assert not hasattr(goose, "LEVEL1_UNATTENDED")
    assert measure_harness.level_1_unattended(goose) == {}
    assert measure_harness.level_1_modes(goose, "http") == [(None, "http")]
    assert measure_harness.level_1_record(goose, "http", "2026-07-28", None, {"exit": 0}) == {
        "id": "http", "level": 1, "revision": "2026-07-28", "exit": 0}
    cursor = harnesses.load("cursor_agent")
    approved = measure_harness.level_1_record(cursor, "http-unattended", "2026-07-28", "unattended", {"exit": 0})
    assert approved["mode"] == "unattended" and approved["mode_args"] == ["--force", "--approve-mcps"]
    try:
        measure_harness.level_1_unattended(type("M", (), {"LEVEL1_UNATTENDED": ""})())
    except measure_harness.HarnessError:
        pass
    else:
        raise AssertionError("an empty reason must be refused")


def test_level_1_cells_read_the_unattended_pass_and_name_the_default(tmp_path: Path) -> None:
    root = _root(tmp_path)

    def nothing(wire, server) -> None:  # the default session: no fixture ever saw a byte
        pass

    default = _session(root, "stdio", 1, build=nothing)
    approved = {**_session(root, "stdio-unattended", 1), "mode": "unattended", "mode_args": ["--force", "--approve-mcps"]}
    run = _write_run(tmp_path, [default, approved])
    # Without the manifest flag the default session is the one read: nothing listed.
    assert cells.HARNESS_RULES["h-lists-tools-at-startup"](run).code == "n"
    manifest = json.loads((root / "session.json").read_text())
    manifest["level_1"]["unattended"] = "--approve-mcps, the documented switch; the default run starts none"
    (root / "session.json").write_text(json.dumps(manifest))
    run = cells.HarnessRun(root / "session.json")
    cell = cells.HARNESS_RULES["h-lists-tools-at-startup"](run)
    assert cell.code == "y"
    line = cells.level_1_default_line(run)
    assert line == ("level 1 read from stdio-unattended, run with --force --approve-mcps (--approve-mcps, the "
                    "documented switch; the default run starts none); the default session stdio, run with no "
                    "switch, opened 0 of 5 fixtures")
    assert line in cell.evidence
    # Level-2 cells do not carry the line.
    assert line not in cells.HARNESS_RULES["h-calls-read-only-tool"](run).evidence


def test_cursor_records_the_model_auto_routed_to(tmp_path: Path) -> None:
    module = harnesses.load("cursor_agent")
    lines = [json.dumps({"type": "system", "subtype": "init", "model": "Auto Balance", "permissionMode": "default"}),
             json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "> Auto routed to Grok 4.7\n\nready"}]}}),
             json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "ready"})]
    events = module.reduce_output("\n".join(lines))
    assert events[1] == {"event": "routed", "model": "Grok 4.7"}
    assert events[2]["event"] == "text" and events[2]["text"].endswith("ready")
    output = tmp_path / "harness.jsonl"
    output.write_text("".join(json.dumps(e) + "\n" for e in [{"format": "x"}, *events]))
    assert measure_harness.observed_models(tmp_path, [{"output": "harness.jsonl"}]) == ["Auto Balance", "routed to Grok 4.7"]
    plain = module.reduce_output(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "ready"}]}}))
    assert [e["event"] for e in plain] == ["text"]


# -- Containment -----------------------------------------------------------------------

SANDBOX = pytest.mark.skipif(not measure_harness.shutil.which("sandbox-exec"), reason="needs macOS sandbox-exec")

# A harness with no stub that does what a live model with a shell can do: read, list,
# write and run outside its session, open a TCP connection, and start one fixture through
# the recorder (the one thing it must still be able to do). It reports each attempt's
# errno, never a file's contents.
FAKE_CONTAINED_HARNESS = r'''
import errno, json, os, socket, subprocess, sys
from pathlib import Path

import pytest

servers = json.loads((Path(".codex") / "servers.json").read_text()) if Path(".codex/servers.json").exists() else {}
if sys.argv[1] == "list":
    for name in servers:
        print(f"{name}: {servers[name].get('command', 'http')} - configured")
    raise SystemExit(0)

def attempt(action):
    try:
        return action()
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc.errno))

canaries = json.loads(os.environ["CANARIES"])
out = {"type": "init", "model": "fake-model-2", "servers": sorted(servers), "home": os.environ["HOME"]}
out["read"] = {name: attempt(lambda p=p: "read:" + open(p).read().strip()) for name, p in canaries["read"].items()}
out["listed"] = attempt(lambda: os.listdir(canaries["dir"]) and "listed")
out["wrote_outside"] = attempt(lambda: open(os.path.join(canaries["dir"], "from-harness"), "w").write("x") and "written")
out["ran_outside"] = attempt(lambda: subprocess.run([canaries["binary"], "-c", "true"]).returncode == 0 and "ran")
out["wrote_inside"] = attempt(lambda: open("note.txt", "w").write("x") and "written")
out["tmpdir_inside"] = os.environ.get("TMPDIR", "").startswith(os.environ["HOME"])
s = socket.socket(); s.settimeout(3)
try:
    s.connect(("192.0.2.1", 443)); out["tcp"] = "connected"
except OSError as exc:
    out["tcp"] = errno.errorcode.get(exc.errno, str(exc.errno))
# One fixture through the recorder, as a real harness starts it: initialize, one answer.
name, server = next(iter(servers.items()))
proc = subprocess.Popen([server["command"], *server["args"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True)
proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "fake", "version": "0"}}}) + "\n")
proc.stdin.flush()
answer = json.loads(proc.stdout.readline() or "{}")
proc.stdin.close()
proc.wait(timeout=60)
out["initialized"] = answer.get("result", {}).get("serverInfo", {}).get("name")
out["server"] = {"argv": [server["command"], *server["args"]], "exit": proc.returncode, "stderr": proc.stderr.read()[-800:]}
print(json.dumps(out))
'''


def _contained_module(tmp_path: Path):
    """The fake harness above, installed in its own directory: the sandbox allows the
    harness's install, so nothing else under tmp_path (the operator's home, the repository)
    may sit beside it."""
    install = tmp_path / "harness-install"
    install.mkdir()
    script = install / "fake_contained.py"
    script.write_text(FAKE_CONTAINED_HARNESS)
    module = _fake_module(tmp_path, checked=False, STUB_KIND=None, LIVE_SERVICE="Fake",
                          LIVE_EGRESS_ALLOWED={"api.fake.example": "the fake service"})
    module.BINARY = str(script)

    def write_config(servers, workdir, env):
        path = workdir / ".codex" / "servers.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(servers))
        return path

    module.write_config = write_config
    module.list_servers = lambda env, workdir: harnesses.names_before_colon(
        harnesses.list_command([sys.executable, str(script), "list"], env, workdir))
    module.argv = lambda prompt, mode, config, max_turns: [sys.executable, str(script), "run"]
    module.env = lambda url: {}

    def reduce(text: str) -> list[dict]:
        events = []
        for line in text.splitlines():
            item = json.loads(line)
            if item.get("type") == "init":
                events.append({"event": "init", "model": item.pop("model"), "mcp_servers": item.pop("servers"),
                               "mcp_tools": [], **item})
        return events

    module.reduce_output = reduce
    assert harnesses.interface_problems(module) == []
    return module


def _canaries(real_home: Path, tmp_path: Path) -> tuple[dict, Path]:
    """Files a live model must not reach: in the operator's home and in the repository
    checkout the runner itself runs from, plus a binary it must not run."""
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (real_home / "secret-canary.txt").write_text("CANARY-HOME\n")
    (repo / "data" / "canary.json").write_text('{"canary": "CANARY-REPO"}\n')
    binary = repo / "tool"
    measure_harness.shutil.copy("/bin/sh", binary)
    return {"read": {"operator_home": str(real_home / "secret-canary.txt"), "operator_codex": str(real_home / ".codex" / "config.toml"),
                     "repo": str(repo / "data" / "canary.json")},
            "dir": str(repo), "binary": str(binary)}, repo


def _contained_session(module, tmp_path: Path, canaries: dict, **kwargs) -> tuple[dict, dict, dict]:
    out_root = tmp_path / "out"
    fixtures = {server: "fixtures/definitions_server.py" for server in cells.SERVERS}
    record = measure_harness.run_session(
        harness=module, fixtures=fixtures, session_dir=out_root / "live", out_root=out_root, prompt="hi",
        max_turns=2, timeout=kwargs.pop("timeout", 120), env_extra={"CANARIES": json.dumps(canaries)},
        secrets={"FAKE_KEY": "fake-secret-for-FAKE_KEY"}, mode=None, transport="stdio", delays={}, stub=None,
        live=True, **kwargs,
    )
    fresh = json.loads((out_root / record["fresh"]).read_text())
    lines = (out_root / record["output"]).read_text().splitlines()[1:]
    init = json.loads(lines[0]) if lines else {}
    return record, fresh, init


@SANDBOX
def test_a_live_session_cannot_reach_the_operator_home_or_the_repository(real_home: Path, tmp_path: Path) -> None:
    canaries, repo = _canaries(real_home, tmp_path)
    module = _contained_module(tmp_path)
    record, fresh, init = _contained_session(module, tmp_path, canaries)
    assert record["exit_code"] == 0 and fresh["passed"], (record["stderr_tail"], fresh)
    sandbox = fresh["checks"]["sandbox"]
    assert sandbox["passed"] and sandbox["profile_sha256"] == measure_harness.LiveSandbox.PROFILE_SHA256
    assert sandbox["probe"] == measure_harness.LIVE_SANDBOX_HELD
    # Every canary: unreadable, the directory unlistable, nothing written there, the binary not run.
    assert init["read"] == {"operator_home": "EPERM", "operator_codex": "EPERM", "repo": "EPERM"}
    assert init["listed"] == "EPERM" and init["wrote_outside"] == "EPERM" and init["ran_outside"] == "EPERM"
    assert not (repo / "from-harness").exists()
    # TCP leaves to loopback only; the session's own home and temp dir still work.
    assert init["tcp"] == "EPERM"
    assert init["wrote_inside"] == "written" and init["tmpdir_inside"] is True and init["home"] == "~"
    # The one thing it must still do: start a fixture through the recorder, which wrote the capture.
    assert init["initialized"] == "tricky-mcp"
    assert "def" in record["captures"] and fresh["checks"]["recorders"]["passed"]
    assert "initialize" in (tmp_path / "out" / record["captures"]["def"]).read_text()
    assert fresh["checks"]["process_group"] == {"passed": True, "timed_out": False, "outlived": [], "still_alive": [], "escaped": []}
    # The allowed paths are listed as published: repository-relative, the home as a label.
    assert sandbox["write"][0] == "<session home>" and "fixtures" in sandbox["read"] and "src" in sandbox["read"]
    assert any(p.endswith("harness-install") for p in sandbox["read"])
    assert sandbox["root_names_only"] == "." and sandbox["unix_sockets"] == "<session home>"
    assert sandbox["mach_lookup_allowed"] == [] and "/usr/bin/open" in sandbox["desktop_exec_denied"]
    assert sandbox["probe"]["exec_pbpaste"] == "EPERM" and sandbox["probe"]["prefs_read"] == "denied"
    # Loopback is open to the sink's port alone, rendered per session; signals and process
    # information (argv, pid info) stop at the session's own processes.
    assert len(sandbox["loopback_ports"]) == 1 and 1024 < sandbox["loopback_ports"][0] < 65536  # stdio: the sink alone
    assert sandbox["probe"]["tcp_sink"] == "connected" and sandbox["probe"]["tcp_other_local"] == "EPERM"
    assert sandbox["probe"]["signal_outside"] == "EPERM" and sandbox["probe"]["signal_child"] == "killed"
    assert sandbox["probe"]["argv_outside"] == "denied" and sandbox["probe"]["pidinfo_outside"] == "denied"
    assert sandbox["signals"] == "<session processes>" and sandbox["process_info"] == "<session processes>"
    assert "CANARY" not in (tmp_path / "out" / record["output"]).read_text()


def test_a_live_session_fails_closed_without_sandbox_exec(real_home: Path, tmp_path: Path, monkeypatch) -> None:
    which = measure_harness.shutil.which
    monkeypatch.setattr(measure_harness.shutil, "which", lambda name, *a, **k: None if name == "sandbox-exec" else which(name, *a, **k))
    canaries, _ = _canaries(real_home, tmp_path)
    module = _contained_module(tmp_path)
    with pytest.raises(harnesses.HarnessError, match="sandbox-exec"):
        _contained_session(module, tmp_path, canaries)
    assert not (tmp_path / "out" / "live" / cells.HARNESS_FRESH).exists()
    assert not list(Path(measure_harness.SHORT_HOME).glob("mcp-mirror-home-*/probe-inside.txt"))


@SANDBOX
def test_a_live_session_does_not_start_when_the_probe_shows_an_allowance(real_home: Path, tmp_path: Path, monkeypatch) -> None:
    leaky = {**measure_harness.LIVE_SANDBOX_HELD, "read_outside": "read"}
    monkeypatch.setattr(measure_harness.LiveSandbox, "probe", lambda self: leaky)
    canaries, _ = _canaries(real_home, tmp_path)
    module = _contained_module(tmp_path)
    record, fresh, init = _contained_session(module, tmp_path, canaries)
    assert record["exit_code"] is None and init == {}
    assert record["stderr_tail"] == ["not started: the pre-run freshness checks failed (fresh.json)"]
    assert not fresh["passed"] and not fresh["checks"]["sandbox"]["passed"]
    assert fresh["checks"]["sandbox"]["probe"]["read_outside"] == "read"
    assert "process_group" not in fresh["checks"]


def test_a_child_that_outlives_the_harness_in_its_group_is_reaped_and_recorded(real_home: Path, tmp_path: Path) -> None:
    from test_harness_fresh import FAKE_HARNESS, _run
    module = _fake_module(tmp_path)
    script = tmp_path / "fake_codex.py"
    # The stubbed fake, plus one child left behind on exit, in the group (what Cursor's
    # CLI does with its worker-server). Reaped a few seconds after the exit: the session
    # passes, with the child on record.
    script.write_text(FAKE_HARNESS + "\n__import__('subprocess').Popen(['sleep', '300'])\n")
    record = _run(module, tmp_path, "orphan")
    fresh = record["fresh_record"]
    group = fresh["checks"]["process_group"]
    assert fresh["passed"] and group["passed"], group
    assert group["timed_out"] is False and group["still_alive"] == [] and group["escaped"] == []
    [child] = [m for m in group["outlived"] if m["command"] == "sleep"]
    assert 5 <= child["after_exit_s"] < 60
    with pytest.raises(ProcessLookupError):
        os.kill(child["pid"], 0)


def test_a_child_that_leaves_the_group_is_found_in_the_session_and_fails_it(real_home: Path, tmp_path: Path) -> None:
    from test_harness_fresh import FAKE_HARNESS, _run
    module = _fake_module(tmp_path)
    script = tmp_path / "fake_codex.py"
    # The child moves itself out of the harness's group (setpgid) but stays in its session.
    script.write_text(FAKE_HARNESS + "\nimport os, subprocess\nsubprocess.Popen(['sleep', '300'], preexec_fn=os.setpgrp)\n")
    record = _run(module, tmp_path, "escaped")
    fresh = record["fresh_record"]
    group = fresh["checks"]["process_group"]
    assert not fresh["passed"] and not group["passed"]
    assert group["outlived"] == [] and group["still_alive"] == []
    [stray] = [m for m in group["escaped"] if m["command"].endswith("sleep")]
    with pytest.raises(ProcessLookupError):
        os.kill(stray["pid"], 0)


def test_a_child_that_survives_the_kill_fails_the_session(real_home: Path, tmp_path: Path, monkeypatch) -> None:
    from test_harness_fresh import FAKE_HARNESS, _run
    module = _fake_module(tmp_path)
    (tmp_path / "fake_codex.py").write_text(FAKE_HARNESS + "\n__import__('subprocess').Popen(['sleep', '300'])\n")
    # A member the kill does not take (reported by the sampler after the SIGKILL too).
    sampler = measure_harness._group_members
    monkeypatch.setattr(measure_harness, "_group_members",
                        lambda pgid: sampler(pgid) or [{"pid": pgid, "command": "immortal"}])
    group = _run(module, tmp_path, "immortal")["fresh_record"]["checks"]["process_group"]
    assert not group["passed"] and [m["command"] for m in group["still_alive"]] == ["immortal"]
    assert group["outlived"] and group["escaped"] == []


def test_the_timeout_kills_the_whole_group(real_home: Path, tmp_path: Path) -> None:
    from test_harness_fresh import FAKE_HARNESS
    module = _fake_module(tmp_path)
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE_HARNESS + "\n__import__('subprocess').Popen(['sleep', '300'])\n__import__('time').sleep(300)\n")
    out_root = tmp_path / "out"
    fixtures = {server: "fixtures/definitions_server.py" for server in cells.SERVERS}
    with measure_harness.ModelStub("anthropic", text="done") as stub:
        record = measure_harness.run_session(
            harness=module, fixtures=fixtures, session_dir=out_root / "slow", out_root=out_root, prompt="hi",
            max_turns=2, timeout=3, env_extra=module.env(stub.url), secrets={}, mode=None, transport="stdio",
            delays={}, stub=stub,
        )
    fresh = json.loads((out_root / record["fresh"]).read_text())
    assert record["timed_out"] is True and record["duration_s"] < 60
    group = fresh["checks"]["process_group"]
    # Killed at the timeout, harness and child alike: nothing outlived the group.
    assert group == {"passed": True, "timed_out": True, "outlived": [], "still_alive": [], "escaped": []}


CURSOR_BINARY = harnesses.resolve_binary(harnesses.load("cursor_agent").BINARY)


@SANDBOX
@pytest.mark.skipif(CURSOR_BINARY is None, reason="needs the project-local Cursor agent CLI (scripts/harnesses/cursor_agent.py INSTALL)")
def test_the_real_cursor_cli_starts_contained_and_reaches_only_the_sink(real_home: Path, tmp_path: Path) -> None:
    """A short run of the pinned cursor-agent under the live profile, with a placeholder
    key and nothing forwarded: the CLI starts, asks the sink for its endpoint, is refused
    and stops. No op, no account, no byte leaves the machine."""
    real = harnesses.load("cursor_agent")
    module = types.ModuleType("cursor_contained")
    module.__dict__.update(vars(real))
    module.LIVE_EGRESS_ALLOWED = {}  # forward nothing: the placeholder key never reaches Cursor
    shared = Path("/tmp/.cursor")
    before = set(os.listdir(shared)) if shared.is_dir() else set()
    out_root = tmp_path / "out"
    fixtures = {server: "fixtures/definitions_server.py" for server in cells.SERVERS}
    record = measure_harness.run_session(
        harness=module, fixtures=fixtures, session_dir=out_root / "cursor", out_root=out_root, prompt="list your tools",
        max_turns=2, timeout=120, env_extra=measure_harness.session_env(module, None),
        secrets={"CURSOR_API_KEY": harnesses.STUB_KEY}, mode="default", transport="stdio", delays={}, stub=None, live=True,
    )
    fresh = json.loads((out_root / record["fresh"]).read_text())
    checks = fresh["checks"]
    assert record["exit_code"] is not None and not record["timed_out"], record
    assert checks["sandbox"]["passed"] and checks["sandbox"]["probe"] == measure_harness.LIVE_SANDBOX_HELD
    assert any(p.endswith(real.BUILD) for p in checks["sandbox"]["exec"])
    # The CLI's one destination, refused by the sink (nothing is forwarded here), so the
    # session is not fresh for exactly that reason and no other.
    assert checks["egress"]["not_allowed"] == ["api2.cursor.sh"]
    assert [name for name, check in checks.items() if not check["passed"]] == ["egress"]
    assert checks["process_group"]["passed"] and checks["secrets"]["passed"]
    # The per-project state stayed in the short home, not in the shared /tmp/.cursor.
    after = set(os.listdir(shared)) if shared.is_dir() else set()
    assert after == before
    assert not (real_home / ".cursor").exists()
