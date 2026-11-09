"""A capture is never rewritten under a live recorder, and a padded one is never published.

Goose 1.54.0 showed the race twice: the harness exited, the runner scrubbed home paths
out of the stdio captures (a rewrite that shortens the file), and the def recorder,
still alive with a stale offset in a write-mode handle, then put its end record past
the new end, leaving a run of NUL bytes that no cell rule could parse.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mcp_mirror import cells
from mcp_mirror.recorder import Capture, read_capture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_harness_cells import ROOT, measure_harness  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "validator"))
import validate  # noqa: E402


def test_a_capture_appends_after_an_external_rewrite(tmp_path: Path) -> None:
    out = tmp_path / "def.jsonl"
    capture = Capture(out, fixture="python fixtures/def.py", transport="stdio")
    capture.write({"type": "message", "session": "stdio", "direction": "client_to_server",
                   "origin": "client", "kind": "request", "method": "tools/call", "id": 1,
                   "revision": None, "batch": False, "message": {"cwd": "/private/var/folders/x/work"}})
    text = out.read_text()
    out.write_text(text.replace("/private/var/folders/x/work", "~/work"))  # the runner's scrub
    capture.close()
    raw = out.read_bytes()
    assert b"\x00" not in raw
    records = read_capture(out)
    assert [r["type"] for r in records] == ["capture", "message", "end"]
    assert records[1]["message"]["cwd"] == "~/work"
    assert cells.padded_captures(tmp_path) == []


def test_the_runner_waits_for_a_recorder_it_did_not_start(tmp_path: Path) -> None:
    capture = tmp_path / "def.jsonl"
    # Stands in for `mcp-mirror record <fixture> --out <capture>`, launched by the harness.
    recorder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)", "--out", str(capture)])
    try:
        started = time.monotonic()
        check = measure_harness._wait_for_recorders([capture], timeout=10)
        waited = time.monotonic() - started
    finally:
        recorder.wait()
    assert check["passed"] and check["still_running"] == []
    assert 1.0 <= waited < 10, waited
    assert check["waited_seconds"] == pytest.approx(waited, abs=0.3)


def test_the_runner_gives_up_on_a_recorder_that_never_exits(tmp_path: Path) -> None:
    capture = tmp_path / "res.jsonl"
    recorder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)", "--out", str(capture)])
    try:
        check = measure_harness._wait_for_recorders([capture], timeout=0.5)
    finally:
        recorder.terminate()
        recorder.wait()
    assert not check["passed"] and check["still_running"] == ["res.jsonl"]


def _padded(session_dir: Path, name: str) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / name
    end = json.dumps({"type": "end", "seq": 2, "messages": 1, "sessions": 1})
    path.write_bytes(b'{"type":"capture","seq":0}\n{"type":"message","seq":1}\n' + b"\x00" * 80 + end.encode() + b"\n")
    return path


def test_the_runner_refuses_a_padded_capture(tmp_path: Path) -> None:
    session_dir = tmp_path / "level2-2026-07-28-destructive-2"
    _padded(session_dir, "def.jsonl")
    (session_dir / "res.jsonl").write_text('{"type":"capture","seq":0}\n')
    assert [p.name for p in cells.padded_captures(session_dir)] == ["def.jsonl"]
    with pytest.raises(measure_harness.HarnessError, match="def.jsonl contains NUL bytes"):
        measure_harness._refuse_padded(session_dir)


def test_the_validator_refuses_a_padded_capture(tmp_path: Path) -> None:
    source = ROOT / "data" / "harnesses"
    document = json.loads((source / "harnesses.json").read_text())
    agent_id, agent = next(iter(document["agents"].items()))
    document["agents"] = {agent_id: agent}
    harness_dir = tmp_path / "harnesses"
    harness_dir.mkdir()
    (harness_dir / "surface.json").write_bytes((source / "surface.json").read_bytes())
    (harness_dir / "harnesses.json").write_text(json.dumps(document))
    manifest_path = source / agent["capture"]
    manifest = json.loads(manifest_path.read_text())
    target = harness_dir / Path(agent["capture"]).parent
    target.mkdir(parents=True)
    (target / "session.json").write_bytes(manifest_path.read_bytes())
    session = next(s for level in ("level_1", "level_2") for s in manifest[level]["sessions"] if s.get("captures"))
    for relative in [*session["captures"].values(), session["output"], session.get("fresh")]:
        if relative:
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            (target / relative).write_bytes((manifest_path.parent / relative).read_bytes())
    victim = next(iter(session["captures"].values()))
    (target / victim).write_bytes((target / victim).read_bytes() + b"\x00" * 80 + b'{"type":"end"}\n')
    _, _, errors = validate.validate_harnesses(harness_dir)
    nul_errors = [e for e in errors if "contains NUL bytes" in e]
    assert nul_errors and victim in nul_errors[0], errors
