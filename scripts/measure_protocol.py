"""Record every adapter's wire exchange with the base-protocol fixture.

    uv run python scripts/measure_protocol.py [--spec-version REV] [--renderers a,b]

The base-protocol rows (the handshake, ping, per-request ``_meta``, ``server/discover``,
the JSON Schema dialect, ``resultType``, ``x-mcp-header``) are questions about what an
adapter puts on the wire, not about the definition it hands a model. So each renderer
runs in its pinned managed environment exactly as the scan does, but the server it is
given is the recorder (``src/mcp_mirror/recorder.py``) in front of
``fixtures/base_protocol_server.py``:

* over stdio, every process the adapter launches is a recorder with ``--ping``, each
  writing its own capture, so an adapter that opens one session per call is still
  recorded whole;
* over Streamable HTTP, one recorder serves every session the adapter opens, and keeps
  the request's ``mcp-*`` headers.

The adapter lists the tools and calls each tool in the fixture's ``CALLS``. Captures go
to ``<snapshot>/measurements/protocol/<renderer>/`` beside a ``run.json`` holding what
the adapter handed back (the rendered definitions and results). Nothing here decides a
support code; ``src/mcp_mirror/cells.py`` reads the captures.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "fixtures"))

FIXTURE = "fixtures/base_protocol_server.py"
RENDERERS = ("crewai", "langchain", "openai_agents", "pydantic_ai", "mastra", "google_adk", "ms_agent_framework", "strands")
TRANSPORTS = ("stdio", "http")


def _snapshot_dir(spec_version: str | None) -> Path:
    if not spec_version or spec_version == json.loads((ROOT / "data" / "frameworks.json").read_text())["mcp_spec"]:
        return ROOT / "data"
    return ROOT / "data" / "specs" / spec_version


def _shown(path: Path | str) -> str:
    try:
        return str(Path(os.path.abspath(path)).relative_to(ROOT))
    except ValueError:
        return str(path)


def record_one_session(directory: Path, fixture: list[str]) -> int:
    """Launcher mode: one stdio recorder with its own capture file in ``directory``."""

    from mcp_mirror.recorder import run

    # The fixture command is written relative to the checkout, as in every capture.
    os.chdir(ROOT)
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(1000):
        path = directory / f"stdio-{index:02d}.jsonl"
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        run(shlex.join(fixture), transport="stdio", out=path, ping=True, announce=lambda _text: None)
        return 0
    raise SystemExit(f"no free capture name in {directory}")


def _worker(renderer_id: str, server: str, omitted: tuple[str, ...] = ()) -> dict:
    from mcp_mirror.cli import _managed_import_root
    from mcp_mirror.runner_manifests import build_managed_worker_command
    from base_protocol_server import CALLS

    handle, out_path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    command = build_managed_worker_command(renderer_id, Path(out_path))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = _managed_import_root()
    # Every declared result boundary: a direct call and the call an agent makes can take
    # different sessions, and only one of them may have listed the tools first.
    calls = [call for call in CALLS if call["tool"] not in omitted]
    request = json.dumps({"server": server, "headers": {}, "calls": calls, "result_boundaries": ["*"]})
    proc = subprocess.run(
        command, input=request, capture_output=True, text=True, timeout=900, env=environment, cwd=ROOT
    )
    raw = Path(out_path).read_text()
    Path(out_path).unlink(missing_ok=True)
    if not raw.strip():
        return {"ok": False, "error": f"worker wrote nothing: {proc.stderr[-800:]}"}
    payload = json.loads(raw)
    if not payload.get("ok"):
        return {"ok": False, "error": payload.get("error")}
    return {
        "ok": True,
        "versions": payload.get("versions"),
        "tools": [{"name": rep["name"], "params": rep.get("params") or {}} for rep in payload.get("reps") or []],
        "result_boundaries": payload.get("result_boundaries") or [],
        "results": payload.get("results") or [],
    }


def _omitting(fixture: str, omitted: tuple[str, ...]) -> str:
    return " ".join([fixture, *(f"--omit {name}" for name in omitted)])


def _stdio(renderer_dir: Path, fixture: str, renderer_id: str, omitted: tuple[str, ...] = ()) -> dict:
    launcher = [sys.executable, str(Path(__file__).resolve()), "--record-into", str(renderer_dir.resolve()),
                "--", *shlex.split(_omitting(fixture, omitted))]
    return _worker(renderer_id, shlex.join(launcher), omitted)


def _http(renderer_dir: Path, fixture: str, renderer_id: str, omitted: tuple[str, ...] = ()) -> dict:
    capture = renderer_dir / ("http.jsonl" if not omitted else f"http-without-{'-'.join(omitted)}.jsonl")
    fixture = _omitting(fixture, omitted)
    recorder = subprocess.Popen(
        [sys.executable, "-m", "mcp_mirror.cli", "record", fixture, "--transport", "http",
         "--port", "0", "--out", _shown(capture)],
        cwd=ROOT, stderr=subprocess.PIPE, text=True,
    )
    url = None
    deadline = time.monotonic() + 60
    assert recorder.stderr is not None
    while time.monotonic() < deadline and url is None:
        line = recorder.stderr.readline()
        if not line and recorder.poll() is not None:
            break
        found = re.search(r"listening on (\S+)", line)
        url = found.group(1) if found else None
    if url is None:
        recorder.kill()
        return {"ok": False, "error": "the HTTP recorder never announced a URL"}
    try:
        return _worker(renderer_id, url, omitted)
    finally:
        recorder.send_signal(signal.SIGINT)
        try:
            recorder.wait(timeout=30)
        except subprocess.TimeoutExpired:
            recorder.kill()


def measure(transport: str, renderer_dir: Path, fixture: str, renderer_id: str) -> dict:
    """One transport's run. If the adapter cannot load the fixture at all, find the one
    tool it cannot load by leaving each out in turn, and measure the rest without it; the
    full run's error is kept beside the retry."""

    from base_protocol_server import TOOLS

    attempt = _stdio if transport == "stdio" else _http
    payload = attempt(renderer_dir, fixture, renderer_id)
    if payload.get("ok"):
        return payload
    for tool in TOOLS:
        retry = attempt(renderer_dir, fixture, renderer_id, (tool.name,))
        if retry.get("ok"):
            return {**retry, "omitted": [tool.name], "full_fixture_error": payload.get("error")}
    # An adapter can refuse more than one tool (LlamaIndex fails on both dialect tools),
    # so no single omission loads; then leave out each pair.
    for pair in itertools.combinations([tool.name for tool in TOOLS], 2):
        retry = attempt(renderer_dir, fixture, renderer_id, pair)
        if retry.get("ok"):
            return {**retry, "omitted": list(pair), "full_fixture_error": payload.get("error")}
    return payload


def describe(payload: dict) -> str:
    if not payload.get("ok"):
        # Spelled the way scripts/check_reproduce_commands.py greps for it.
        return f"FAILED: {payload.get('error')}"
    calls = ", ".join(
        f"{entry['tool']}@{entry.get('boundary', 'default')}={'ok' if entry.get('ok') else 'raised'}" for entry in payload.get("results") or []
    )
    omitted = f" (without {', '.join(payload['omitted'])})" if payload.get("omitted") else ""
    return f"{len(payload.get('tools') or [])} tools{omitted}; {calls}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-version", default=None, help="protocol revision the fixture serves")
    parser.add_argument("--fixture-python", default=os.environ.get("MCP_MIRROR_FIXTURE_PYTHON", sys.executable))
    parser.add_argument("--renderers", default=",".join(RENDERERS))
    parser.add_argument("--transports", default=",".join(TRANSPORTS))
    parser.add_argument("--out", default=None, help="directory for the captures")
    parser.add_argument("--record-into", default=None, help=argparse.SUPPRESS)
    args, rest = parser.parse_known_args()
    if args.record_into:
        return record_one_session(Path(args.record_into), [a for a in rest if a != "--"])
    if args.spec_version:
        sys.path.insert(0, str(ROOT / "scripts"))
        from _fixture import fixture_python

        args.fixture_python = fixture_python(args.spec_version)
    fixture = f"{_shown(args.fixture_python)} {FIXTURE}"
    out = Path(args.out) if args.out else _snapshot_dir(args.spec_version) / "measurements" / "protocol"
    transports = [t for t in args.transports.replace(" ", "").split(",") if t]
    print(f"fixture: {fixture}")
    for renderer_id in [r for r in args.renderers.replace(" ", "").split(",") if r]:
        renderer_dir = out / renderer_id
        if renderer_dir.exists():
            shutil.rmtree(renderer_dir)
        renderer_dir.mkdir(parents=True)
        print(f"\n=== {renderer_id} ===")
        runs: dict[str, dict] = {}
        for transport in transports:
            payload = measure(transport, renderer_dir, fixture, renderer_id)
            runs[transport] = payload
            print(f"  {transport:<6} {describe(payload)}")
        versions = next((p.get("versions") for p in runs.values() if p.get("versions")), None)
        (renderer_dir / "run.json").write_text(
            json.dumps(
                {
                    "schema": "mcp-mirror/protocol-run@1",
                    "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "renderer": renderer_id,
                    "fixture": FIXTURE,
                    "spec_version": args.spec_version,
                    "versions": versions,
                    "captures": sorted(p.name for p in renderer_dir.glob("*.jsonl")),
                    "transports": runs,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        print(f"  wrote {_shown(renderer_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
