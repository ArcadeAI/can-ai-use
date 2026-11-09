"""Run the definition, result and probe instruments over every transport, and record both ends.

The scan, result-boundary and probe captures are all taken over stdio. Whether an adapter
reaches the same fixture as well over HTTP is a separate question, and the transport
rows answer it by asking every renderer the same things over each transport and keeping
what came back beside the wire the recorder saw:

* ``stdio``: the fixture served through ``mcp-mirror record`` as the server command.
* ``http``: the same fixture behind ``mcp-mirror record --transport http`` (Streamable
  HTTP; the recorder mints an ``Mcp-Session-Id`` unless the revision has none).
* ``sse``: the same fixture behind ``python -m mcp_mirror.legacy_sse``, the deprecated
  HTTP+SSE transport. Only for revisions that still define it.

    uv run python scripts/measure_transports.py
    uv run python scripts/measure_transports.py --spec-version 2026-07-28 \\
        --out data/specs/2026-07-28/measurements/transports/all-frameworks.json

Each renderer makes two requests per transport, in its pinned managed environment:
``fixtures/tricky_server.py`` for tool definitions and the resource and prompt probe,
and ``fixtures/results_server.py`` for every result call at every boundary. The capture
keeps each answer normalised (no raw reprs, no ports, no object addresses) so the cell
rules in ``src/mcp_mirror/cells.py`` can compare transports field by field, and writes
the recorder's wire logs under ``wire/`` beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mcp_mirror.cli import _managed_import_root  # noqa: E402
from mcp_mirror.runner_manifests import build_managed_worker_command  # noqa: E402
from measure_result_boundaries import CALLS  # noqa: E402

RENDERERS = ("crewai", "langchain", "openai_agents", "pydantic_ai", "google_adk", "mastra", "strands")
FIXTURES = {
    "definitions": "fixtures/tricky_server.py",
    "results": "fixtures/results_server.py",
}
# The first revision that no longer defines the HTTP+SSE transport.
SSE_REMOVED = "2026-07-28"
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_URL = re.compile(r"http://127\.0\.0\.1:\d+")
# An adapter may open a fresh stdio connection per call, and each one launches the
# recorder anew, so each writes its own part; the parts are joined in order afterwards.
_STDIO_RECORDER = (
    "import sys, time; from pathlib import Path; from mcp_mirror.recorder import run; "
    "run(sys.argv[1], out=Path(f'{sys.argv[2]}.{time.time_ns()}.part'))"
)


def _requests(server: str) -> dict[str, dict]:
    return {
        "definitions": {"server": server, "headers": {}, "probe": True},
        "results": {"server": server, "headers": {}, "calls": CALLS, "result_boundaries": ["*"]},
    }


def _scrub(value, replacements: list[tuple[str, str]]):
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return _URL.sub("http://127.0.0.1:<port>", _ADDRESS.sub("", value))
    if isinstance(value, list):
        return [_scrub(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _scrub(item, replacements) for key, item in value.items()}
    return value


def normalise(payload: dict, replacements: list[tuple[str, str]]) -> dict:
    """What a transport could change: definitions, results, probe answers. Raw reprs and
    renderer evidence are dropped, since they name objects rather than content."""

    out: dict = {}
    if payload.get("reps") is not None:
        out["tools"] = [
            {key: rep.get(key) for key in ("name", "description", "params", "annotations", "icons")}
            for rep in payload.get("reps") or []
        ]
    if payload.get("results") is not None:
        results = []
        for entry in payload.get("results") or []:
            item = {"tool": entry.get("tool"), "boundary": entry.get("boundary"), "ok": entry.get("ok")}
            if entry.get("ok"):
                rep = entry.get("rep") or {}
                item["rep"] = {
                    key: rep.get(key)
                    for key in ("blocks", "block_kinds", "structured_content", "is_error", "text", "error_channel")
                }
            else:
                item["error"] = entry.get("error")
            results.append(item)
        out["results"] = results
    if payload.get("probe") is not None:
        out["probe"] = payload.get("probe")
    return _scrub(out, replacements)


class Served:
    """A fixture behind the recorder (stdio or Streamable HTTP) or the legacy SSE bridge."""

    def __init__(self, transport: str, fixture: str, capture: Path) -> None:
        self.transport = transport
        self.capture = capture
        self.process: subprocess.Popen | None = None
        capture.parent.mkdir(parents=True, exist_ok=True)
        capture.unlink(missing_ok=True)
        python = sys.executable
        if transport == "stdio":
            # The recorder is the server command the adapter launches.
            self.server = shlex.join([python, "-c", _STDIO_RECORDER, fixture, str(capture)])
            return
        if transport == "http":
            argv = [python, "-m", "mcp_mirror.cli", "record", fixture, "--transport", "http", "--out", str(capture)]
        else:
            argv = [python, "-m", "mcp_mirror.legacy_sse", fixture, "--out", str(capture)]
        self.process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 60
        url = None
        assert self.process.stderr is not None
        while time.monotonic() < deadline:
            line = self.process.stderr.readline()
            if not line:
                break
            found = re.search(r"listening on (\S+)", line)
            if found:
                url = found.group(1)
                break
        if url is None:
            self.stop()
            raise SystemExit(f"FAILED: the {transport} server for {fixture} never announced a URL")
        self.server = f"sse+{url}" if transport == "sse" else url
        # The fixtures inherit this pipe; left unread it fills and every fixture blocks.
        threading.Thread(target=self.process.stderr.read, daemon=True).start()

    def stop(self) -> None:
        if self.process is None:
            parts = sorted(self.capture.parent.glob(f"{self.capture.name}.*.part"))
            if parts:
                self.capture.write_text("".join(part.read_text() for part in parts))
                for part in parts:
                    part.unlink()
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None


def run_worker(renderer_id: str, request: dict) -> dict:
    handle, out_path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    command = build_managed_worker_command(renderer_id, Path(out_path))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = _managed_import_root()
    try:
        proc = subprocess.run(
            command, input=json.dumps(request), capture_output=True, text=True, timeout=600,
            env=environment, cwd=ROOT,
        )
    except subprocess.TimeoutExpired:
        Path(out_path).unlink(missing_ok=True)
        return {"ok": False, "error": "worker timed out after 600s"}
    raw = Path(out_path).read_text()
    Path(out_path).unlink(missing_ok=True)
    if not raw.strip():
        return {"ok": False, "error": f"worker wrote nothing: {proc.stderr.strip()[-600:]}"}
    return json.loads(raw)


def scrub_capture(path: Path, replacements: list[tuple[str, str]]) -> None:
    if not path.exists():
        return
    text = path.read_text()
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec-version", default=None, help="protocol revision the fixture serves")
    parser.add_argument("--fixture-python", default=os.environ.get("MCP_MIRROR_FIXTURE_PYTHON", sys.executable))
    parser.add_argument("--renderers", default=",".join(RENDERERS))
    parser.add_argument("--transports", default=None,
                        help="comma-separated: stdio,http,sse (default: every transport the revision defines)")
    parser.add_argument("--out", default=str(ROOT / "data" / "measurements" / "transports" / "all-frameworks.json"))
    args = parser.parse_args()
    if args.spec_version:
        from _fixture import fixture_python

        args.fixture_python = fixture_python(args.spec_version)
    revision = args.spec_version or "2025-11-25"
    transports = (args.transports.split(",") if args.transports
                  else ["stdio", "http"] + (["sse"] if revision < SSE_REMOVED else []))
    renderers = [r for r in args.renderers.replace(" ", "").split(",") if r]
    out_path = Path(args.out).resolve()
    wire_dir = out_path.parent / "wire"
    replacements = [(str(ROOT) + "/", ""), (str(ROOT), "."), (str(Path.home()), "~")]

    command = "uv run python scripts/measure_transports.py"
    if args.spec_version:
        command += f" --spec-version {args.spec_version}"
    if args.renderers != ",".join(RENDERERS):
        command += f" --renderers {','.join(renderers)}"
    if args.transports:
        command += f" --transports {args.transports}"
    if out_path != (ROOT / "data" / "measurements" / "transports" / "all-frameworks.json"):
        command += f" --out {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}"

    frameworks: dict[str, dict] = {}
    for renderer_id in renderers:
        print(f"\n=== {renderer_id} ===", flush=True)
        runs: dict[str, dict] = {}
        for transport in transports:
            runs[transport] = {}
            for kind, fixture_path in FIXTURES.items():
                fixture = shlex.join([args.fixture_python, str(ROOT / fixture_path)])
                capture = wire_dir / f"{renderer_id}-{Path(fixture_path).stem}-{transport}.jsonl"
                served = Served(transport, fixture, capture)
                try:
                    payload = run_worker(renderer_id, _requests(served.server)[kind])
                finally:
                    served.stop()
                scrub_capture(capture, replacements)
                entry = {
                    "ok": bool(payload.get("ok")),
                    "fixture": fixture_path,
                    "capture": str(capture.relative_to(out_path.parent)),
                }
                if payload.get("ok"):
                    entry["version"] = (payload.get("versions") or {}).get("framework")
                    entry["answers"] = normalise(payload, replacements)
                    failed = [r for r in entry["answers"].get("results") or [] if not r.get("ok")]
                    detail = f"{len(entry['answers'].get('tools') or [])} tools"
                    if kind == "results":
                        detail = f"{len(entry['answers']['results']) - len(failed)}/{len(entry['answers']['results'])} calls answered"
                else:
                    entry["error"] = _scrub(payload.get("error") or "", replacements)
                    detail = f"no session: {entry['error'][:160]}"
                runs[transport][kind] = entry
                print(f"  {transport:<6} {kind:<12} {detail}", flush=True)
        frameworks[renderer_id] = {"transports": runs}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "schema": "mcp-mirror/transports@1",
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spec_version": revision,
        "fixtures": FIXTURES,
        "transports": transports,
        "command": command,
        "frameworks": frameworks,
    }, indent=2) + "\n")
    print(f"\nwrote {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
