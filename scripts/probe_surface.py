"""Ask every adapter for the fixture's resources and prompts, and record what came back.

    uv run python scripts/probe_surface.py [--fixture-python PY] [--renderers a,b] [--out FILE]

Resources and prompts are not measured by diffing a tool definition. The question is
whether the adapter gives an agent a route to them and what survives the trip: does a
read keep its URI and MIME type, does a binary resource decode to the same bytes, do
prompt arguments keep their required flags. Every adapter runs the same six calls (see
``src/mcp_mirror/probe.py``) in its pinned environment, and the server is asked the
same questions through the protocol-matched SDK so the two can be compared.

Each adapter reaches the fixture through the wire recorder (``mcp-mirror record``), so
the capture also keeps what the adapter declared about itself when it opened the
session: its client capabilities, and in them any ``extensions`` it negotiates. The
recorder forwards every message unchanged.

The capture is written as a measurement artifact the cell generator reads; nothing
here decides a support code.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp_mirror.cli import _managed_import_root  # noqa: E402
from mcp_mirror.runner_manifests import build_managed_worker_command  # noqa: E402

RENDERERS = ("crewai", "langchain", "openai_agents", "pydantic_ai", "mastra")


def capture_source(fixture_python: str, server: str) -> dict:
    script = (
        "import json,sys;"
        "sys.path.insert(0, %r);"
        "from mcp_mirror.source import parse_server, probe_source;"
        "print(json.dumps(probe_source(parse_server(%r))))" % (str(ROOT / "src"), server)
    )
    proc = subprocess.run([fixture_python, "-c", script], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise SystemExit(f"source probe failed:\n{proc.stderr[-1500:]}")
    return json.loads(proc.stdout)


def recorded(server: str, capture: Path) -> str:
    """The server command wrapped in the wire recorder from this project's environment."""
    recorder = Path(sys.executable).with_name("mcp-mirror.exe" if os.name == "nt" else "mcp-mirror")
    return shlex.join([str(recorder), "record", server, "--out", str(capture)])


# What the capture keeps from each handshake the recorder saw. Client info is left out:
# it names the adapter, which the capture already does.
NEGOTIATION_KEYS = ("via", "requested_revision", "negotiated_revision", "client_capabilities", "server_capabilities")


def negotiations(capture: Path) -> list[dict]:
    """Every handshake in the capture. An adapter that opens a session per call starts
    one recorder per session on the same path; the last to open it wins, which still
    holds a whole handshake, and a line cut short by a concurrent writer is skipped."""
    if not capture.exists():
        return []
    records = []
    for line in capture.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return [
        {key: record.get(key) for key in NEGOTIATION_KEYS}
        for record in records
        if record.get("type") == "negotiation"
    ]


def capture_managed(renderer_id: str, server: str) -> dict:
    handle, out_path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    handle, wire_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(handle)
    command = build_managed_worker_command(renderer_id, Path(out_path))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = _managed_import_root()
    request = json.dumps({"server": recorded(server, Path(wire_path)), "headers": {}, "probe": True})
    proc = subprocess.run(command, input=request, capture_output=True, text=True, timeout=900, env=environment)
    raw = Path(out_path).read_text()
    Path(out_path).unlink(missing_ok=True)
    wire = negotiations(Path(wire_path))
    Path(wire_path).unlink(missing_ok=True)
    if not raw.strip():
        raise SystemExit(f"{renderer_id}: worker wrote nothing\n{proc.stderr[-1500:]}")
    payload = json.loads(raw)
    if not payload.get("ok"):
        raise SystemExit(f"{renderer_id}: {payload.get('error')}")
    return {"ok": True, "versions": payload.get("versions"), "probe": payload.get("probe"), "wire": {"negotiations": wire}}


def describe(feature: str, answer: dict) -> str:
    if answer.get("ok"):
        keys = {k: v for k, v in answer.items() if k != "ok"}
        if "entries" in keys:
            return f"{len(keys['entries'])} entries"
        if "messages" in keys:
            return f"{len(keys['messages'])} messages"
        return ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in keys.items() if k in ("uri", "mimeType", "text", "blob_sha256"))
    return f"{'unsupported' if 'unsupported' in answer else 'FAILED:'} {answer.get('unsupported') or answer.get('error')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-version", default=None, help="protocol revision the fixture serves")
    parser.add_argument("--fixture-python", default=os.environ.get("MCP_MIRROR_FIXTURE_PYTHON", sys.executable))
    parser.add_argument("--renderers", default=",".join(RENDERERS))
    parser.add_argument("--out", default=str(ROOT / "data" / "measurements" / "probe" / "all-frameworks.json"))
    args = parser.parse_args()
    if args.spec_version:
        sys.path.insert(0, str(ROOT / "scripts"))
        from _fixture import fixture_python

        args.fixture_python = fixture_python(args.spec_version)
    server = f"{args.fixture_python} fixtures/tricky_server.py"
    print(f"fixture: {server}")
    print("\n=== source ===")
    source = capture_source(args.fixture_python, server)
    for feature, answer in source.items():
        print(f"    {feature:<22} {describe(feature, answer)}")
    captured: dict[str, dict] = {}
    for renderer_id in [r for r in args.renderers.replace(" ", "").split(",") if r]:
        print(f"\n=== {renderer_id} ===")
        payload = capture_managed(renderer_id, server)
        captured[renderer_id] = payload
        if not payload.get("ok"):
            print(f"  unavailable: {payload.get('error')}")
            continue
        print(f"  version {(payload.get('versions') or {}).get('framework', '?')}")
        for feature, answer in (payload.get("probe") or {}).items():
            print(f"    {feature:<22} {describe(feature, answer)}")
        declared = sorted(
            {name for n in payload["wire"]["negotiations"] for name in ((n.get("client_capabilities") or {}).get("extensions") or {})}
        )
        print(f"    {'handshakes':<22} {len(payload['wire']['negotiations'])}, extensions declared: {declared or 'none'}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema": "mcp-mirror/surface-probe@1",
                "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "fixture": "fixtures/tricky_server.py",
                "fixture_python": args.fixture_python,
                "spec_version": args.spec_version,
                "source": source,
                "frameworks": captured,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out_path.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
