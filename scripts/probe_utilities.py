"""Ask every adapter to use the fixture's server utilities, and record the wire.

    uv run python scripts/measure_locked.py -- \\
        uv run python scripts/probe_utilities.py [--spec-version REV] [--renderers a,b] [--out FILE]

Pagination, progress, logging, completion, cancellation, tasks and caching are not
properties of a tool definition. They are things a client does, or fails to do, while
talking to the server. So each question gets its own fixture process behind the
recording proxy (``mcp-mirror record``), the adapter is driven through its own public
API (see ``probe_utilities`` in each renderer and ``src/mcp_mirror/probe.py``), and the
capture keeps both halves: what the server saw on the wire, and what the adapter handed
its caller.

The fixture's own catalogue (every tool name across all pages, the completion values it
would give) is read from the fixture module with the interpreter that serves it, so the
comparison does not depend on any client.

The capture is written as a measurement artifact the cell generator reads; nothing here
decides a support code.
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
from mcp_mirror.probe import UTILITY_FEATURES  # noqa: E402
from mcp_mirror.recorder import read_capture  # noqa: E402
from mcp_mirror.runner_manifests import build_managed_worker_command  # noqa: E402

RENDERERS = ("crewai", "langchain", "openai_agents", "pydantic_ai", "mastra")
FIXTURE = "fixtures/utilities_server.py"

SOURCE_SCRIPT = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, sys.argv[1])
import utilities_server as U
from _dual import SDK2
colour, record = sys.argv[2], sys.argv[3]
print(json.dumps({
    "sdk": 2 if SDK2 else 1,
    "tools": [t.name for t in U.TOOLS],
    "page_size": U.PAGE_SIZE,
    "progress_steps": U.PROGRESS_STEPS,
    "log_levels": U.LOG_LEVELS,
    "completion": {
        "prompt": [v for v in U.COLOURS if v.startswith(colour)],
        "resource": [v for v in U.RECORD_IDS if v.startswith(record)],
    },
    "cache_ttl_ms": U.CACHE_TTL_MS if SDK2 else None,
    "tasks_extension": U.TASKS_EXTENSION,
}))
"""


def capture_source(fixture_python: str) -> dict:
    from mcp_mirror.probe import COMPLETION_PROMPT, COMPLETION_RESOURCE

    proc = subprocess.run(
        [fixture_python, "-c", SOURCE_SCRIPT, str(ROOT / "fixtures"), COMPLETION_PROMPT[2], COMPLETION_RESOURCE[2]],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise SystemExit(f"source read failed:\n{proc.stderr[-1500:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _reduce(record: dict) -> dict | None:
    """A wire record small enough to publish: list results keep names, not definitions."""
    if record.get("type") not in ("message", "negotiation"):
        return None
    record = dict(record)
    message = record.get("message")
    if isinstance(message, dict) and isinstance(message.get("result"), dict):
        result = dict(message["result"])
        for key, identity in (("tools", "name"), ("resources", "uri"), ("prompts", "name")):
            if isinstance(result.get(key), list):
                result[key] = [item.get(identity) for item in result[key] if isinstance(item, dict)]
        record["message"] = {**message, "result": result}
    return record


def _recorded(fixture_command: str, prefix: Path) -> str:
    """A server command that records each process it starts into its own capture.

    An adapter may open more than one session for one question (LangChain's
    ``get_tools`` opens one per call), and the recorder truncates its ``--out`` file, so
    every launch gets a fresh ``<prefix>-XXXXXX`` file instead of sharing one.
    """
    record = " ".join(
        [shlex.quote(sys.executable), "-m", "mcp_mirror.cli", "record", shlex.quote(fixture_command), "--out"]
    )
    return f"/bin/sh -c {shlex.quote(f'exec {record} \"$(mktemp {shlex.quote(str(prefix))}-XXXXXX)\"')}"


def _wire(prefix: Path) -> list[dict]:
    """Every capture one question produced, in launch order, each record tagged with it."""
    captures = [read_capture(path) for path in sorted(prefix.parent.glob(f"{prefix.name}-*"))]
    captures = sorted((c for c in captures if c), key=lambda records: records[0].get("t") or "")
    wire = []
    for index, records in enumerate(captures):
        for record in records:
            reduced = _reduce(record)
            if reduced is not None:
                wire.append({"capture": index, **reduced})
    return wire


MODERN = "2026-07-28"


def questions(spec_version: str | None) -> tuple[str, ...]:
    """The questions a revision defines: ``logging/setLevel`` is gone at 2026-07-28 and
    caching hints arrive with it, so each is asked only where it exists."""
    modern = (spec_version or "") >= MODERN
    skip = {"set_level"} if modern else {"caching"}
    return tuple(q for q in UTILITY_FEATURES if q not in skip)


def capture_managed(renderer_id: str, fixture_command: str, workdir: Path, asked: tuple[str, ...]) -> dict:
    prefixes = {feature: workdir / f"{renderer_id}-{feature}" for feature in asked}
    recorded = {feature: _recorded(fixture_command, prefix) for feature, prefix in prefixes.items()}
    handle, out_path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    command = build_managed_worker_command(renderer_id, Path(out_path))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = _managed_import_root()
    request = json.dumps({"server": fixture_command, "headers": {}, "utilities": recorded})
    proc = subprocess.run(command, input=request, capture_output=True, text=True, timeout=900, env=environment)
    raw = Path(out_path).read_text()
    Path(out_path).unlink(missing_ok=True)
    if not raw.strip():
        raise SystemExit(f"{renderer_id}: worker wrote nothing\n{proc.stderr[-1500:]}")
    payload = json.loads(raw)
    if not payload.get("ok"):
        raise SystemExit(f"{renderer_id}: {payload.get('error')}")
    wire = {feature: _wire(prefix) for feature, prefix in prefixes.items()}
    return {"ok": True, "versions": payload.get("versions"), "utilities": payload.get("utilities"), "wire": wire}


def describe(answer: dict, wire: list[dict]) -> str:
    sent = sum(1 for r in wire if r.get("direction") == "client_to_server" and r.get("kind") != "response")
    if answer.get("ok"):
        keys = [k for k in answer if k != "ok"]
        return f"answered ({', '.join(keys)}); client sent {sent} messages"
    kind = "unsupported" if "unsupported" in answer else "FAILED:"
    return f"{kind} {answer.get('unsupported') or answer.get('error')}; client sent {sent} messages"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-version", default=None, help="protocol revision the fixture serves")
    parser.add_argument("--fixture-python", default=os.environ.get("MCP_MIRROR_FIXTURE_PYTHON", sys.executable))
    parser.add_argument("--renderers", default=",".join(RENDERERS))
    parser.add_argument("--out", default=str(ROOT / "data" / "measurements" / "utilities" / "all-frameworks.json"))
    args = parser.parse_args()
    if args.spec_version:
        sys.path.insert(0, str(ROOT / "scripts"))
        from _fixture import fixture_python

        args.fixture_python = fixture_python(args.spec_version)
    fixture_command = f"{shlex.quote(args.fixture_python)} {shlex.quote(str(ROOT / FIXTURE))}"
    print(f"fixture: {fixture_command}")
    source = capture_source(args.fixture_python)
    print(f"source: {len(source['tools'])} tools in pages of {source['page_size']}, SDK {source['sdk']}.x")
    asked = questions(args.spec_version)
    captured: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="mcp-mirror-utilities-") as scratch:
        for renderer_id in [r for r in args.renderers.replace(" ", "").split(",") if r]:
            print(f"\n=== {renderer_id} ===")
            payload = capture_managed(renderer_id, fixture_command, Path(scratch), asked)
            captured[renderer_id] = payload
            print(f"  version {(payload.get('versions') or {}).get('framework', '?')}")
            for feature in asked:
                answer = (payload.get("utilities") or {}).get(feature) or {}
                print(f"    {feature:<13} {describe(answer, payload['wire'].get(feature) or [])}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema": "mcp-mirror/utilities-probe@1",
                "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "fixture": FIXTURE,
                "fixture_python": args.fixture_python,
                "spec_version": args.spec_version,
                "source": source,
                "frameworks": captured,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
