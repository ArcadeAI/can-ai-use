"""Capture every framework's tool-result boundaries, not just one per framework.

``measure_results.py`` asks each adapter for a result once, at whichever boundary the
renderer captures by default. That is enough when a framework has one answer, and
misleading when it has two: LangChain returns the same failed call as plain content
blocks through a direct invocation and as a ``ToolMessage`` with ``status="error"``
through the invocation an agent makes.

This script asks every renderer at every boundary it declares, so a published cell can
cite what was observed at each one rather than at whichever was measured first. Each
renderer runs in its pinned managed environment, exactly as the definition scan does.

    uv run python scripts/measure_result_boundaries.py

Writes the full capture to ``--out`` (default
``data/measurements/results/all-frameworks.json``, which the cell generator reads) and
prints one line per framework, boundary and call.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp_mirror.cli import _managed_import_root  # noqa: E402
from mcp_mirror.models import ResultRep  # noqa: E402
from mcp_mirror.runner_manifests import build_managed_worker_command  # noqa: E402

FIXTURE = "fixtures/results_server.py"

# One call per content kind, so an adapter that raises on one kind (LangChain on
# audio) cannot hide how it treats the others. The mixed result, the tool-reported
# error and the text-plus-structured result are measured as their own calls, because
# what an adapter does with a combination is a separate observation. Every tool on the
# fixture takes no arguments.
CALLS = [
    {"tool": tool, "arguments": {}}
    for tool in (
        "text_content",
        "image_content",
        "audio_content",
        "embedded_resource_content",
        "resource_link_content",
        "annotated_content",
        "mixed_content",
        "text_and_structured",
        "tool_error",
    )
]

RENDERERS = ("crewai", "langchain", "openai_agents", "pydantic_ai", "mastra")


def _fixture_command(fixture_python: str) -> str:
    return f"{fixture_python} {FIXTURE}"


def capture_managed(renderer_id: str, server: str) -> dict:
    """Run one renderer in its manifest environment at every boundary."""

    handle, out_path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    command = build_managed_worker_command(renderer_id, Path(out_path))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = _managed_import_root()
    request = json.dumps(
        {"server": server, "headers": {}, "calls": CALLS, "result_boundaries": ["*"]}
    )

    proc = subprocess.run(
        command, input=request, capture_output=True, text=True, timeout=900, env=environment
    )
    raw = Path(out_path).read_text()
    Path(out_path).unlink(missing_ok=True)
    if not raw.strip():
        raise SystemExit(f"{renderer_id}: worker wrote nothing\n{proc.stderr[-1500:]}")
    payload = json.loads(raw)
    if not payload.get("ok"):
        raise SystemExit(f"{renderer_id}: {payload.get('error')}")
    return payload


def capture_source(fixture_python: str, server: str) -> dict[str, dict]:
    """Call each tool directly on the fixture through the protocol-matched SDK.

    The interpreter that runs the fixture is the one that speaks its revision, so the
    source capture runs there too; the project venv may pin a different SDK line.
    """
    script = (
        "import json,sys;"
        "sys.path.insert(0, %r);"
        "from mcp_mirror.source import parse_server, call_source_tool;"
        "h=parse_server(%r);"
        "print(json.dumps({c['tool']: call_source_tool(h, c['tool'], c['arguments']).model_dump(mode='json')"
        " for c in %r}))" % (str(ROOT / "src"), server, CALLS)
    )
    proc = subprocess.run(
        [fixture_python, "-c", script], capture_output=True, text=True, timeout=300
    )
    if proc.returncode != 0:
        raise SystemExit(f"source capture failed:\n{proc.stderr[-1500:]}")
    return json.loads(proc.stdout)


def describe(entry: dict) -> str:
    if not entry.get("ok"):
        # Spelled the way scripts/check_reproduce_commands.py greps for it, so a
        # capture that dies still fails the published-command gate.
        return f"FAILED: {entry.get('error')}"
    rep = ResultRep(**entry["rep"])
    return (
        f"kinds={rep.block_kinds} structured={rep.structured_content is not None} "
        f"isError={rep.is_error} text={(rep.text or '')[:60]!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec-version",
        default=None,
        help="protocol revision the fixture serves; builds a matching interpreter "
        "under .mcp-mirror/ on first use",
    )
    parser.add_argument(
        "--fixture-python",
        default=os.environ.get("MCP_MIRROR_FIXTURE_PYTHON", sys.executable),
        help="interpreter that runs the results fixture, which fixes the "
        "protocol revision the adapters negotiate against",
    )
    parser.add_argument(
        "--renderers",
        default=",".join(RENDERERS),
        help="comma-separated renderer ids to capture",
    )
    parser.add_argument(
        "--out", default=str(ROOT / "data" / "measurements" / "results" / "all-frameworks.json")
    )
    args = parser.parse_args()
    if args.spec_version:
        sys.path.insert(0, str(ROOT / "scripts"))
        from _fixture import fixture_python

        args.fixture_python = fixture_python(args.spec_version)

    requested = [r for r in args.renderers.replace(" ", "").split(",") if r]
    server = _fixture_command(args.fixture_python)
    print(f"fixture: {server}")

    print("\n=== source ===")
    source = capture_source(args.fixture_python, server)
    for tool, rep in source.items():
        print(f"    {tool:<20} {describe({'ok': True, 'rep': rep})}")
    captured: dict[str, dict] = {}
    for renderer_id in requested:
        print(f"\n=== {renderer_id} ===")
        payload = capture_managed(renderer_id, server)
        captured[renderer_id] = payload
        if not payload.get("ok"):
            print(f"  unavailable: {payload.get('error')}")
            continue
        print(f"  version {(payload.get('versions') or {}).get('framework', '?')}")
        for boundary in payload.get("result_boundaries") or []:
            marker = "agent path" if boundary["agent_path"] else "not the agent path"
            print(f"  boundary {boundary['id']:<28} {boundary['capture_api']}  [{marker}]")
        for entry in payload.get("results") or []:
            label = f"{entry['tool']} @ {entry.get('boundary', 'default')}"
            print(f"    {label:<52} {describe(entry)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema": "mcp-mirror/result-boundaries@2",
                "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "fixture": FIXTURE,
                "fixture_python": args.fixture_python,
                "spec_version": args.spec_version,
                "calls": CALLS,
                "source": source,
                "frameworks": captured,
            },
            indent=2,
        )
        + "\n"
    )
    try:
        shown = out_path.resolve().relative_to(ROOT)
    except ValueError:
        shown = out_path
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
