"""Answer the fixture's sampling, elicitation and roots requests through every adapter.

    uv run python scripts/probe_client_features.py [--spec-version REV] [--renderers a,b] [--out FILE]

These are requests the server sends to the client, so the question is not what an
adapter shows the agent but whether it answers at all. Each adapter registers the hooks
it offers (see ``probe_client_features`` on each renderer), calls the fixture tools in
``fixtures/client_features_server.py`` that trigger each request, and the tool result
records what the server got back. At 2026-07-28 the same tools return an
``InputRequiredResult`` instead, so the capture measures whether the adapter answers
the embedded request and retries.

The capture is written as a measurement artifact the cell generator reads; nothing
here decides a support code.
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
from mcp_mirror.runner_manifests import build_managed_worker_command  # noqa: E402

RENDERERS = ("crewai", "langchain", "openai_agents", "pydantic_ai", "mastra")
FIXTURE = "fixtures/client_features_server.py"


def capture_managed(renderer_id: str, server: str) -> dict:
    handle, out_path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    command = build_managed_worker_command(renderer_id, Path(out_path))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = _managed_import_root()
    request = json.dumps({"server": server, "headers": {}, "client_probe": True})
    proc = subprocess.run(command, input=request, capture_output=True, text=True, timeout=900, env=environment)
    raw = Path(out_path).read_text()
    Path(out_path).unlink(missing_ok=True)
    if not raw.strip():
        raise SystemExit(f"{renderer_id}: worker wrote nothing\n{proc.stderr[-1500:]}")
    payload = json.loads(raw)
    if not payload.get("ok"):
        raise SystemExit(f"{renderer_id}: {payload.get('error')}")
    return {
        "ok": True,
        "versions": payload.get("versions"),
        "negotiated_mcp_spec_version": (payload.get("evidence") or {}).get("negotiated_mcp_spec_version"),
        "client_probe": payload.get("client_probe"),
    }


def describe(entry: dict) -> str:
    text = (entry.get("text") or "").replace("\n", " ")
    hook = entry.get("hook") or "no hook"
    return f"[{hook}] {text[:110]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-version", default=None, help="protocol revision the fixture serves")
    parser.add_argument("--fixture-python", default=os.environ.get("MCP_MIRROR_FIXTURE_PYTHON", sys.executable))
    parser.add_argument("--renderers", default=",".join(RENDERERS))
    parser.add_argument("--out", default=str(ROOT / "data" / "measurements" / "client" / "all-frameworks.json"))
    args = parser.parse_args()
    if args.spec_version:
        sys.path.insert(0, str(ROOT / "scripts"))
        from _fixture import fixture_python

        args.fixture_python = fixture_python(args.spec_version)
    server = f"{args.fixture_python} {FIXTURE}"
    print(f"fixture: {server}")
    captured: dict[str, dict] = {}
    for renderer_id in [r for r in args.renderers.replace(" ", "").split(",") if r]:
        print(f"\n=== {renderer_id} ===")
        payload = capture_managed(renderer_id, server)
        captured[renderer_id] = payload
        print(
            f"  version {(payload.get('versions') or {}).get('framework', '?')}, "
            f"negotiated {payload.get('negotiated_mcp_spec_version')}"
        )
        for feature, entry in (payload.get("client_probe") or {}).items():
            if isinstance(entry, dict) and "tool" in entry:
                print(f"    {feature:<20} {describe(entry)}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema": "mcp-mirror/client-probe@1",
                "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "fixture": FIXTURE,
                "fixture_python": args.fixture_python,
                "spec_version": args.spec_version,
                "frameworks": captured,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
