"""Change the fixture's catalogue mid-session and record whether each adapter follows.

    uv run python scripts/probe_notifications.py [--spec-version REV] [--renderers a,b] [--out FILE]

fixtures/notify_server.py has tools that add a tool, a resource or a prompt, or change a
watched resource, and then fire the matching notification. Every adapter runs the same
sequence through its own API (see ``src/mcp_mirror/notify.py``). It lists, mutates,
waits for any hook it exposes to hear the notification, and lists again. Then it
subscribes to the watched resource and unsubscribes through whatever route it offers.
The same sequence runs straight through the protocol-matched SDK as the control.

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
FIXTURE = "fixtures/notify_server.py"


def capture_source(fixture_python: str, server: str) -> dict:
    script = (
        "import json,sys,warnings;"
        "warnings.simplefilter('ignore');"
        "sys.path.insert(0, %r);"
        "from mcp_mirror.source import parse_server;"
        "from mcp_mirror.notify import probe_source;"
        "print(json.dumps(probe_source(parse_server(%r))))" % (str(ROOT / "src"), server)
    )
    proc = subprocess.run([fixture_python, "-c", script], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise SystemExit(f"source probe failed:\n{proc.stderr[-1500:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def capture_managed(renderer_id: str, server: str) -> dict:
    handle, out_path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    command = build_managed_worker_command(renderer_id, Path(out_path))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = _managed_import_root()
    request = json.dumps({"server": server, "headers": {}, "notify": True})
    proc = subprocess.run(command, input=request, capture_output=True, text=True, timeout=900, env=environment)
    raw = Path(out_path).read_text()
    Path(out_path).unlink(missing_ok=True)
    if not raw.strip():
        raise SystemExit(f"{renderer_id}: worker wrote nothing\n{proc.stderr[-1500:]}")
    payload = json.loads(raw)
    if not payload.get("ok"):
        raise SystemExit(f"{renderer_id}: {payload.get('error')}")
    return {"ok": True, "versions": payload.get("versions"), "notify": payload.get("notify")}


def describe(answer: dict) -> str:
    if not isinstance(answer, dict):
        return str(answer)
    if not answer.get("ok"):
        return f"{'unsupported' if 'unsupported' in answer else 'FAILED:'} {answer.get('unsupported') or answer.get('error')}"
    if "sentinel" in answer:
        followed = any(answer["sentinel"] in e for e in answer.get("after") or [])
        return f"hook={answer.get('hook')} notified={answer.get('notified')} list_followed={followed}"
    if "subscribed" in answer:
        return (
            f"route={answer.get('route')} subscribed={answer.get('subscribed')} notified={answer.get('notified')} "
            f"unsubscribed={answer.get('unsubscribed')} after_unsubscribe={answer.get('notified_after_unsubscribe')}"
            + (f" subscribe_error={answer['subscribe_error']}" if answer.get("subscribe_error") else "")
        )
    return f"route={answer.get('route')} opened={answer.get('opened')} events={answer.get('events')}"


def show(notify: dict) -> None:
    print(f"    negotiated {notify.get('negotiated')}")
    for key, answer in notify.items():
        if key != "negotiated":
            print(f"    {key:<24} {describe(answer)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-version", default=None, help="protocol revision the fixture serves")
    parser.add_argument("--fixture-python", default=os.environ.get("MCP_MIRROR_FIXTURE_PYTHON", sys.executable))
    parser.add_argument("--renderers", default=",".join(RENDERERS))
    parser.add_argument("--out", default=str(ROOT / "data" / "measurements" / "notify" / "all-frameworks.json"))
    args = parser.parse_args()
    if args.spec_version:
        sys.path.insert(0, str(ROOT / "scripts"))
        from _fixture import fixture_python

        args.fixture_python = fixture_python(args.spec_version)
    server = f"{args.fixture_python} {FIXTURE}"
    print(f"fixture: {server}")
    print("\n=== source ===")
    source = capture_source(args.fixture_python, server)
    show(source)
    captured: dict[str, dict] = {}
    for renderer_id in [r for r in args.renderers.replace(" ", "").split(",") if r]:
        print(f"\n=== {renderer_id} ===")
        payload = capture_managed(renderer_id, server)
        captured[renderer_id] = payload
        print(f"  version {(payload.get('versions') or {}).get('framework', '?')}")
        show(payload.get("notify") or {})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema": "mcp-mirror/notify-probe@1",
                "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "fixture": FIXTURE,
                "fixture_python": args.fixture_python,
                "spec_version": args.spec_version,
                "source": source,
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
