"""Run the official MCP conformance suite's client scenarios against every adapter.

    uv run python scripts/run_conformance.py [--renderers a,b] [--suites auth,core]

For each renderer this runs ``npx @modelcontextprotocol/conformance client --suite
<suite>`` with ``scripts/conformance_client.py <renderer>`` as the client command, and
writes what the suite's scenario servers recorded to
``data/measurements/conformance/<renderer>/checks.json``. The cell generator reads
those files; nothing here decides a support code.

The suite version is pinned (``SUITE_VERSION``) so a re-run grades against the same
scenarios. Its client scenarios are tagged with the protocol revisions they test, and
as of 0.1.16 every scenario in ``auth`` and ``core`` is tagged 2025-11-25, which is why
the artifacts live in the 2025-11-25 snapshot (``data/``). Each artifact records the
version that ran and every scenario's tags, so a later suite that adds a 2026 tag is
visible in the diff.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp_mirror.renderers import canonical_id  # noqa: E402

SUITE_PACKAGE = "@modelcontextprotocol/conformance"
SUITE_VERSION = "0.1.16"
SPEC_VERSION = "2025-11-25"
RENDERERS = ("crewai", "langchain", "openai_agents", "pydantic_ai", "mastra", "vercel_ai")
SUITES = ("auth", "core")
OUT_DIR = ROOT / "data" / "measurements" / "conformance"
# The suite names each result directory `<scenario>-<ISO timestamp>`.
STAMP = re.compile(r"-\d{4}-\d{2}-\d{2}T[\d-]+Z$")


def suite(*arguments: str, timeout: int = 900) -> subprocess.CompletedProcess:
    npx = shutil.which("npx")
    if not npx:
        raise SystemExit("the conformance suite runs through npx; install Node.js first")
    return subprocess.run(
        [npx, "-y", f"{SUITE_PACKAGE}@{SUITE_VERSION}", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def scenario_tags() -> dict[str, list[str]]:
    """``{scenario: [revision tags]}`` as the suite lists its client scenarios."""

    listing = suite("list", "--client", timeout=300).stdout
    return {
        match.group(1): match.group(2).split(",")
        for match in re.finditer(r"^\s*-\s+(\S+)\s+\[([^\]]+)\]", listing, re.MULTILINE)
    }


def client_summary(stdout: str) -> dict | None:
    """The one JSON line conformance_client.py prints: versions, auth hook, tools, call."""

    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def run_suite(renderer: str, suite_name: str, timeout_ms: int) -> dict:
    command = " ".join([sys.executable, str(Path(__file__).with_name("conformance_client.py")), renderer])
    if len(command.split(" ")) != 3:
        # The suite splits the client command on single spaces before running it.
        raise SystemExit(f"a path in the client command contains a space: {command!r}")
    with tempfile.TemporaryDirectory(prefix="mcp-mirror-conformance-") as out:
        proc = suite(
            "client", "--suite", suite_name, "--command", command,
            "--timeout", str(timeout_ms), "-o", out,
        )
        scenarios: dict[str, dict] = {}
        for checks_path in sorted(Path(out).rglob("checks.json")):
            name = STAMP.sub("", str(checks_path.parent.relative_to(out)))
            stdout = (checks_path.parent / "stdout.txt").read_text() if (checks_path.parent / "stdout.txt").exists() else ""
            stderr = (checks_path.parent / "stderr.txt").read_text() if (checks_path.parent / "stderr.txt").exists() else ""
            scenarios[name] = {
                "checks": [slim(c) for c in json.loads(checks_path.read_text())],
                "client": client_summary(stdout),
                "client_stderr_tail": stderr.strip()[-600:] or None,
            }
    return {"exit_code": proc.returncode, "scenarios": scenarios}


def slim(check: dict) -> dict:
    """An INFO check is the suite's wire log: one request or response. Keep what a rule
    can read (method, route, status, whether the server issued a session ID, and the
    error a rejected request got) and drop the bodies, which are most of the bytes."""

    if check.get("status") != "INFO":
        # Name, timestamp and spec references are fixed by the suite version.
        return {k: check[k] for k in ("id", "status", "description", "errorMessage", "details") if k in check}
    details = check.get("details") or {}
    kept = {k: details[k] for k in ("method", "path", "mcpMethod", "statusCode", "sessionIdIssued") if k in details}
    if "mcp-session-id" in (details.get("headers") or {}):
        kept["sessionIdIssued"] = True
    if (details.get("statusCode") or 0) >= 400 and "body" in details:
        body = details["body"]
        kept["body"] = (body if isinstance(body, str) else json.dumps(body))[:300]
    return {"id": check.get("id"), "status": "INFO", "details": kept}


def dumps(artifact: dict) -> str:
    """Indented JSON with one check per line, so a re-measurement diffs check by check
    without every details field taking a line of its own."""

    compact: list[str] = []

    def mark(checks: list[dict]) -> list[str]:
        out = []
        for check in checks:
            compact.append(json.dumps(check, separators=(", ", ": ")))
            out.append(f"@@check{len(compact) - 1}@@")
        return out

    shaped = json.loads(json.dumps(artifact))
    for run in shaped.get("suites", {}).values():
        for scenario in run.get("scenarios", {}).values():
            scenario["checks"] = mark(scenario["checks"])
    text = json.dumps(shaped, indent=2)
    return re.sub(r'"@@check(\d+)@@"', lambda m: compact[int(m.group(1))], text) + "\n"


def describe(checks: list[dict]) -> str:
    graded = [c for c in checks if c.get("status") != "INFO"]
    passed = sum(1 for c in graded if c.get("status") == "SUCCESS")
    failed = sorted({c["id"] for c in graded if c.get("status") != "SUCCESS"})
    return f"{passed}/{len(graded)} passed" + (f"  not passed: {', '.join(failed)}" if failed else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderers", default=",".join(RENDERERS))
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--timeout-ms", type=int, default=120000, help="per-scenario client timeout")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    version = suite("--version", timeout=300).stdout.strip()
    if version != SUITE_VERSION:
        raise SystemExit(f"expected {SUITE_PACKAGE} {SUITE_VERSION}, npx ran {version!r}")
    tags = scenario_tags()
    renderers = [canonical_id(r) for r in args.renderers.replace(" ", "").split(",") if r]
    suites = [s for s in args.suites.replace(" ", "").split(",") if s]
    command = "uv run python scripts/run_conformance.py"
    if set(renderers) != set(RENDERERS):
        command += f" --renderers {','.join(renderers)}"
    if suites != list(SUITES):
        command += f" --suites {','.join(suites)}"

    mastra_node = ROOT / "src" / "mcp_mirror" / "renderers" / "mastra_node"
    if "mastra" in renderers and not (mastra_node / "node_modules" / "@mastra" / "mcp").exists():
        # Once, up front: the suite runs its scenarios in parallel, and parallel
        # installs into one directory race.
        subprocess.run(["npm", "ci", "--prefix", str(mastra_node)], check=True)
    vercel_ai_node = ROOT / "src" / "mcp_mirror" / "renderers" / "vercel_ai_node"
    if "vercel_ai" in renderers and not (vercel_ai_node / "node_modules" / "@ai-sdk" / "mcp").exists():
        subprocess.run(["npm", "ci", "--prefix", str(vercel_ai_node)], check=True)
    agents_js_node = ROOT / "src" / "mcp_mirror" / "renderers" / "openai_agents_js_node"
    if "openai_agents_js" in renderers and not (agents_js_node / "node_modules" / "@openai" / "agents").exists():
        subprocess.run(["npm", "ci", "--prefix", str(agents_js_node)], check=True)
    pi_node = ROOT / "src" / "mcp_mirror" / "renderers" / "pi_node"
    if "pi" in renderers and not (pi_node / "node_modules" / "@earendil-works" / "pi-mcp").exists():
        subprocess.run(["npm", "ci", "--prefix", str(pi_node)], check=True)
    adk_js_node = ROOT / "src" / "mcp_mirror" / "renderers" / "google_adk_js_node"
    if "google_adk_js" in renderers and not (adk_js_node / "node_modules" / "@google" / "adk").exists():
        subprocess.run(["npm", "ci", "--prefix", str(adk_js_node)], check=True)
    langchain_js_node = ROOT / "src" / "mcp_mirror" / "renderers" / "langchain_js_node"
    if "langchain_js" in renderers and not (langchain_js_node / "node_modules" / "@langchain" / "mcp-adapters").exists():
        subprocess.run(["npm", "ci", "--prefix", str(langchain_js_node)], check=True)
    genkit_js_node = ROOT / "src" / "mcp_mirror" / "renderers" / "genkit_js_node"
    if "genkit_js" in renderers and not (genkit_js_node / "node_modules" / "@genkit-ai" / "mcp").exists():
        subprocess.run(["npm", "ci", "--prefix", str(genkit_js_node)], check=True)
    copilot_sdk_node = ROOT / "src" / "mcp_mirror" / "renderers" / "copilot_sdk_node"
    if "copilot_sdk" in renderers and not (copilot_sdk_node / "node_modules" / "@github" / "copilot-sdk").exists():
        subprocess.run(["npm", "ci", "--prefix", str(copilot_sdk_node)], check=True)

    for renderer in renderers:
        print(f"\n=== {renderer} ===")
        runs = {}
        for suite_name in suites:
            runs[suite_name] = run_suite(renderer, suite_name, args.timeout_ms)
            print(f"  --suite {suite_name} (exit {runs[suite_name]['exit_code']})")
            for name, scenario in sorted(runs[suite_name]["scenarios"].items()):
                print(f"    {name:<40} {describe(scenario['checks'])}")
        ran = sorted({name for run in runs.values() for name in run["scenarios"]})
        versions = next(
            (
                s["client"]["versions"]
                for run in runs.values()
                for s in run["scenarios"].values()
                if s.get("client") and s["client"].get("versions")
            ),
            None,
        )
        out_path = Path(args.out_dir) / renderer / "checks.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            dumps(
                {
                    "schema": "mcp-mirror/conformance@1",
                    "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "renderer": renderer,
                    "versions": versions,
                    "suite": {"package": SUITE_PACKAGE, "version": version},
                    "spec_version": SPEC_VERSION,
                    "spec_note": (
                        f"Every scenario that ran is tagged {SPEC_VERSION} by the suite "
                        "(see scenario_tags); the suite has no 2026-07-28 client scenarios."
                        if all(SPEC_VERSION in tags.get(name, []) for name in ran)
                        else f"Not every scenario that ran is tagged {SPEC_VERSION}; see scenario_tags."
                    ),
                    "scenario_tags": {name: tags.get(name, []) for name in ran},
                    "client_command": f"python scripts/conformance_client.py {renderer} <url>",
                    "command": command,
                    "suites": runs,
                }
            )
        )
        print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
