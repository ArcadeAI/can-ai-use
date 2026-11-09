"""Refresh each framework's last-month download count into data/usage.json.

The site's default view keeps the grid readable by showing the frameworks people
actually install, and "people actually install" needs a number with a source behind
it rather than a judgment call. The number is last month's downloads of the package
that carries the framework's MCP support:

    PyPI  https://pypistats.org/api/packages/<pkg>/recent       (last_month)
    npm   https://api.npmjs.org/downloads/point/last-month/<pkg>

The package is the first pin in src/mcp_mirror/runner-manifests.json that is not the
MCP SDK itself or zod. That is the adapter where MCP ships separately
(langchain-mcp-adapters) and the framework's own package where MCP ships inside it
(strands-agents). Maven Central, NuGet and the Go module proxy publish no comparable
monthly figure, so those rows are recorded as null with the reason.

    uv run python scripts/refresh_usage.py
"""

from __future__ import annotations

import json
import tomllib
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "src" / "mcp_mirror" / "runner-manifests.json"
FRAMEWORK_FILES = (
    ROOT / "data" / "frameworks.json",
    ROOT / "data" / "specs" / "2026-07-28" / "frameworks.json",
)
OUTPUT = ROOT / "data" / "usage.json"

# The MCP SDK and the schema library are shared plumbing, not the framework.
SKIPPED_PACKAGES = {
    "mcp",
    "zod",
    "@modelcontextprotocol/sdk",
    "@modelcontextprotocol/client",
    "@modelcontextprotocol/core",
}
NO_MONTHLY_FIGURE = {
    "java": "Maven Central publishes no per-artifact monthly download count",
    "dotnet": "NuGet publishes all-time totals only, not a monthly count",
    "go": "The Go module proxy publishes no download counts",
}
SOURCES = {"python": "pypistats.org", "node": "api.npmjs.org"}
TIMEOUT = 30
ATTEMPTS = 6
# The repository URL is set once in pyproject.toml.
REPOSITORY = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["urls"]["Repository"]
USER_AGENT = f"can-ai-use usage refresh ({REPOSITORY})"


def package_name(pin: str, runtime: str) -> str:
    """The bare package name from a manifest pin, without version or extras."""

    if runtime == "python":
        return re.split(r"[\[=<>!~]", pin, maxsplit=1)[0]
    if runtime == "java":
        return ":".join(pin.split(":")[:2])
    # npm, NuGet and Go pins are name@version, and an npm scope also starts with @.
    head, separator, _ = pin.rpartition("@")
    return head if separator and head else pin


def framework_package(runner: dict) -> str:
    """The first pinned package that is the framework rather than shared plumbing."""

    runtime = runner["runtime"]
    for pin in runner["packages"]:
        name = package_name(pin, runtime)
        if name not in SKIPPED_PACKAGES:
            return name
    raise SystemExit(f"no framework package among {runner['packages']}")


def runner_id(agent: dict, agents: dict) -> str | None:
    """The runner-manifest key behind one site row; an alias uses its source's runner."""

    if agent.get("aliases_of"):
        return runner_id(agents[agent["aliases_of"]], agents)
    renderer = agent.get("renderer")
    if not renderer:
        return None
    return Path(renderer).stem.removesuffix("_renderer")


def _fetch(url: str) -> dict:
    """GET JSON, backing off on rate limits and transient server errors."""

    delay = 5.0
    for attempt in range(ATTEMPTS):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == ATTEMPTS - 1:
                raise
            retry_after = error.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
            print(f"  {error.code} from {url}; retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            delay *= 2
    raise AssertionError("unreachable")


def monthly_downloads(package: str, runtime: str) -> int:
    if runtime == "python":
        document = _fetch(f"https://pypistats.org/api/packages/{package.lower()}/recent")
        # pypistats asks clients to space requests out; one per second keeps it quiet.
        time.sleep(1)
        return int(document["data"]["last_month"])
    document = _fetch(f"https://api.npmjs.org/downloads/point/last-month/{package}")
    return int(document["downloads"])


def site_agents() -> dict[str, dict]:
    """Every row any snapshot publishes, newest snapshot winning on conflicts."""

    agents: dict[str, dict] = {}
    for path in FRAMEWORK_FILES:
        agents.update(json.loads(path.read_text())["agents"])
    return agents


def main() -> int:
    runners = json.loads(MANIFESTS.read_text())["runners"]
    agents = site_agents()
    fetched = date.today().isoformat()
    frameworks: dict[str, dict] = {}
    cache: dict[str, int] = {}

    for agent_id in sorted(agents):
        key = runner_id(agents[agent_id], agents)
        if key not in runners:
            raise SystemExit(f"{agent_id}: no runner manifest entry for {key!r}")
        runner = runners[key]
        runtime = runner["runtime"]
        package = framework_package(runner)
        entry = {"package": package, "runner": key, "runtime": runtime}
        if runtime in NO_MONTHLY_FIGURE:
            entry |= {"source": None, "monthly_downloads": None, "reason": NO_MONTHLY_FIGURE[runtime]}
        else:
            if package not in cache:
                print(f"{agent_id}: {package}", file=sys.stderr)
                cache[package] = monthly_downloads(package, runtime)
            entry |= {"source": SOURCES[runtime], "monthly_downloads": cache[package], "reason": None}
        entry["fetched"] = fetched
        frameworks[agent_id] = entry

    document = {
        "schema": "mcp-mirror/usage@1",
        "fetched": fetched,
        "metric": "last-month downloads of the package that carries the framework's MCP support",
        "frameworks": frameworks,
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(frameworks)} frameworks)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
