"""Refresh each framework's and agent's published release window from its registry.

PyPI, npm, Maven Central, NuGet, the Go module proxy and GitHub releases are each asked
in their own format; a framework whose registry is none of those stops the run with its
name. An agent with no registry (Cursor ships from its own installer) is skipped.

`current_version` is the release a scan actually ran against, so it is never touched
here: moving it would re-label evidence as belonging to a version nobody measured. What
does go stale is everything around it. `stable_version` and the releases after era 0 are
registry facts with a shelf life, and when they lag the registry the grid quietly
implies the measurement sits at the frontier when it is twenty-one releases behind.

    uv run python scripts/refresh_versions.py          # write
    uv run python scripts/refresh_versions.py --check  # report drift, change nothing

Entries at or before era 0 are left alone. Each snapshot chose how much history to
publish, and that is an editorial decision rather than something a registry can answer.
"""

from __future__ import annotations

import argparse
import email.utils
import gzip
import json
import re
import urllib.request
from collections.abc import Callable
from datetime import date, timezone
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent.parent
FRAMEWORK_FILES = (
    ROOT / "data" / "frameworks.json",
    ROOT / "data" / "specs" / "2026-07-28" / "frameworks.json",
)
# Agents are measured on the wire and keep their own file; the release window works the same.
HARNESS_FILE = ROOT / "data" / "harnesses" / "harnesses.json"

# Only final releases are published. A prerelease would sit in the middle of the window
# and shift every era after it, and @mastra/mcp 2.0.0-alpha.4 is exactly that case.
STABLE_VERSION = re.compile(r"^\d+(?:\.\d+)*$")
NUMERIC_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)(.*)$")
TIMEOUT = 30
MAVEN_CENTRAL = "https://repo1.maven.org/maven2"
NUGET_FLAT = "https://api.nuget.org/v3-flatcontainer"
NUGET_REGISTRATION = "https://api.nuget.org/v3/registration5-gz-semver2"
GO_PROXY = "https://proxy.golang.org"

# A registry answers with (version, date) pairs. The date is a callable because Maven
# Central and the Go proxy cost one request per version, and only the releases after
# the measured one are ever published.
Release = tuple[str, Callable[[], str]]


def _open(url: str, method: str = "GET"):
    request = urllib.request.Request(url, method=method)
    return urllib.request.urlopen(request, timeout=TIMEOUT)


def _fetch(url: str) -> dict:
    with _open(url) as response:
        body = response.read()
    if body[:2] == b"\x1f\x8b":
        # NuGet's semver2 registration hive is served gzip-compressed regardless of
        # Accept-Encoding.
        body = gzip.decompress(body)
    return json.loads(body)


def _fetch_text(url: str) -> str:
    with _open(url) as response:
        return response.read().decode()


def _last_modified(url: str) -> str:
    with _open(url, method="HEAD") as response:
        return response.headers["Last-Modified"]


def _known(timestamp: str) -> Callable[[], str]:
    return lambda: timestamp


def stable_pattern(current: str) -> re.Pattern[str]:
    """Which versions count as releases of the line this row measures.

    A row measured at a final release only sees final releases. A few adapters ship
    nothing but one qualifier line, langchain4j-mcp as -betaNN and Koog as -beta, so
    for those the measured version's own qualifier is the stable line: 1.22.0-beta32
    is a release of langchain4j-mcp, and 1.0.0-beta-preview3 of Koog is not.
    """

    match = NUMERIC_PREFIX.match(current)
    if not match or not match.group(2):
        return STABLE_VERSION
    qualifier = re.sub(r"\d+", r"\\d+", re.escape(match.group(2)))
    return re.compile(rf"^\d+(?:\.\d+)*(?:{qualifier})?$")


def pypi_releases(package: str) -> list[Release]:
    document = _fetch(f"https://pypi.org/pypi/{package}/json")
    return [
        (version, _known(min(file["upload_time_iso_8601"] for file in files)))
        for version, files in document["releases"].items()
        if files
    ]


def npm_releases(package: str) -> list[Release]:
    document = _fetch(f"https://registry.npmjs.org/{package}")
    times = document["time"]
    return [
        (version, _known(times[version]))
        for version in document["versions"]
        if version in times
    ]


def maven_releases(package: str) -> list[Release]:
    """Versions from maven-metadata.xml, dates from the POM's Last-Modified.

    search.maven.org would answer dates in one request, but its index lags Central by
    months (it lists six langchain4j-mcp betas out of thirty). The POM's Last-Modified
    is the moment Central published the artifact.
    """

    group, artifact = package.split(":")
    base = f"{MAVEN_CENTRAL}/{group.replace('.', '/')}/{artifact}"
    metadata = ElementTree.fromstring(_fetch_text(f"{base}/maven-metadata.xml"))

    def published(version: str) -> Callable[[], str]:
        pom = f"{base}/{version}/{artifact}-{version}.pom"
        return lambda: _http_date(_last_modified(pom))

    return [
        (version, published(version))
        for version in (node.text for node in metadata.iterfind("./versioning/versions/version"))
        if version
    ]


def nuget_releases(package: str) -> list[Release]:
    """Versions from the flat container, published dates from the registration index."""

    lowered = package.lower()
    versions = _fetch(f"{NUGET_FLAT}/{lowered}/index.json")["versions"]
    published = {}
    for page in _fetch(f"{NUGET_REGISTRATION}/{lowered}/index.json")["items"]:
        # Large packages page their registration; the index then links each page
        # instead of inlining it.
        leaves = page["items"] if "items" in page else _fetch(page["@id"])["items"]
        for leaf in leaves:
            entry = leaf["catalogEntry"]
            if entry.get("listed", True):
                published[entry["version"].lower()] = entry["published"]
    return [
        (version, _known(published[version]))
        for version in versions
        if version in published
    ]


def _go_escape(module: str) -> str:
    """The proxy's case encoding: an uppercase letter becomes ! and its lowercase."""

    return re.sub(r"[A-Z]", lambda match: "!" + match.group(0).lower(), module)


def go_releases(package: str) -> list[Release]:
    """Tags from @v/list, each one's Time from @v/<version>.info.

    Published versions carry no leading v, the way frameworks.json already stores
    google.golang.org/adk/v2 2.4.0.
    """

    base = f"{GO_PROXY}/{_go_escape(package)}/@v"

    def published(tag: str) -> Callable[[], str]:
        return lambda: _fetch(f"{base}/{tag}.info")["Time"]

    return [
        (tag.removeprefix("v"), published(tag))
        for tag in _fetch_text(f"{base}/list").split()
        if tag.startswith("v")
    ]


def github_releases(repository: str) -> list[Release]:
    """Published, non-prerelease GitHub releases, tags read without their leading v.

    Asked without a token: the anonymous limit is 60 requests an hour, and two agents
    (Goose, Crush) need a page or two each.
    """

    releases: list[Release] = []
    for page in range(1, 4):
        batch = json.loads(_fetch_text(f"https://api.github.com/repos/{repository}/releases?per_page=100&page={page}"))
        releases.extend(
            (entry["tag_name"].removeprefix("v"), _known(entry["published_at"]))
            for entry in batch
            if not entry["draft"] and not entry["prerelease"] and entry.get("published_at")
        )
        if len(batch) < 100:
            break
    return releases


REGISTRIES: dict[str, Callable[[str], list[Release]]] = {
    "pypi": pypi_releases,
    "npm": npm_releases,
    "maven": maven_releases,
    "nuget": nuget_releases,
    "go": go_releases,
    "github": github_releases,
}


def _as_tuple(version: str) -> tuple:
    """Version order with a qualifier ranked below the final it precedes.

    1.20.2-beta30 sorts by 1.20.2 and then 30; a bare 1.20.2 would sort after every
    qualified 1.20.2.
    """

    match = NUMERIC_PREFIX.match(version)
    numbers = tuple(int(part) for part in match.group(1).split("."))
    qualifier = tuple(int(part) for part in re.findall(r"\d+", match.group(2)))
    return numbers, not match.group(2), qualifier


def releases_for(agent: dict) -> list[Release]:
    """Every stable release of one agent's package, in version order.

    Ordering by upload date would be wrong: pydantic-ai still ships 1.107.x patches,
    and the newest of those went out after 2.44.0. A maintenance backport is not a
    release after the 2.40.0 this project measured, so era offsets follow the version
    line rather than the calendar.
    """

    registry = agent["registry"]
    if registry not in REGISTRIES:
        raise SystemExit(f"{agent['package']}: no reader for registry {registry!r}")
    stable = stable_pattern(agent["current_version"])
    releases = [
        release for release in REGISTRIES[registry](agent["package"]) if stable.match(release[0])
    ]
    if not releases:
        raise SystemExit(f"{agent['package']}: no stable releases found on {registry}")
    return sorted(releases, key=lambda entry: _as_tuple(entry[0]))


def _http_date(value: str) -> str:
    return email.utils.parsedate_to_datetime(value).astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _iso(timestamp: str) -> str:
    """Second-precision UTC, which is the format already published and parsed.

    PyPI and NuGet report fractional seconds and npm does not, so all are trimmed to
    the one shape scripts/build_data.py can read back.
    """

    return re.sub(r"\.\d+", "", timestamp).replace("+00:00", "Z")


def refresh_agent(agent_id: str, agent: dict) -> list[str]:
    """Rewrite one agent's forward release window. Returns human-readable changes."""

    releases = releases_for(agent)
    current = agent["current_version"]
    versions = [version for version, _ in releases]
    if current not in versions:
        raise SystemExit(
            f"{agent_id}: measured version {current} is not a stable release of "
            f"{agent['package']}; refusing to guess an era offset"
        )

    index = versions.index(current)
    history = [entry for entry in agent["version_list"] if entry["era"] <= 0]
    changes = []
    # A row written by a measurement records the version it ran but not when that
    # version shipped; without the date the pages can't tell a newer release from an older one.
    for entry in history:
        if entry["version"] == current and not entry.get("release_date"):
            entry["release_date"] = _iso(releases[index][1]())
            changes.append(f"release_date of the measured {current} -> {entry['release_date']}")
    forward = [
        {"version": version, "release_date": _iso(uploaded()), "era": offset}
        for offset, (version, uploaded) in enumerate(releases[index + 1 :], start=1)
    ]
    latest = versions[-1]

    if agent.get("stable_version") != latest:
        changes.append(f"stable_version {agent.get('stable_version')} -> {latest}")
    published = [entry["version"] for entry in agent["version_list"] if entry["era"] > 0]
    if published != [entry["version"] for entry in forward]:
        behind = len(forward)
        changes.append(
            f"version_list now carries {behind} release(s) after the measured {current}"
            + (f", newest {latest}" if behind else "")
        )

    agent["version_list"] = history + forward
    agent["stable_version"] = latest
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift and exit nonzero instead of writing",
    )
    arguments = parser.parse_args()

    drifted = False
    for path in (*FRAMEWORK_FILES, HARNESS_FILE):
        document = json.loads(path.read_text())
        print(f"\n{path.relative_to(ROOT)}")
        file_changes = False
        for agent_id, agent in document["agents"].items():
            if agent.get("aliases_of"):
                # An alias inherits its source's versions at build time; nothing to refresh.
                print(f"  {agent_id}: alias of {agent['aliases_of']}")
                continue
            if path == HARNESS_FILE and not agent.get("registry"):
                print(f"  {agent_id}: no registry")
                continue
            changes = refresh_agent(agent_id, agent)
            if changes:
                file_changes = True
                for change in changes:
                    print(f"  {agent_id}: {change}")
            else:
                print(f"  {agent_id}: current")
        if not file_changes:
            continue
        drifted = True
        if arguments.check:
            continue
        # The harness file's "updated" is when its captures changed, which the Agents page
        # reports, so a registry refresh leaves it alone.
        if path != HARNESS_FILE:
            document["updated"] = date.today().isoformat()
        # The harness file is written with its own characters, as scripts/measure_harness.py does.
        path.write_text(json.dumps(document, indent=2, ensure_ascii=path != HARNESS_FILE) + "\n")
        print(f"  wrote {path.relative_to(ROOT)}")

    if arguments.check and drifted:
        print("\nversion metadata is behind the registries")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
