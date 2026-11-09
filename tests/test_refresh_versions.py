"""refresh_versions against recorded registry answers, no network.

The bodies below are trimmed copies of what each registry returned on 2026-10-09.
"""

import gzip
import io
import json
import sys
from email.message import Message
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import refresh_versions  # noqa: E402

MAVEN = "https://repo1.maven.org/maven2"
LANGCHAIN4J_MCP = f"{MAVEN}/dev/langchain4j/langchain4j-mcp"
KOOG = f"{MAVEN}/ai/koog/agents-mcp-jvm"
SPRING_AI = f"{MAVEN}/org/springframework/ai/spring-ai-starter-mcp-client"
NUGET_FLAT = "https://api.nuget.org/v3-flatcontainer/microsoft.agents.ai/index.json"
NUGET_REGISTRATION = "https://api.nuget.org/v3/registration5-gz-semver2/microsoft.agents.ai/index.json"
NUGET_PAGE = "https://api.nuget.org/v3/registration5-gz-semver2/microsoft.agents.ai/page/1.0.0-rc1/1.24.0.json"
GO_ADK = "https://proxy.golang.org/google.golang.org/adk/v2/@v"
GOOSE = "https://api.github.com/repos/aaif-goose/goose/releases?per_page=100&page=1"


def _metadata(*versions: str) -> str:
    listed = "".join(f"<version>{version}</version>" for version in versions)
    return (
        '<?xml version="1.0" encoding="UTF-8"?><metadata><versioning>'
        f"<versions>{listed}</versions></versioning></metadata>"
    )


def _leaf(version: str, published: str, listed: bool = True) -> dict:
    return {"catalogEntry": {"version": version, "published": published, "listed": listed}}


NUGET_LEAVES = [
    _leaf("1.23.0-preview.1", "2026-09-25T09:00:00.1+00:00"),
    _leaf("1.23.0", "2026-09-29T10:31:41.523+00:00"),
    # An unlisted version reports a placeholder date and is not a release anyone installs.
    _leaf("1.23.1", "1900-01-01T00:00:00+00:00", listed=False),
    _leaf("1.24.0", "2026-10-07T13:23:37.313+00:00"),
]

GETS = {
    f"{LANGCHAIN4J_MCP}/maven-metadata.xml": _metadata(
        "1.0.0-beta5", "1.20.1-beta30", "1.20.2-beta30", "1.21.0-beta31", "1.22.0-beta32"
    ),
    f"{KOOG}/maven-metadata.xml": _metadata(
        "0.8.0",
        "1.0.0-beta",
        "0.4.3-dev-22092025-feat-856",
        "1.0.0-beta-preview3",
        "1.0.0-beta-preview7",
        "1.2.0-beta",
        "1.3.0-beta",
    ),
    f"{SPRING_AI}/maven-metadata.xml": _metadata(
        "2.0.0-M8", "2.0.0-RC2", "2.0.0", "2.0.1", "2.1.0-M1"
    ),
    NUGET_FLAT: json.dumps(
        {"versions": ["1.23.0-preview.1", "1.23.0", "1.23.1", "1.24.0", "1.25.0-rc.1"]}
    ),
    # The registration index links its page instead of inlining it, as a large package's does.
    NUGET_REGISTRATION: gzip.compress(json.dumps({"items": [{"@id": NUGET_PAGE}]}).encode()),
    NUGET_PAGE: gzip.compress(json.dumps({"items": NUGET_LEAVES}).encode()),
    f"{GO_ADK}/list": "v2.3.0\nv2.2.0\nv2.4.0\nv2.5.0-pre.1\nv2.5.0\n",
    f"{GO_ADK}/v2.5.0.info": json.dumps({"Version": "v2.5.0", "Time": "2026-09-30T14:44:44Z"}),
    # Trimmed from what GitHub returned for Goose on 2026-10-19, plus a draft and a nightly.
    GOOSE: json.dumps(
        [
            {"tag_name": "nightly", "draft": False, "prerelease": True, "published_at": "2026-10-19T01:04:18Z"},
            {"tag_name": "v1.56.0", "draft": True, "prerelease": False, "published_at": None},
            {"tag_name": "v1.55.0", "draft": False, "prerelease": False, "published_at": "2026-10-16T16:58:21Z"},
            {"tag_name": "v1.54.0", "draft": False, "prerelease": False, "published_at": "2026-10-08T18:44:54Z"},
        ]
    ),
}

HEADS = {
    f"{LANGCHAIN4J_MCP}/1.21.0-beta31/langchain4j-mcp-1.21.0-beta31.pom": "Fri, 02 Oct 2026 11:31:56 GMT",
    f"{LANGCHAIN4J_MCP}/1.22.0-beta32/langchain4j-mcp-1.22.0-beta32.pom": "Wed, 07 Oct 2026 21:41:12 GMT",
}


class _Response(io.BytesIO):
    def __init__(self, body: bytes, headers: Message):
        super().__init__(body)
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def requests(monkeypatch):
    """Serve the recorded answers and remember every URL asked for."""

    seen = []

    def fake_open(url, method="GET"):
        seen.append((method, url))
        headers = Message()
        if method == "HEAD":
            headers["Last-Modified"] = HEADS[url]
            return _Response(b"", headers)
        body = GETS[url]
        return _Response(body if isinstance(body, bytes) else body.encode(), headers)

    monkeypatch.setattr(refresh_versions, "_open", fake_open)
    return seen


def _agent(registry, package, current, history=()):
    return {
        "registry": registry,
        "package": package,
        "current_version": current,
        "stable_version": current,
        "version_list": [
            {"version": version, "era": era, "release_date": "2026-01-01T00:00:00Z"}
            for version, era in [*history, (current, 0)]
        ],
    }


def test_maven_keeps_the_beta_line_langchain4j_mcp_ships(requests):
    agent = _agent("maven", "dev.langchain4j:langchain4j-mcp", "1.20.2-beta30", [("1.20.1-beta30", -1)])
    changes = refresh_versions.refresh_agent("langchain4j", agent)

    assert agent["stable_version"] == "1.22.0-beta32"
    assert agent["version_list"][-2:] == [
        {"version": "1.21.0-beta31", "release_date": "2026-10-02T11:31:56Z", "era": 1},
        {"version": "1.22.0-beta32", "release_date": "2026-10-07T21:41:12Z", "era": 2},
    ]
    assert agent["version_list"][0]["version"] == "1.20.1-beta30"
    assert changes[0] == "stable_version 1.20.2-beta30 -> 1.22.0-beta32"
    # Only the releases after the measured one cost a POM request.
    assert [url for method, url in requests if method == "HEAD"] == list(HEADS)


def test_maven_drops_koog_previews_and_dev_builds(requests):
    releases = refresh_versions.releases_for(_agent("maven", "ai.koog:agents-mcp-jvm", "1.3.0-beta"))
    assert [version for version, _ in releases] == ["0.8.0", "1.0.0-beta", "1.2.0-beta", "1.3.0-beta"]


def test_maven_drops_milestones_and_release_candidates(requests):
    agent = _agent("maven", "org.springframework.ai:spring-ai-starter-mcp-client", "2.0.1")
    assert refresh_versions.refresh_agent("spring_ai", agent) == []
    assert agent["stable_version"] == "2.0.1"
    assert not any(method == "HEAD" for method, _ in requests)


def test_nuget_reads_paged_registration_and_skips_previews_and_unlisted(requests):
    agent = _agent("nuget", "Microsoft.Agents.AI", "1.23.0")
    refresh_versions.refresh_agent("ms_agent_framework_dotnet", agent)

    assert agent["stable_version"] == "1.24.0"
    assert agent["version_list"][-1] == {
        "version": "1.24.0",
        "release_date": "2026-10-07T13:23:37Z",
        "era": 1,
    }
    assert ("GET", NUGET_PAGE) in requests


def test_go_proxy_strips_the_v_and_skips_prereleases(requests):
    agent = _agent("go", "google.golang.org/adk/v2", "2.4.0")
    refresh_versions.refresh_agent("google_adk_go", agent)

    assert agent["stable_version"] == "2.5.0"
    assert agent["version_list"][-1] == {
        "version": "2.5.0",
        "release_date": "2026-09-30T14:44:44Z",
        "era": 1,
    }
    assert ("GET", f"{GO_ADK}/v2.5.0-pre.1.info") not in requests


def test_github_reads_published_releases_without_the_v(requests):
    agent = _agent("github", "aaif-goose/goose", "1.54.0")
    refresh_versions.refresh_agent("goose", agent)

    assert agent["stable_version"] == "1.55.0"
    assert agent["version_list"][-1] == {
        "version": "1.55.0",
        "release_date": "2026-10-16T16:58:21Z",
        "era": 1,
    }


def test_a_measured_agent_gets_its_release_date(requests):
    agent = _agent("github", "aaif-goose/goose", "1.54.0")
    agent["version_list"][-1]["release_date"] = None
    changes = refresh_versions.refresh_agent("goose", agent)

    assert agent["version_list"][0]["release_date"] == "2026-10-08T18:44:54Z"
    assert "release_date of the measured 1.54.0 -> 2026-10-08T18:44:54Z" in changes


def test_every_agent_registry_has_a_reader():
    for agent_id, agent in json.loads(refresh_versions.HARNESS_FILE.read_text())["agents"].items():
        if agent.get("registry"):
            assert agent["registry"] in refresh_versions.REGISTRIES, agent_id
            measured = [entry for entry in agent["version_list"] if entry["version"] == agent["current_version"]]
            # Without the measured release's date the pages can't mark a newer one.
            assert measured and measured[0].get("release_date"), agent_id


def test_go_escapes_uppercase_module_paths():
    assert refresh_versions._go_escape("github.com/Azure/azure-sdk") == "github.com/!azure/azure-sdk"


def test_qualified_versions_sort_below_their_final():
    versions = ["1.20.2", "1.20.10-beta31", "1.20.2-beta30", "1.20.2-beta4"]
    assert sorted(versions, key=refresh_versions._as_tuple) == [
        "1.20.2-beta4",
        "1.20.2-beta30",
        "1.20.2",
        "1.20.10-beta31",
    ]


def test_an_unknown_registry_names_itself():
    with pytest.raises(SystemExit, match="cargo"):
        refresh_versions.releases_for(_agent("cargo", "rmcp", "0.1.0"))


def test_every_published_registry_has_a_reader():
    for path in refresh_versions.FRAMEWORK_FILES:
        for agent_id, agent in json.loads(path.read_text())["agents"].items():
            if not agent.get("aliases_of"):
                assert agent["registry"] in refresh_versions.REGISTRIES, agent_id
