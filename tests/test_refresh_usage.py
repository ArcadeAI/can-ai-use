import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import refresh_usage  # noqa: E402


def test_package_name_strips_versions_extras_and_keeps_npm_scopes():
    assert refresh_usage.package_name("pydantic-ai-slim[mcp]==2.51.0", "python") == "pydantic-ai-slim"
    assert refresh_usage.package_name("@ai-sdk/mcp@2.0.62", "node") == "@ai-sdk/mcp"
    assert refresh_usage.package_name("dev.langchain4j:langchain4j-mcp:1.20.2-beta30", "java") == (
        "dev.langchain4j:langchain4j-mcp"
    )
    assert refresh_usage.package_name("google.golang.org/adk/v2@v2.4.0", "go") == "google.golang.org/adk/v2"


def test_framework_package_skips_the_mcp_sdk_and_zod():
    runners = json.loads(refresh_usage.MANIFESTS.read_text())["runners"]
    assert refresh_usage.framework_package(runners["strands"]) == "strands-agents"
    assert refresh_usage.framework_package(runners["langchain"]) == "langchain-mcp-adapters"
    assert refresh_usage.framework_package(runners["vercel_ai"]) == "@ai-sdk/mcp"


def test_usage_covers_every_site_row_with_a_figure_or_a_reason():
    usage = json.loads(refresh_usage.OUTPUT.read_text())
    agents = refresh_usage.site_agents()
    runners = json.loads(refresh_usage.MANIFESTS.read_text())["runners"]
    assert set(usage["frameworks"]) == set(agents)
    for agent_id, entry in usage["frameworks"].items():
        runner = runners[refresh_usage.runner_id(agents[agent_id], agents)]
        assert entry["package"] == refresh_usage.framework_package(runner), agent_id
        if entry["monthly_downloads"] is None:
            assert entry["reason"] and entry["source"] is None, agent_id
        else:
            assert entry["source"] in ("pypistats.org", "api.npmjs.org"), agent_id
            assert entry["monthly_downloads"] >= 0, agent_id
