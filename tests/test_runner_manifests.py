from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_mirror.runner_manifests import (
    NODE_RUNNER_DIR_ENV,
    ManagedRunnerUnavailable,
    build_managed_source_command,
    build_managed_worker_command,
    install_node_runner,
    load_runner_manifests,
    node_install_digest,
)


def test_runner_manifests_cover_the_launch_renderers_with_pinned_packages():
    manifests = load_runner_manifests()

    assert set(manifests) == {
        "langchain",
        "pydantic_ai",
        "crewai",
        "crewai_native",
        "openai_agents",
        "mastra",
        "google_adk",
        "vercel_ai",
        "ms_agent_framework",
        "autogen",
        "strands",
        "semantic_kernel",
        "openai_agents_js",
        "llamaindex",
        "pi",
        "google_adk_js",
        "agno",
        "haystack",
        "smolagents",
        "dspy",
        "spring_ai",
        "langchain_js",
        "genkit_js",
        "beeai",
        "copilot_sdk",
        "google_adk_java",
        "langchain4j",
        "ms_agent_framework_dotnet",
        "google_adk_go",
        "koog",
    }
    for renderer_id, manifest in manifests.items():
        assert manifest.module.startswith(".")
        assert manifest.protocols
        assert manifest.packages
        assert len(manifest.digest) == 64
        for package in manifest.packages:
            if manifest.runtime == "python":
                assert "==" in package, (renderer_id, package)
            elif manifest.runtime == "java":
                assert package.count(":") == 2 and package.rsplit(":", 1)[-1][0].isdigit(), (
                    renderer_id,
                    package,
                )
            elif manifest.runtime == "go":
                module, version = package.rsplit("@", 1)
                assert version.startswith("v") and version[1].isdigit(), (renderer_id, package)
            else:
                assert package.rsplit("@", 1)[-1][0].isdigit(), (
                    renderer_id,
                    package,
                )
        for package in manifest.host_packages:
            assert "==" in package, (renderer_id, package)


def test_managed_only_runners_say_why_and_leave_the_local_registry():
    from mcp_mirror.renderers import install_hint, local_renderer_ids

    manifests = load_runner_manifests()
    managed_only = {rid for rid, manifest in manifests.items() if manifest.managed_only}

    assert "agno" in managed_only
    assert "dspy" in managed_only
    assert "beeai" in managed_only
    for renderer_id in managed_only:
        assert manifests[renderer_id].managed_only_reason
        assert renderer_id not in local_renderer_ids()
        assert install_hint(renderer_id).startswith("run it with --runner managed")
    assert set(local_renderer_ids()) == set(manifests) - managed_only


def test_a_managed_only_flag_needs_a_reason():
    from mcp_mirror.runner_manifests import RunnerManifest

    with pytest.raises(ValueError, match="managed_only_reason"):
        RunnerManifest.model_validate(
            {
                "id": "x",
                "module": ".x",
                "runtime": "python",
                "python": "3.11",
                "packages": ["x==1"],
                "protocols": ["2025-11-25"],
                "capture_stage": "provider_format",
                "provider_request_captured": False,
                "managed_only": True,
            }
        )


def test_runners_without_the_flag_keep_the_digest_their_measurements_record():
    # The flag postdates these captures; adding it to the schema must not move them.
    root = Path(__file__).resolve().parents[1]
    scan = (root / "data" / "measurements" / "scan" / "strands.json").read_text()

    assert load_runner_manifests()["strands"].digest in scan


def test_managed_python_command_is_argument_safe_and_manifest_driven():
    command = build_managed_worker_command(
        "pydantic_ai",
        Path("/tmp/report worker.json"),
        uv_executable="/opt/homebrew/bin/uv",
    )

    assert command[:6] == [
        "/opt/homebrew/bin/uv",
        "run",
        "--isolated",
        "--no-project",
        "--python",
        "3.11",
    ]
    assert "mcp==2.0.0" in command
    assert "pydantic-ai-slim[mcp]==2.54.0" in command
    assert command[-5:] == [
        "python",
        "-m",
        "mcp_mirror._render_worker",
        "pydantic_ai",
        "/tmp/report worker.json",
    ]


FAKE_NPM = """#!/bin/sh
# Stands in for `npm install`: records the call and lays out what a real install
# leaves behind, a pinned node binary and the @mastra/mcp package.
echo "$@" >> "$FAKE_NPM_LOG"
cp package.json "$FAKE_NPM_SEEN"
[ -n "$FAKE_NPM_FAIL" ] && { echo "npm ERR! $FAKE_NPM_FAIL" >&2; exit 1; }
mkdir -p node_modules/.bin node_modules/@mastra/mcp
printf '#!/bin/sh\\necho %s\\n' "${FAKE_NODE_VERSION:-v24.21.0}" > node_modules/.bin/node
chmod +x node_modules/.bin/node
echo '{"version": "2.2.0"}' > node_modules/@mastra/mcp/package.json
"""


@pytest.fixture
def fake_npm(tmp_path, monkeypatch):
    npm = tmp_path / "npm"
    npm.write_text(FAKE_NPM)
    npm.chmod(0o755)
    monkeypatch.setenv("FAKE_NPM_LOG", str(tmp_path / "npm.log"))
    monkeypatch.setenv("FAKE_NPM_SEEN", str(tmp_path / "package.json"))
    return npm


def _npm_calls(tmp_path) -> int:
    log = tmp_path / "npm.log"
    return len(log.read_text().splitlines()) if log.exists() else 0


def test_managed_node_command_installs_the_manifest_and_points_the_worker_at_it(
    tmp_path, fake_npm
):
    cache = tmp_path / "cache"
    command = build_managed_worker_command(
        "mastra",
        Path("/tmp/out.json"),
        uv_executable="/usr/bin/uv",
        npm_executable=str(fake_npm),
        cache_root=cache,
    )

    manifest = load_runner_manifests()["mastra"]
    install = cache / "node-runners" / node_install_digest(manifest)
    assert (install / ".complete").read_text().strip() == "v24.21.0"
    assert (install / "_mastra_worker.mjs").exists()
    assert (install / "runner.env").read_text() == (
        f"{NODE_RUNNER_DIR_ENV}='{install}'\n"
    )

    # npm resolved exactly the manifest's packages plus the manifest's Node.
    seen = json.loads((tmp_path / "package.json").read_text())
    assert seen["dependencies"] == {
        "@mastra/mcp": "2.2.0",
        "@modelcontextprotocol/client": "2.0.0",
        "node": "24",
    }

    assert command[:6] == ["/usr/bin/uv", "run", "--isolated", "--no-project", "--python", "3.11"]
    assert command[6:8] == ["--env-file", str(install / "runner.env")]
    withs = [command[i + 1] for i, arg in enumerate(command) if arg == "--with"]
    assert withs == list(manifest.host_packages)
    assert command[-5:] == [
        "python",
        "-m",
        "mcp_mirror._render_worker",
        "mastra",
        "/tmp/out.json",
    ]


def test_managed_node_install_is_cached_by_manifest_digest(tmp_path, fake_npm):
    cache = tmp_path / "cache"
    for _ in range(2):
        build_managed_worker_command(
            "mastra",
            Path("/tmp/out.json"),
            uv_executable="/usr/bin/uv",
            npm_executable=str(fake_npm),
            cache_root=cache,
        )

    assert _npm_calls(tmp_path) == 1
    assert [p.name for p in (cache / "node-runners").iterdir()] == [
        node_install_digest(load_runner_manifests()["mastra"])
    ]


def test_managed_node_install_refreshes_an_edited_worker(tmp_path, fake_npm):
    cache = tmp_path / "cache"
    manifest = load_runner_manifests()["mastra"]
    install = install_node_runner(
        manifest, npm_executable=str(fake_npm), cache_root=cache
    )
    (install / "_mastra_worker.mjs").write_text("// stale\n")

    install_node_runner(manifest, npm_executable=str(fake_npm), cache_root=cache)

    assert "MCPClient" in (install / "_mastra_worker.mjs").read_text()
    assert _npm_calls(tmp_path) == 1


@pytest.mark.parametrize(
    ("variable", "value", "match"),
    [
        ("FAKE_NPM_FAIL", "E404 not found", "E404 not found"),
        ("FAKE_NODE_VERSION", "v26.8.1", "pinned Node 24 installed as v26.8.1"),
    ],
)
def test_failed_node_install_is_unavailable_and_leaves_no_cache(
    tmp_path, fake_npm, monkeypatch, variable, value, match
):
    monkeypatch.setenv(variable, value)
    cache = tmp_path / "cache"

    with pytest.raises(ManagedRunnerUnavailable, match=match):
        build_managed_worker_command(
            "mastra",
            Path("/tmp/out.json"),
            uv_executable="/usr/bin/uv",
            npm_executable=str(fake_npm),
            cache_root=cache,
        )

    assert list((cache / "node-runners").iterdir()) == []


def test_managed_node_runner_requires_npm(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(ManagedRunnerUnavailable, match="npm"):
        build_managed_worker_command(
            "mastra",
            Path("/tmp/out.json"),
            uv_executable="/usr/bin/uv",
            cache_root=tmp_path,
        )


def test_mastra_renderer_uses_the_managed_install_and_its_pinned_node(
    tmp_path, fake_npm, monkeypatch
):
    from mcp_mirror.renderers import mastra_renderer

    install = install_node_runner(
        load_runner_manifests()["mastra"],
        npm_executable=str(fake_npm),
        cache_root=tmp_path,
    )
    monkeypatch.setenv(NODE_RUNNER_DIR_ENV, str(install))

    assert mastra_renderer.available()
    assert mastra_renderer._node() == str(install / "node_modules" / ".bin" / "node")
    assert mastra_renderer.get_renderer().versions() == {
        "framework": "2.2.0",
        "adapter": "node v24.21.0",
    }


@pytest.mark.parametrize(
    ("spec_version", "mcp_package"),
    [
        ("2025-11-25", "mcp==1.26.0"),
        ("2026-07-28", "mcp==2.0.0"),
        (None, "mcp==2.0.0"),
    ],
)
def test_managed_source_command_selects_the_protocol_sdk(
    spec_version,
    mcp_package,
):
    command = build_managed_source_command(
        spec_version,
        Path("/tmp/source.json"),
        uv_executable="/usr/bin/uv",
    )

    assert mcp_package in command
    assert command[-4:] == [
        "-m",
        "mcp_mirror._source_worker",
        "/tmp/source.json",
        spec_version or "2026-07-28",
    ]
