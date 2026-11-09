"""Alias rows reuse their source's derived cells; preview rows stay out of public counts."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "validator"))
from validate import validate_aliases, validate_all, validate_capability  # noqa: E402

SNAPSHOTS = [ROOT / "data", ROOT / "data" / "specs" / "2026-07-28"]


def _code(capability: dict, agent_id: str, agents: dict) -> str:
    return capability["stats"][agent_id][agents[agent_id]["current_version"]].split()[0]


@pytest.mark.parametrize("snapshot", SNAPSHOTS, ids=lambda p: p.name)
def test_alias_row_equals_its_source_cells(snapshot: Path) -> None:
    root, capabilities, errors = validate_all(snapshot)
    assert errors == []
    agents = root["agents"]
    aliases = {a: agent["aliases_of"] for a, agent in agents.items() if agent.get("aliases_of")}
    assert aliases.get("langgraph") == "langchain"
    for capability in capabilities:
        for alias, source in aliases.items():
            assert _code(capability, alias, agents) == _code(capability, source, agents), capability["id"]
            if source in capability.get("boundary_dependent", []):
                assert alias in capability["boundary_dependent"]
    # The alias inherits the measurement, so the version it shows is the one measured.
    assert agents["langgraph"]["current_version"] == agents["langchain"]["current_version"]
    assert agents["langgraph"]["package"] == agents["langchain"]["package"]


@pytest.mark.parametrize("snapshot", SNAPSHOTS, ids=lambda p: p.name)
def test_preview_rows_stay_out_of_verdicts_and_aliases_are_named_with_their_source(snapshot: Path) -> None:
    root, capabilities, _errors = validate_all(snapshot)
    agents = root["agents"]
    previews = [agent["name"] for agent in agents.values() if agent.get("visibility") == "preview"]
    aliases = [(agent["name"], agents[agent["aliases_of"]]["name"]) for agent in agents.values() if agent.get("aliases_of")]
    assert aliases
    for capability in capabilities:
        text = capability["verdict"]["headline"] + capability["verdict"]["detail"]
        for name in previews:
            assert f"{name} " not in text, capability["id"]
        for alias, source in aliases:
            assert (f"{alias} " in text) == (f"{source} " in text), capability["id"]


def test_preview_notes_are_numbered_after_published_ones() -> None:
    built = json.loads((ROOT / "data" / "fulldata" / "data-2025-11-25.json").read_text())
    agents = built["agents"]
    public = [a for a, agent in agents.items() if agent.get("visibility") != "preview"]
    for capability in built["data"].values():
        refs = [
            int(ref.lstrip("#"))
            for a in public
            for ref in capability["stats"][a][agents[a]["current_version"]].split()[1:]
        ]
        assert refs == list(range(1, len(refs) + 1)), capability["id"]


def test_specs_index_counts_exclude_preview_rows() -> None:
    index = json.loads((ROOT / "data" / "fulldata" / "specs.json").read_text())
    for entry in index["specs"]:
        built = json.loads((ROOT / "data" / "fulldata" / entry["file"]).read_text())
        public = [agent for agent in built["agents"].values() if agent.get("visibility") != "preview"]
        assert len(public) <= len(built["agents"])
        assert entry["total_frameworks"] == len(public)
        assert entry["measured_frameworks"] == sum(
            1 for agent in public if agent.get("measurement_status", "measured") == "measured"
        )


def test_validator_rejects_malformed_aliases_and_visibility() -> None:
    raw = json.loads((ROOT / "data" / "frameworks.json").read_text())["agents"]

    def errors_for(**changes: dict) -> list[str]:
        agents = copy.deepcopy(raw)
        for agent_id, entry in changes.items():
            agents[agent_id] = entry
        errors: list[str] = []
        validate_aliases(agents, errors)
        return errors

    assert errors_for() == []
    alias = raw["langgraph"]
    assert any("unknown agent" in e for e in errors_for(langgraph={**alias, "aliases_of": "nope"}))
    assert any("cannot declare" in e for e in errors_for(langgraph={**alias, "version_list": []}))
    assert any("itself an alias" in e for e in errors_for(second={**alias, "aliases_of": "langgraph"}))
    assert any("alias_note" in e for e in errors_for(langgraph={**alias, "alias_note": ""}))
    assert any("visibility" in e for e in errors_for(crewai={**raw["crewai"], "visibility": "hidden"}))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_site_hides_preview_rows_unless_flagged() -> None:
    proc = subprocess.run(
        ["node", "--test", "tests/test_visibility.mjs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_validator_rejects_capability_files_that_are_not_overlays(tmp_path: Path) -> None:
    # A row in any other schema would carry its own cells into the build.
    root, _capabilities, _errors = validate_all(ROOT / "data")
    path = tmp_path / "destructive-hint.json"
    overlay = json.loads((ROOT / "data" / "capabilities" / "destructive-hint.json").read_text())
    path.write_text(json.dumps({**overlay, "schema": "mcp-mirror/capability@3"}))
    errors: list[str] = []
    assert validate_capability(path, root["agents"], root["statuses"], errors, root, {}) is None
    assert any("is not mcp-mirror/capability@4" in error for error in errors)
