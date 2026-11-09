from __future__ import annotations

import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "validator"))
from validate import validate_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "specs" / "2026-07-28"
BEHAVIOR_ANNOTATIONS = {
    "destructive-hint",
    "idempotent-hint",
    "open-world-hint",
    "read-only-hint",
}


def test_2026_snapshot_keeps_same_protocol_evidence_separate() -> None:
    frameworks = json.loads((SNAPSHOT / "frameworks.json").read_text())
    report = json.loads((SNAPSHOT / "report.json").read_text())
    attempts = json.loads((SNAPSHOT / "attempts.json").read_text())

    assert frameworks["mcp_spec"] == report["mcp_spec_version"] == "2026-07-28"
    assert report["mcp_spec_version_evidence"] == (
        "direct source connection server/discover result"
    )

    runs = {run["framework"]: run for run in report["runs"]}
    # A re-measured row moves out of the shared report into its own scan file.
    for name in ("mastra", "pydantic_ai", "openai_agents"):
        own = SNAPSHOT / "measurements" / "scan" / f"{name}.json"
        if own.exists():
            runs.update({run["framework"]: run for run in json.loads(own.read_text())["runs"]})
    assert set(runs) == {"mastra", "pydantic_ai", "openai_agents"}
    assert {
        run["evidence"]["negotiated_mcp_spec_version"]
        for run in runs.values()
    } == {"2026-07-28"}
    # Every adapter that reaches the revision keeps the parameter contract intact;
    # what moves is identity, annotations, icons and the output schema. Mastra is the
    # only one that renames.
    assert {
        difference["dimension"]
        for run in runs.values()
        for difference in run["differences"]
    } == {"annotation", "icons", "name", "output_schema"}
    assert {
        difference["dimension"]
        for run in (runs["pydantic_ai"], runs["openai_agents"])
        for difference in run["differences"]
    } == {"annotation", "icons", "output_schema"}

    agents = {a: agent for a, agent in frameworks["agents"].items() if not agent.get("aliases_of")}
    assert agents["pydantic-ai"]["current_version"] == "2.54.0"
    assert agents["openai-agents"]["current_version"] == "0.23.1"
    assert agents["mastra"]["current_version"] == "2.2.0"
    for agent_id, run_id in (("pydantic-ai", "pydantic_ai"), ("openai-agents", "openai_agents"), ("mastra", "mastra")):
        assert runs[run_id]["framework_version"] == agents[agent_id]["current_version"]
    assert {
        agent_id
        for agent_id, agent in agents.items()
        if agent["measurement_status"] != "measured"
    } == {"autogen", "beeai", "crewai", "crewai-native", "genkit_js", "google_adk", "google_adk_java", "google_adk_js", "haystack", "koog", "langchain", "llamaindex", "ms_agent_framework", "pi", "semantic_kernel", "smolagents", "spring_ai"}
    assert set(attempts["attempts"]) == {"autogen", "beeai", "crewai", "crewai-native", "genkit_js", "google_adk", "google_adk_java", "google_adk_js", "haystack", "koog", "langchain", "llamaindex", "ms_agent_framework", "pi", "semantic_kernel", "smolagents", "spring_ai"}
    for agent_id, attempt in attempts["attempts"].items():
        assert attempt["status"] == agents[agent_id]["measurement_status"]
        assert attempt["framework_version"] == agents[agent_id]["current_version"]
        assert attempt["evidence"]


def test_2026_capability_cells_match_the_saved_report() -> None:
    agents = {
        a: agent
        for a, agent in json.loads((SNAPSHOT / "frameworks.json").read_text())["agents"].items()
        if not agent.get("aliases_of")
    }
    blocked = {
        agent_id: (agent["current_version"], agent["measurement_note"])
        for agent_id, agent in agents.items()
        if agent["measurement_status"] != "measured"
    }

    _root, capabilities, errors = validate_all(SNAPSHOT)
    assert errors == [], errors
    for capability in capabilities:
        capability_id = capability["id"]
        stats = capability["stats"]

        # The revision and fixture are fixed for this snapshot; the run date is not,
        # because features are measured on different days as coverage grows.
        measured = capability["measured"]
        assert measured["mcp_spec"] == "2026-07-28"
        # Each row names the fixture its instruments' captures ran against.
        assert all(part.startswith("fixtures/") and part.endswith(".py") for part in measured["fixture"].split(", ")), measured["fixture"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", measured["run_date"] or ""), (
            capability_id,
            measured["run_date"],
        )

        # An adapter that cannot open a session at this revision reaches none of its
        # features. That is a measured outcome carrying the evidence for it, not the
        # 'u' we owe a feature nobody has probed.
        for agent_id, (version, note) in blocked.items():
            cell = stats[agent_id][version]
            assert cell.startswith("x #"), (capability_id, agent_id, cell)
            assert capability["notes_by_num"][cell.split("#")[1]] == note

        openai_code = stats["openai-agents"][agents["openai-agents"]["current_version"]].split()[0]
        pydantic_code = stats["pydantic-ai"][agents["pydantic-ai"]["current_version"]].split()[0]
        mastra_code = stats["mastra"][agents["mastra"]["current_version"]].split()[0]

        # Only the rows derived from the saved tool-definition scan are pinned to exact
        # codes; that report is what they are a transcription of. Features measured by
        # other means (result capture, adapter surface probes) have their own evidence,
        # so this check holds them to the vocabulary and the note requirement instead.
        # This used to test for a "uvx mcp-mirror scan" prefix no published row ever
        # carried, so the pinned branch below never ran. Rows now say which
        # instrument produced them.
        if "scan" not in capability["instruments"]:
            for agent_id, code in (
                ("openai-agents", openai_code),
                ("pydantic-ai", pydantic_code),
                ("mastra", mastra_code),
            ):
                assert code in {"y", "a", "n"}, (capability_id, agent_id, code)
                if code != "y":
                    cell = stats[agent_id][agents[agent_id]["current_version"]]
                    assert "#" in cell, (capability_id, agent_id, cell)
                    assert capability["notes_by_num"][cell.split("#")[1]].strip()
            continue

        codes = (openai_code, pydantic_code, mastra_code)
        if capability_id in BEHAVIOR_ANNOTATIONS:
            assert codes == ("n", "a", "a")
        elif capability_id == "title-annotation":
            assert codes == ("a", "a", "a")
        elif capability_id == "tool-icons":
            # The only part of a tool definition aimed at a human, and the only one
            # no adapter carries.
            assert codes == ("n", "n", "n")
        elif capability_id == "tool-output-schema":
            # Dropped by the OpenAI Agents SDK; kept by Pydantic AI as a return schema
            # the model is not shown by default; declared by Mastra's tool action.
            assert codes == ("n", "a", "y")
        elif capability_id == "tool-name":
            # Mastra namespaces every tool as <serverKey>_<toolName>.
            assert codes == ("y", "y", "a")
        else:
            assert codes == ("y", "y", "y")
