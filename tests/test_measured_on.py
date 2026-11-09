"""Every published row says when it was measured, and the date comes from its captures."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "validator"))
from validate import _cells_module  # noqa: E402

cells = _cells_module()
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _load_build_data():
    spec = importlib.util.spec_from_file_location("build_data", ROOT / "scripts" / "build_data.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_framework_dates_read_each_capture_and_let_the_later_file_win(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap"
    _write(snapshot / "measurements" / "scan" / "a.json", {
        "provenance": {"generated_at": "2026-09-30T22:19:53Z"},
        "runs": [
            {"framework": "pydantic_ai", "status": "measured"},
            {"framework": "mastra", "status": "measured"},
            {"framework": "skipped", "status": "failed"},
        ],
    })
    _write(snapshot / "measurements" / "scan" / "b.json", {
        "provenance": {"generated_at": "2026-10-09T19:16:55.203333Z"},
        "runs": [{"framework": "mastra", "status": "measured"}],
    })
    _write(snapshot / "measurements" / "results" / "all-frameworks.json", {
        "captured_at": "2026-10-01T18:26:06Z",
        "frameworks": {"pydantic_ai": {"ok": True}, "mastra": {"ok": False}},
    })
    _write(snapshot / "measurements" / "transports" / "run.json", {
        "captured_at": "2026-10-02T10:00:00Z",
        "frameworks": {"mastra": {"transports": {}}},
    })
    _write(snapshot / "measurements" / "protocol" / "mastra" / "run.json", {
        "renderer": "mastra", "captured_at": "2026-10-03T10:00:00Z",
    })
    _write(snapshot / "measurements" / "conformance" / "pydantic_ai" / "checks.json", {
        "renderer": "pydantic_ai", "captured_at": "2026-10-04T10:00:00Z",
    })
    _write(snapshot / "attempts.json", {
        "run_date": "2026-09-29", "attempts": {"crewai_native": {"status": "sdk_incompatible"}},
    })

    dates = cells._framework_dates(snapshot)

    # Renderer ids are published ids; a run that did not measure carries no date.
    assert dates == {
        "pydantic-ai": {"scan": "2026-09-30", "results": "2026-10-01", "conformance": "2026-10-04"},
        "mastra": {"scan": "2026-10-09", "transports": "2026-10-02", "protocol": "2026-10-03"},
        "crewai-native": {"attempt": "2026-09-29"},
    }


def test_merge_overlay_dates_each_cell_from_the_instruments_that_fed_it() -> None:
    agents = {
        "mastra": {"name": "Mastra", "current_version": "2.2.0", "version_list": []},
        "langchain": {"name": "LangChain", "current_version": "0.3.2", "version_list": []},
        "langgraph": {"name": "LangGraph", "current_version": "1.0", "version_list": [], "aliases_of": "langchain"},
        "late": {"name": "Late", "current_version": "1", "version_list": [], "visibility": "preview"},
    }
    context = {
        "cells": {"cap": {
            "mastra": cells.Cell("y", [], "scan+protocol"),
            "langchain": cells.Cell("a", ["note"], "results"),
        }},
        "dependent": {},
        "dates": {"scan": "2026-10-09", "protocol": "2026-10-03", "results": "2026-09-30"},
        "framework_dates": {
            "mastra": {"scan": "2026-10-01", "protocol": "2026-10-03"},
            "langchain": {"results": "2026-09-30"},
        },
        "fixtures": {},
        "reproduce": {},
    }
    overlay = {"id": "cap", "title": "Cap", "description": "d", "notes": {"langchain": "why"}}
    published = cells.merge_overlay(overlay, {"agents": agents, "mcp_spec": "2025-11-25"}, context)

    # The newest day across the cell's own instruments, not the snapshot-wide latest;
    # the alias carries its source's date; a row without a cell has no date.
    assert published["measured"]["measured_on"] == {
        "mastra": "2026-10-03", "langchain": "2026-09-30", "langgraph": "2026-09-30",
    }
    assert published["measured"]["run_date"] == "2026-10-09"


def test_build_refuses_a_published_row_without_a_date() -> None:
    build_data = _load_build_data()
    agents = {"a": {}, "b": {}, "p": {"visibility": "preview"}}
    capabilities = [
        {"measured": {"measured_on": {"a": "2026-10-01"}}},
        {"measured": {"measured_on": {"a": "2026-10-09"}}},
    ]
    dates, missing = build_data.measured_on(agents, capabilities)
    assert dates == {"a": "2026-10-09"}
    assert missing == ["b"]


def test_an_agent_with_no_session_has_no_date_and_carries_the_reason(tmp_path: Path) -> None:
    build_data = _load_build_data()
    root = tmp_path / "measurements" / "wire" / "demo"
    root.mkdir(parents=True)
    reason = "scripts/harnesses/demo.py has no STUB_KIND; level 1 needs one"
    (root / "session.json").write_text(json.dumps({
        "harness": "demo", "version": "1.0.0", "captured_at": "2026-09-30T03:43:10+00:00",
        "level_1": {"status": "skipped", "reason": reason, "sessions": []},
        "level_2": {"status": "skipped", "reason": "no STUB_KIND", "sessions": []},
    }))
    run = cells.HarnessRun(root / "session.json")
    assert run.blocked() == reason
    agents = {"demo": {}, "ran": {}}
    rows = [{"measured": {"measured_on": {"ran": "2026-10-01"}}}]
    dates, blocked, missing = build_data.harness_dates(agents, rows, {"demo": run})
    assert dates == {"ran": "2026-10-01"}
    assert blocked == {"demo": reason}
    assert missing == []
    # A harness that did run and still has no date is a build error, as before.
    assert build_data.harness_dates({"ran": {}, "other": {}}, rows, {})[2] == ["other"]
    # The committed Cursor run opened no session, so the module's one line is the reason.
    assert build_data.unmeasured_reason("cursor_agent", "fallback").startswith("The CLI only talks to Cursor")
    assert build_data.unmeasured_reason("no_such_harness", "fallback") == "fallback"


def test_every_published_row_carries_an_iso_measured_date() -> None:
    for path in sorted((ROOT / "data" / "fulldata").glob("data-*.json")):
        database = json.loads(path.read_text())
        assert ISO_DATE.fullmatch(database["measured_on"]), path.name
        for agent_id, agent in database["agents"].items():
            if cells.is_preview(agent):
                continue
            assert ISO_DATE.fullmatch(agent.get("measured_on") or ""), (path.name, agent_id)
        for capability in database["data"].values():
            measured_on = capability["measured"]["measured_on"]
            for agent_id, agent in database["agents"].items():
                code = (capability["stats"].get(agent_id, {}).get(agent["current_version"]) or "u").split()[0]
                if code != "u":
                    assert ISO_DATE.fullmatch(measured_on.get(agent_id) or ""), (path.name, capability["id"], agent_id)
                assert measured_on.get(agent_id, "") <= database["measured_on"]

    harnesses = json.loads((ROOT / "data" / "fulldata" / "harnesses.json").read_text())
    assert ISO_DATE.fullmatch(harnesses["measured_on"])
    # An agent whose run opened no session has no date and says why instead; the
    # rest carry a date on every row.
    measured = {a for a, harness in harnesses["agents"].items() if not harness.get("measurement_blocked")}
    for harness_id, harness in harnesses["agents"].items():
        if harness_id in measured:
            assert ISO_DATE.fullmatch(harness.get("measured_on") or ""), harness_id
        else:
            assert harness["measured_on"] is None and harness["measurement_blocked"], harness_id
    for row in harnesses["data"].values():
        assert set(row["measured"]["measured_on"]) == measured, row["id"]

    index = json.loads((ROOT / "data" / "fulldata" / "specs.json").read_text())
    assert all(ISO_DATE.fullmatch(spec["measured_on"]) for spec in index["specs"])
    assert ISO_DATE.fullmatch(index["harnesses"]["measured_on"])
