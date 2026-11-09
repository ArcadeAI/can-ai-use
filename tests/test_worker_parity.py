"""The Worker's TypeScript catalog (worker/catalog.ts) must not drift from catalog.py.

Both implementations answer the cases in worker_parity_cases.json from the built site
data in docs/data, and every answer, including each error message, must be identical.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from mcp_mirror import catalog

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "data"
CASES = Path(__file__).with_name("worker_parity_cases.json")
DUMP = Path(__file__).with_name("worker_parity_dump.mjs")


@pytest.fixture
def built_site_data(monkeypatch):
    """Point catalog.py at docs/data the way the Worker reads it: index first, then files."""

    monkeypatch.setenv("MCP_MIRROR_DATA_URL", DATA_DIR.as_uri())
    monkeypatch.setattr(catalog, "_cache", {"at": 0.0, "snapshots": None, "default": None})
    yield
    catalog._cache.update(at=0.0, snapshots=None, default=None)


def python_answer(case: dict) -> dict:
    args = case["arguments"]
    revision = args.get("revision", "default")
    frameworks = args.get("frameworks")
    try:
        if case["tool"] == "list_snapshots":
            result = catalog.list_snapshots_data()
        elif case["tool"] == "search_capabilities":
            result = catalog.search_capabilities_data(args["query"], revision)
        elif case["tool"] == "lookup_capability":
            result = catalog.capability_data(args["capability_id"], revision, frameworks)
        elif case["tool"] == "inspect_tool_definition":
            result = catalog.inspect_tool_definition_data(args["tool"], revision, frameworks)
        elif case["tool"] == "inspect_tool_result":
            result = catalog.inspect_tool_result_data(args["result"], revision, frameworks)
        else:
            raise AssertionError(f"unknown tool {case['tool']}")
    except ValueError as error:
        return {"ok": False, "error": str(error)}
    return {"ok": True, "result": result}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_worker_catalog_matches_catalog_py(built_site_data) -> None:
    cases = json.loads(CASES.read_text())
    tools = {case["tool"] for case in cases}
    assert tools == {
        "list_snapshots",
        "search_capabilities",
        "lookup_capability",
        "inspect_tool_definition",
        "inspect_tool_result",
    }

    completed = subprocess.run(
        ["node", str(DUMP), str(DATA_DIR)],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    worker_answers = json.loads(completed.stdout)
    assert len(worker_answers) == len(cases)

    for case, worker_answer in zip(cases, worker_answers):
        expected = json.loads(json.dumps(python_answer(case)))
        assert worker_answer == expected, f"{case['tool']} {json.dumps(case['arguments'])[:120]}"

    # The cases exercise both outcomes, so an implementation that errors everywhere
    # (or never) cannot pass by accident.
    assert any(answer["ok"] for answer in worker_answers)
    assert any(not answer["ok"] for answer in worker_answers)
