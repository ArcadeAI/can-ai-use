"""The home page's new list is what each revision adds, not a sort by a shared run date."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_build_data():
    spec = importlib.util.spec_from_file_location("build_data", ROOT / "scripts" / "build_data.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _aggregate(revision: str, ids: list[str], hidden: tuple[str, ...] = ()) -> tuple:
    data = {capability_id: {"title": capability_id, "shown": capability_id not in hidden} for capability_id in ids}
    return ({"mcp_spec": revision, "data": data}, [], {})


def test_each_revision_lists_what_the_older_one_lacks():
    build_data = _load_build_data()
    added = build_data.new_capabilities(
        [
            _aggregate("2026-07-28", ["ping", "server-discover", "hidden-new"], hidden=("hidden-new",)),
            _aggregate("2025-11-25", ["ping", "initialize-handshake"]),
        ]
    )
    assert added == {
        "2025-11-25": [],
        "2026-07-28": [{"id": "server-discover", "title": "server-discover"}],
    }


def test_published_index_matches_the_published_snapshots():
    index = json.loads((ROOT / "docs" / "data" / "specs.json").read_text())
    ordered = sorted(index["specs"], key=lambda entry: entry["id"])
    for older, newer in zip([None, *ordered], ordered):
        if older is None:
            assert newer["new_capabilities"] == []
            continue
        previous = json.loads((ROOT / "docs" / "data" / older["file"]).read_text())["data"]
        current = json.loads((ROOT / "docs" / "data" / newer["file"]).read_text())["data"]
        expected = [
            capability_id
            for capability_id, capability in current.items()
            if capability_id not in previous and capability.get("shown", True)
        ]
        assert [entry["id"] for entry in newer["new_capabilities"]] == expected
