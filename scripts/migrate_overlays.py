"""Convert capability@3 files into capability@4 overlays, once.

    uv run python scripts/migrate_overlays.py [--write]

An overlay keeps what a person wrote (description, question, why, known issues,
links, per-framework notes, and any verdict prose that says more than the generated
template would) and drops what the artifacts now produce (stats, note numbering,
run date, reproduce command, boundary dependence).

Without --write it only reports how the published rows would change, which is the
review step: every change must be either a correction the instruments found or a
generated field replacing a hand-typed one.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp_mirror import cells  # noqa: E402

SNAPSHOTS = [ROOT / "data", *sorted(p for p in (ROOT / "data" / "specs").iterdir() if p.is_dir())]
GENERIC_PREFIXES = (
    "Exposed by both adapters",
    "Exposed by all",
    "Mixed across the",
    "Not yet",
)
KEEP = (
    "id", "title", "description", "question", "spec", "status", "categories",
    "keywords", "usage_perc_y", "usage_perc_a", "usage_note", "links",
    "known_issues", "why", "shown", "docs_url", "docs_label", "example",
)


def to_overlay(published: dict) -> dict:
    overlay: dict = {"schema": cells.OVERLAY_SCHEMA}
    for key in KEEP:
        if key in published:
            overlay[key] = copy.deepcopy(published[key])
    notes: dict[str, str] = {}
    for agent_id, versions in published["stats"].items():
        raw = next(iter(versions.values()))
        if "#" not in raw or raw.startswith("x"):
            continue  # 'x' evidence comes from attempts.json / measurement_note
        number = raw.split("#", 1)[1].strip()
        note = published["notes_by_num"].get(number)
        if note:
            notes[agent_id] = note
    if notes:
        overlay["notes"] = notes
    verdict = published.get("verdict") or {}
    headline = verdict.get("headline", "")
    if headline and not headline.startswith(GENERIC_PREFIXES):
        # Bespoke verdict prose is kept. Its code is not: a typed code drifts from the
        # cells, and the site shows the counts instead.
        overlay["verdict"] = {
            "headline": headline,
            "detail": verdict.get("detail", ""),
        }
    summary = published.get("notes", "")
    if summary and not summary.startswith(GENERIC_PREFIXES) and summary != headline:
        overlay["summary"] = summary
    return overlay


def codes(capability: dict) -> dict[str, str]:
    return {a: next(iter(v.values())).split()[0] for a, v in capability["stats"].items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    for snapshot in SNAPSHOTS:
        directory = snapshot / "capabilities"
        if not directory.exists():
            continue
        root_data = json.loads((snapshot / "frameworks.json").read_text())
        context = cells.snapshot_context(snapshot, ROOT)
        print(f"\n## {snapshot.relative_to(ROOT)}")
        for path in sorted(directory.glob("*.json")):
            published = json.loads(path.read_text())
            if published.get("schema") == cells.OVERLAY_SCHEMA:
                continue
            overlay = to_overlay(published)
            merged = cells.merge_overlay(overlay, root_data, context)
            before, after = codes(published), codes(merged)
            changed = {a: (before.get(a), after.get(a)) for a in after if before.get(a) != after.get(a)}
            if changed:
                print(f"  {published['id']:<28} cells {changed}")
            if sorted(published.get("boundary_dependent") or []) != merged["boundary_dependent"]:
                print(f"  {published['id']:<28} boundary_dependent {published.get('boundary_dependent')} -> {merged['boundary_dependent']}")
            if arguments.write:
                path.write_text(json.dumps(overlay, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
