"""A capability's hand-written prose has to stay true as rows are added.

The build opens every capability's notes with a tally derived from its cells, so the
summary after it, the why and any hand-written verdict may only state a mechanism. A count of adapters or a list of
frameworks goes stale the moment a row lands, which is how summaries written for five
frameworks came to be wrong for thirty-one.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_mirror.cells import published_notes, tally  # noqa: E402

SNAPSHOTS = [ROOT / "data", *sorted((ROOT / "data" / "specs").glob("*/"))]

NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|\\d+"
)
SUBJECTS = (
    r"(?:adapters?|frameworks?|capture|captured|boundar(?:y|ies)|renderers?|releases?|rows?|"
    r"definitions?|objects?|SDKs?)"
)
# A quantifier over these is a claim about the set of rows. "Only the captured
# definition" is about one boundary, so capture words are left out here.
ROWS = r"(?:adapters?|frameworks?|renderers?|SDKs?|boundar(?:y|ies))"
COUNTING = [
    # "three of five", "2 of 31", "all 5"
    re.compile(rf"\b(?:{NUMBER_WORDS})\s+of\b|\ball\s+(?:{NUMBER_WORDS})\b", re.IGNORECASE),
    # "Four frameworks", "five declared capture objects"
    re.compile(rf"\b(?:{NUMBER_WORDS})\b(?:\s+[\w-]+){{0,2}}?\s+{SUBJECTS}", re.IGNORECASE),
    # "rewritten in one.", "changed at one,"
    re.compile(rf"\b(?:at|in|by|to|from|for)\s+(?:{NUMBER_WORDS})\s*[.,;:]", re.IGNORECASE),
    # "every adapter", "only Mastra"-style exhaustive claims about the set of rows
    re.compile(rf"\b(?:all|every|each|no|none|only|most|other|remaining)\b(?:\s+[\w-]+){{0,3}}?\s+{ROWS}", re.IGNORECASE),
    # "both adapters", "both declared definitions"; "both sides" is not a count of rows
    re.compile(rf"\bboth\b(?:\s+[\w-]+){{0,2}}?\s+{SUBJECTS}", re.IGNORECASE),
    # "every captured tool definition" quantifies over the rows as surely as "every adapter"
    re.compile(r"\b(?:all|every|each|no|none)\b(?:\s+of)?(?:\s+the)?\s+captured\b", re.IGNORECASE),
    re.compile(r"\b(?:unanimous|unanimously|everywhere|nowhere)\b", re.IGNORECASE),
]


def framework_names() -> list[str]:
    """Every row name in any snapshot, without its qualifier, longest first."""
    names: set[str] = set()
    for snapshot in SNAPSHOTS:
        for agent in json.loads((snapshot / "frameworks.json").read_text())["agents"].values():
            names.add(re.sub(r"\s*\(.*\)$", "", agent["name"]))
    return sorted(names, key=len, reverse=True)


def named_frameworks(text: str, names: list[str]) -> set[str]:
    found: set[str] = set()
    for name in names:
        pattern = re.compile(rf"(?<![\w.]){re.escape(name)}(?![\w])")
        if pattern.search(text):
            found.add(name)
            text = pattern.sub(" ", text)
    return found


# Every hand-written overlay field that is published as prose.
FIELDS = (("summary",), ("why",), ("verdict", "headline"), ("verdict", "detail"))


def hand_written() -> list[tuple[str, str]]:
    out = []
    for snapshot in SNAPSHOTS:
        for path in sorted((snapshot / "capabilities").glob("*.json")):
            overlay = json.loads(path.read_text())
            for field in FIELDS:
                value = overlay
                for key in field:
                    value = value.get(key) if isinstance(value, dict) else None
                if value:
                    out.append((f"{path.relative_to(ROOT)}:{'.'.join(field)}", value))
    return out


@pytest.mark.parametrize(("where", "text"), hand_written(), ids=lambda value: value if ".json:" in value else "")
def test_prose_names_no_count_and_at_most_one_framework(where: str, text: str) -> None:
    for pattern in COUNTING:
        match = pattern.search(text)
        assert match is None, f"{where}: {match.group(0)!r} counts rows; the tally derives that"
    named = named_frameworks(text, framework_names())
    assert len(named) <= 1, f"{where}: names {sorted(named)}; per-framework facts belong in its note"


@pytest.mark.parametrize(
    "text",
    [
        "The annotation is absent from all five declared capture objects.",
        "Answered by three of five adapters.",
        "Four frameworks pass the description through byte for byte.",
        "Present in all 5 adapters.",
        "Both adapters keep it.",
        "Measured as present at every declared framework capture boundary.",
        "No adapter announces a roots change through its own API.",
        "2 of 31 keep it.",
        "The constraint remains intact in every captured tool definition.",
    ],
)
def test_the_guard_catches_counts(text: str) -> None:
    assert any(pattern.search(text) for pattern in COUNTING)


def test_the_guard_catches_lists_of_frameworks() -> None:
    names = framework_names()
    assert len(named_frameworks("CrewAI and OpenAI Agents SDK drop it.", names)) == 2
    assert named_frameworks("LangChain.js keeps it.", names) == {"LangChain.js"}


def test_tally_states_present_changed_dropped_and_only_the_rest_that_occur() -> None:
    assert tally(["y", "a", "n", "n"], "2025-11-25") == (
        "1 of 4 frameworks present at the capture boundary, 1 changed or retained elsewhere, 2 dropped."
    )
    assert tally(["y", "u", "x", "x"], "2026-07-28") == (
        "1 of 4 frameworks present at the capture boundary, 0 changed or retained elsewhere, "
        "0 dropped, 1 unmeasured, 2 cannot reach MCP 2026-07-28."
    )


def test_published_notes_count_every_public_row_and_keep_the_summary_apart() -> None:
    agents = {
        "public": {"current_version": "1"},
        "preview": {"current_version": "1", "visibility": "preview"},
        "alias": {"current_version": "1", "aliases_of": "public"},
    }
    stats = {"public": {"1": "y"}, "preview": {"1": "n #1"}, "alias": {"1": "y"}}
    assert published_notes(stats, agents, "2025-11-25", "A mechanism.") == {
        "notes": "2 of 2 frameworks present at the capture boundary, 0 changed or retained elsewhere, "
        "0 dropped. A mechanism.",
        "summary": "A mechanism.",
    }
