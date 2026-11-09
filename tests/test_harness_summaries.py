"""Each published harness note has a one-line summary read off its evidence.

The summary is what a reader sees first; the evidence sits behind it. So it must exist
for every note, say nothing the cell's code and label contradict, and carry only counts
the evidence itself carries.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _build_data():
    spec = importlib.util.spec_from_file_location("build_data", ROOT / "scripts" / "build_data.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_data", module)
    spec.loader.exec_module(module)
    return module


build_data = _build_data()
PUBLISHED = json.loads((ROOT / "data" / "fulldata" / "harnesses.json").read_text())["data"]


def _noted():
    for row_id, row in PUBLISHED.items():
        for agent_id, cell in row["cells"].items():
            if cell["note"]:
                yield row_id, agent_id, cell, row["evidence"][agent_id], row["values"].get(agent_id)


def test_every_published_note_has_a_summary() -> None:
    noted = list(_noted())
    assert noted
    missing = [(row_id, agent_id) for row_id, agent_id, cell, _, _ in noted if not cell.get("summary")]
    assert not missing
    for row in PUBLISHED.values():
        for cell in row["cells"].values():
            assert bool(cell["note"]) == bool(cell.get("summary"))


def test_summaries_are_the_ones_build_data_derives() -> None:
    for row_id, agent_id, cell, evidence, value in _noted():
        assert cell["summary"] == build_data.harness_summary(row_id, cell["code"], evidence, value), (row_id, agent_id)


def test_summaries_agree_with_the_cell_code_and_label() -> None:
    for row_id, agent_id, cell, evidence, _ in _noted():
        summary, where = cell["summary"], (row_id, agent_id, cell["summary"])
        if cell["label"] == "ran without asking":
            assert "held" not in summary.lower(), where
            assert "with no prompt" in summary, where
        if cell["label"] == "blocked by prompt":
            assert "prompt" in summary, where
        if cell["label"] == "none offered":
            assert "none of the" in summary, where
        if cell["code"] == "n":
            assert not re.search(r"\b(every run|all runs|all \d+ runs)\b", summary), where
        counts = build_data.harness_run_counts(row_id, evidence)
        if counts and cell["code"] in "yn":
            # The outcome a summary leads with is the one its code was voted for.
            assert counts[0]["code"] == cell["code"], where
            assert sum(entry["count"] for entry in counts if entry["code"] == cell["code"]) > sum(
                entry["count"] for entry in counts if entry["code"] != cell["code"]
            ), where
        if cell["code"] == "u" and not counts:
            context = any(line.startswith(build_data.cells.CONTEXT_NOTE) for line in evidence)
            assert context or summary.startswith("Not measured: "), where


def test_summary_counts_match_the_evidence() -> None:
    for row_id, agent_id, cell, evidence, value in _noted():
        where = (row_id, agent_id, cell["summary"])
        # Every number a summary quotes is one its evidence states. Tallies ("k of n",
        # "in k", "changed k") are counted off the evidence lines, and checked below.
        source = " ".join(evidence) + " " + (value or "")
        quoted = re.sub(r"(?:in )?\d+ of \d+|none of \d+ pings|\bin \d+\b|changed \d+", "", cell["summary"])
        for number in re.findall(r"(?<![\w.-])\d+(?![\w.-])", quoted):
            assert re.search(rf"(?<!\d){number}(?!\d)", source), (number, where)
        if row_id in ("h-reacts-to-list-changed", "h-relist-next-turn") and cell["code"] != "u":
            quiet = sum(line.endswith("list_changed: no tools/list after it") for line in evidence)
            fired = quiet + sum("notifications/tools/list_changed -> " in line for line in evidence)
            assert f"in {quiet} of {fired} " in cell["summary"], where
            if row_id == "h-relist-next-turn":
                lacked = sum(line.endswith("did not offer the model the added tool") for line in evidence)
                offered = sum(bool(re.search(r"request \d+ offered the model ", line)) for line in evidence)
                if lacked or offered:
                    assert f"lacked the added tool in {lacked}" in cell["summary"], where
                else:
                    # A live row: no model request is visible, and the summary says so.
                    assert "what the next model request offered is not visible" in cell["summary"], where
        if row_id == "h-answers-ping":
            for transport in ("stdio", "HTTP"):
                pings = [line for line in evidence if line.startswith(f"{transport}: ")]
                answered = sum(not line.endswith("no answer within 5s") for line in pings)
                if pings:
                    tally = f"{answered} of {len(pings)}" if answered else f"none of {len(pings)}"
                    assert f"{tally} pings on {transport}" in cell["summary"], where
        if row_id == "h-tools-offered-to-model" and value:
            offered, listed, same = re.search(r"(\d+)/(\d+) tools offered; descriptions \d+/\d+ unchanged; "
                                              r"schemas (\d+)/\d+ unchanged", value).groups()
            if offered != "0":
                assert f"Offered {offered} of {listed}" in cell["summary"], where
                if same != offered:
                    assert f"changed {int(offered) - int(same)} input schema" in cell["summary"], where
        counts = build_data.harness_run_counts(row_id, evidence)
        if not counts:
            continue
        # Every run is counted once: per revision, the outcomes add up to the runs it
        # says it made, and its y outcomes to the passes it reports.
        by_revision, _, _, _ = build_data._split_evidence(evidence)
        for revision, lines in by_revision.items():
            made = sum(int(m[1]) for line in lines if (m := build_data.RUN_TOTAL.match(line)))
            only = build_data.harness_run_counts(row_id, [f"{revision} fixture: {line}" for line in lines])
            assert sum(entry["count"] for entry in only) == made, (revision, where)
            for line in lines:
                if m := build_data.RUN_PASSED.match(line):
                    assert sum(e["count"] for e in only if e["code"] == "y") == int(m[1]), (revision, where)
        total = sum(entry["count"] for entry in counts)
        assert f"{counts[0]['count']} of {total} runs" in cell["summary"], where


def test_summary_leads_with_the_context_and_keeps_the_counts() -> None:
    evidence = [
        "2025-11-25 fixture: 3/3 runs",
        "2025-11-25 fixture: run 1: reached the server: def seq 4 tools/call",
        "2025-11-25 fixture: run 2: reached the server: def seq 4 tools/call",
        "2025-11-25 fixture: run 3: reached the server: def seq 4 tools/call",
        "2025-11-25 fixture: 3 runs, stub:x (scripted); default mode (none); invoked with prompt 'go'",
        "context: measured headless (x run), which approves its own permission requests (see x.go).",
    ]
    summary = build_data.harness_summary("h-destructive-blocked-by-default", "n", evidence, None)
    assert summary == (
        "Measured headless (x run), which approves its own permission requests. "
        "Sent delete_account to the server with no prompt in 3 of 3 runs."
    )


def test_summary_of_a_held_call_says_nothing_reached_the_server() -> None:
    evidence = [
        f"{revision} fixture: {line}"
        for revision in ("2025-11-25", "2026-07-28")
        for line in [
            "0/3 runs",
            *(f"run {i}: held: def carries a permission event, no tools/call reached def" for i in (1, 2, 3)),
            "3 runs, stub:x (scripted); default mode (none); invoked with prompt 'go'",
        ]
    ]
    assert build_data.harness_summary("h-destructive-blocked-by-default", "y", evidence, None) == (
        "Held delete_account behind a permission prompt in 6 of 6 runs, on both fixture revisions; "
        "no tools/call reached the server."
    )


def test_summary_of_followed_pagination_counts_the_pages() -> None:
    evidence = [
        "2025-11-25 fixture: level1-2025-11-25-stdio/util.jsonl seq 6 tools/list cursor=None -> nextCursor='20'",
        "2025-11-25 fixture: level1-2025-11-25-stdio/util.jsonl seq 9 tools/list cursor='20' -> nextCursor='40'",
        "2025-11-25 fixture: level1-2025-11-25-stdio/util.jsonl seq 11 tools/list cursor='40' -> nextCursor=None",
        "2026-07-28 fixture: level1-2026-07-28-stdio/util.jsonl seq 6 tools/list cursor=None -> nextCursor='20'",
        "2026-07-28 fixture: level1-2026-07-28-stdio/util.jsonl seq 9 tools/list cursor='20' -> nextCursor='40'",
        "2026-07-28 fixture: level1-2026-07-28-stdio/util.jsonl seq 11 tools/list cursor='40' -> nextCursor=None",
        "context: 0.162.0 started following nextCursor in legacy protocol mode.",
    ]
    summary = build_data.harness_summary("h-follows-pagination", "y", evidence, None)
    assert summary == (
        "0.162.0 started following nextCursor in legacy protocol mode. "
        "Followed every tools/list nextCursor to the last page, on both fixture revisions."
    )


def test_summary_of_a_listed_slow_server_reads_the_wait() -> None:
    evidence = [
        "2025-11-25 fixture: level1-2025-11-25-slow-start/util.jsonl seq 6 tools/list; util answered its first request after 8.2s",
        "2026-07-28 fixture: level1-2026-07-28-slow-start/util.jsonl seq 6 tools/list; util answered its first request after 8.3s",
        "fresh.json: all four freshness checks (home, servers_listed, tools_offered, egress) passed in 72/72 sessions",
        "context: level 1 held the stub's reply 10 s, which matches Pi's documented first-prompt wait. More words here.",
    ]
    assert build_data.harness_summary("h-startup-timeout", "y", evidence, None) == (
        "Level 1 held the stub's reply 10 s, which matches Pi's documented first-prompt wait. "
        "Listed util's tools after it answered its first request 8.2s and 8.3s late."
    )
