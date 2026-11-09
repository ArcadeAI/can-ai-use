#!/usr/bin/env python3
"""Validate the support data and generate the caniuse-shaped aggregate.

Same split caniuse uses: per-entry files are the source of truth (so `git log` on
one file is that capability's changelog), a single aggregate is generated for
consumers, and the website is just one consumer of that aggregate.

    data/frameworks.json                  legacy source snapshot
    data/capabilities/*.json
    data/specs/<revision>/frameworks.json separate protocol snapshots
    data/specs/<revision>/capabilities/*
    data/harnesses/                       the harness table, measured on the wire

Each source snapshot produces ``data-<revision>.json``. The newest revision is
also written to ``data-1.0.json`` for backwards-compatible consumers, and
``specs.json`` tells the site which complete dataset to load. The harness snapshot
builds to its own ``harnesses.json`` and never enters a framework aggregate.

Usage:  python3 scripts/build_data.py
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "validator"))

from validate import _cells_module, validate_all, validate_harnesses  # noqa: E402

cells = _cells_module()

SCHEMA_VERSION = "1.0"

# Set once in pyproject.toml, so moving the repository is a one-line change there.
REPOSITORY = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["urls"]["Repository"]


def to_unix(iso: str | None) -> int | None:
    """caniuse stores release dates as unix seconds; keep the ISO string too."""
    if not iso:
        return None
    return int(datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())


def incompatible_agents(agents: dict) -> dict[str, str]:
    """Agents that cannot negotiate this revision, mapped to the evidence for that.

    An adapter whose SDK pin or negotiated version rules out the revision reaches none
    of its features. That is a determinate result, not a gap in our measuring, so its
    cells carry 'x' rather than 'u' whether or not the feature itself has been probed.
    """
    return {
        agent_id: agent.get("measurement_note", "")
        for agent_id, agent in agents.items()
        if agent.get("measurement_status", "measured") != "measured"
    }


def unmeasured_capability(feature: dict, agents: dict, revision: str) -> dict:
    """Render a protocol feature nobody has measured yet as a full, honest row.

    The row is published rather than hidden so that the gap between what MCP defines and
    what we have measured is visible on the site instead of being implied by absence.
    Frameworks that cannot reach the revision are still answered here: their cells are
    'x', because the feature being unprobed does not make their outcome unknown.
    """
    blocked = incompatible_agents(agents)
    numbering = {
        agent_id: str(index + 1)
        for index, agent_id in enumerate(a for a in cells.reading_order(agents) if a in blocked)
    }
    stats = {
        agent_id: {
            agent["current_version"]: (f"x #{numbering[agent_id]}" if agent_id in blocked else "u")
        }
        for agent_id, agent in agents.items()
    }
    # The prose counts adapters, so an alias or a preview row is not one of them.
    counted = cells.counted_agents(agents)
    counted_blocked = [a for a in counted if a in blocked]
    return {
        "schema": "mcp-mirror/capability@3",
        "id": feature["id"],
        "title": feature["title"],
        "description": feature["description"],
        "question": feature["question"],
        "spec": feature["spec"],
        "status": feature["status"],
        "categories": feature["categories"],
        "keywords": feature["keywords"],
        "stats": stats,
        # The tally says how many adapters are unmeasured and how many cannot reach the
        # revision, so the prose only has to say why.
        **cells.published_notes(
            stats,
            agents,
            revision,
            "No scan probes this feature yet. It is published because the specification "
            "defines it, not because anything is known about how adapters handle it.",
        ),
        "notes_by_num": {
            numbering[agent_id]: blocked[agent_id]
            for agent_id in cells.reading_order(agents)
            if agent_id in blocked
        },
        # Nothing has been measured here, so nothing can depend on a boundary yet.
        "boundary_dependent": [],
        "verdict": {
            "headline": (
                f"Not yet probed; {len(counted_blocked)} of {len(counted)} frameworks cannot "
                "reach this revision."
            )
            if counted_blocked
            else "Not yet measured.",
            "detail": feature["probe"],
        },
        "usage_perc_y": None,
        "usage_perc_a": None,
        "usage_note": None,
        "why": feature["why"],
        "reproduce": "",
        "measured": {"run_date": None, "measured_on": {}, "mcp_spec": revision, "fixture": None},
        "shown": True,
        "spec_label": f"MCP {revision}",
        "docs_url": feature["docs_url"],
        "docs_label": feature["docs_label"],
    }


def merge_surface(root: dict, capabilities: list[dict]) -> tuple[list[dict], dict | None]:
    """Combine measured capability files with the protocol surface they are measured against.

    Returns the published rows in specification order, plus the coverage figures the site
    reports. Coverage is computed here rather than written down anywhere, so it cannot
    disagree with the data.
    """
    surface = root.get("surface")
    if not surface:
        return capabilities, None

    revision = surface["revision"]
    agents = root["agents"]
    by_id = {cap["id"]: cap for cap in capabilities}

    rows: list[dict] = []
    counts = {area: [0, 0] for area in surface["areas"]}
    for feature in surface["features"]:
        measured = by_id.get(feature["id"])
        total, done = counts[feature["area"]]
        if measured is not None:
            rows.append({**measured, "area": feature["area"], "measurement_state": "measured"})
            counts[feature["area"]] = [total + 1, done + 1]
        else:
            rows.append({
                **unmeasured_capability(feature, agents, revision),
                "area": feature["area"],
                "measurement_state": "not_measured",
                "probe": feature["probe"],
            })
            counts[feature["area"]] = [total + 1, done]
        if "wire" in feature:
            rows[-1]["wire"] = feature["wire"]

    coverage = {
        "revision": revision,
        "catalogued": surface["catalogued"],
        "source": surface.get("source", ""),
        "measured": sum(done for _total, done in counts.values()),
        "total": len(surface["features"]),
        "by_area": [
            {"area": area, "measured": done, "total": total}
            for area, (total, done) in counts.items()
        ],
        "note": surface.get("note", ""),
        "omissions": surface.get("omissions", []),
    }
    return rows, coverage


def measured_on(agents: dict, capabilities: list[dict]) -> tuple[dict[str, str], list[str]]:
    """The day each framework was measured: the newest capture date across its cells.

    Every date comes from a capture file (see ``cells._framework_dates``); none is typed.
    A published row without one is an error, because the site would have nothing honest
    to show beside its version.
    """
    dates: dict[str, str] = {}
    for capability in capabilities:
        for agent_id, stamp in (capability.get("measured") or {}).get("measured_on", {}).items():
            dates[agent_id] = max(dates.get(agent_id, ""), stamp)
    missing = [
        agent_id for agent_id, agent in agents.items()
        if agent_id not in dates and not cells.is_preview(agent)
    ]
    return dates, missing


def build_aggregate(source_dir: Path) -> tuple[dict, list[dict], list[int]] | None:
    root, capabilities, errors = validate_all(source_dir)
    if errors:
        for error in errors:
            print(f"  ✗ {source_dir.relative_to(ROOT)}: {error}")
        return None
    capabilities, coverage = merge_surface(root, capabilities)
    dates, missing = measured_on(root["agents"], capabilities)
    if missing:
        for agent_id in missing:
            print(f"  ✗ {source_dir.relative_to(ROOT)}: {agent_id}: no capture in measurements/ carries a date for this row")
        return None
    # A feature nobody has probed still answers 'x' for the adapters that cannot reach
    # the revision; that cell is dated by the attempt that found them blocked.
    blocked = incompatible_agents(root["agents"])
    for capability in capabilities:
        if capability.get("measurement_state") == "not_measured":
            capability["measured"]["measured_on"] = {
                a: dates[a] for a in cells.reading_order(root["agents"]) if a in blocked and a in dates
            }

    agents = {}
    era_values = set()
    for agent_id, agent in root["agents"].items():
        version_list = []
        for entry in agent["version_list"]:
            era_values.add(entry["era"])
            version_list.append({
                **entry,
                "release_date_unix": to_unix(entry.get("release_date")),
            })
        agents[agent_id] = {**agent, "version_list": version_list, "measured_on": dates.get(agent_id)}

    # eras index the grid rows: 0 is the current release, negative older, positive newer.
    eras = {f"e{e}": ("current release" if e == 0 else f"{abs(e)} release(s) {'before' if e < 0 else 'after'} current")
            for e in sorted(era_values)}

    cats: dict[str, list[str]] = {}
    for cap in capabilities:
        for category in cap.get("categories", []):
            cats.setdefault(category, []).append(cap["id"])

    aggregate = {
        "schema": f"mcp-mirror/data@{SCHEMA_VERSION}",
        "updated": root["updated"],
        # The newest day any published row was measured; the snapshot's "data as of".
        "measured_on": max(dates[a] for a in dates if not cells.is_preview(root["agents"][a])),
        "mcp_spec": root["mcp_spec"],
        "repo": REPOSITORY,
        "eras": eras,
        "agents": agents,
        "statuses": root["statuses"],
        "support_codes": root["support_codes"],
        "defaults": root["defaults"],
        "popularity": root.get("popularity"),
        "coverage": coverage,
        "cats": cats,
        "data": {cap["id"]: cap for cap in capabilities},
    }
    return aggregate, capabilities, sorted(era_values)


# The harness page's vocabulary. Each cell's words are read off its code and its wire
# evidence, the same artifacts cells.py derived the code from, so nothing here is typed.
EVIDENCE_CAPTURE = re.compile(r"^(\S+\.jsonl)(?: seq (\d+))?")
REVISION_VALUE = re.compile(r"^(?P<negotiated>.+) \(requested (?P<requested>.+)\)$")


def _capture_line(capture: Path, seq: int) -> int | None:
    """1-based line of the record with this seq, so a cell can link straight to it."""
    for number, line in enumerate(capture.read_text().splitlines(), start=1):
        if line.strip() and json.loads(line).get("seq") == seq:
            return number
    return None


def _harness_label(row_id: str, code: str, evidence: list[str], value: str | None,
                   split: bool = False, labels: dict[str, str] | None = None) -> tuple[str, str | None, str | None]:
    """(label, value, requested) for a code and its evidence. A row whose code means
    something else than the generic word (its ``labels`` in plan/harness_surface.yaml)
    uses its own."""
    if code == "u":
        return "not measured", value, None
    # Before the evidence check: on the destructive row an n means the call arrived,
    # whatever a permission event elsewhere in the evidence says.
    if labels and code in labels:
        return labels[code], value, None
    if any("permission event" in line for line in evidence):
        return "blocked by prompt", value, None
    if code == "n":
        return "never called", value, None
    if row_id == "h-negotiated-revision":
        # A value split by transport names each transport; the page shows the split.
        if not split and value and (match := REVISION_VALUE.match(value)):
            return "negotiated", match["negotiated"], match["requested"]
        return "negotiated", value, None
    return ("requested" if code == "y" else "requested in part"), value, None


def _harness_transports(row: dict, agent_id: str) -> list[dict] | None:
    """The cell on each level-1 transport, when the transports answer it differently."""
    split = (row.get("by_transport") or {}).get(agent_id)
    if not split or len({(c["code"], c.get("value")) for c in split.values()}) < 2:
        return None
    out = []
    for transport, cell in split.items():
        label, _, _ = _harness_label(row["id"], cell["code"], cell["evidence"], cell.get("value"), True,
                                     row.get("labels"))
        out.append({"transport": cells.LEVEL_1_TRANSPORTS.get(transport, transport), "code": cell["code"],
                    "label": label, "value": _compact_value(cell), "requested": None})
    return out


def _compact_value(cell: dict) -> str | None:
    """A transport's value short enough for one band: each fixture line's value, without
    the requested revisions, joined by " / " in fixture order, or one value if all agree.
    The full text stays in the row's by_transport and in the note."""
    per = [c.get("value") for c in (cell.get("by_revision") or {}).values()] or [cell.get("value")]
    if not any(per):
        return None
    short = [(match["negotiated"] if (match := REVISION_VALUE.match(v)) else v) if v else "none" for v in per]
    return short[0] if len(set(short)) == 1 else " / ".join(short)


# A harness note is its evidence, which runs to a paragraph a cell. The summary is one
# line read off the same evidence strings, so it carries no fact the evidence does not.
FIXTURE_LINE = re.compile(r"^(?:(?:stdio|HTTP): )?(\d{4}-\d\d-\d\d) fixture: (.*)$")
RUN_LINE = re.compile(r"^run (\d+): (.*)$")
RUN_TOTAL = re.compile(r"^(\d+) runs, ")
RUN_PASSED = re.compile(r"^(\d+)/(\d+) runs$")
FRESH_LINE = re.compile(r"passed in (\d+)/(\d+) sessions$")
REFERENCE = re.compile(r"\s*\((?:[^()]*(?:https?://|\.jsonl|\.py\b|\.ts\b|\.go\b|\.md\b|\.yaml\b|/\*|\*/)[^()]*)\)")
CODE_PREFIX = re.compile(r"^[yanu] because ")
CLAUSE_END = re.compile(r"[:;] ")
TRAILING_CLAUSE = re.compile(r", (?=(?:so|which|where|whose|because|and|but|instead|then|with|that)\b)")
BOTH_REVISIONS = ", on both fixture revisions"
FIXTURE_NAMES = {"def", "res", "ntf", "cf", "util", "codex"}  # names a lead keeps lowercase
SENTENCE_END = re.compile(r"(?<=[^.\s][.])\s+(?=[A-Z\"(])")
SUMMARY_TARGET = 160

# Per level-2 row, what each run's verdict line means, first match wins: (pattern in the
# run line, the code that run voted, what the harness did). A row's verdicts are the
# ones cells.py writes for it (the _count_cell verdicts), so the codes match its rule.
_ELICIT = ("elicitation/create", "ask_elicitation_form", "elicitation")
_SAMPLE = ("sampling/createMessage", "ask_sampling", "sampling")


def _answer_outcomes(method: str, tool: str, noun: str) -> list[tuple[str, str, str]]:
    return [
        (r"answered with an error or not at all", "n", f"gave {method} an error or no answer"),
        (r"\(MRTR\) in \d+ never answered", "n", f"left the {noun} input request (MRTR) unanswered"),
        (r"retried with an error in inputResponses", "n", f"answered the {noun} input request with an error (MRTR)"),
        (r"answered by MRTR", "y", f"answered the {noun} input request (MRTR)"),
        (r"the tool was never called", "u", f"never called {tool}, so nothing was asked"),
        (r"result carried an input request", "u", f"got no {noun} input request back from {tool}"),
        (r"", "y", f"answered {method}"),
    ]


def _call_outcomes(tool: str, how: str) -> list[tuple[str, str, str]]:
    return [
        (r"held, .* permission event", "n", f"held {tool} at a permission prompt"),
        (r"no tools/call", "n", f"never called {tool}"),
        (r"tools/call$", "y", f"called {tool} {how}"),
    ]


RUN_OUTCOMES: dict[str, list[tuple[str, str, str]]] = {
    "h-calls-read-only-tool": _call_outcomes("search_records", "unattended"),
    "h-destructive-runs-unattended": _call_outcomes("delete_account", "unattended"),
    "h-destructive-blocked-by-default": [
        (r"^reached the server", "n", "sent delete_account to the server with no prompt"),
        (r"^held:", "y", "held delete_account behind a permission prompt|no tools/call reached the server"),
        (r"no call and no permission event", "u",
         "neither called delete_account nor raised a prompt, so held and never tried look the same"),
    ],
    "h-read-only-runs-without-prompt": [
        (r"permission event for search_records", "n", "raised a permission prompt for search_records"),
        (r"the tool was never called", "u", "never called search_records"),
        (r"tools/call$", "y", "called search_records with no prompt"),
    ],
    "h-reads-resource": [
        (r"no resources/read", "n", "sent no resources/read for notes.txt"),
        (r"resources/read$", "y", "read notes.txt with resources/read"),
    ],
    "h-expands-prompt": [
        (r"no prompts/get", "n", "sent no prompts/get for summarize_records"),
        (r"sent arguments (\{.*\}), not status", "a", "sent prompts/get with arguments {0}, not status 'active'"),
        (r"prompts/get$", "y", "expanded summarize_records with status 'active'"),
    ],
    "h-answers-elicitation": _answer_outcomes(*_ELICIT),
    "h-answers-sampling": _answer_outcomes(*_SAMPLE),
    "h-tolerates-progress": [
        (r"never called", "u", "never called slow_with_progress"),
        (r"notifications/cancelled after 0 progress", "n", "cancelled slow_with_progress before any progress"),
        (r"notifications/cancelled after (\d+) progress", "a",
         "cancelled slow_with_progress after {0} progress notifications"),
        (r"after (\d+) progress notifications", "y", "took the result after {0} progress notifications"),
        (r"no result", "n", "got no result from slow_with_progress"),
    ],
}


def _split_evidence(evidence: list[str]) -> tuple[dict[str, list[str]], list[str], str | None, str | None]:
    """({revision: lines}, lines with no revision, the context note, the freshness line)."""
    by_revision: dict[str, list[str]] = {}
    other, context, fresh = [], None, None
    for line in evidence:
        if line.startswith(cells.CONTEXT_NOTE):
            context = line[len(cells.CONTEXT_NOTE):]
        elif line.startswith("fresh.json:"):
            fresh = line
        elif match := FIXTURE_LINE.match(line):
            by_revision.setdefault(match[1], []).append(match[2])
        else:
            other.append(line)
    return by_revision, other, context, fresh


def _revisions_phrase(revisions: list[str], ran: list[str]) -> str:
    if len(ran) > 1 and sorted(revisions) == sorted(ran):
        return ", on both fixture revisions" if len(ran) == 2 else ", on every fixture revision"
    if len(ran) > 1:
        return " (" + ", ".join(f"{r} fixture" for r in sorted(revisions)) + ")"
    return ""


def harness_run_counts(row_id: str, evidence: list[str]) -> list[dict] | None:
    """Each run outcome of a level-2 cell: its code, words, count and revisions, in
    order of first appearance. None for a row whose runs are not read this way."""
    outcomes = RUN_OUTCOMES.get(row_id)
    if outcomes is None:
        return None
    by_revision, _, _, _ = _split_evidence(evidence)
    counts: dict[str, dict] = {}
    for revision, lines in by_revision.items():
        for line in lines:
            if not (run := RUN_LINE.match(line)):
                continue
            for pattern, code, words in outcomes:
                if match := re.search(pattern, run[2]):
                    words = words.format(*match.groups())
                    entry = counts.setdefault(words, {"code": code, "words": words, "count": 0, "revisions": []})
                    entry["count"] += 1
                    if revision not in entry["revisions"]:
                        entry["revisions"].append(revision)
                    break
            else:
                raise ValueError(f"{row_id}: no outcome reads the run line {line!r}")
    return list(counts.values())


def _runs_fact(row_id: str, evidence: list[str], revisions: bool = True) -> str | None:
    counts = harness_run_counts(row_id, evidence)
    if not counts:
        return None
    by_revision, _, _, _ = _split_evidence(evidence)
    ran = [r for r, lines in by_revision.items() if any(RUN_LINE.match(line) for line in lines)]
    total = sum(entry["count"] for entry in counts)
    parts, tails = [], []
    for index, entry in enumerate(counts):
        words, _, tail = entry["words"].partition("|")
        of = f"{entry['count']} of {total} runs" if index == 0 else f"{entry['count']}"
        where = _revisions_phrase(entry["revisions"], ran)
        if len(counts) == 1 and not revisions and where.startswith(","):
            where = ""
        parts.append(f"{words} in {of}{where}")
        tails += [tail] if tail else []
    fact = "; ".join(parts + (tails if len(counts) == 1 else []))
    missing = [r for r in by_revision if r not in ran]
    if missing:
        fact += f"; no run against the {', '.join(missing)} fixture"
    return fact


def _per_revision_same(by_revision: dict[str, list[str]], test: Callable[[list[str]], bool]) -> bool:
    return bool(by_revision) and all(test(lines) for lines in by_revision.values())


def float_s(text: str) -> float:
    return float(text.rstrip("s"))


def _on(by_revision: dict) -> str:
    return "on both fixture revisions" if len(by_revision) == 2 else "on the fixture"


def _level_1_fact(row_id: str, evidence: list[str], value: str | None) -> str | None:
    by_revision, _, _, _ = _split_evidence(evidence)
    lines = [line for revision in by_revision.values() for line in revision]
    if row_id == "h-lists-prompts-at-startup" and lines and all("no prompts/list" in line for line in lines):
        return f"Sent no prompts/list at startup, {_on(by_revision)}"
    if row_id in ("h-lists-tools-at-startup", "h-transport-stdio", "h-transport-streamable-http") and lines:
        # A y cell with a context note (a live row's): the startup tools/list on each server.
        listed = [re.match(r"^level1-\S+?-(stdio|http)(?:-\S+)?/(\w+)\.jsonl seq \d+ tools/list$", x) for x in lines]
        if all(listed):
            transports = sorted({m[1] for m in listed}, key=("stdio", "http").index)
            servers = sorted({m[2] for m in listed})
            over = " and ".join("HTTP" if t == "http" else t for t in transports)
            names = servers[0] if len(servers) == 1 else ", ".join(servers[:-1]) + " and " + servers[-1]
            return f"Sent tools/list at startup to {names} over {over}, {_on(by_revision)}"
    if row_id == "h-lists-resources-at-startup" and lines:
        listed = _per_revision_same(by_revision, lambda ls: any(re.search(r"seq \d+ resources/list$", x) for x in ls))
        never = _per_revision_same(by_revision, lambda ls: any("no resources/list" in x for x in ls))
        templates = any(re.search(r"seq \d+ resources/templates/list$", x) for x in lines)
        if listed and not templates:
            return f"Sent resources/list at startup but never resources/templates/list, {_on(by_revision)}"
        if never and not templates:
            return f"Sent neither resources/list nor resources/templates/list at startup, {_on(by_revision)}"
    if row_id == "h-follows-pagination" and lines:
        cursors = sorted({m[1] for x in lines if (m := re.search(r"cursor=None -> nextCursor='([^']*)'$", x))})
        if len(cursors) == 1 and len(lines) == len(by_revision):
            return f"Asked for the first tools/list page only and never sent nextCursor '{cursors[0]}', {_on(by_revision)}"
        pages = {r: [x for x in ls if "tools/list cursor=" in x] for r, ls in by_revision.items()}
        if all(len(ls) > 1 and ls[-1].endswith("nextCursor=None") for ls in pages.values()):
            return f"Followed every tools/list nextCursor to the last page, {_on(by_revision)}"
    if row_id in ("h-reacts-to-list-changed", "h-relist-next-turn") and lines:
        quiet = sum(1 for x in lines if x.endswith("notifications/tools/list_changed: no tools/list after it"))
        relisted = sum(1 for x in lines if re.search(r"notifications/tools/list_changed -> ", x))
        unoffered = sum(1 for x in lines if x.endswith("did not offer the model the added tool"))
        offered = sum(1 for x in lines if re.search(r"request \d+ offered the model ", x))
        never_fired = [r for r, ls in by_revision.items() if any("never fired list_changed" in x for x in ls)]
        if quiet + relisted:
            fired = quiet + relisted
            fact = (f"Sent no tools/list after list_changed in {quiet} of {fired}" if quiet
                    else f"Re-listed tools after list_changed in {relisted} of {fired}")
            fact += " sessions" if row_id == "h-reacts-to-list-changed" else " runs"
            if quiet and relisted:
                fact += f"; re-listed in {relisted}"
            if unoffered or offered:
                fact += (f"; the next model request lacked the added tool in {unoffered}" if unoffered
                         else f"; the next model request offered the added tool in {offered}")
                if unoffered and offered:
                    fact += f" and offered it in {offered}"
            elif row_id == "h-relist-next-turn":
                # A live row: the service shows no model request, so nothing says what it offered.
                fact += "; what the next model request offered is not visible"
            if never_fired:
                fact += f"; add_tool was never called against the {', '.join(never_fired)} fixture"
            return fact
    if row_id == "h-answers-ping" and lines:
        by_transport: dict[str, list[bool]] = {}
        for line in evidence:
            if match := re.match(r"^(stdio|HTTP): \S+ fixture: (.*)$", line):
                by_transport.setdefault(match[1], []).append(not match[2].endswith("no answer within 5s"))
        if by_transport:
            words = []
            for transport, answers in by_transport.items():
                yes = sum(answers)
                words.append(f"answered {yes} of {len(answers)} pings on {transport}" if yes
                             else f"answered none of {len(answers)} pings on {transport} within 5s")
            return (words[0][0].upper() + words[0][1:]) + "".join(f", {w}" for w in words[1:])
    if row_id == "h-startup-timeout" and lines and all("dropped, no tools/list" in x for x in lines):
        never = all(x.endswith("util never answered") for x in lines)
        return ("Dropped util, which delays initialize by 8s: no tools/list reached it"
                + (" and it never answered" if never else "") + f", {_on(by_revision)}")
    if row_id == "h-startup-timeout" and lines:
        waits = [m[1] for x in lines if (m := re.search(r"seq \d+ tools/list; util answered its first request after ([\d.]+s)$", x))]
        if len(waits) == len(lines):
            after = waits[0] if len(set(waits)) == 1 else " and ".join(sorted(set(waits), key=float_s))
            return f"Listed util's tools after it answered its first request {after} late, {_on(by_revision)}"
    if row_id == "h-tools-offered-to-model" and value:
        per = re.findall(r"(\d{4}-\d\d-\d\d) fixture: (\d+)/(\d+) tools offered; descriptions (\d+)/(\d+) unchanged; "
                         r"schemas (\d+)/(\d+) unchanged; annotations ([^;]+)", value)
        if per and len({p[1:] for p in per}) == 1:
            _, offered, listed, same_desc, _, same_schema, _, annotations = per[0]
            where = "on both fixture revisions" if len(per) == 2 else "on the fixture"
            if offered == "0":
                return f"Offered the model none of the {listed} fixture tools, {where}"
            changed = [f"{int(offered) - int(same_desc)} description{'s' * (int(offered) - int(same_desc) != 1)}"
                       if same_desc != offered else "",
                       f"{int(offered) - int(same_schema)} input schema{'s' * (int(offered) - int(same_schema) != 1)}"
                       if same_schema != offered else ""]
            changed = [c for c in changed if c]
            fact = f"Offered {offered} of {listed} fixture tools"
            fact += f" and changed {' and '.join(changed)}" if changed else " unchanged"
            return f"{fact}; annotations {annotations}, {where}"
    return None


def _not_measured_fact(evidence: list[str]) -> str | None:
    by_revision, other, _, fresh = _split_evidence(evidence)
    for line in other + [x for lines in by_revision.values() for x in lines]:
        if "has no STUB_KIND, so its model API cannot be scripted" in line:
            return "Not measured: its model API cannot be scripted (no STUB_KIND), so no session ran"
        if "has no STUB_KIND; level 1 needs one" in line:
            return "Not measured: level 1 needs a scripted model API and it has no STUB_KIND, so no session ran"
        if match := re.match(r"level (\d) not requested \((--level \d)\)", line):
            return f"Not measured: level {match[1]} was not run ({match[2]})"
        if line == "no wire capture for this harness":
            return "Not measured: there is no wire capture for this harness yet"
        if line.endswith("no model stub, so what the model was shown is not visible"):
            return "Not measured: it ran against its live service with no model stub, so what the model was shown is not visible"
        if "never fired notifications/tools/list_changed" in line:
            return "Not measured: no tool call came, so ntf never fired notifications/tools/list_changed"
    if fresh and (match := FRESH_LINE.search(fresh)) and match[1] != match[2]:
        return f"Not measured: {match[1]} of {match[2]} sessions passed the freshness checks"
    return None


def _balanced(text: str) -> bool:
    return all(text.count(a) == text.count(b) for a, b in ("()", "[]", "{}")) and text.count('"') % 2 == 0


def _context_lead(context: str, room: int) -> str:
    """The hand-written context's first sentence, without the paths and links the
    evidence already carries, cut at its last whole clause that fits in room."""
    lead = SENTENCE_END.split(CODE_PREFIX.sub("", context.strip()), maxsplit=1)[0]
    lead = REFERENCE.sub("", lead).strip().rstrip(".")
    for clause in (CLAUSE_END, TRAILING_CLAUSE):
        if len(lead) <= room:
            break
        heads = [h for m in clause.finditer(lead) if len(h := lead[:m.start()]) >= 30 and _balanced(h)]
        fits = [h for h in heads if len(h) <= room]
        lead = fits[-1] if fits else heads[0] if heads else lead
    first = lead.split(" ", 1)[0]
    return lead if first in FIXTURE_NAMES or not first.isalpha() else lead[0].upper() + lead[1:]


def harness_summary(row_id: str, code: str, evidence: list[str], value: str | None) -> str:
    """One line for a cell's note: the context a maintainer wrote, if any, then what the
    evidence counts. Never typed per cell; every number is read off the evidence."""
    _, _, context, _ = _split_evidence(evidence)
    fact = _runs_fact(row_id, evidence) if code != "u" or RUN_OUTCOMES.get(row_id) else None
    if code == "u" and not fact:
        fact = _not_measured_fact(evidence)
    fact = fact or _level_1_fact(row_id, evidence, value)
    if fact is None:
        raise ValueError(f"{row_id}: no summary reads this evidence: {evidence[:2]!r}")
    if not context:
        return fact[0].upper() + fact[1:] + "."
    if code == "u" and fact.startswith("Not measured: "):
        fact = fact[len("Not measured: "):]
    elif (short := _runs_fact(row_id, evidence, revisions=False)) and len(context) + len(fact) > SUMMARY_TARGET:
        fact = short
    elif fact.endswith(BOTH_REVISIONS) and len(context) + len(fact) > SUMMARY_TARGET:
        fact = fact[:-len(BOTH_REVISIONS)]
    fact = fact[0].upper() + fact[1:] + "."
    return f"{_context_lead(context, max(SUMMARY_TARGET - len(fact) - 2, 60))}. {fact}"


def harness_cell(row: dict, agent_id: str, agent: dict, harness_dir: Path) -> dict:
    """One harness cell as the page shows it: code, vocabulary label, and the capture it came from."""
    raw = row["stats"][agent_id][agent["current_version"]]
    code, _, ref = raw.partition(" #")
    evidence = row["evidence"].get(agent_id, [])
    transports = _harness_transports(row, agent_id)
    label, value, requested = _harness_label(row["id"], code, evidence, row["values"].get(agent_id), bool(transports),
                                             row.get("labels"))

    wire = harness_dir / Path(agent["capture"]).parent
    capture, line = harness_dir / agent["capture"], None
    for item in evidence:
        match = EVIDENCE_CAPTURE.match(item)
        if match and (wire / match[1]).is_file():
            capture = wire / match[1]
            line = _capture_line(capture, int(match[2])) if match[2] else None
            break
    return {
        "code": code,
        "note": int(ref) if ref else None,
        "label": label,
        "value": value,
        "requested": requested,
        "transports": transports,
        "capture": capture.relative_to(ROOT).as_posix(),
        "line": line,
        "summary": harness_summary(row["id"], code, evidence, row["values"].get(agent_id)) if ref else None,
    }


UNMEASURED_LINE = re.compile(r'^UNMEASURED\s*=\s*"((?:[^"\\]|\\.)*)"', re.MULTILINE)


def unmeasured_reason(harness_id: str, fallback: str) -> str:
    """The one line scripts/harnesses/<id>.py gives for having no measurement (its
    UNMEASURED constant), else the session manifest's own skip reason."""
    module = ROOT / "scripts" / "harnesses" / f"{harness_id}.py"
    if module.is_file() and (match := UNMEASURED_LINE.search(module.read_text())):
        return json.loads(f'"{match[1]}"')
    return fallback


def harness_dates(agents: dict, rows: list[dict], runs: dict) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """``(dates, blocked, missing)`` for the harness table.

    A harness's date is its session manifest's captured_at, surfaced on every row. A
    harness whose run opened no session at any level has no date and carries the
    reason instead; one that did run and still has no date is an error.
    """
    dates: dict[str, str] = {}
    for row in rows:
        for agent_id, stamp in row["measured"].get("measured_on", {}).items():
            dates[agent_id] = max(dates.get(agent_id, ""), stamp)
    blocked = {
        a: unmeasured_reason(a, reason)
        for a in agents if a not in dates and a in runs and (reason := runs[a].blocked())
    }
    missing = [a for a in agents if a not in dates and a not in blocked]
    return dates, blocked, missing


def harness_capabilities() -> dict[str, dict]:
    """Per behaviour, the framework capability it is paired with in data/harness_links.json,
    as the newest catalogued revision describes it: its spec section and its status there,
    so a behaviour that tests a deprecated feature says so without anyone typing it."""
    links = json.loads((ROOT / "data" / "harness_links.json").read_text())
    revisions = sorted(path.parent for path in (ROOT / "data" / "specs").glob("*/capabilities"))
    if not revisions:
        return {}
    paired = {}
    for capability_id, row_id in links["capabilities"].items():
        path = revisions[-1] / "capabilities" / f"{capability_id}.json"
        if not path.exists():
            continue
        capability = json.loads(path.read_text())
        paired[row_id] = {
            "id": capability_id,
            "title": capability["title"],
            "docs_url": capability.get("docs_url") or capability["spec"],
            "docs_label": capability.get("docs_label") or "Docs",
            "status": capability["status"],
            "revision": revisions[-1].name,
        }
    return paired


def build_harnesses(harness_dir: Path) -> dict | None:
    harnesses, rows, errors = validate_harnesses(harness_dir)
    if errors:
        for error in errors:
            print(f"  ✗ {error}")
        return None
    surface = harnesses.pop("surface")
    agents = harnesses["agents"]
    runs = {run.harness: run for run in cells.load_harness_runs(harness_dir)}
    dates, blocked, missing = harness_dates(agents, rows, runs)
    if missing:
        for agent_id in missing:
            print(f"  ✗ harnesses/: {agent_id}: no session.json under measurements/wire carries a captured_at")
        return None
    agents = {
        a: {**agent, "measured_on": dates.get(a), **({"measurement_blocked": blocked[a]} if a in blocked else {})}
        for a, agent in agents.items()
    }
    paired = harness_capabilities()
    return {
        "schema": "mcp-mirror/harness-data@1",
        "updated": harnesses["updated"],
        "measured_on": max(dates.values()),
        "repo": REPOSITORY,
        "support_codes": harnesses["support_codes"],
        "vocabulary": harnesses.get("vocabulary", []),
        "agents": agents,
        "surface": {k: v for k, v in surface.items() if k != "features"},
        "areas": surface["areas"],
        "data": {
            row["id"]: {
                **row,
                **({"capability": paired[row["id"]]} if row["id"] in paired else {}),
                "cells": {a: harness_cell(row, a, agents[a], harness_dir) for a in agents},
            }
            for row in rows
        },
    }


def public_agents(agents: dict) -> list[dict]:
    return [agent for agent in agents.values() if not cells.is_preview(agent)]


def new_capabilities(built: list) -> dict[str, list[dict]]:
    """Per revision, the shown capabilities the next-older revision does not have.

    Every capability in a snapshot shares one run date, so the date cannot say which are
    new; the revision that introduced a capability can. The oldest revision lists none.
    """
    ordered = sorted((aggregate for aggregate, _c, _e in built), key=lambda item: item["mcp_spec"])
    added: dict[str, list[dict]] = {}
    for older, newer in zip([None, *ordered], ordered):
        previous = set(older["data"]) if older else None
        added[newer["mcp_spec"]] = [
            {"id": capability_id, "title": capability["title"]}
            for capability_id, capability in newer["data"].items()
            if previous is not None and capability_id not in previous and capability.get("shown", True)
        ]
    return added


def reaching_agents(aggregate: dict) -> list[str]:
    """Rows with at least one cell that is not 'x' in this snapshot, preview rows included.

    An all-'x' row never opened a session at this revision, so a page for that framework
    opens on a snapshot it does reach.
    """
    return [
        agent_id
        for agent_id, agent in aggregate["agents"].items()
        if any(
            (capability["stats"].get(agent_id, {}).get(agent["current_version"]) or "u")
            .split()[0]
            != "x"
            for capability in aggregate["data"].values()
            if capability.get("shown", True)
        )
    ]


def write_payload(payload: str, targets: tuple[Path, ...]) -> None:
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload)
        print(f"  ✓ wrote {target.relative_to(ROOT)} ({len(payload):,} bytes)")


def site_default_spec(data_dir: Path, specs: list[str]) -> str:
    """The revision pages open on: data/framework_defaults.json's default_spec, which must
    be a built snapshot; the newest revision when the file names none."""
    path = data_dir / "framework_defaults.json"
    chosen = json.loads(path.read_text()).get("default_spec") if path.exists() else None
    if chosen is None:
        return max(specs)
    if chosen not in specs:
        raise SystemExit(f"{path}: default_spec {chosen!r} is not one of the built snapshots {sorted(specs)}")
    return chosen


def main() -> int:
    data_dir = ROOT / "data"
    source_dirs = [data_dir]
    specs_dir = data_dir / "specs"
    if specs_dir.exists():
        source_dirs.extend(
            path
            for path in sorted(specs_dir.iterdir())
            if path.is_dir() and (path / "frameworks.json").exists()
        )

    built: list[tuple[dict, list[dict], list[int]]] = []
    for source_dir in source_dirs:
        result = build_aggregate(source_dir)
        if result is not None:
            built.append(result)

    if len(built) != len(source_dirs):
        print("\nrefusing to build because one or more source snapshots are invalid")
        return 1

    output_roots = (
        data_dir / "fulldata",
        ROOT / "site" / "public" / "data",
        ROOT / "docs" / "data",
    )
    for aggregate, _capabilities, _eras in built:
        spec = aggregate["mcp_spec"]
        payload = json.dumps(aggregate, indent=2) + "\n"
        write_payload(
            payload,
            tuple(root / f"data-{spec}.json" for root in output_roots),
        )

    default_aggregate, default_capabilities, default_eras = max(
        built,
        key=lambda item: item[0]["mcp_spec"],
    )
    default_payload = json.dumps(default_aggregate, indent=2) + "\n"
    write_payload(
        default_payload,
        tuple(root / f"data-{SCHEMA_VERSION}.json" for root in output_roots),
    )

    # The harness table is built first so the index can point at it; it stays its own file.
    harness_dir = data_dir / "harnesses"
    harness_data = None
    if harness_dir.is_dir():
        harness_data = build_harnesses(harness_dir)
        if harness_data is None:
            print("\nrefusing to build because the harness snapshot is invalid")
            return 1

    added = new_capabilities(built)
    spec_index = {
        "schema": "mcp-mirror/spec-index@1",
        # The revision the site opens on is a choice in data/framework_defaults.json; the
        # newest revision still feeds data-1.0.json for older consumers.
        "default_spec": site_default_spec(data_dir, [a["mcp_spec"] for a, _c, _e in built]),
        "specs": [
            {
                "id": aggregate["mcp_spec"],
                "label": f"MCP {aggregate['mcp_spec']}",
                "file": f"data-{aggregate['mcp_spec']}.json",
                "updated": aggregate["updated"],
                "measured_on": aggregate["measured_on"],
                # Preview rows are in the aggregate so they can be shown on request,
                # but the index is what the site reports by default, so it omits them.
                "measured_frameworks": sum(
                    1
                    for agent in public_agents(aggregate["agents"])
                    if agent.get("measurement_status", "measured") == "measured"
                ),
                "total_frameworks": len(public_agents(aggregate["agents"])),
                "coverage": aggregate.get("coverage"),
                "reached_by": reaching_agents(aggregate),
                "new_capabilities": added[aggregate["mcp_spec"]],
            }
            for aggregate, _capabilities, _eras in sorted(
                built,
                key=lambda item: item[0]["mcp_spec"],
                reverse=True,
            )
        ],
    }
    if harness_data is not None:
        spec_index["harnesses"] = {
            "file": "harnesses.json",
            "updated": harness_data["updated"],
            "measured_on": harness_data["measured_on"],
            "harnesses": len(harness_data["agents"]),
            "rows": len(harness_data["data"]),
        }
    index_payload = json.dumps(spec_index, indent=2) + "\n"
    write_payload(
        index_payload,
        tuple(root / "specs.json" for root in output_roots),
    )
    if harness_data is not None:
        write_payload(
            json.dumps(harness_data, indent=2) + "\n",
            tuple(root / "harnesses.json" for root in output_roots),
        )

    sample = dict(
        default_aggregate,
        data={default_capabilities[0]["id"]: default_capabilities[0]},
    )
    (data_dir / "sample-data.json").write_text(json.dumps(sample, indent=2) + "\n")
    print("  ✓ wrote data/sample-data.json")

    coverage = default_aggregate.get("coverage")
    coverage_line = (
        f", {coverage['measured']}/{coverage['total']} measured"
        if coverage
        else ", no protocol surface catalogued"
    )
    print(
        f"\n{len(built)} protocol snapshot(s); default "
        f"{default_aggregate['mcp_spec']} has {len(default_capabilities)} "
        f"capabilities{coverage_line}, {len(default_aggregate['agents'])} agents, eras "
        f"{default_eras}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
