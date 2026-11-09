"""Derive support cells from measurement artifacts.

A capability cell (``y``, ``a``, ``n``, ``x``) is a transcription of what an instrument
observed. Until now that transcription was done by hand, one capability file at a
time, which made adding a framework a day of editing and made the cells impossible to
audit against the artifact they claim to summarise. This module makes the cell a
function of the artifact.

Instruments, one per artifact kind under ``<snapshot>/measurements/``:

``scan/*.json``
    ``mcp-mirror scan`` reports. Tool-definition capabilities are derived from the
    categorised differences at the paths the capability is about, weighed against how
    often the source tools exercise that keyword.

``attempts.json`` (snapshot root)
    Frameworks that could not open a session at this revision. Every capability cell
    for such a framework is ``x`` with the attempt's evidence.

The derivation rule for a keyword capability is deliberately simple so a maintainer
can check it by hand:

* no difference at the capability's paths -> ``y``
* differences, none of them lossy -> ``a`` (changed or retained elsewhere)
* lossy at every place the source exercises the keyword -> ``n``
* lossy at some places, not all -> ``a``
* a source tool the framework did not render loses every place it exercises the
  keyword, so a keyword only dropped tools carry reads ``n``

Nothing here writes prose. Notes live in the capability overlay and are attached by
the build; a cell that is not ``y`` and has no note is a validation error, which is
the signal that a human still owes an explanation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

Code = str  # "y" | "a" | "n" | "x" | "u"

BEHAVIOR_HINTS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")


@dataclass(frozen=True)
class Rule:
    """How one capability reads the scan differences.

    ``paths`` is a regular expression over a difference path. ``count`` walks a source
    tool and returns how many places exercise the keyword, so a partial loss can be told
    from a total one. ``categories`` restricts which difference categories count at all;
    ``long-descriptions`` for example only cares about a truncation, not an injection.
    """

    paths: str
    count: Callable[[dict], int]
    categories: frozenset[str] = frozenset({"lossy", "transformative", "additive"})
    tools: frozenset[str] | None = None
    # A structural capability (nested objects, arrays of objects) is about the shape
    # surviving, not about every keyword that happens to sit inside that shape; those
    # keywords have their own rows. Restricting by dimension keeps them apart.
    dimensions: frozenset[str] | None = None
    # A capability older scans never exercised (the fixture gained the keyword later)
    # reads ``u`` from them instead of a ``y`` nothing was measured for.
    requires_exercise: bool = False

    def matches(self, difference: dict) -> bool:
        if difference["category"] not in self.categories:
            return False
        if self.dimensions is not None and difference["dimension"] not in self.dimensions:
            return False
        if self.tools is not None and difference["tool"] not in self.tools:
            return False
        return re.search(self.paths, difference["path"]) is not None


def _walk(node: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else key
            yield child, value
            yield from _walk(value, child)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            child = f"{path}[{index}]"
            yield child, value
            yield from _walk(value, child)


def _keyword_count(*keywords: str) -> Callable[[dict], int]:
    # One place per keyword, in the unit the difference paths use: not once more for
    # each element of its list (`required[0]`), and not for a property that merely
    # shares its name (`properties.format`).
    def count(tool: dict) -> int:
        return sum(
            1
            for path, _value in _walk(tool.get("params") or {}, "params")
            if path.rsplit(".", 1)[-1] in keywords
            and not re.search(r"(^|\.)properties\.[^.]+$", path)
        )

    return count


def _property_description_count(tool: dict) -> int:
    return sum(
        1
        for path, _value in _walk(tool.get("params") or {}, "params")
        if re.search(r"properties\.[^.]+(\.anyOf\[\d+\]|\.oneOf\[\d+\]|\.items)*\.description$", path)
    )


def _nullable_count(tool: dict) -> int:
    """Places where the source says a value may be null."""
    total = 0
    for path, value in _walk(tool.get("params") or {}, "params"):
        leaf = path.rsplit(".", 1)[-1]
        if leaf == "type" and (value == "null" or (isinstance(value, list) and "null" in value)):
            total += 1
        elif leaf == "nullable" and value:
            total += 1
    return total


def _nested_object_count(tool: dict) -> int:
    return sum(
        1
        for path, value in _walk(tool.get("params") or {}, "params")
        if re.search(r"properties\.[^.]+\.properties$", path) and isinstance(value, dict)
    )


def _annotation_count(key: str) -> Callable[[dict], int]:
    return lambda tool: 1 if (tool.get("annotations") or {}).get(key) is not None else 0


def _one(tool: dict) -> int:
    return 1


def _has_icons(tool: dict) -> int:
    return 1 if tool.get("icons") else 0


def _long_description(tool: dict) -> int:
    return 1 if len(tool.get("description") or "") > 1024 else 0


def _has_output_schema(tool: dict) -> int:
    return 1 if tool.get("output_schema") else 0


SCAN_RULES: dict[str, Rule] = {
    "tool-name": Rule(r"^name$", _one),
    "description-fidelity": Rule(r"^description$", _one),
    "long-descriptions": Rule(
        r"^description$", _long_description, categories=frozenset({"lossy"})
    ),
    "property-descriptions": Rule(
        r"^params\..*\.description$", _property_description_count
    ),
    "enum": Rule(r"\.enum(\[\d+\])?$", _keyword_count("enum")),
    "format": Rule(r"\.format$", _keyword_count("format")),
    "default-value": Rule(r"\.default$", _keyword_count("default")),
    "required": Rule(r"\.required(\[\d+\])?$", _keyword_count("required")),
    "numeric-range": Rule(
        r"\.(minimum|maximum|exclusiveMinimum|exclusiveMaximum)$",
        _keyword_count("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"),
    ),
    "null-acceptance": Rule(r"\.(null|nullable|type)$", _nullable_count),
    "additional-properties": Rule(
        r"\.additionalProperties$", _keyword_count("additionalProperties")
    ),
    "nested-objects": Rule(
        r"properties\.[^.]+\.properties",
        _nested_object_count,
        dimensions=frozenset({"structure"}),
    ),
    "arrays-of-objects": Rule(
        r"\.items(\.|$)", _keyword_count("items"), dimensions=frozenset({"structure"})
    ),
    "one-of-any-of": Rule(r"\.(anyOf|oneOf)(\[\d+\])?$", _keyword_count("anyOf", "oneOf")),
    "title-annotation": Rule(r"^annotations\.title$", _annotation_count("title")),
    "tool-icons": Rule(r"^icons$", _has_icons),
    # A rendering that keeps every keyword and adds its own (a `$schema` dialect) is
    # additive: the schema survived, so only a loss or a change decides the cell.
    "tool-output-schema": Rule(
        r"^outputSchema$",
        _has_output_schema,
        categories=frozenset({"lossy", "transformative"}),
        requires_exercise=True,
    ),
}
for _hint in BEHAVIOR_HINTS:
    _slug = re.sub(r"(?<!^)(?=[A-Z])", "-", _hint).lower()  # readOnlyHint -> read-only-hint
    SCAN_RULES[_slug] = Rule(rf"^annotations\.{_hint}$", _annotation_count(_hint))


@dataclass
class Cell:
    code: Code
    evidence: list[str] = field(default_factory=list)
    instrument: str = ""

    def as_dict(self) -> dict:
        return {"code": self.code, "instrument": self.instrument, "evidence": self.evidence}


def dropped_tools(differences: list[dict]) -> set[str]:
    """Source tools the framework did not render at all.

    The diff records such a tool once, as a lossy ``name`` with no rendered value, and
    walks nothing else in it, so a rule reading paths alone would never see the rest of
    what the tool carried.
    """
    return {
        d["tool"]
        for d in differences
        if d["path"] == "name" and d["category"] == "lossy" and d.get("rendered_value") is None
    }


def derive_scan_cell(rule: Rule, source_tools: list[dict], differences: list[dict]) -> Cell:
    exercised = sum(rule.count(tool) for tool in source_tools)
    if rule.requires_exercise and not exercised:
        return Cell("u", ["the scanned fixture does not declare it"], "scan")
    # A dropped tool loses every place it exercises the keyword; its own differences
    # (the lost name) are folded into that count rather than counted beside it.
    gone = dropped_tools(differences)
    dropped = {
        tool["name"]: places
        for tool in source_tools
        if tool["name"] in gone
        and (rule.tools is None or tool["name"] in rule.tools)
        and (places := rule.count(tool))
    }
    matched = [d for d in differences if rule.matches(d) and d["tool"] not in gone]
    if not matched and not dropped:
        return Cell("y", instrument="scan")
    lossy = [d for d in matched if d["category"] == "lossy"]
    evidence = sorted(
        {f"{d['tool']}: {d['path']} {d['category']}" for d in matched}
        | {f"{tool}: not rendered, so all {places} counted places are lost" for tool, places in dropped.items()}
    )
    if not lossy and not dropped:
        return Cell("a", evidence, "scan")
    # Distinct places lost, so five tools each losing the same hint count as five.
    lost = len({(d["tool"], d["path"]) for d in lossy}) + sum(dropped.values())
    if exercised and lost >= exercised:
        return Cell("n", evidence, "scan")
    return Cell("a", evidence, "scan")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_scan_runs(snapshot: Path) -> tuple[list[dict], dict[str, dict]]:
    """Merge every scan report in the snapshot into one set of runs keyed by framework.

    A snapshot may hold one report per framework, one per runner, or one for all; the
    generator does not care. Later files win on a repeated framework so a re-measurement
    can be dropped in beside the old one and take over.
    """
    reports: list[Path] = []
    legacy = snapshot / "report.json"
    if legacy.exists():
        reports.append(legacy)
    reports.extend(sorted((snapshot / "measurements" / "scan").glob("*.json")))
    source_tools: list[dict] = []
    runs: dict[str, dict] = {}
    for report_path in reports:
        report = _load_json(report_path)
        if not source_tools:
            source_tools = report.get("source_observation", {}).get("tools") or []
        for run in report.get("runs", []):
            if run.get("status", "measured") != "measured":
                continue
            runs[run["framework"]] = {**run, "_report": str(report_path)}
    return source_tools, runs


PUBLISHED_IDS = {
    "crewai": "crewai",
    "crewai_native": "crewai-native",
    "langchain": "langchain",
    "openai_agents": "openai-agents",
    "pydantic_ai": "pydantic-ai",
    "mastra": "mastra",
    "vercel_ai": "vercel_ai",
}


# ---------------------------------------------------------------------------
# Alias rows and visibility
# ---------------------------------------------------------------------------

# What an alias may say for itself. Everything else describes the measurement, and the
# measurement belongs to the source row: an alias that declared its own version list or
# capture boundary would be claiming a run nobody made.
ALIAS_OWN_KEYS = {"name", "abbr", "aliases_of", "alias_note", "visibility"}
VISIBILITIES = {"published", "preview"}


def source_id(agent_id: str, agent: dict) -> str:
    """The row whose derived cells this row publishes: itself, or what it aliases."""
    return agent.get("aliases_of") or agent_id


def resolve_agents(agents: dict) -> dict:
    """Expand alias entries into full rows that inherit their source's measurement.

    An alias is a framework that reaches MCP through another framework's adapter
    (LangGraph through langchain-mcp-adapters). It gets its own name and its own notes,
    but every field that says what was measured, and against which release, is the
    source's, so its cells cannot drift from the run that produced them.
    """
    resolved: dict = {}
    for agent_id, agent in agents.items():
        target = agent.get("aliases_of")
        if target and target in agents:
            inherited = {k: v for k, v in agents[target].items() if k not in ALIAS_OWN_KEYS}
            resolved[agent_id] = {**inherited, **agent}
        else:
            resolved[agent_id] = agent
    return resolved


def is_preview(agent: dict) -> bool:
    return agent.get("visibility", "published") == "preview"


def reading_order(agents: dict) -> list[str]:
    """The order note numbers are assigned in: published rows, then preview rows.

    Numbering preview rows last means landing one leaves every public note number where
    it was, and hiding them leaves the public numbering contiguous.
    """
    return [a for a in agents if not is_preview(agents[a])] + [a for a in agents if is_preview(agents[a])]


def counted_agents(agents: dict) -> list[str]:
    """Rows a published verdict counts and names: every public framework row.

    The verdict counts frameworks, the unit the header and the tally use, so the numbers
    on one card add up. An alias row (LangGraph, measured through LangChain's adapter) is
    counted and named like any framework, since a reader looking it up needs its answer;
    its own page says whose adapter it runs. A preview row is not public yet, so a verdict
    that named it would publish it through the headline.
    """
    return [a for a, agent in agents.items() if not is_preview(agent)]


def blocked_frameworks(snapshot: Path) -> dict[str, dict]:
    path = snapshot / "attempts.json"
    if not path.exists():
        return {}
    return _load_json(path).get("attempts", {})


def derive_snapshot(snapshot: Path) -> dict[str, dict[str, Cell]]:
    """Every derivable cell in a snapshot: ``{capability_id: {published_agent_id: Cell}}``."""
    source_tools, runs = load_scan_runs(snapshot)
    blocked = blocked_frameworks(snapshot)
    cells: dict[str, dict[str, Cell]] = {}
    for capability_id, rule in SCAN_RULES.items():
        row: dict[str, Cell] = {}
        for framework, run in runs.items():
            row[PUBLISHED_IDS.get(framework, framework)] = derive_scan_cell(
                rule, source_tools, run.get("differences") or []
            )
        for framework, attempt in blocked.items():
            row[PUBLISHED_IDS.get(framework, framework)] = Cell(
                "x", [attempt.get("evidence", "")], "attempt"
            )
        cells[capability_id] = row
    result_cells, _dependent = derive_result_cells(snapshot)
    result_cells.update(derive_probe_cells(snapshot))
    result_cells.update(derive_utility_cells(snapshot))
    result_cells.update(derive_notify_cells(snapshot))
    result_cells.update(derive_conformance_cells(snapshot))
    result_cells.update(derive_client_cells(snapshot))
    for capability_id, row in derive_protocol_cells(snapshot).items():
        existing = result_cells.setdefault(capability_id, {})
        for agent, cell in row.items():
            existing[agent] = _both_instruments(existing[agent], cell) if agent in existing else cell
    _merge_instruments(result_cells, derive_transport_cells(snapshot))
    for capability_id, row in result_cells.items():
        merged = dict(row)
        for framework, attempt in blocked.items():
            merged[PUBLISHED_IDS.get(framework, framework)] = Cell(
                "x", [attempt.get("evidence", "")], "attempt"
            )
        if merged:
            cells[capability_id] = merged
    return cells


# ---------------------------------------------------------------------------
# Results instrument: measurements/results/*.json from measure_result_boundaries.py
# ---------------------------------------------------------------------------

KIND_CAPABILITIES = {
    "tool-text-content": "text",
    "tool-image-content": "image",
    "tool-audio-content": "audio",
    "embedded-resource-content": "resource",
    "resource-link-content": "resource_link",
}
# What has to survive, somewhere in what the agent received, for a block that no longer
# arrives as a block to count as flattened (``a``) rather than lost (``n``). A resource
# is identified by its URI; media by its MIME type, since payloads are redacted in the
# capture; text by its own characters.
_KIND_TOKENS = {
    "text": ("text",),
    "image": ("mimeType",),
    "audio": ("mimeType",),
    "resource": ("uri",),
    "resource_link": ("uri",),
}
_KIND_WORDS = {
    "image": ("image",),
    "audio": ("audio",),
    "resource": ("embeddedresource", "embedded resource", "resourcecontents"),
    "resource_link": ("resourcelink", "resource_link", "resource link"),
    "text": ("textcontent",),
}
SEVERITY = {"n": 3, "a": 2, "y": 1, "u": 0}


def _tokens(block: dict, kind: str) -> list[str]:
    out = []
    for key in _KIND_TOKENS[kind]:
        value = block.get(key)
        if kind == "resource" and key == "uri":
            value = (block.get("resource") or {}).get("uri", value)
        if isinstance(value, str) and value:
            out.append(value)
    return out


def _rendered_text(rep: dict) -> str:
    parts = [rep.get("text") or ""]
    for block in rep.get("blocks") or []:
        parts.append(json.dumps(block))
    return "\n".join(parts)


def _combine(codes: Iterable[Code]) -> Code:
    """One code for a capability exercised by several calls.

    Unanimous answers stand. A mix is ``a``: a kind that arrives on one call and not
    another is delivered on the adapter's terms, which is exactly what partial means.
    Calls that could not be judged (``u``) do not vote unless nothing else did.
    """
    known = [code for code in codes if code != "u"]
    if not known:
        return "u"
    if len(set(known)) == 1:
        return known[0]
    return "a"


def _invocation_failed(source: dict, rep: dict) -> bool:
    """Nothing the tool produced came back: no blocks, no typed value, no flag, and
    none of the source text. The adapter failed to make the call at all (CrewAI's
    generated schema rejecting the arguments, for instance), which is a finding for
    the parameter rows, not evidence about any content kind."""
    if rep.get("blocks") or rep.get("structured_content") is not None or rep.get("is_error"):
        return False
    text = rep.get("text") or ""
    # Every identifying token the source sent, not just text: a single-kind media call has
    # no source text, and an adapter that stringifies the block still carries its MIME
    # type or URI. Only when none of it came back did the call itself fail.
    tokens = [b.get("text") for b in source.get("blocks") or [] if b.get("text")]
    for block in source.get("blocks") or []:
        kind = block.get("type")
        if kind in _KIND_TOKENS:
            tokens.extend(_tokens(block, kind))
    return not any(token in text for token in tokens)


def _conversion_failed(source: dict, rep: dict) -> bool:
    """The adapter raised while turning a successful result into its own shape."""
    return bool(rep.get("is_error")) and not source.get("is_error") and rep.get(
        "error_channel"
    ) == "exception"


def _kind_code(source: dict, rep: dict | None, kind: str) -> Code | None:
    """How one content kind fared in one call, or None if the call does not exercise it."""
    source_blocks = [b for b in source.get("blocks") or [] if b.get("type") == kind]
    if not source_blocks:
        return None
    if rep is None:
        return "u"
    if _conversion_failed(source, rep):
        message = (rep.get("text") or "").lower()
        named = any(word in message for word in _KIND_WORDS[kind])
        return "n" if named else "u"
    if rep.get("error_channel") == "exception":
        return None  # an exception carries no content; nothing to judge here
    if _invocation_failed(source, rep):
        return "u"
    if kind in (rep.get("block_kinds") or []):
        return "y"
    haystack = _rendered_text(rep)
    if kind == "text":
        if any(b.get("text") == rep.get("text") for b in source_blocks):
            return "y"  # bare string, byte-identical to the block
        return "a" if any(t in haystack for b in source_blocks for t in _tokens(b, kind)) else "n"
    return "a" if any(t in haystack for b in source_blocks for t in _tokens(b, kind)) else "n"


def _annotations_code(source: dict, rep: dict | None) -> Code | None:
    annotated = [b for b in source.get("blocks") or [] if b.get("annotations")]
    if not annotated:
        return None
    if rep is None or _conversion_failed(source, rep) or _invocation_failed(source, rep):
        return "u"
    if rep.get("error_channel") == "exception":
        return None
    for block in annotated:
        for rendered in rep.get("blocks") or []:
            if rendered.get("type") == block.get("type") and rendered.get("annotations"):
                return "y"
    return "n"


def _structured_code(source: dict, rep: dict | None) -> Code | None:
    if source.get("structured_content") is None:
        return None
    if rep is None or _conversion_failed(source, rep) or _invocation_failed(source, rep):
        return "u"
    if rep.get("error_channel") == "exception":
        return None
    rendered = rep.get("structured_content")
    if rendered is None:
        return "n"
    return "y" if rendered == source.get("structured_content") else "a"


def _error_code(source: dict, rep: dict | None) -> Code | None:
    if not source.get("is_error"):
        return None
    if rep is None:
        return "u"
    if not rep.get("is_error"):
        return "n"
    return "a" if rep.get("error_channel") == "exception" else "y"


def _call_code(source: dict, rep: dict | None) -> Code | None:
    return "y" if rep is not None else "n"


RESULT_RULES: dict[str, Callable[[dict, dict | None], Code | None]] = {
    "tools-call": _call_code,
    "structured-content": _structured_code,
    "content-annotations": _annotations_code,
    "tool-execution-errors": _error_code,
    **{
        capability: (lambda kind: (lambda s, r: _kind_code(s, r, kind)))(kind)
        for capability, kind in KIND_CAPABILITIES.items()
    },
}


def load_result_captures(snapshot: Path) -> list[dict]:
    return [_load_json(p) for p in sorted((snapshot / "measurements" / "results").glob("*.json"))]


def _counted_boundaries(payload: dict) -> list[str]:
    """Boundaries whose answer decides the cell.

    A boundary an adapter exposes by default counts, whether or not it is the agent's
    own path, because a caller can read either. A boundary that only exists under a
    documented non-default option is published as a note, not as the cell.
    """
    return [
        b["id"]
        for b in payload.get("result_boundaries") or []
        if b.get("agent_path") or not b.get("limitation")
    ]


def derive_result_cells(snapshot: Path) -> tuple[dict[str, dict[str, Cell]], dict[str, list[str]]]:
    """``{capability: {agent: Cell}}`` plus ``{capability: [agents whose boundaries disagree]}``."""
    cells: dict[str, dict[str, Cell]] = {cap: {} for cap in RESULT_RULES}
    dependent: dict[str, list[str]] = {cap: [] for cap in RESULT_RULES}
    for capture in load_result_captures(snapshot):
        source = capture.get("source") or {}
        for framework, payload in (capture.get("frameworks") or {}).items():
            agent = PUBLISHED_IDS.get(framework, framework)
            if not payload.get("ok"):
                continue
            by_boundary: dict[str, dict[str, dict | None]] = {}
            for entry in payload.get("results") or []:
                boundary = entry.get("boundary") or "default"
                by_boundary.setdefault(boundary, {})[entry["tool"]] = (
                    entry.get("rep") if entry.get("ok") else None
                )
            counted = _counted_boundaries(payload) or list(by_boundary)
            for capability, rule in RESULT_RULES.items():
                per_boundary: dict[str, Code] = {}
                evidence: list[str] = []
                for boundary, reps in by_boundary.items():
                    codes = []
                    for tool, source_rep in source.items():
                        code = rule(source_rep, reps.get(tool))
                        if code is not None:
                            codes.append(code)
                            evidence.append(f"{boundary}/{tool}: {code}")
                    if codes:
                        per_boundary[boundary] = _combine(codes)
                decisive = {b: c for b, c in per_boundary.items() if b in counted} or per_boundary
                if not decisive:
                    continue
                if len(set(decisive.values())) > 1:
                    dependent[capability].append(agent)
                    code = "a"
                else:
                    code = next(iter(decisive.values()))
                cells[capability][agent] = Cell(code, evidence, "results")
    return cells, dependent


# ---------------------------------------------------------------------------
# Surface probe instrument: measurements/probe/*.json from probe_surface.py
# ---------------------------------------------------------------------------


def _probe_status(answer: dict | None) -> Code | None:
    """``n`` for no route, ``u`` for a route that raised, None when it answered."""
    if not answer:
        return "u"
    if answer.get("ok"):
        return None
    return "n" if "unsupported" in answer else "u"


def _entries_code(source: dict, rendered: dict, key: str, identity: str, extras: tuple[str, ...]) -> Cell:
    status = _probe_status(rendered)
    if status:
        return Cell(status, [rendered.get("unsupported") or rendered.get("error") or "no answer"], "probe")
    want = {e.get(identity) for e in source.get("entries") or [] if e.get(identity)}
    have = {e.get(identity) for e in rendered.get("entries") or [] if e.get(identity)}
    if not want:
        return Cell("u", ["fixture publishes nothing here"], "probe")
    if not want <= have:
        missing = sorted(want - have)
        return Cell("n" if not have & want else "a", [f"{key}: missing {missing}"], "probe")
    lost = []
    for entry in source.get("entries") or []:
        twin = next((e for e in rendered["entries"] if e.get(identity) == entry.get(identity)), {})
        for field_name in extras:
            if entry.get(field_name) not in (None, "") and twin.get(field_name) in (None, ""):
                lost.append(f"{entry.get(identity)}.{field_name}")
    if lost:
        return Cell("a", [f"{key}: dropped {sorted(set(lost))}"], "probe")
    return Cell("y", [f"{key}: {len(want)} entries intact"], "probe")


def _contents_code(
    source: dict, rendered: dict, key: str, payload: str, identity: tuple[str, ...] = ("uri", "mimeType")
) -> Cell:
    status = _probe_status(rendered)
    if status:
        return Cell(status, [rendered.get("unsupported") or rendered.get("error") or "no answer"], "probe")
    if source.get(payload) != rendered.get(payload):
        return Cell("n", [f"{key}: {payload} differs"], "probe")
    lost = [f for f in identity if source.get(f) and not rendered.get(f)]
    if lost:
        return Cell("a", [f"{key}: {payload} intact, dropped {lost}"], "probe")
    return Cell("y", [f"{key}: contents and identity intact"], "probe")


def _arguments_code(source: dict, rendered: dict) -> Cell:
    status = _probe_status(rendered)
    if status:
        return Cell(status, [rendered.get("unsupported") or rendered.get("error") or "no answer"], "probe")
    want = {(p["name"], a["name"]): a for p in source.get("entries") or [] for a in p.get("arguments") or []}
    have = {(p["name"], a["name"]): a for p in rendered.get("entries") or [] for a in p.get("arguments") or []}
    if not want:
        return Cell("u", ["fixture publishes no prompt arguments"], "probe")
    if not set(want) <= set(have):
        return Cell("n" if not set(have) & set(want) else "a", [f"prompt-arguments: missing {sorted(set(want) - set(have))}"], "probe")
    lost = [k for k, a in want.items() if a.get("required") is not None and have[k].get("required") is None]
    if lost:
        return Cell("a", [f"prompt-arguments: required flag dropped for {sorted(lost)}"], "probe")
    changed = [k for k, a in want.items() if a.get("required") is not None and have[k].get("required") not in (None, a["required"])]
    if changed:
        return Cell("a", [f"prompt-arguments: required flag changed for {sorted(changed)}"], "probe")
    return Cell("y", [f"prompt-arguments: {len(want)} arguments with required flags intact"], "probe")


def _messages_code(source: dict, rendered: dict) -> Cell:
    status = _probe_status(rendered)
    if status:
        return Cell(status, [rendered.get("unsupported") or rendered.get("error") or "no answer"], "probe")
    want = [m.get("text") for m in source.get("messages") or []]
    have = [m.get("text") for m in rendered.get("messages") or []]
    if not want:
        return Cell("u", ["fixture prompt expands to nothing"], "probe")
    if want != have:
        return Cell("n" if not set(want) & set(have) else "a", ["prompts.get: message text differs"], "probe")
    if any(not m.get("role") for m in rendered.get("messages") or []):
        return Cell("a", ["prompts.get: text intact, role dropped"], "probe")
    return Cell("y", [f"prompts.get: {len(want)} messages with role and text intact"], "probe")


def derive_probe_cells(snapshot: Path) -> dict[str, dict[str, Cell]]:
    cells: dict[str, dict[str, Cell]] = {}
    captures = [_load_json(path) for path in sorted((snapshot / "measurements" / "probe").glob("*.json"))]
    # The fixture offers the same extensions to every session, and a client that opens
    # with its first request (no server/discover) never asks, so the offer is read from
    # every handshake in the snapshot, not only those in the client's own capture file.
    offered = _extensions_of(
        [
            n
            for capture in captures
            for p in (capture.get("frameworks") or {}).values()
            for n in (p.get("wire") or {}).get("negotiations") or []
        ],
        "server_capabilities",
    )
    for capture in captures:
        source = capture.get("source") or {}
        for framework, payload in (capture.get("frameworks") or {}).items():
            if not payload.get("ok"):
                continue
            agent = PUBLISHED_IDS.get(framework, framework)
            probe = payload.get("probe") or {}
            derived = {
                "resources-list": _entries_code(
                    source.get("resources.list", {}), probe.get("resources.list", {}),
                    "resources.list", "uri", ("name", "mimeType"),
                ),
                "resources-read": _contents_code(
                    source.get("resources.read.text", {}), probe.get("resources.read.text", {}),
                    "resources.read", "text",
                ),
                # Binary contents are about the bytes and their type; identity is the
                # resources-read row's question.
                "resource-binary-contents": _contents_code(
                    source.get("resources.read.blob", {}), probe.get("resources.read.blob", {}),
                    "resources.read", "blob_sha256", identity=("mimeType",),
                ),
                "resource-templates": _entries_code(
                    source.get("resources.templates", {}), probe.get("resources.templates", {}),
                    "resources.templates", "uriTemplate", ("description", "mimeType"),
                ),
                "prompts-list": _entries_code(
                    source.get("prompts.list", {}), probe.get("prompts.list", {}),
                    "prompts.list", "name", ("title", "description"),
                ),
                "prompt-arguments": _arguments_code(
                    source.get("prompts.list", {}), probe.get("prompts.list", {})
                ),
                "prompts-get": _messages_code(
                    source.get("prompts.get", {}), probe.get("prompts.get", {})
                ),
            }
            if "wire" in payload:
                derived.update(_extension_cells(payload["wire"], offered))
            for capability_id, cell in derived.items():
                cells.setdefault(capability_id, {})[agent] = cell
    return cells


# Extensions: what the adapter declared about itself when it opened the session, read
# from the handshakes the recorder saw during the probe (probe_surface.py). Declaring is
# the whole of what a client does to negotiate; what it then does with an extension's
# content is a separate question these rows do not answer.

APPS_EXTENSION = "io.modelcontextprotocol/ui"


def _extensions_of(negotiations: list[dict], side: str) -> set[str]:
    names: set[str] = set()
    for negotiation in negotiations:
        names |= set(((negotiation.get(side) or {}).get("extensions") or {}))
    return names


def _extension_cells(wire: dict, offered: set[str] | None = None) -> dict[str, Cell]:
    negotiations = wire.get("negotiations") or []
    if not negotiations:
        missing = Cell("u", ["the recorder saw no handshake"], "probe")
        return {"extension-negotiation": missing, "extension-apps": missing}
    offered = (offered or set()) | _extensions_of(negotiations, "server_capabilities")
    declared = _extensions_of(negotiations, "client_capabilities")
    said = f"client declared {sorted(declared) or 'no extensions'}; fixture offered {sorted(offered) or 'none'}"
    if not offered:
        negotiation = Cell("u", [f"{said}: nothing to negotiate"], "probe")
    elif declared & offered:
        negotiation = Cell("y", [said], "probe")
    else:
        negotiation = Cell("a" if declared else "n", [said], "probe")
    apps = Cell("y" if APPS_EXTENSION in declared else "n", [said], "probe")
    return {"extension-negotiation": negotiation, "extension-apps": apps}


# ---------------------------------------------------------------------------
# Utilities instrument: measurements/utilities/*.json from probe_utilities.py. Each
# question ran against its own recorded fixture process, so a framework's payload
# carries two halves per question: ``utilities[q]``, what the adapter handed its
# caller, and ``wire[q]``, what the server saw (recorder records tagged with the
# launch they came from). A utility counts as supported (``y``) only through the
# adapter's own API; one reachable only on a raw SDK session the adapter hands out,
# or only with a hand-built progress token, is ``a``.
# ---------------------------------------------------------------------------

UTILITY_CAPABILITIES = (
    "pagination-cursor",
    "progress-notifications",
    "logging-notifications",
    "logging-set-level",
    "completion-complete",
    "cancellation",
    "tasks-experimental",
    "extension-tasks",
    "caching-hints",
)
LOG_LEVEL_META = "io.modelcontextprotocol/logLevel"


def _launches(wire: list[dict]) -> list[list[dict]]:
    by_launch: dict[int, list[dict]] = {}
    for record in wire:
        by_launch.setdefault(record.get("capture", 0), []).append(record)
    return [by_launch[k] for k in sorted(by_launch)]


def _refused(feature: str, answer: dict) -> Cell | None:
    """``n`` for no route, ``u`` for a route that raised, None when it answered."""
    if answer.get("ok"):
        return None
    if "unsupported" in answer:
        return Cell("n", [f"{feature}: no route: {answer['unsupported']}"], "utilities")
    return Cell("u", [f"{feature}: {answer.get('error') or 'no answer'}"], "utilities")


def _by_route(answer: dict, evidence: list[str]) -> Cell:
    route = answer.get("route")
    if route == "adapter":
        return Cell("y", evidence, "utilities")
    return Cell("a", evidence + [f"reachable only through {answer.get('api') or route} (route: {route})"], "utilities")


def _meta(record: dict) -> dict:
    return ((record.get("message") or {}).get("params") or {}).get("_meta") or {}


def _u_pagination(source: dict, answer: dict, wire: list[dict]) -> Cell:
    refused = _refused("tools.list", answer)
    if refused:
        return refused
    want = list(source.get("tools") or [])
    have = set(answer.get("names") or [])
    pages = []
    for launch in _launches(wire):
        for request in _c2s(launch, "tools/list"):
            reply = next((r for r in _s2c(launch, "tools/list") if r.get("id") == request.get("id") and r["seq"] > request["seq"]), None)
            if reply is None:
                continue
            cursor = (request["message"].get("params") or {}).get("cursor")
            pages.append(f"cursor={cursor!r} -> nextCursor={((reply['message'].get('result') or {}).get('nextCursor'))!r}")
        if pages:
            break
    exposed = len(have & set(want))
    evidence = [f"tools.list: {exposed} of {len(want)} fixture tools exposed", f"wire: {len(pages)} tools/list pages: {'; '.join(pages)}"]
    if want and exposed == len(want):
        return _by_route(answer, evidence)
    return Cell("n" if exposed <= int(source.get("page_size") or 0) else "a", evidence, "utilities")


def _u_progress(source: dict, answer: dict, wire: list[dict]) -> Cell:
    calls = _tool_calls(wire, "slow_with_progress")
    if not calls:
        return _refused("progress", answer) or Cell("u", ["progress: slow_with_progress never called"], "utilities")
    token = next((_meta(c).get("progressToken") for c in calls if _meta(c).get("progressToken") is not None), None)
    sent = len(_s2c(wire, "notifications/progress", "notification"))
    if token is None:
        return Cell("n", ["wire: tools/call slow_with_progress carried no _meta.progressToken; the server sent no progress"]
                    + ([f"progress: {answer['unsupported']}"] if "unsupported" in answer else []), "utilities")
    seen = len(answer.get("events") or []) if answer.get("ok") else 0
    evidence = [f"wire: progressToken={token!r}, server sent {sent} notifications/progress", f"progress: caller received {seen}"]
    if sent and seen >= sent:
        return _by_route(answer, evidence)
    refused = _refused("progress", answer)
    return Cell("a", evidence + (refused.evidence if refused else []), "utilities")


def _u_logging(source: dict, answer: dict, wire: list[dict]) -> Cell:
    calls = _tool_calls(wire, "emit_logs")
    if not calls:
        return _refused("logging", answer) or Cell("u", ["logging: emit_logs never called"], "utilities")
    sent = [r for r in _s2c(wire, "notifications/message", "notification")]
    if not sent:
        level = next((_meta(c).get(LOG_LEVEL_META) for c in calls if _meta(c).get(LOG_LEVEL_META)), None)
        why = "the server sent none" if level else f"tools/call emit_logs carried no {LOG_LEVEL_META} in _meta, so the server sent none"
        return Cell("n", [f"wire: {why}"] + ([f"logging: {answer['unsupported']}"] if "unsupported" in answer else []), "utilities")
    texts = [str(m.get("data")) for m in (answer.get("messages") or [])] if answer.get("ok") else []
    seen = sum(1 for t in texts if "SENTINEL_LOG_" in t)
    evidence = [f"wire: server sent {len(sent)} notifications/message", f"logging: caller received {seen}"]
    if seen >= len(sent):
        return _by_route(answer, evidence)
    refused = _refused("logging", answer)
    return Cell("n" if seen == 0 else "a", evidence + (refused.evidence if refused else []), "utilities")


def _u_set_level(source: dict, answer: dict, wire: list[dict]) -> Cell:
    requests = _c2s(wire, "logging/setLevel")
    if not requests:
        return _refused("set_level", answer) or Cell("n", ["wire: no logging/setLevel sent"], "utilities")
    level = (requests[0]["message"].get("params") or {}).get("level")
    answered = [r for r in _s2c(wire, "logging/setLevel", "response")]
    evidence = [f"wire: logging/setLevel level={level!r}, {'answered' if answered else 'not answered'}",
                f"set_level: emit_logs then said {answer.get('text')!r}"]
    if answered and f"at level>={level}" in str(answer.get("text") or ""):
        return _by_route(answer, evidence)
    return Cell("a", evidence, "utilities")


def _u_completion(source: dict, answer: dict, wire: list[dict]) -> Cell:
    refused = _refused("completion", answer)
    if refused:
        return refused
    want = source.get("completion") or {}
    matched = [k for k in ("prompt", "resource") if want.get(k) and list(answer.get(k) or []) == want[k]]
    evidence = [f"wire: {len(_c2s(wire, 'completion/complete'))} completion/complete requests",
                f"completion: prompt={answer.get('prompt')}, resource={answer.get('resource')}"]
    if len(matched) == 2:
        return _by_route(answer, evidence)
    return Cell("a" if matched else "n", evidence, "utilities")


def _u_cancellation(source: dict, answer: dict, wire: list[dict]) -> Cell:
    refused = _refused("cancellation", answer)
    if refused and refused.code == "n":
        return refused
    for launch in _launches(wire):
        calls = _tool_calls(launch, "cancellable_wait")
        if not calls:
            continue
        call = calls[0]
        notices = [n for n in _c2s(launch, "notifications/cancelled", "notification")
                   if (n["message"].get("params") or {}).get("requestId") == call.get("id")]
        finished = any("SENTINEL_NOT_CANCELLED" in json.dumps(r.get("message")) for r in _s2c(launch, "tools/call"))
        evidence = [f"cancellation: caller abandoned via {answer.get('mechanism')} after {answer.get('abandoned_after_s')}s ({answer.get('outcome')})"]
        if notices:
            reason = (notices[0]["message"].get("params") or {}).get("reason")
            return Cell("y", evidence + [f"wire: notifications/cancelled for request {call.get('id')!r} (reason {reason!r}) on stdio"], "utilities")
        tail = "the call ran on and returned SENTINEL_NOT_CANCELLED" if finished else "the session ended with the call unanswered"
        return Cell("n", evidence + [f"wire: no notifications/cancelled for request {call.get('id')!r} on stdio; {tail}"], "utilities")
    return refused or Cell("u", ["cancellation: cancellable_wait never called"], "utilities")


def _u_tasks(source: dict, answer: dict, wire: list[dict], modern: bool) -> Cell:
    extension = source.get("tasks_extension") or "io.modelcontextprotocol/tasks"
    declared = []
    for negotiation in (r for r in wire if r.get("type") == "negotiation"):
        capabilities = negotiation.get("client_capabilities") or {}
        if modern and extension in (capabilities.get("extensions") or {}):
            declared.append(f"extension {extension}")
        if not modern and capabilities.get("tasks") is not None:
            declared.append("capabilities.tasks")
    augmented = [c for c in _tool_calls(wire, "task_capable") if (c["message"].get("params") or {}).get("task") is not None]
    polled = sorted({r.get("method") for r in _c2s(wire) if str(r.get("method") or "").startswith("tasks/")})
    if not (declared or augmented or polled):
        called = bool(_tool_calls(wire, "task_capable"))
        return Cell("n", [
            f"wire: client declared no {'Tasks extension' if modern else 'tasks capability'}"
            + (", called task_capable without task augmentation" if called else ""),
            "probe only: the fixture advertises task-augmented tools/call and answers synchronously",
        ], "utilities")
    return Cell("u", [f"wire: {', '.join(declared + [f'{len(augmented)} task-augmented calls'] + polled)}",
                      "probe only: the fixture never returns a task, so completion is not judged"], "utilities")


def _u_caching(source: dict, answer: dict, wire: list[dict]) -> Cell:
    refused = _refused("caching", answer)
    if refused:
        return refused
    first_pages = [r for r in _c2s(wire, "tools/list") if not (r["message"].get("params") or {}).get("cursor")]
    hinted = [r for r in _s2c(wire, "tools/list", "response") if "ttlMs" in (r["message"].get("result") or {})]
    if not hinted:
        return Cell("u", ["wire: the fixture's tools/list carried no ttlMs"], "utilities")
    ttl = hinted[0]["message"]["result"]["ttlMs"]
    marks = {m["call"]: m for m in answer.get("marks") or []}

    def fetched(call: str) -> bool:
        mark = marks.get(call) or {}
        return any(mark.get("started", "~") <= (r.get("t") or "") <= mark.get("finished", "") for r in first_pages)

    within, after = fetched("within_ttl"), fetched("after_ttl")
    evidence = [
        f"wire: tools/list answered with ttlMs={ttl}; {len(first_pages)} first-page fetches",
        f"caching: re-list inside the ttl {'went to the server' if within else 'was served locally'}; "
        f"re-list after it {'went to the server' if after else 'was served locally (stale)'}",
        "cacheScope partitioning not exercised: one authorization context",
    ]
    passed = (not within) + after
    return Cell({2: "y", 1: "a", 0: "n"}[passed], evidence, "utilities")


def derive_utility_cells(snapshot: Path) -> dict[str, dict[str, Cell]]:
    surface_path = snapshot / "surface.json"
    defined = {f["id"] for f in _load_json(surface_path).get("features", [])} if surface_path.exists() else set()
    cells: dict[str, dict[str, Cell]] = {}
    for path in sorted((snapshot / "measurements" / "utilities").glob("*.json")):
        capture = _load_json(path)
        source = capture.get("source") or {}
        modern = (capture.get("spec_version") or "") >= "2026-07-28"
        for framework, payload in (capture.get("frameworks") or {}).items():
            if not payload.get("ok"):
                continue
            agent = PUBLISHED_IDS.get(framework, framework)
            answers = payload.get("utilities") or {}
            wire = payload.get("wire") or {}

            def ask(key: str) -> tuple[dict, list[dict]]:
                return answers.get(key) or {}, wire.get(key) or []

            derived = {
                "pagination-cursor": _u_pagination(source, *ask("tools.list")),
                "progress-notifications": _u_progress(source, *ask("progress")),
                "logging-notifications": _u_logging(source, *ask("logging")),
                "logging-set-level": _u_set_level(source, *ask("set_level")),
                "completion-complete": _u_completion(source, *ask("completion")),
                "cancellation": _u_cancellation(source, *ask("cancellation")),
                "tasks-experimental": _u_tasks(source, *ask("tasks"), modern=False),
                "extension-tasks": _u_tasks(source, *ask("tasks"), modern=True),
                "caching-hints": _u_caching(source, *ask("caching")),
            }
            for capability_id, cell in derived.items():
                if capability_id in defined:
                    cells.setdefault(capability_id, {})[agent] = cell
    return cells


# ---------------------------------------------------------------------------
# Notification instrument: measurements/notify/*.json from probe_notifications.py,
# which changes fixtures/notify_server.py's catalogue mid-session (see notify.py)
# ---------------------------------------------------------------------------

# probe key -> (capability, event name a listen stream delivers for it)
NOTIFY_LIST_CHANGES = {
    "tools.list_changed": ("tools-list-changed", "ToolsListChanged"),
    "resources.list_changed": ("resources-list-changed", "ResourcesListChanged"),
    "prompts.list_changed": ("prompts-list-changed", "PromptsListChanged"),
}
LISTEN_EVENTS = ("ToolsListChanged", "ResourcesListChanged", "PromptsListChanged", "ResourceUpdated")


def _notify_status(answer: dict | None) -> Cell | None:
    """``n`` for no route, ``u`` for a run that raised or never ran, None when it answered."""
    if not answer:
        return Cell("u", ["no answer"], "notify")
    if answer.get("ok"):
        return None
    if "unsupported" in answer:
        return Cell("n", [f"no route: {answer['unsupported']}"], "notify")
    return Cell("u", [f"probe raised: {answer.get('error')}"], "notify")


def _source_fired(source: dict, key: str, event: str) -> bool:
    """Did the fixture deliver this change to a plain SDK client on some channel?"""
    return bool((source.get(key) or {}).get("notified")) or event in (
        (source.get("subscriptions.listen") or {}).get("events") or []
    )


def _list_changed_code(source: dict, rendered: dict, key: str, event: str) -> Cell:
    if not _source_fired(source, key, event):
        return Cell("u", [f"{key}: the fixture delivered no notification to the control client"], "notify")
    status = _notify_status(rendered)
    if status:
        return status
    sentinel = rendered.get("sentinel") or ""
    followed = any(sentinel in e for e in rendered.get("after") or []) and not any(
        sentinel in e for e in rendered.get("before") or []
    )
    hook = rendered.get("hook")
    heard = bool(rendered.get("notified"))
    evidence = [f"{key}: fixture said {rendered.get('server_said')!r}"]
    if rendered.get("hook_error"):
        evidence.append(f"{key}: registering {hook} raised {rendered['hook_error']}")
    if heard and followed:
        evidence.insert(0, f"{key}: {hook} heard it and the next list shows the change")
        return Cell("a" if rendered.get("hook_error") else "y", evidence, "notify")
    if heard:
        evidence.insert(0, f"{key}: {hook} heard it, but the next list is stale")
        return Cell("a", evidence, "notify")
    heard_by = f"{hook} heard nothing" if hook else "the adapter offers no hook"
    if followed:
        evidence.insert(0, f"{key}: {heard_by}; the next list refetches and shows the change")
        return Cell("a", evidence, "notify")
    evidence.insert(0, f"{key}: {heard_by} and the next list is stale")
    return Cell("n", evidence, "notify")


def _subscription_code(source: dict, rendered: dict) -> Cell:
    key = "resources.subscribe"
    if not (source.get(key) or {}).get("notified"):
        return Cell("u", [f"{key}: the control client got no update from the fixture"], "notify")
    status = _notify_status(rendered)
    if status:
        return status
    api = rendered.get("api")
    if not rendered.get("subscribed"):
        return Cell("n", [f"{key}: {api} raised {rendered.get('subscribe_error')}"], "notify")
    if not rendered.get("notified"):
        return Cell("n", [f"{key}: subscribed through {api}, but no update reached the caller"], "notify")
    problems = []
    if not rendered.get("unsubscribed"):
        problems.append(f"unsubscribe failed: {rendered.get('unsubscribe_error') or 'no unsubscribe route'}")
    elif rendered.get("notified_after_unsubscribe"):
        problems.append("updates kept arriving after unsubscribe")
    if rendered.get("route") != "adapter":
        problems.append(f"only through the raw MCP session the adapter exposes ({api})")
    if problems:
        return Cell("a", [f"{key}: update delivered; " + "; ".join(problems)], "notify")
    return Cell("y", [f"{key}: {api} subscribed, delivered the update, and unsubscribed"], "notify")


def _listen_code(source: dict, rendered: dict) -> Cell:
    key = "subscriptions.listen"
    if not (source.get(key) or {}).get("events"):
        return Cell("u", [f"{key}: the control client's stream delivered nothing"], "notify")
    status = _notify_status(rendered)
    if status:
        return status
    api = rendered.get("api")
    if not rendered.get("opened"):
        return Cell("n", [f"{key}: {api} could not open a stream: {rendered.get('open_error')}"], "notify")
    events = set(rendered.get("events") or [])
    missing = [e for e in LISTEN_EVENTS if e not in events]
    if not events:
        return Cell("n", [f"{key}: {api} opened a stream that delivered nothing"], "notify")
    problems = []
    if missing:
        problems.append(f"missing {missing}")
    if rendered.get("route") != "adapter":
        problems.append(f"only through the raw MCP session the adapter exposes ({api})")
    if problems:
        return Cell("a", [f"{key}: stream delivered {sorted(events)}; " + "; ".join(problems)], "notify")
    return Cell("y", [f"{key}: {api} delivered all {len(LISTEN_EVENTS)} event kinds"], "notify")


def derive_notify_cells(snapshot: Path) -> dict[str, dict[str, Cell]]:
    cells: dict[str, dict[str, Cell]] = {}
    for path in sorted((snapshot / "measurements" / "notify").glob("*.json")):
        capture = _load_json(path)
        source = capture.get("source") or {}
        for framework, payload in (capture.get("frameworks") or {}).items():
            if not payload.get("ok"):
                continue
            agent = PUBLISHED_IDS.get(framework, framework)
            rendered = payload.get("notify") or {}
            derived = {
                capability: _list_changed_code(source, rendered.get(key) or {}, key, event)
                for key, (capability, event) in NOTIFY_LIST_CHANGES.items()
            }
            derived["resource-subscriptions"] = _subscription_code(source, rendered.get("resources.subscribe") or {})
            # The stream exists only at 2026-07-28; an earlier capture has no such row.
            if "subscriptions.listen" in source:
                derived["subscriptions-listen"] = _listen_code(source, rendered.get("subscriptions.listen") or {})
            for capability_id, cell in derived.items():
                cells.setdefault(capability_id, {})[agent] = cell
    return cells


# ---------------------------------------------------------------------------
# Conformance instrument: measurements/conformance/<renderer>/checks.json from
# run_conformance.py, which runs the official MCP conformance suite's client scenarios
# with each adapter as the client
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConformanceRule:
    """Which scenarios answer a capability, and which of their checks count.

    ``scenarios`` is a regular expression over the suite's scenario name. ``checks`` is
    a regular expression over a check id, or None to count every graded check the
    scenario records. INFO checks are the suite's wire log and are never graded.
    """

    scenarios: str
    checks: str | None = None


CONFORMANCE_RULES: dict[str, ConformanceRule] = {
    "initialize-handshake": ConformanceRule(r"^initialize$"),
    # A full round trip: POST each message, read the answer from a request-scoped SSE
    # stream, and call a tool on the far side.
    "transport-streamable-http": ConformanceRule(r"^tools_call$"),
    # The row asks whether a broken stream is resumed: reconnect, and say where from.
    # How promptly the client honours `retry` is graded too, but by wall-clock time on
    # a machine running the whole suite in parallel, so it does not decide this cell.
    "sse-resumability": ConformanceRule(
        r"^sse-retry$", r"^client-sse-(graceful-reconnect|last-event-id)$"
    ),
    # Every authorization scenario, every check: discovery, registration or a
    # pre-registered or URL-based client id, PKCE, the token request, scope handling
    # and the bearer token on each MCP request.
    "authorization-oauth": ConformanceRule(r"^auth/"),
    # Only the discovery checks: did the client find the authorization server from
    # the resource's metadata rather than being told where it was.
    "authorization-resource-metadata": ConformanceRule(
        r"^auth/metadata-", r"^(prm-|authorization-server-metadata)"
    ),
}

# The one client scenario whose server mints an Mcp-Session-Id and rejects a request
# that does not carry it back, so its wire log answers the session-identifier row.
SESSION_SCENARIO = "elicitation-sep1034-client-defaults"
SESSION_OPENERS = ("initialize", "server/discover")


def load_conformance_captures(snapshot: Path) -> list[dict]:
    return [
        _load_json(p) for p in sorted((snapshot / "measurements" / "conformance").glob("*/checks.json"))
    ]


def _scenario_runs(capture: dict) -> dict[str, list[list[dict]]]:
    """``{scenario: [checks from each suite run]}``. ``auth`` and ``core`` overlap, so
    most authorization scenarios ran twice; each run is a vote."""

    runs: dict[str, list[list[dict]]] = {}
    for run in (capture.get("suites") or {}).values():
        for name, scenario in (run.get("scenarios") or {}).items():
            runs.setdefault(name, []).append(scenario.get("checks") or [])
    return runs


def _checks_code(checks: list[dict], pattern: str | None) -> tuple[Code, str]:
    graded = [
        c for c in checks
        if c.get("status") != "INFO" and (pattern is None or re.search(pattern, c.get("id", "")))
    ]
    if any(c.get("description") == "Failed to run scenario" for c in graded):
        return "u", "the suite could not run the scenario"
    if not graded:
        return "u", "no graded checks"
    passed = [c for c in graded if c.get("status") == "SUCCESS"]
    missed = sorted({c["id"] for c in graded if c.get("status") != "SUCCESS"})
    summary = f"{len(passed)}/{len(graded)} checks passed" + (f", not {', '.join(missed)}" if missed else "")
    if len(passed) == len(graded):
        return "y", summary
    return ("n" if not passed else "a"), summary


def _session_code(checks: list[dict]) -> tuple[Code, str]:
    """Did every request after the server minted a session ID get past the session check?"""

    wire = [c.get("details") or {} for c in checks if c.get("status") == "INFO"]
    # A request that opens a connection (the 2.x SDKs probe with server/discover before
    # initialize) precedes any session, so it owes no session ID.
    responses = [d for d in wire if "statusCode" in d]
    minted = next((i for i, d in enumerate(responses) if d.get("sessionIdIssued")), None)
    if minted is None:
        return "u", "the server never minted a session ID"
    later = [d for d in responses[minted + 1 :] if d.get("mcpMethod") not in SESSION_OPENERS]
    rejected = [d for d in later if d.get("statusCode") == 400 and "session" in (d.get("body") or "").lower()]
    if not later:
        return "u", "no request followed the session ID"
    if not rejected:
        return "y", f"{len(later)} requests after the session ID was minted, none rejected for it"
    if len(rejected) == len(later):
        return "n", f"all {len(later)} later requests rejected for a missing session ID"
    return "a", f"{len(rejected)} of {len(later)} later requests rejected for a missing session ID"


def derive_conformance_cells(snapshot: Path) -> dict[str, dict[str, Cell]]:
    cells: dict[str, dict[str, Cell]] = {}
    for capture in load_conformance_captures(snapshot):
        agent = PUBLISHED_IDS.get(capture.get("renderer", ""), capture.get("renderer", ""))
        runs = _scenario_runs(capture)
        for capability_id, rule in CONFORMANCE_RULES.items():
            codes: list[Code] = []
            evidence: list[str] = []
            for name in sorted(runs):
                if not re.search(rule.scenarios, name):
                    continue
                for checks in runs[name]:
                    code, summary = _checks_code(checks, rule.checks)
                    codes.append(code)
                    evidence.append(f"{name}: {summary}")
            if codes:
                cells.setdefault(capability_id, {})[agent] = Cell(
                    _combine(codes), sorted(set(evidence)), "conformance"
                )
        session = [_session_code(checks) for checks in runs.get(SESSION_SCENARIO, [])]
        if session:
            cells.setdefault("session-identifier", {})[agent] = Cell(
                _combine(code for code, _ in session),
                sorted({f"{SESSION_SCENARIO}: {summary}" for _, summary in session}),
                "conformance",
            )
    return cells


# ---------------------------------------------------------------------------
# Client-features instrument: measurements/client/*.json from
# probe_client_features.py, which answers the fixture's sampling, elicitation and roots
# requests through each adapter's own hooks
# ---------------------------------------------------------------------------

CLIENT_FEATURES = {
    "sampling-create-message": "sampling",
    "sampling-tools": "sampling.tools",
    "elicitation-form-mode": "elicitation.form",
    "elicitation-url-mode": "elicitation.url",
    "roots-list": "roots.list",
    "roots-list-changed": "roots.list_changed",
}
# 2026-07-28 removed the notification with the rest of the roots machinery.
HANDSHAKE_ONLY = {"roots-list-changed"}
# The requests MRTR still carries as stable features; sampling and roots ride it too,
# but deprecated, so they do not decide whether an adapter implements the mechanism.
MRTR_FEATURES = ("elicitation.form", "elicitation.url")
CLIENT_ANSWERED = "SENTINEL_CLIENT_ANSWERED"
CLIENT_UNANSWERED = "SENTINEL_CLIENT_FEATURE_UNAVAILABLE"
# An answer through the adapter's own API is support. One that needs the MCP SDK's
# session keywords forwarded unread, or a method on the library the adapter wraps,
# works but is not something the adapter offers: partial.
ROUTE_CODES = {"adapter": "y", "passthrough": "a", "underlying": "a"}
ROUTE_WORDS = {
    "adapter": "the adapter's own API",
    "passthrough": "session keywords the adapter forwards to the MCP SDK unread",
    "underlying": "the client library the adapter wraps",
}


def load_client_captures(snapshot: Path) -> list[dict]:
    return [_load_json(p) for p in sorted((snapshot / "measurements" / "client").glob("*.json"))]


def _via(entry: dict) -> str:
    hook = entry.get("hook")
    if not hook:
        return "no hook offered"
    return f"via {hook} ({ROUTE_WORDS.get(entry.get('route'), entry.get('route'))})"


def _client_answered(entry: dict) -> bool:
    text = entry.get("text") or ""
    if CLIENT_UNANSWERED in text:
        return False
    return (entry.get("expect") or CLIENT_ANSWERED) in text


def _client_code(entry: dict | None) -> Cell:
    if not entry or "tool" not in entry:
        return Cell("u", ["not probed"], "client")
    tool, text, via = entry["tool"], entry.get("text") or "", _via(entry)
    if tool == "roots_changes_seen":
        # Adapters that re-serialize structuredContent (LangChain4j uses Jackson) drop
        # the space after the colon.
        found = re.search(r'(?:SENTINEL_ROOTS_CHANGES_SEEN |"seen": ?)(\d+)', text)
        if not found:
            return Cell("u", [f"{tool}: no count returned: {text[:160]}"], "client")
        seen = int(found.group(1))
        if not seen:
            return Cell("n", [f"{tool}: the server received no notifications/roots/list_changed, {via}"], "client")
        return Cell(
            ROUTE_CODES.get(entry.get("route"), "u"),
            [f"{tool}: the server received {seen} notifications/roots/list_changed {via}"],
            "client",
        )
    if _client_answered(entry):
        return Cell(ROUTE_CODES.get(entry.get("route"), "u"), [f"{tool}: the server received the answer {via}"], "client")
    if (CLIENT_ANSWERED in text or f'"answered": "{tool}"' in text) and CLIENT_UNANSWERED not in text:
        # An answer came back but not the one the handler gives at the end, which for
        # the sampling tool loop means it stopped after one turn.
        return Cell("a", [f"{tool}: the server received an answer without {entry.get('expect')}, {via}: {text[:200]}"], "client")
    if not entry.get("hook") or CLIENT_UNANSWERED in text:
        # With no hook registered, whatever the adapter reports, the request went
        # unanswered: method not found on the wire, or an input request refused.
        return Cell("n", [f"{tool}: the server got no answer, {via}: {text[:200]}"], "client")
    return Cell("u", [f"{tool}: could not tell whether the server got an answer {via}: {text[:200]}"], "client")


def _mrtr_code(probe: dict) -> Cell:
    entries = [probe.get(feature) or {} for feature in MRTR_FEATURES]
    answered = [e["tool"] for e in entries if e.get("tool") and _client_answered(e)]
    missed = [e.get("tool", "?") for e in entries if not (e.get("tool") and _client_answered(e))]
    evidence = [f"{e.get('tool', '?')}: {'answered' if e.get('tool') in answered else 'unanswered'}, {_via(e)}" for e in entries]
    if not missed:
        routes = {e.get("route") for e in entries}
        return Cell("y" if routes == {"adapter"} else "a", evidence, "client")
    return Cell("a" if answered else "n", evidence, "client")


def derive_client_cells(snapshot: Path) -> dict[str, dict[str, Cell]]:
    cells: dict[str, dict[str, Cell]] = {}
    for capture in load_client_captures(snapshot):
        for framework, payload in (capture.get("frameworks") or {}).items():
            if not payload.get("ok"):
                continue
            agent = PUBLISHED_IDS.get(framework, framework)
            probe = payload.get("client_probe") or {}
            modern = (payload.get("negotiated_mcp_spec_version") or "") >= MRTR_REVISION
            for capability_id, feature in CLIENT_FEATURES.items():
                if modern and capability_id in HANDSHAKE_ONLY:
                    continue
                cells.setdefault(capability_id, {})[agent] = _client_code(probe.get(feature))
            if modern:
                cells.setdefault("mrtr-input-required", {})[agent] = _mrtr_code(probe)
    return cells


# ---------------------------------------------------------------------------
# Protocol instrument: measurements/protocol/<renderer>/ from measure_protocol.py. Each
# renderer ran against fixtures/base_protocol_server.py behind the recorder, so every
# stdio-NN.jsonl is one session and http.jsonl holds every Streamable HTTP session.
# run.json keeps what the adapter handed back. The base-protocol rows are functions of
# those files.
# ---------------------------------------------------------------------------

PROTOCOL_RUN = "run.json"
PING_WINDOW = 5.0
REQUEST_META_KEYS = (
    "io.modelcontextprotocol/protocolVersion",
    "io.modelcontextprotocol/clientCapabilities",
    "io.modelcontextprotocol/clientInfo",
)
# Keywords whose spelling or meaning belongs to one dialect, per fixture tool. `$ref`
# is listed because its target (`$defs` or `definitions`) is dialect-specific.
DIALECT_KEYWORDS = {
    "dialect_2020_12": ("$schema", "$defs", "$ref", "prefixItems", "items", "dependentRequired"),
    "dialect_draft_07": ("$schema", "definitions", "$ref", "items", "additionalItems"),
}
HEADER_TOOL, HEADER_PARAM, HEADER_NAME = "routed_lookup", "region", "mcp-param-region"
MALFORMED_HEADER_TOOL = "bad_header"
INPUT_TOOL = "needs_input"


@dataclass
class ProtocolRun:
    """One renderer's captures and the adapter's own answers from one protocol run."""

    agent: str
    directory: Path
    run: dict
    captures: dict[str, list[dict]]
    root: Path

    def where(self, name: str, record: dict | None = None) -> str:
        path = _relative(str(self.directory / name), self.root)
        if record is None:
            return path
        return f"{path} seq {record['seq']} {record.get('method') or record.get('via') or record.get('type')}"

    def stdio(self) -> dict[str, list[dict]]:
        return {n: r for n, r in sorted(self.captures.items()) if n.startswith("stdio-")}

    def http_name(self) -> str:
        # A run that had to leave a tool out was recorded beside the full run's capture.
        omitted = self.transport("http").get("omitted") or []
        return f"http-without-{'-'.join(omitted)}.jsonl" if omitted else "http.jsonl"

    def http(self) -> list[dict] | None:
        return self.captures.get(self.http_name())

    def transport(self, name: str) -> dict:
        return (self.run.get("transports") or {}).get(name) or {}


def _checkout_root(path: Path) -> Path:
    root = path.resolve()
    while root.name and not (root / "fixtures").exists():
        root = root.parent
    return root


def load_protocol_runs(snapshot: Path) -> list[ProtocolRun]:
    runs = []
    root = _checkout_root(snapshot)
    for run_path in sorted((snapshot / "measurements" / "protocol").glob(f"*/{PROTOCOL_RUN}")):
        run = _load_json(run_path)
        renderer = run.get("renderer") or run_path.parent.name
        captures = {
            path.name: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for path in sorted(run_path.parent.glob("*.jsonl"))
        }
        runs.append(ProtocolRun(PUBLISHED_IDS.get(renderer, renderer), run_path.parent, run, captures, root))
    return runs


def _snapshot_features(snapshot: Path) -> set[str]:
    path = snapshot / "surface.json"
    return {f["id"] for f in _load_json(path).get("features", [])} if path.exists() else set()


def _later_client_message(records: list[dict], after: dict) -> bool:
    return any(r["seq"] > after["seq"] for r in _wire_messages(records, direction="client_to_server"))


def _p_initialize(run: ProtocolRun, revision: str) -> Cell:
    codes: list[Code] = []
    evidence: list[str] = []
    probes = 0
    for name, records in run.stdio().items():
        negotiation = next((r for r in records if r.get("type") == "negotiation" and r.get("via") == "initialize"), None)
        sent = {r.get("method") for r in _c2s(records) if r.get("origin") == "client"}
        if negotiation is None and sent == {"server/discover"}:
            # A discovery probe the server refused; the adapter reconnects to initialize.
            probes += 1
            continue
        if negotiation is None:
            codes.append("n")
            evidence.append(f"{run.where(name)}: no initialize request")
            continue
        negotiated = negotiation.get("negotiated_revision")
        initialized = _c2s(records, "notifications/initialized", "notification")
        said = f"requested {negotiation.get('requested_revision')}, negotiated {negotiated}"
        if not negotiated:
            codes.append("n")
            evidence.append(f"{run.where(name, negotiation)}: {said}, error {negotiation.get('error')}")
        elif not initialized:
            codes.append("a")
            evidence.append(f"{run.where(name, negotiation)}: {said}, never sent notifications/initialized")
        elif negotiated != revision:
            codes.append("a")
            evidence.append(f"{run.where(name, negotiation)}: {said}, not {revision}")
        else:
            codes.append("y")
            evidence.append(f"{run.where(name, negotiation)}: {said}, notifications/initialized sent")
    if probes:
        evidence.append(f"{probes} session(s) sent only server/discover and closed when it was refused")
    if not codes:
        return Cell("u", [f"{run.where('')}: no stdio session initialized"], "protocol")
    return Cell(_combine(codes), evidence, "protocol")


def _p_ping(run: ProtocolRun, revision: str) -> Cell:
    sent = answered = 0
    own = 0
    evidence: list[str] = []
    for name, records in run.stdio().items():
        own += len([r for r in _c2s(records, "ping") if r.get("origin") == "client"])
        for ping in [r for r in _s2c(records, "ping", "request") if r.get("origin") == "recorder"]:
            answer = _answer_to(records, ping)
            if answer is not None and answer["kind"] == "response" and answer["elapsed"] - ping["elapsed"] <= PING_WINDOW:
                sent += 1
                answered += 1
                evidence.append(f"{run.where(name, answer)} answered after {answer['elapsed'] - ping['elapsed']:.3f}s")
            elif answer is not None or _later_client_message(records, ping):
                # The client kept talking after the ping, so silence is a choice.
                sent += 1
                evidence.append(f"{run.where(name, ping)}: {'error' if answer else 'no answer'}")
    if not sent:
        return Cell("u", [f"{run.where('')}: the recorder injected no ping any client could have answered"], "protocol")
    evidence.append(f"client issued {own} ping(s) of its own")
    return Cell(_by_fraction(answered, sent), evidence, "protocol")


def _listed_tools(records: list[dict]) -> dict[str, dict]:
    for response in _s2c(records, "tools/list", "response"):
        return {t["name"]: t for t in (response["message"].get("result") or {}).get("tools") or []}
    return {}


def _rendered_tool(tools: list[dict], name: str) -> dict | None:
    # Some adapters namespace a tool with the server's name (Mastra: `src_<tool>`,
    # Semantic Kernel: `<plugin>-<tool>`).
    return next((t for t in tools if t["name"] == name), None) or next(
        (t for t in tools if t["name"].endswith((f"_{name}", f"-{name}"))), None
    )


def _json_pointer(schema: dict, ref: Any) -> Any:
    node: Any = schema
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _p_dialect(run: ProtocolRun, revision: str) -> Cell:
    listed: dict[str, dict] = {}
    for records in run.stdio().values():
        listed = listed or _listed_tools(records)
    rendered = run.transport("stdio").get("tools") or []
    if not listed or not run.transport("stdio").get("ok"):
        return Cell("u", [f"{run.where(PROTOCOL_RUN)}: no listed and rendered definitions to compare"], "protocol")
    codes: list[Code] = []
    evidence: list[str] = []
    omitted = run.transport("stdio").get("omitted") or []
    for tool, keywords in DIALECT_KEYWORDS.items():
        source = (listed.get(tool) or {}).get("inputSchema") or {}
        if tool in omitted:
            codes.append("n")
            error = run.transport("stdio").get("full_fixture_error")
            evidence.append(f"{run.where(PROTOCOL_RUN)}: the adapter loaded no tool while {tool} was listed ({error})")
            continue
        twin = _rendered_tool(rendered, tool)
        if twin is None:
            codes.append("n")
            evidence.append(f"{run.where(PROTOCOL_RUN)}: {tool} not offered")
            continue
        places = [(path, value) for path, value in _walk(source) if path.rsplit(".", 1)[-1].split("[")[0] in keywords]
        have = dict(_walk(twin.get("params") or {}))
        lost = sorted(path for path, value in places if have.get(path) != value)
        # A `$ref` replaced by the definition it points at was resolved, not lost; once
        # every `$ref` is resolved that way the definitions container is not needed.
        refs = [path for path in lost if path.endswith(".$ref")]
        inlined = [path for path in refs
                   if have.get(path[: -len(".$ref")]) == _json_pointer(source, dict(places)[path])]
        if inlined and len(inlined) == len([path for path, _value in places if path.endswith("$ref")]):
            lost = [path for path in lost if path not in inlined and path not in ("$defs", "definitions")]
        resolved = f", resolving {len(inlined)} $ref inline" if inlined else ""
        if not lost:
            codes.append("y")
            evidence.append(f"{run.where(PROTOCOL_RUN)}: {tool} kept all {len(places)} dialect keywords{resolved}")
        elif lost == ["$schema"]:
            codes.append("a")
            evidence.append(f"{run.where(PROTOCOL_RUN)}: {tool} kept its keywords but dropped $schema")
        else:
            codes.append("n" if len(lost) == len(places) else "a")
            evidence.append(f"{run.where(PROTOCOL_RUN)}: {tool} changed or dropped {lost}")
    return Cell(_combine(codes), evidence, "protocol")


def _p_request_meta(run: ProtocolRun, revision: str) -> Cell:
    total = complete = 0
    missing: dict[str, int] = {}
    where = ""
    for name, records in run.captures.items():
        for request in [r for r in _c2s(records) if r.get("origin") == "client"]:
            total += 1
            meta = (request["message"].get("params") or {}).get("_meta") or {}
            absent = [k.rsplit("/", 1)[-1] for k in REQUEST_META_KEYS if k not in meta]
            if absent:
                missing[f"{request.get('method')} without {', '.join(absent)}"] = missing.get(
                    f"{request.get('method')} without {', '.join(absent)}", 0) + 1
                where = where or run.where(name, request)
            else:
                complete += 1
    if not total:
        return Cell("u", [f"{run.where('')}: no client request captured"], "protocol")
    evidence = [f"{run.where('')}: {complete} of {total} requests carried all three _meta keys"]
    evidence += [f"{count} x {what}" for what, count in sorted(missing.items())]
    if where:
        evidence.append(f"first: {where}")
    return Cell(_by_fraction(complete, total), evidence, "protocol")


def _p_discover(run: ProtocolRun, revision: str) -> Cell:
    """Did each working session use a version the server listed in ``server/discover``?

    Some adapters discover over a connection of its own and then open a fresh one for the
    work (Mastra over stdio), so a session that only discovered hands its answer to the
    next session in capture order."""

    sessions = issued = chose = 0
    evidence: list[str] = []
    carried: tuple[str, dict] | None = None
    for name, records in run.stdio().items():
        discover = next((r for r in records if r.get("type") == "negotiation" and r.get("via") == "server/discover"), None)
        work = [r for r in _c2s(records) if r.get("origin") == "client" and r.get("method") != "server/discover"]
        if discover is not None:
            issued += 1
        if not work:
            carried = (name, discover) if discover is not None else carried
            continue
        sessions += 1
        source = (name, discover) if discover is not None else carried
        carried = None
        used = ((work[0]["message"].get("params") or {}).get("_meta") or {}).get(REQUEST_META_KEYS[0])
        if source is None:
            evidence.append(f"{run.where(name, work[0])}: used {used} without discovering")
            continue
        supported = source[1].get("supported_versions") or []
        via = "" if source[0] == name else f" discovered over {source[0]}"
        if used in supported:
            chose += 1
            evidence.append(f"{run.where(name, work[0])}: used {used} from supportedVersions {supported}{via}")
        else:
            evidence.append(f"{run.where(name, work[0])}: used {used}, not in supportedVersions {supported}{via}")
    if not sessions:
        return Cell("u", [f"{run.where('')}: no stdio session did any work"], "protocol")
    if not issued:
        return Cell("n", [f"{run.where('')}: none of {sessions} sessions issued server/discover"], "protocol")
    code = "y" if chose == sessions else ("n" if not chose else "a")
    return Cell(code, [f"{chose} of {sessions} working sessions used a discovered version", *evidence], "protocol")


def _input_answer(entry: dict) -> tuple[Code, str]:
    """How the adapter handed back a result whose ``resultType`` is ``input_required``."""
    if not entry.get("ok"):
        text = entry.get("error") or ""
    else:
        rep = entry.get("rep") or {}
        text = rep.get("text") or ""
        if "SENTINEL_INPUT_ANSWERED" in text:
            return "y", f"completed after answering the input request: {text[:80]!r}"
        if not rep.get("is_error") and not rep.get("error_channel"):
            return "n", f"handed back as a completed result: {text[:80]!r}"
    if re.search(r"input|elicit", text, re.IGNORECASE):
        return "y", f"refused as unfinished: {text[:120]!r}"
    return "u", f"failed for an unrelated reason: {text[:120]!r}"


def _p_result_type(run: ProtocolRun, revision: str) -> Cell:
    codes: list[Code] = []
    evidence: list[str] = []
    for name, records in run.captures.items():
        calls = _tool_calls(records, INPUT_TOOL)
        retries = [c for c in calls if "inputResponses" in (c["message"].get("params") or {})]
        for retry in retries:
            codes.append("y")
            evidence.append(f"{run.where(name, retry)}: retried with inputResponses")
    for transport in ("stdio", "http"):
        for entry in run.transport(transport).get("results") or []:
            if entry.get("tool") != INPUT_TOOL:
                continue
            code, said = _input_answer(entry)
            codes.append(code)
            evidence.append(f"{run.where(PROTOCOL_RUN)} {transport}/{entry.get('boundary', 'default')}: {said}")
    if not codes:
        return Cell("u", [f"{run.where(PROTOCOL_RUN)}: {INPUT_TOOL} was never called"], "protocol")
    return Cell(_combine(codes), evidence, "protocol")


def _p_header(run: ProtocolRun, revision: str) -> Cell:
    records, capture = run.http(), run.http_name()
    if records is None or not run.transport("http").get("ok"):
        return Cell("u", [f"{run.where(capture)}: no Streamable HTTP run"], "protocol")
    calls = _tool_calls(records, HEADER_TOOL)
    if not calls:
        return Cell("u", [f"{run.where(capture)}: {HEADER_TOOL} was never called"], "protocol")
    evidence: list[str] = []
    mirrored = 0
    for call in calls:
        value = ((call.get("http") or {}).get("headers") or {}).get(HEADER_NAME)
        argument = ((call["message"].get("params") or {}).get("arguments") or {}).get(HEADER_PARAM)
        if value is not None and value == argument:
            mirrored += 1
        evidence.append(f"{run.where(capture, call)}: {HEADER_NAME} {'= ' + repr(value) if value is not None else 'absent'}")
    offered = _rendered_tool(run.transport("http").get("tools") or [], MALFORMED_HEADER_TOOL) is not None
    evidence.append(f"{run.where(PROTOCOL_RUN)} http: {MALFORMED_HEADER_TOOL} {'offered' if offered else 'excluded'}")
    codes = [_by_fraction(mirrored, len(calls)), "n" if offered else "y"]
    return Cell("y" if codes == ["y", "y"] else ("n" if codes == ["n", "n"] else "a"), evidence, "protocol")


PROTOCOL_RULES: dict[str, Callable[[ProtocolRun, str], Cell]] = {
    "initialize-handshake": _p_initialize,
    "ping": _p_ping,
    "json-schema-dialect": _p_dialect,
    "per-request-metadata": _p_request_meta,
    "server-discover": _p_discover,
    "result-type": _p_result_type,
    "x-mcp-header": _p_header,
}


def derive_protocol_cells(snapshot: Path) -> dict[str, dict[str, Cell]]:
    """Base-protocol cells for the rows this snapshot's surface defines."""
    frameworks = snapshot / "frameworks.json"
    if not frameworks.exists():
        return {}
    revision = _load_json(frameworks).get("mcp_spec", "")
    features = _snapshot_features(snapshot)
    cells: dict[str, dict[str, Cell]] = {}
    for run in load_protocol_runs(snapshot):
        for capability_id, rule in PROTOCOL_RULES.items():
            if capability_id in features:
                cells.setdefault(capability_id, {})[run.agent] = rule(run, revision)
    return cells


def _both_instruments(first: Cell, second: Cell) -> Cell:
    """One row measured two ways: the worse answer stands, and both are cited."""
    return Cell(
        _combine([first.code, second.code]),
        [*first.evidence, *second.evidence],
        "+".join(sorted({*first.instrument.split("+"), *second.instrument.split("+")})),
    )


# ---------------------------------------------------------------------------
# Transport instrument: measurements/transports/*.json from measure_transports.py, which
# asks every renderer for the same definitions, results and probe answers over stdio,
# Streamable HTTP and (where the revision still defines it) HTTP+SSE, with the recorder
# or the legacy SSE bridge logging the wire under measurements/transports/wire/
# ---------------------------------------------------------------------------

# Each transport row reads the runs over its transport. stdio is the reference: it is the
# transport every other instrument measures over, so a transport whose answers differ
# from stdio's is the one marked down.
TRANSPORT_FEATURES = {
    "transport-stdio": "stdio",
    "transport-streamable-http": "http",
    "transport-http-sse": "sse",
}
# 2026-07-28 removed sessions and requires these headers on every Streamable HTTP POST;
# the name header only on the methods that address one named thing.
STATELESS_REVISION = "2026-07-28"
NAMED_METHODS = {"tools/call": "name", "prompts/get": "name", "resources/read": "uri"}


def load_transport_captures(snapshot: Path) -> list[tuple[Path, dict]]:
    return [(p, _load_json(p)) for p in sorted((snapshot / "measurements" / "transports").glob("*.json"))]


# An id an adapter mints for each block it builds (LangChain's ``lc_<uuid>``) differs on
# every call over any transport, so it is not something a transport changed.
_MINTED_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _stable(value: Any) -> Any:
    if isinstance(value, str):
        return _MINTED_ID.sub("<id>", value)
    if isinstance(value, list):
        return [_stable(item) for item in value]
    if isinstance(value, dict):
        return {key: _stable(item) for key, item in value.items()}
    return value


def _answer_items(answers: dict) -> dict[str, Any]:
    """The comparable pieces of one run, keyed so a difference names what moved."""
    answers = _stable(answers)
    items: dict[str, Any] = {}
    for tool in answers.get("tools") or []:
        items[f"tool {tool.get('name')}"] = tool
    for result in answers.get("results") or []:
        items[f"call {result.get('tool')}@{result.get('boundary') or 'default'}"] = result
    for feature, answer in (answers.get("probe") or {}).items():
        items[f"probe {feature}"] = answer
    return items


def _transport_code(runs: dict[str, dict], transport: str, where: Callable[[dict], str]) -> tuple[Code, list[str]] | None:
    target = runs.get(transport)
    if not target:
        return None
    reference = runs.get("stdio") or {}
    opened = [kind for kind, entry in target.items() if entry.get("ok")]
    evidence: list[str] = []
    differing = 0
    for kind, entry in sorted(target.items()):
        if not entry.get("ok"):
            evidence.append(f"{where(entry)}: no session ({(entry.get('error') or '')[:120]})")
            continue
        items = _answer_items(entry.get("answers") or {})
        base = reference.get(kind)
        if transport == "stdio" or not base or not base.get("ok"):
            evidence.append(f"{where(entry)}: session opened, {len(items)} answers")
            continue
        expected = _answer_items(base.get("answers") or {})
        moved = sorted(k for k in set(items) | set(expected) if items.get(k) != expected.get(k))
        if moved:
            differing += 1
            evidence.append(f"{where(entry)}: {len(moved)} of {len(expected)} answers differ from stdio ({', '.join(moved[:4])})")
        else:
            evidence.append(f"{where(entry)}: all {len(expected)} answers match stdio")
    if not opened:
        return "n", evidence
    if len(opened) < len(target) or differing:
        return "a", evidence
    return "y", evidence


def _capture_records(snapshot: Path, entry: dict) -> list[dict]:
    path = snapshot / "measurements" / "transports" / entry.get("capture", "")
    if not entry.get("capture") or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _client_posts(records: list[dict]) -> list[dict]:
    return [
        r for r in records
        if r.get("type") == "message" and r.get("direction") == "client_to_server"
        and r.get("kind") in ("request", "notification") and r.get("http")
    ]


def _header_code(records: list[dict], where: str) -> tuple[Code, str] | None:
    """On a stateless revision, did every POST carry ``Mcp-Method`` (and ``Mcp-Name``)?

    Only sessions that settled on such a revision count: a 2.x client probes a 2025
    server with ``server/discover`` at 2026-07-28 and then falls back to initialize.
    """
    settled = {
        r.get("session") for r in records
        if r.get("type") == "negotiation" and (r.get("negotiated_revision") or "") >= STATELESS_REVISION
    }
    posts = [
        r for r in _client_posts(records)
        if r.get("session") in settled and (r.get("revision") or "") >= STATELESS_REVISION
    ]
    if not posts:
        return None
    lacking = []
    for record in posts:
        headers = {k.lower(): v for k, v in ((record.get("http") or {}).get("headers") or {}).items()}
        if headers.get("mcp-method") != record.get("method"):
            lacking.append(f"{record.get('method')} without Mcp-Method")
        elif record.get("method") in NAMED_METHODS and not headers.get("mcp-name"):
            lacking.append(f"{record.get('method')} without Mcp-Name")
    if not lacking:
        return "y", f"{where}: all {len(posts)} POSTs carried Mcp-Method, and Mcp-Name where the method names a target"
    return ("n" if len(lacking) == len(posts) else "a"), (
        f"{where}: {len(lacking)} of {len(posts)} POSTs lacked a required header ({', '.join(sorted(set(lacking))[:3])})"
    )


def _recorded_session_code(records: list[dict], where: str) -> tuple[Code, str] | None:
    """After the recorder minted an ``Mcp-Session-Id``, did every later request carry it back?

    The recorder files a POST under the id it carries, under ``stateless`` when it carries
    none, and logs ``unknown_session`` for an id it never issued. A request that opens a
    connection (``initialize``, or the ``server/discover`` probe a 2.x client sends
    first) precedes any session, so it owes no session ID.
    """
    minted: set[str] = set()
    carried = missing = 0
    for record in records:
        if record.get("type") == "http" and record.get("event") == "unknown_session":
            missing += 1
            continue
        if record.get("type") != "message" or record.get("direction") != "client_to_server":
            continue
        session = record.get("session")
        if record.get("method") in SESSION_OPENERS and record.get("kind") == "request":
            if record.get("method") == "initialize" and session not in (None, "stateless"):
                minted.add(session)
            continue
        if record.get("kind") not in ("request", "notification") or not minted:
            continue
        if session in minted:
            carried += 1
        else:
            missing += 1
    if not minted:
        return None
    total = carried + missing
    if not total:
        return "u", f"{where}: no request followed the session ID"
    if not missing:
        return "y", f"{where}: {total} requests after initialize, every one carried the minted Mcp-Session-Id"
    return ("n" if not carried else "a"), f"{where}: {missing} of {total} requests after initialize lacked the minted Mcp-Session-Id"


def derive_transport_cells(snapshot: Path) -> dict[str, dict[str, Cell]]:
    cells: dict[str, dict[str, Cell]] = {}
    for path, capture in load_transport_captures(snapshot):
        prefix = path.parent.relative_to(snapshot).as_posix()

        def where(entry: dict) -> str:
            return f"{prefix}/{entry.get('capture') or entry.get('fixture')}"

        for framework, payload in (capture.get("frameworks") or {}).items():
            agent = PUBLISHED_IDS.get(framework, framework)
            runs = payload.get("transports") or {}
            for capability_id, transport in TRANSPORT_FEATURES.items():
                answered = _transport_code(runs, transport, where)
                if answered is None:
                    continue
                code, evidence = answered
                if transport == "http":
                    headers = [
                        _header_code(_capture_records(snapshot, entry), where(entry))
                        for entry in (runs.get("http") or {}).values() if entry.get("ok")
                    ]
                    headers = [h for h in headers if h]
                    if headers:
                        code = _verdict_code([code, *(c for c, _ in headers)])
                        evidence += [line for _, line in headers]
                cells.setdefault(capability_id, {})[agent] = Cell(code, evidence, "transports")
            sessions = [
                _recorded_session_code(_capture_records(snapshot, entry), where(entry))
                for entry in (runs.get("http") or {}).values() if entry.get("ok")
            ]
            sessions = [s for s in sessions if s]
            if sessions:
                cells.setdefault("session-identifier", {})[agent] = Cell(
                    _combine(c for c, _ in sessions), [line for _, line in sessions], "transports"
                )
    return cells


def _merge_instruments(into: dict[str, dict[str, Cell]], other: dict[str, dict[str, Cell]]) -> None:
    """Fold a second instrument's cells into the first's. Where both measured the same
    cell the codes combine and the cell names both instruments, joined by ``+``."""
    for capability_id, row in other.items():
        target = into.setdefault(capability_id, {})
        for agent, cell in row.items():
            have = target.get(agent)
            if have is None or have.code == "x":
                target[agent] = cell
                continue
            target[agent] = _both_instruments(have, cell)

# ---------------------------------------------------------------------------
# Overlay + derived cells -> the published capability shape (capability@3)
# ---------------------------------------------------------------------------

OVERLAY_SCHEMA = "mcp-mirror/capability@4"
PUBLISHED_SCHEMA = "mcp-mirror/capability@3"
FIXTURE = "fixtures/tricky_server.py"
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _artifact_dates(snapshot: Path) -> dict[str, str]:
    """Latest capture date per instrument, so a row carries the day it was measured."""
    dates: dict[str, str] = {}
    legacy = snapshot / "report.json"
    for report_path in ([legacy] if legacy.exists() else []) + sorted((snapshot / "measurements" / "scan").glob("*.json")):
        report = _load_json(report_path)
        stamp = ((report.get("provenance") or {}).get("generated_at") or "")[:10]
        if stamp:
            dates["scan"] = max(dates.get("scan", ""), stamp)
    for kind in ("results", "probe", "client", "utilities", "notify"):
        for path in sorted((snapshot / "measurements" / kind).glob("*.json")):
            stamp = (_load_json(path).get("captured_at") or "")[:10]
            if stamp:
                dates[kind] = max(dates.get(kind, ""), stamp)
    for capture in load_conformance_captures(snapshot):
        stamp = (capture.get("captured_at") or "")[:10]
        if stamp:
            dates["conformance"] = max(dates.get("conformance", ""), stamp)
    for run in load_protocol_runs(snapshot):
        stamp = (run.run.get("captured_at") or "")[:10]
        if stamp:
            dates["protocol"] = max(dates.get("protocol", ""), stamp)
    for _path, capture in load_transport_captures(snapshot):
        stamp = (capture.get("captured_at") or "")[:10]
        if stamp:
            dates["transports"] = max(dates.get("transports", ""), stamp)
    return dates


def _framework_dates(snapshot: Path) -> dict[str, dict[str, str]]:
    """``{published_agent_id: {instrument: date}}``: the day each instrument captured each
    framework, read from the capture that produced its cells.

    Walks the files in the same order the derive functions do, so when two captures hold
    the same framework the later file's date wins alongside its cells. A framework's
    published ``measured_on`` is the newest of these across the instruments that fed it.
    """
    dates: dict[str, dict[str, str]] = {}

    def note(framework: str, instrument: str, stamp: str | None) -> None:
        stamp = (stamp or "")[:10]
        if stamp:
            dates.setdefault(PUBLISHED_IDS.get(framework, framework), {})[instrument] = stamp

    legacy = snapshot / "report.json"
    for report_path in ([legacy] if legacy.exists() else []) + sorted((snapshot / "measurements" / "scan").glob("*.json")):
        report = _load_json(report_path)
        stamp = (report.get("provenance") or {}).get("generated_at")
        for run in report.get("runs", []):
            if run.get("status", "measured") == "measured":
                note(run["framework"], "scan", stamp)
    for kind in ("results", "probe", "client", "utilities", "notify"):
        for path in sorted((snapshot / "measurements" / kind).glob("*.json")):
            capture = _load_json(path)
            for framework, payload in (capture.get("frameworks") or {}).items():
                if payload.get("ok"):
                    note(framework, kind, capture.get("captured_at"))
    for capture in load_conformance_captures(snapshot):
        note(capture.get("renderer", ""), "conformance", capture.get("captured_at"))
    for run in load_protocol_runs(snapshot):
        note(run.agent, "protocol", run.run.get("captured_at"))
    for _path, capture in load_transport_captures(snapshot):
        for framework in capture.get("frameworks") or {}:
            note(framework, "transports", capture.get("captured_at"))
    attempts_path = snapshot / "attempts.json"
    if attempts_path.exists():
        attempts = _load_json(attempts_path)
        for framework in attempts.get("attempts", {}):
            note(framework, "attempt", attempts.get("run_date"))
    return dates


def _artifact_fixtures(snapshot: Path) -> dict[str, str]:
    """The fixture each instrument's capture says it ran against."""
    fixtures: dict[str, str] = {}
    legacy = snapshot / "report.json"
    reports = ([legacy] if legacy.exists() else []) + sorted((snapshot / "measurements" / "scan").glob("*.json"))
    for report_path in reports:
        server = _load_json(report_path).get("mcp_server") or ""
        found = re.search(r"fixtures/[\w./-]+\.py", server)
        if found:
            fixtures["scan"] = found.group(0)
    for kind in ("results", "probe", "client", "utilities", "notify"):
        for path in sorted((snapshot / "measurements" / kind).glob("*.json")):
            fixture = _load_json(path).get("fixture")
            if fixture:
                fixtures[kind] = fixture
    for run in load_protocol_runs(snapshot):
        if run.run.get("fixture"):
            fixtures["protocol"] = run.run["fixture"]
    for _path, capture in load_transport_captures(snapshot):
        if capture.get("fixtures"):
            fixtures["transports"] = ", ".join(sorted(capture["fixtures"].values()))
    return fixtures


def _relative(path_text: str, root: Path) -> str:
    try:
        return str(Path(path_text).resolve().relative_to(root.resolve()))
    except ValueError:
        return path_text


def _managed_mastra_version() -> str | None:
    # Read as plain JSON: the validator imports this module without the package's
    # dependencies, so it cannot go through the pydantic manifest loader.
    manifests = _load_json(Path(__file__).with_name("runner-manifests.json"))
    for package in manifests["runners"]["mastra"]["packages"]:
        name, _, version = package.rpartition("@")
        if name == "@mastra/mcp":
            return version
    return None


# The renderers each script runs when given no --renderers, so a capture of exactly that
# set publishes the bare command that wrote it.
_DEFAULT_RENDERERS = {"crewai", "langchain", "openai_agents", "pydantic_ai", "mastra"}
_DEFAULT_TRANSPORT_RENDERERS = {"crewai", "langchain", "openai_agents", "pydantic_ai", "google_adk", "mastra", "strands"}


def _capture_command(
    script: str, path: Path, root: Path, kind: str, defaults: set[str] = _DEFAULT_RENDERERS
) -> str:
    """The command that rewrites one capture file."""
    capture = _load_json(path)
    command = f"uv run python scripts/{script}"
    revision = capture.get("spec_version")
    if revision is None and ".venv" not in (capture.get("fixture_python") or ""):
        # Captures written before the flag existed: the interpreter path names it.
        revision = Path(capture.get("fixture_python") or "").parent.parent.name.removeprefix("fixture-") or None
    if revision:
        command += f" --spec-version {revision}"
    frameworks = sorted(capture.get("frameworks") or {})
    if set(frameworks) != defaults:
        command += f" --renderers {','.join(frameworks)}"
    if path != root / "data" / "measurements" / kind / "all-frameworks.json":
        command += f" --out {_relative(str(path), root)}"
    return command


def _reproduce_commands(snapshot: Path, root: Path) -> dict[str, str]:
    """The commands that re-derive each instrument, taken from what the artifacts say they ran.

    An instrument's commands are chained with ``&&``, one per framework wherever the
    artifacts allow it, so adding or re-measuring one framework changes only its own
    part and check_reproduce_commands --changed-from reruns only that. A capture file
    that already holds several frameworks (an ``all-frameworks.json``, the scan's
    ``python-adapters.json``) keeps the one command that wrote it, because splitting it
    would change the measurement layout rather than the command that reproduces it."""
    commands: dict[str, str] = {}
    source_tools, runs = load_scan_runs(snapshot)
    scans = sorted((snapshot / "measurements" / "scan").glob("*.json"))
    legacy = snapshot / "report.json"
    reports = ([legacy] if legacy.exists() else []) + scans
    if reports:
        # The report that carries the most runs names the server and revision the managed
        # runner reproduces. The Node row joins it only when the managed manifest pins the
        # Mastra version that was measured; an older row has no runner to re-derive it.
        # A scan writes no file, so each framework is scanned on its own.
        report = max((_load_json(r) for r in reports), key=lambda r: len(r.get("runs") or []))
        revision = report.get("mcp_spec_version", "")
        managed_runs = sorted(
            f for f in runs
            if f != "mastra" or runs[f].get("framework_version") == _managed_mastra_version()
        )
        server = report.get("mcp_server") or f"python {FIXTURE}"
        commands["scan"] = " && ".join(
            f'uv run mcp-mirror scan "{server}" --runner managed '
            f"--frameworks {framework} --spec-version {revision}"
            for framework in managed_runs
        )
    for kind, script in (
        ("results", "measure_result_boundaries.py"),
        ("probe", "probe_surface.py"),
        ("client", "probe_client_features.py"),
        ("utilities", "probe_utilities.py"),
        ("notify", "probe_notifications.py"),
    ):
        captures = sorted((snapshot / "measurements" / kind).glob("*.json"))
        if not captures:
            continue
        # A framework measured on its own writes its own file, so the rows are re-derived
        # only by running every capture's command.
        commands[kind] = " && ".join(_capture_command(script, path, root, kind) for path in captures)
    conformance = load_conformance_captures(snapshot)
    if conformance:
        # Each renderer writes its own checks.json, so each is re-run on its own; a
        # capture written by the bare all-renderer command keeps its suites.
        parts = []
        for capture in sorted(conformance, key=lambda c: c.get("renderer") or ""):
            recorded = capture.get("command") or ""
            if not recorded or not capture.get("renderer"):
                continue
            suites = re.search(r" --suites \S+", recorded)
            parts.append(
                f"uv run python scripts/run_conformance.py --renderers {capture['renderer']}"
                + (suites.group(0) if suites else "")
            )
        if parts:
            commands["conformance"] = " && ".join(parts)
    protocol = load_protocol_runs(snapshot)
    if protocol:
        # Each renderer writes its own directory, so each is re-run on its own.
        parts = []
        for run in sorted(protocol, key=lambda r: r.run.get("renderer") or r.directory.name):
            command = "uv run python scripts/measure_protocol.py"
            revision = run.run.get("spec_version")
            if revision:
                command += f" --spec-version {revision}"
            parts.append(f"{command} --renderers {run.run.get('renderer') or run.directory.name}")
        commands["protocol"] = " && ".join(parts)
    transports = load_transport_captures(snapshot)
    if transports:
        commands["transports"] = " && ".join(
            _capture_command("measure_transports.py", path, root, "transports", _DEFAULT_TRANSPORT_RENDERERS)
            for path, _capture in transports
        )
    return commands


def snapshot_context(snapshot: Path, root: Path | None = None) -> dict:
    root = root or snapshot.resolve().parents[1 if snapshot.name != "data" else 0]
    while root.name and not (root / "fixtures").exists():
        root = root.parent
    _cells_by_result, dependent = derive_result_cells(snapshot)
    return {
        "cells": derive_snapshot(snapshot),
        "dependent": dependent,
        "dates": _artifact_dates(snapshot),
        "framework_dates": _framework_dates(snapshot),
        "fixtures": _artifact_fixtures(snapshot),
        "reproduce": _reproduce_commands(snapshot, root),
    }


def _display(agent_id: str, agent: dict) -> str:
    return f"{agent.get('name', agent_id)} {agent.get('current_version', '')}".strip()


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _verdict_code(codes: list[Code]) -> Code:
    known = [c for c in codes if c in ("y", "a", "n")]
    if not known:
        return "u"
    if "n" in known:
        return "n"
    if "a" in known:
        return "a"
    return "y"


def _templates(revision: str, measured: dict[str, str], blocked: list[str], agents: dict) -> tuple[str, str]:
    n, m = len(measured), len(blocked)
    reach = f"all {WORDS.get(n, n)} frameworks that reach {revision}" if n != 1 else f"the one framework that reaches {revision}"
    tail = f"; {m} cannot reach the revision." if m else "."
    blocked_names = _join([agents[a].get("name", a) for a in blocked])
    blocked_clause = f" {blocked_names} cannot negotiate MCP {revision}, so {'they reach' if m > 1 else 'it reaches'} none of it." if m else ""
    by_code: dict[str, list[str]] = {}
    for agent_id, code in measured.items():
        by_code.setdefault(code, []).append(_display(agent_id, agents[agent_id]))
    codes = set(measured.values())
    if codes == {"y"}:
        return (
            f"Exposed by {reach}{tail}",
            f"{_join(by_code['y'])} expose it at the declared capture boundary.{blocked_clause}",
        )
    if codes == {"n"}:
        return (
            f"Dropped by {reach}{tail}",
            f"{_join(by_code['n'])} drop it during adaptation.{blocked_clause}",
        )
    if codes == {"a"}:
        return (
            f"Retained elsewhere or changed by {reach}{tail}",
            f"{_join(by_code['a'])} change it or retain it outside the captured definition.{blocked_clause}",
        )
    parts = []
    if "y" in by_code:
        parts.append(f"{_join(by_code['y'])} expose it")
    if "a" in by_code:
        parts.append(f"{_join(by_code['a'])} change it or retain it elsewhere")
    if "n" in by_code:
        parts.append(f"{_join(by_code['n'])} drop it entirely")
    if "u" in by_code:
        parts.append(f"{_join(by_code['u'])} could not be judged")
    return (
        f"Mixed across the {n} frameworks that reach {revision}{tail}",
        "; ".join(parts) + "." + blocked_clause,
    )


TALLY_LABELS = (
    ("a", "changed or retained elsewhere"),
    ("n", "dropped"),
    ("u", "unmeasured"),
)


def tally(codes: list[str], revision: str) -> str:
    """The counting sentence that opens a capability's notes, derived from its cells.

    Hand-written prose cannot say how many frameworks do something, because that goes
    stale the moment a row is added. Present, changed and dropped are always stated so
    the three read as one breakdown; unmeasured and unreachable only when they occur.
    """
    if not codes:
        return ""
    count = {code: codes.count(code) for code in "yanxu"}
    noun = "framework" if len(codes) == 1 else "frameworks"
    parts = [f"{count['y']} of {len(codes)} {noun} present at the capture boundary"]
    parts += [f"{count[code]} {label}" for code, label in TALLY_LABELS if code in ("a", "n") or count[code]]
    if count["x"]:
        parts.append(f"{count['x']} cannot reach MCP {revision}")
    return ", ".join(parts) + "."


def published_notes(stats: dict, agents: dict, revision: str, summary: str | None) -> dict[str, str]:
    """The notes a capability page opens with: a tally of the public rows, then the summary.

    The summary is hand-written mechanism prose and is published on its own as well, so
    a page that shows a different set of rows (?preview=1, a framework filter) can
    recount the tally from the cells it shows and keep the prose.
    """
    # Every public row counts, an alias included, so the tally agrees with the page's
    # Observed header, which counts the rows it shows.
    codes = [next(iter(stats[a].values())).split()[0] for a in agents if not is_preview(agents[a])]
    return {
        "notes": " ".join(part for part in (tally(codes, revision), summary) if part),
        "summary": summary or "",
    }


def merge_overlay(overlay: dict, root_data: dict, context: dict) -> dict:
    """Publish one capability from its hand-written overlay and the derived cells."""
    agents: dict = root_data["agents"]
    revision = root_data["mcp_spec"]
    capability_id = overlay["id"]
    derived: dict[str, Cell] = context["cells"].get(capability_id, {})
    overlay_notes: dict = overlay.get("notes") or {}
    stats: dict = {}
    notes_by_num: dict = {}
    measured: dict[str, str] = {}
    blocked: list[str] = []
    instruments: set[str] = set()
    measured_on: dict[str, str] = {}
    framework_dates: dict[str, dict[str, str]] = context.get("framework_dates", {})
    number = 0
    counted = set(counted_agents(agents))
    for agent_id in reading_order(agents):
        agent = agents[agent_id]
        source = source_id(agent_id, agent)
        cell = derived.get(source)
        if cell is not None:
            # The day this cell's own captures were taken: the newest across the
            # instruments that fed it. An alias carries its source's date.
            cell_dates = [framework_dates.get(source, {}).get(i, "") for i in cell.instrument.split("+")]
            if any(cell_dates):
                measured_on[agent_id] = max(cell_dates)
        # An alias's own note wins; without one it carries its source's, because the
        # mechanism the note names is the same adapter's.
        own_note = overlay_notes.get(agent_id) or (overlay_notes.get(source) if source != agent_id else None)
        if cell is None:
            code, note = "u", own_note
        elif cell.code == "x":
            code, note = "x", agent.get("measurement_note") or (cell.evidence[0] if cell.evidence else "")
            if agent_id in counted:
                blocked.append(agent_id)
        else:
            code, note = cell.code, own_note
            if agent_id in counted:
                measured[agent_id] = code
            instruments.update(cell.instrument.split("+"))
        if note:
            number += 1
            notes_by_num[str(number)] = note
            stats[agent_id] = {agent["current_version"]: f"{code} #{number}"}
        else:
            stats[agent_id] = {agent["current_version"]: code}
    stats = {agent_id: stats[agent_id] for agent_id in agents}
    measured = {agent_id: measured[agent_id] for agent_id in agents if agent_id in measured}
    blocked = [agent_id for agent_id in agents if agent_id in blocked]
    headline, detail = _templates(revision, measured, blocked, agents)
    verdict_overlay = overlay.get("verdict") or {}
    dates = [context["dates"].get(i) for i in instruments if context["dates"].get(i)]
    published = {
        "schema": PUBLISHED_SCHEMA,
        "id": capability_id,
        "title": overlay["title"],
        "description": overlay["description"],
        "question": overlay.get("question", ""),
        "spec": overlay.get("spec", "https://modelcontextprotocol.io/specification"),
        "status": overlay.get("status", "stable"),
        "categories": overlay.get("categories", []),
        "keywords": overlay.get("keywords", ""),
        "stats": stats,
        **published_notes(stats, agents, revision, overlay.get("summary")),
        "notes_by_num": notes_by_num,
        "boundary_dependent": [a for a in context["dependent"].get(capability_id, []) if a in agents]
        + [
            a
            for a, agent in agents.items()
            if agent.get("aliases_of") in context["dependent"].get(capability_id, [])
        ],
        "verdict": {
            "headline": verdict_overlay.get("headline") or headline,
            "detail": verdict_overlay.get("detail") or detail,
        },
        "usage_perc_y": overlay.get("usage_perc_y"),
        "usage_perc_a": overlay.get("usage_perc_a"),
        "usage_note": overlay.get("usage_note"),
        "links": overlay.get("links", []),
        "known_issues": overlay.get("known_issues", []),
        "why": overlay.get("why", ""),
        "reproduce": " && ".join(context["reproduce"][i] for i in sorted(instruments) if i in context["reproduce"]),
        "measured": {
            "run_date": max(dates) if dates else None,
            # Per framework, the day its cell in this row was captured, in agent order.
            "measured_on": {agent_id: measured_on[agent_id] for agent_id in agents if agent_id in measured_on},
            "mcp_spec": revision,
            "fixture": ", ".join(
                sorted({f for i in instruments for f in context.get("fixtures", {}).get(i, FIXTURE).split(", ")})
            )
            or FIXTURE,
        },
        "instruments": sorted(instruments),
        "shown": overlay.get("shown", True),
        "spec_label": f"MCP {revision}",
        "docs_url": overlay.get("docs_url", ""),
        "docs_label": overlay.get("docs_label", "MCP Docs"),
    }
    if overlay.get("example"):
        published["example"] = overlay["example"]
    return published


# ---------------------------------------------------------------------------
# Wire instrument: harness sessions under measurements/wire/<harness>/ written by
# scripts/measure_harness.py. Each session.json names the recorder captures (one JSONL
# per mounted fixture, see recorder.py for the shape) and a reduced copy of the
# harness's own machine-readable output. Harness cells are functions of those files;
# plan/harness_surface.yaml says what each one observes.
# ---------------------------------------------------------------------------

HARNESS_SESSION = "session.json"
# One per session directory, written by the runner: the four freshness checks (isolated
# home, servers listed, tools offered, egress). A run in which any session has none, or
# failed one, publishes no cell.
HARNESS_FRESH = "fresh.json"


def padded_captures(session_dir: Path) -> list[Path]:
    """Capture files in ``session_dir`` that hold a NUL byte. A capture is JSON text,
    so a NUL can only come from a writer appending at a stale offset after another
    process rewrote the file shorter; such a file is never published."""

    return sorted(path for path in session_dir.glob("*.jsonl") if b"\x00" in path.read_bytes())
FRESH_CHECKS = ("home", "servers_listed", "tools_offered", "egress")
LEVEL_2_SKIPPED = "level 2 needs model credentials (.env.harness)"
# Cells whose answer is a value (a revision, a list, a count) rather than a y/a/n. The
# code says whether the value was observed; the value itself is published beside it.
HARNESS_VALUE_CELLS = frozenset({
    "h-negotiated-revision", "h-declared-capabilities", "h-tool-count-limit", "h-error-surfaced",
    "h-tools-offered-to-model",
})
SERVERS = ("def", "res", "ntf", "cf", "util")
HINT_KEYS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")


def offered_name(names: Iterable[str], server: str, tool: str) -> str | None:
    """The name a harness offered its model for ``tool`` on ``server``, if it offered one.

    Harnesses prefix the server name their own way (mcp__def__search_records,
    mcp--def--search_records, def_search_records, def.search_records); an exact bare name
    also counts, for a harness that does not prefix at all.
    """
    names = list(names)
    for separator in ("__", "--", "_", "-", ".", "/", ":"):
        for name in names:
            qualified = f"{server}{separator}{tool}"
            if name == qualified or name.endswith(f"_{qualified}") or name.endswith(f"-{qualified}"):
                return name
    return tool if tool in names else None


@dataclass
class WireCell(Cell):
    value: str | None = None

    def as_dict(self) -> dict:
        return {**super().as_dict(), "value": self.value}


class HarnessRun:
    """One harness's session.json with its captures loaded on demand."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.root = path.parent
        self.manifest: dict = _load_json(path)
        self._cache: dict[str, list[dict]] = {}
        self._freshness: tuple[bool, str] | None = None

    @property
    def harness(self) -> str:
        return self.manifest["harness"]

    @property
    def live(self) -> dict | None:
        """The manifest's live block (env file, forwarded hosts, models) for a run made
        against the harness's real service; None for a stubbed run."""
        return self.manifest.get("live") or None

    def records(self, relative: str | None) -> list[dict]:
        if not relative:
            return []
        if relative not in self._cache:
            target = self.root / relative
            self._cache[relative] = (
                [json.loads(line) for line in target.read_text().splitlines() if line.strip()]
                if target.exists()
                else []
            )
        return self._cache[relative]

    def sessions(self, level: str, session_id: str | None = None) -> list[dict]:
        block = self.manifest.get(level) or {}
        return [s for s in block.get("sessions") or [] if session_id is None or s["id"] == session_id]

    def level_status(self, level: str) -> tuple[str, str]:
        block = self.manifest.get(level) or {}
        return block.get("status", "missing"), block.get("reason", "")

    def blocked(self) -> str | None:
        """Why no session ran at any level, or None once one did.

        Such a run measured nothing: its captured_at is the day the skip was recorded,
        not a measurement date, so the published row carries no date for it.
        """
        if self.sessions("level_1") or self.sessions("level_2"):
            return None
        for level in ("level_1", "level_2"):
            _, reason = self.level_status(level)
            if reason:
                return reason
        return "no session ran at any level"

    def freshness(self) -> tuple[bool, str]:
        """(every session ran fresh, the evidence line that says so or names the failure).

        Read from each session's fresh.json, not the manifest's summary, so a hand edit of
        session.json cannot publish a run that failed.
        """
        if self._freshness is None:
            self._freshness = self._read_freshness()
        return self._freshness

    def _read_freshness(self) -> tuple[bool, str]:
        sessions = self.sessions("level_1") + self.sessions("level_2")
        for session in sessions:
            where = session.get("fresh") or f"{session.get('output', session['id']).rsplit('/', 1)[0]}/{HARNESS_FRESH}"
            record = _load_json(self.root / where) if (self.root / where).is_file() else None
            if record is None:
                return False, f"{where}: missing, so this run is not known to be fresh and publishes no cell"
            checks = record.get("checks") or {}
            # The four named checks must be there and pass; any further check the runner
            # wrote (a live session's sandbox, the process group) must pass too.
            failed = [name for name in FRESH_CHECKS if not (checks.get(name) or {}).get("passed")]
            failed += [name for name, check in checks.items()
                       if name not in FRESH_CHECKS and not (check or {}).get("passed")]
            if failed or not record.get("passed"):
                return False, f"{where}: failed {', '.join(failed) or 'freshness'}, so this run publishes no cell"
        return True, (f"{HARNESS_FRESH}: all four freshness checks ({', '.join(FRESH_CHECKS)}) passed "
                      f"in {len(sessions)}/{len(sessions)} sessions")

    def revisions(self, level: str = "level_1") -> list[str]:
        """Fixture revisions level ran against, baseline first. [] for a one-line run."""
        return list((self.manifest.get(level) or {}).get("revisions") or [])

    def of_revision(self, sessions: list[dict], revision: str | None) -> list[dict]:
        """Sessions against ``revision``; None is the baseline, and so is a session with no revision."""
        revisions = self.revisions()
        wanted = revision or (revisions[0] if revisions else None)
        baseline = revisions[0] if revisions else None
        return [s for s in sessions if s.get("revision", baseline) == wanted]


def _wire_messages(records: list[dict], method: str | None = None, direction: str | None = None,
                   kind: str | None = None) -> list[dict]:
    return [
        r for r in records
        if r.get("type") == "message"
        and (method is None or r.get("method") == method)
        and (direction is None or r.get("direction") == direction)
        and (kind is None or r.get("kind") == kind)
    ]


def _c2s(records: list[dict], method: str | None = None, kind: str = "request") -> list[dict]:
    return _wire_messages(records, method, "client_to_server", kind)


def _s2c(records: list[dict], method: str | None = None, kind: str | None = None) -> list[dict]:
    return _wire_messages(records, method, "server_to_client", kind)


def _where(session: dict, server: str, record: dict | None = None) -> str:
    path = (session.get("captures") or {}).get(server, f"<no {server} capture>")
    if record is None:
        return path
    return f"{path} seq {record['seq']} {record.get('method') or record.get('via') or record.get('type')}"


def _answer_to(records: list[dict], request: dict) -> dict | None:
    """The response or error, in the opposite direction, that carries the request's id."""
    opposite = "server_to_client" if request["direction"] == "client_to_server" else "client_to_server"
    return next(
        (
            r for r in records
            if r.get("type") == "message" and r.get("direction") == opposite
            and r.get("kind") in ("response", "error") and r.get("id") == request.get("id")
            and r["seq"] > request["seq"]
        ),
        None,
    )


def _tool_calls(records: list[dict], name: str) -> list[dict]:
    return [r for r in _c2s(records, "tools/call") if ((r["message"].get("params") or {}).get("name")) == name]


def _output_events(run: HarnessRun, session: dict, event: str | None = None) -> list[dict]:
    return [e for e in run.records(session.get("output")) if event is None or e.get("event") == event]


def _permission_denied(run: HarnessRun, session: dict, tool: str) -> list[dict]:
    return [e for e in _output_events(run, session, "permission_denied") if tool in (e.get("tool") or "")]


def _by_fraction(hits: int, total: int) -> Code:
    if total == 0:
        return "u"
    return "y" if hits == total else ("n" if hits == 0 else "a")


# -- level 1 ------------------------------------------------------------------


LEVEL_1_UNATTENDED_SUFFIX = "-unattended"


def _level_1_unattended(run: HarnessRun) -> str | None:
    """Why level 1 ran a second, unattended pass (session.json level_1 ``unattended``, from
    the module's LEVEL1_UNATTENDED), or None when the plan's sessions are the only ones."""
    return (run.manifest.get("level_1") or {}).get("unattended") or None


def _level_1(run: HarnessRun, session_id: str = "stdio", revision: str | None = None) -> dict | None:
    """The level-1 session a cell reads. With an unattended pass (``_level_1_unattended``),
    the ``<id>-unattended`` session, since the default one started no MCP server, so the
    wire the cell needs is only there; the default session is reported beside it by
    ``level_1_default_line``."""
    if _level_1_unattended(run):
        sessions = run.of_revision(run.sessions("level_1", session_id + LEVEL_1_UNATTENDED_SUFFIX), revision)
        if sessions:
            return sessions[0]
    sessions = run.of_revision(run.sessions("level_1", session_id), revision)
    return sessions[0] if sessions else None


def level_1_default_line(run: HarnessRun, session_id: str = "stdio", revision: str | None = None) -> str | None:
    """One evidence line on the default level-1 session when an unattended pass exists:
    which switch the cell's session ran with, and how many fixtures the default session
    reached, so a reader sees both the documented default and the mode the cell reads."""
    reason = _level_1_unattended(run)
    if not reason:
        return None
    unattended = run.of_revision(run.sessions("level_1", session_id + LEVEL_1_UNATTENDED_SUFFIX), revision)
    default = run.of_revision(run.sessions("level_1", session_id), revision)
    if not unattended or not default:
        return None
    reached = sum(1 for server in SERVERS
                  if any(r.get("method") in SESSION_OPENERS
                         for r in _c2s(run.records((default[0].get("captures") or {}).get(server)))))
    switch = " ".join(unattended[0].get("mode_args") or [])
    return (f"level 1 read from {unattended[0].get('id')}, run with {switch or 'the unattended switch'} ({reason}); "
            f"the default session {default[0].get('id')}, run with no switch, opened {reached} of {len(SERVERS)} fixtures")


def _level_1_missing(run: HarnessRun, session_id: str = "stdio", revision: str | None = None) -> WireCell:
    status, reason = run.level_status("level_1")
    against = f" against the {revision} fixture" if revision else ""
    return WireCell("u", [reason or f"no level-1 {session_id} session{against} in {run.path.name} ({status})"], "wire")


def _negotiations(run: HarnessRun, session: dict, server: str) -> list[dict]:
    return [r for r in run.records((session.get("captures") or {}).get(server)) if r.get("type") == "negotiation"]


def _h_negotiated_revision(run: HarnessRun, session_id: str, revision: str | None = None) -> WireCell:
    session = _level_1(run, session_id, revision)
    if session is None:
        return _level_1_missing(run, session_id, revision)
    evidence: list[str] = []
    negotiated: list[str] = []
    requested: list[str] = []
    for server in SERVERS:
        for record in _negotiations(run, session, server):
            asked, got = record.get("requested_revision"), record.get("negotiated_revision")
            outcome = f"negotiated {got}" if got else ("error" if record.get("error") else "client picks next")
            evidence.append(f"{_where(session, server, record)} via {record.get('via')}: requested {asked}, {outcome}")
            if server == "def":
                requested += [asked] if asked and asked not in requested else []
                negotiated += [got] if got and got not in negotiated else []
    if not negotiated:
        return WireCell("n", evidence or [f"{_where(session, 'def')}: no revision was negotiated"], "wire")
    # The fixture's SDK line bounds what can be negotiated, so the request order is the
    # part that says what the harness itself prefers.
    return WireCell("y", evidence, "wire", f"{', '.join(negotiated)} (requested {', then '.join(requested)})")


def _h_declared_capabilities(run: HarnessRun, session_id: str, revision: str | None = None) -> WireCell:
    session = _level_1(run, session_id, revision)
    if session is None:
        return _level_1_missing(run, session_id, revision)
    declared: set[str] = set()
    evidence: list[str] = []
    for record in _negotiations(run, session, "def"):
        capabilities = record.get("client_capabilities")
        if capabilities is None:
            continue
        declared.update(capabilities)
        evidence.append(f"{_where(session, 'def', record)} via {record.get('via')}: {sorted(capabilities) or 'none'}")
    if not evidence:
        return WireCell("u", [f"{_where(session, 'def')}: no negotiation record"], "wire")
    return WireCell("y", evidence, "wire", ", ".join(sorted(declared)) or "none")


def _h_lists_tools_at_startup(run: HarnessRun, revision: str | None = None) -> WireCell:
    session = _level_1(run, revision=revision)
    if session is None:
        return _level_1_missing(run, revision=revision)
    listed, evidence = 0, []
    for server in SERVERS:
        records = run.records((session.get("captures") or {}).get(server))
        first_call = next((r["seq"] for r in _c2s(records, "tools/call")), None)
        lists = [r for r in _c2s(records, "tools/list") if first_call is None or r["seq"] < first_call]
        if lists:
            listed += 1
            evidence.append(_where(session, server, lists[0]))
        else:
            evidence.append(f"{_where(session, server)}: no tools/list before the session ended")
    return WireCell(_by_fraction(listed, len(SERVERS)), evidence, "wire")


def _h_lists_prompts_at_startup(run: HarnessRun, revision: str | None = None) -> WireCell:
    session = _level_1(run, revision=revision)
    if session is None:
        return _level_1_missing(run, revision=revision)
    evidence = [
        _where(session, server, r)
        for server in SERVERS
        for r in _c2s(run.records((session.get("captures") or {}).get(server)), "prompts/list")[:1]
    ]
    if evidence:
        return WireCell("y", evidence, "wire")
    return WireCell("n", [f"no prompts/list on any of {', '.join(SERVERS)} in {session.get('id')} session"], "wire")


def _h_lists_resources_at_startup(run: HarnessRun, revision: str | None = None) -> WireCell:
    session = _level_1(run, revision=revision)
    if session is None:
        return _level_1_missing(run, revision=revision)
    seen: dict[str, str] = {}
    for method in ("resources/list", "resources/templates/list"):
        for server in SERVERS:
            hit = _c2s(run.records((session.get("captures") or {}).get(server)), method)
            if hit:
                seen[method] = _where(session, server, hit[0])
                break
    evidence = [seen.get(m, f"no {m} on any server") for m in ("resources/list", "resources/templates/list")]
    return WireCell({2: "y", 1: "a", 0: "n"}[len(seen)], evidence, "wire")


def _util_pages(run: HarnessRun, session: dict) -> tuple[int, bool, list[str]]:
    """Pages of util's tools/list the harness fetched, and whether it reached the end."""
    records = run.records((session.get("captures") or {}).get("util"))
    pages, complete, evidence = 0, False, []
    for request in _c2s(records, "tools/list"):
        answer = _answer_to(records, request)
        cursor = (request["message"].get("params") or {}).get("cursor")
        if answer is None or answer.get("kind") != "response":
            continue
        pages += 1
        following = (answer["message"].get("result") or {}).get("nextCursor")
        evidence.append(f"{_where(session, 'util', request)} cursor={cursor!r} -> nextCursor={following!r}")
        if not following:
            complete = True
            break
    return pages, complete, evidence


def _h_follows_pagination(run: HarnessRun, revision: str | None = None) -> WireCell:
    session = _level_1(run, revision=revision)
    if session is None:
        return _level_1_missing(run, revision=revision)
    pages, complete, evidence = _util_pages(run, session)
    if pages == 0:
        return WireCell("n", [f"{_where(session, 'util')}: no tools/list answered"], "wire")
    return WireCell("y" if complete else ("a" if pages > 1 else "n"), evidence, "wire")


def _h_tool_count_limit(run: HarnessRun, revision: str | None = None) -> WireCell:
    session = _level_1(run, revision=revision)
    if session is None:
        return _level_1_missing(run, revision=revision)
    pages, complete, evidence = _util_pages(run, session)
    if pages == 0:
        return WireCell("u", [f"{_where(session, 'util')}: no tools/list answered"], "wire")
    value = f"{pages} page{'s' if pages != 1 else ''}" + (", all of them" if complete else ", stopped early")
    return WireCell("y", evidence, "wire", value)


def _list_changed_reaction(run: HarnessRun, sessions: list[dict], window: float | None) -> tuple[int, int, list[str]]:
    fired = reacted = 0
    evidence: list[str] = []
    for session in sessions:
        records = run.records((session.get("captures") or {}).get("ntf"))
        for notice in _s2c(records, "notifications/tools/list_changed", "notification"):
            fired += 1
            relist = next(
                (
                    r for r in _c2s(records, "tools/list")
                    if r["seq"] > notice["seq"] and (window is None or r["elapsed"] - notice["elapsed"] <= window)
                ),
                None,
            )
            if relist is not None:
                reacted += 1
                evidence.append(f"{_where(session, 'ntf', notice)} -> {_where(session, 'ntf', relist)}")
            else:
                evidence.append(f"{_where(session, 'ntf', notice)}: no tools/list after it")
    return fired, reacted, evidence


def _h_reacts_to_list_changed(run: HarnessRun, revision: str | None = None) -> WireCell:
    sessions = run.of_revision(run.sessions("level_1") + run.sessions("level_2"), revision)
    fired, reacted, evidence = _list_changed_reaction(run, sessions, 30.0)
    if not fired:
        return WireCell("u", ["ntf never fired notifications/tools/list_changed; only a tool call makes it fire"], "wire")
    return WireCell(_by_fraction(reacted, fired), evidence, "wire")


def _h_answers_ping(run: HarnessRun, session_id: str, revision: str | None = None) -> WireCell:
    session = _level_1(run, session_id, revision)
    if session is None:
        return _level_1_missing(run, session_id, revision)
    answered = sent = 0
    evidence: list[str] = []
    for server in SERVERS:
        records = run.records((session.get("captures") or {}).get(server))
        for ping in [r for r in _s2c(records, "ping", "request") if r.get("origin") == "recorder"]:
            sent += 1
            answer = _answer_to(records, ping)
            if answer is not None and answer["kind"] == "response" and answer["elapsed"] - ping["elapsed"] <= 5:
                answered += 1
                evidence.append(f"{_where(session, server, answer)} after {answer['elapsed'] - ping['elapsed']:.3f}s")
            else:
                evidence.append(f"{_where(session, server, ping)}: no answer within 5s")
    if not sent:
        if _stateless(run, session):
            why = "the session negotiated 2026-07-28, which has no notifications/initialized, so the recorder injected no ping"
        else:
            why = "the recorder sent no ping (handshake never completed)"
        return WireCell("u", [f"{session.get('id')}: {why}"], "wire")
    return WireCell(_by_fraction(answered, sent), evidence, "wire")


def _stateless(run: HarnessRun, session: dict) -> bool:
    """Every server settled on 2026-07-28 or later: no session, so no Mcp-Session-Id either."""
    settled = []
    for server in SERVERS:
        revisions = [r.get("negotiated_revision") for r in _negotiations(run, session, server) if r.get("negotiated_revision")]
        settled.append(bool(revisions) and revisions[-1] >= "2026-07-28")
    return all(settled)


def _started(records: list[dict]) -> bool:
    """The handshake completed and the harness asked for the tool list afterwards."""
    opened = any(r.get("type") == "negotiation" and (r.get("negotiated_revision") or r.get("via") == "request-meta")
                 for r in records)
    return opened and bool(_c2s(records, "tools/list"))


def _h_transport(run: HarnessRun, session_id: str, revision: str | None = None) -> WireCell:
    session = _level_1(run, session_id, revision)
    if session is None:
        return _level_1_missing(run, session_id, revision)
    started, evidence, session_ids = 0, [], 0
    for server in SERVERS:
        records = run.records((session.get("captures") or {}).get(server))
        if _started(records):
            started += 1
            evidence.append(_where(session, server, _c2s(records, "tools/list")[0]))
        else:
            evidence.append(f"{_where(session, server)}: startup did not reach tools/list")
        if session_id == "http":
            opened = next((r for r in _c2s(records) if r.get("method") in ("initialize", "server/discover")), None)
            later = [r for r in _c2s(records) + _c2s(records, kind="notification")
                     if opened is not None and r["seq"] > opened["seq"] and r.get("method") not in ("initialize", "server/discover")]
            if later and all("mcp-session-id" in ((r.get("http") or {}).get("headers") or {}) for r in later):
                session_ids += 1
            elif later:
                evidence.append(f"{_where(session, server)}: a request after the handshake carried no Mcp-Session-Id")
    code = _by_fraction(started, len(SERVERS))
    if session_id == "http" and _stateless(run, session):
        # 2026-07-28 has no sessions, so there is no Mcp-Session-Id to echo.
        evidence = [e for e in evidence if "Mcp-Session-Id" not in e]
        evidence.append("negotiated 2026-07-28 on every server: stateless, so Mcp-Session-Id is not expected")
    elif session_id == "http" and code == "y" and session_ids < len(SERVERS):
        code = "a"
    return WireCell(code, evidence, "wire")


def _h_startup_timeout(run: HarnessRun, revision: str | None = None) -> WireCell:
    session = _level_1(run, "slow-start", revision)
    if session is None:
        return _level_1_missing(run, "slow-start", revision)
    records = run.records((session.get("captures") or {}).get("util"))
    opener = next((r for r in _c2s(records) if r.get("method") in ("initialize", "server/discover")), None)
    answer = _answer_to(records, opener) if opener else None
    wait = f"util answered its first request after {answer['elapsed'] - opener['elapsed']:.1f}s" if answer else "util never answered"
    listed = _c2s(records, "tools/list")
    if listed:
        return WireCell("y", [f"{_where(session, 'util', listed[0])}; {wait}"], "wire")
    return WireCell("n", [f"{_where(session, 'util')}: dropped, no tools/list; {wait}"], "wire")


def _listed_definitions(run: HarnessRun, session: dict) -> dict[tuple[str, str], dict]:
    """Every tool each server answered in tools/list, all pages, keyed by (server, name)."""
    listed: dict[tuple[str, str], dict] = {}
    for server in SERVERS:
        records = run.records((session.get("captures") or {}).get(server))
        for answer in _s2c(records, "tools/list", "response"):
            for tool in (answer["message"].get("result") or {}).get("tools") or []:
                listed.setdefault((server, tool.get("name")), tool)
    return listed


def _description_change(listed: str | None, offered: str | None) -> str | None:
    listed, offered = listed or "", offered or ""
    if listed == offered:
        return None
    if listed.startswith(offered):
        return f"truncated to {len(offered)} of {len(listed)} characters"
    if listed in offered:
        return f"extended: {offered.replace(listed, '<server description>')[:200]!r}"
    at = next((i for i, (a, b) in enumerate(zip(listed, offered)) if a != b), min(len(listed), len(offered)))
    return (f"changed from character {at} ({len(listed)} -> {len(offered)} characters): "
            f"{listed[at:at + 12]!r} became {offered[at:at + 12]!r}")


def _h_tools_offered_to_model(run: HarnessRun, revision: str | None = None) -> WireCell:
    """What the harness sent its model for the fixtures' tools, against their tools/list."""
    session = _level_1(run, revision=revision)
    if session is None:
        return _level_1_missing(run, revision=revision)
    if not session.get("model"):
        return WireCell("u", [f"{session.get('id')}: no model stub, so what the model was shown is not visible"], "wire")
    requests = [r for r in run.records(session["model"]) if r.get("request")]
    first = next((r for r in requests if isinstance(r.get("fixture_tools"), list) and r.get("offered")), None)
    if first is None:
        return WireCell("n", [f"{session['model']}: {len(requests)} model request(s), none offered a tool"], "wire")
    where = f"{session['model']} request {first['request']} ({first.get('api')})"
    listed = _listed_definitions(run, session)
    offered = {(t["server"], t["fixture_tool"]): t for t in first["fixture_tools"]}
    evidence = [f"{where}: {len(first['offered'])} tools offered, {len(offered)} of them fixture tools; "
                f"{len(listed)} listed by the fixtures"]
    same_description = same_schema = 0
    annotated, as_field = [], []
    for key, tool in sorted(listed.items()):
        name = f"{key[0]}/{key[1]}"
        hints = tool.get("annotations") or {}
        if hints:
            annotated.append(name)
        sent = offered.get(key)
        if sent is None:
            evidence.append(f"{name}: not offered")
            continue
        change = _description_change(tool.get("description"), sent.get("description"))
        if change is None:
            same_description += 1
        else:
            evidence.append(f"{name}: description {change}")
        if sent.get("input_schema") == tool.get("inputSchema"):
            same_schema += 1
        else:
            evidence.append(f"{name}: input schema differs ({json.dumps(sent.get('input_schema'))[:200]})")
        extra = json.dumps(sent.get("extra") or {})
        if "annotations" in extra or any(k in extra for k in HINT_KEYS):
            as_field.append(name)
    as_text = first.get("hint_keys_in_request") or []
    if as_field:
        annotations = f"annotations sent as fields on {len(as_field)} of {len(annotated)} annotated tools"
    elif as_text:
        annotations = f"annotation hint names appear in the request text ({', '.join(as_text)})"
    else:
        annotations = "annotations not sent"
    evidence.append(
        f"annotated in tools/list: {', '.join(annotated) or 'none'}; "
        + (f"reached the model as fields on {', '.join(as_field)}" if as_field
           else "no annotation field on any offered tool, and no hint name anywhere in the request"
           if not as_text else f"hint names in the request text: {', '.join(as_text)}")
    )
    count = len(offered)
    value = (f"{count}/{len(listed)} tools offered; descriptions {same_description}/{count} unchanged; "
             f"schemas {same_schema}/{count} unchanged; {annotations}")
    if count == 0:
        code: Code = "n"
    elif count == len(listed) and same_description == count and same_schema == count:
        code = "y"
    else:
        code = "a"
    return WireCell(code, evidence, "wire", value)


# -- level 2 ------------------------------------------------------------------
#
# Level 2 runs once per fixture line too, so every level-2 rule takes a revision like
# the level-1 ones. A run written before that (no level_2.revisions) has one line, and
# its level-2 cells have no per-revision form.


def _level_2_runs(run: HarnessRun, session_id: str, revision: str | None = None) -> tuple[list[dict], WireCell | None]:
    status, reason = run.level_status("level_2")
    sessions = run.of_revision(run.sessions("level_2", session_id), revision)
    if status != "ran":
        return [], WireCell("u", [reason or LEVEL_2_SKIPPED], "wire")
    if not sessions:
        against = f" against the {revision} fixture" if revision else ""
        return [], WireCell("u", [f"no level-2 {session_id} session{against} ran"], "wire")
    return sessions, None


def _run_context(run: HarnessRun, sessions: list[dict]) -> str:
    """How the runs were made: the model, the approval mode and its switch, the invocation."""
    first = sessions[0]
    model = (run.manifest.get("level_2") or {}).get("model") or "model"
    parts = [f"{len(sessions)} runs, {model}"]
    if first.get("mode"):
        switch = " ".join(first.get("mode_args") or [])
        parts.append(f"{first['mode']} mode" + (f" ({switch})" if switch else ""))
    if first.get("invocation"):
        parts.append(f"invoked with {first['invocation']} {first.get('prompt', '')!r}")
    return "; ".join(parts)


def _count_cell(run: HarnessRun, session_id: str, verdict: Callable[[dict], tuple[Code, str]],
                revision: str | None = None) -> WireCell:
    """One verdict per run, combined the way calls are: unanimous stands, a mix is a."""
    sessions, missing = _level_2_runs(run, session_id, revision)
    if missing:
        return missing
    codes: list[Code] = []
    evidence: list[str] = []
    for session in sessions:
        code, why = verdict(session)
        codes.append(code)
        evidence.append(f"run {session.get('run')}: {why}")
    code = _combine(codes)
    if code != "u":
        count = f"{codes.count('y')}/{len(sessions)} runs"
        if len({c for c in codes if c != "u"}) > 1:
            # Several runs of one session, and they did not all say the same: on a live
            # service that is the variance the row warns about, and it is said here.
            count += " (runs disagreed)"
        evidence.insert(0, count)
    evidence.append(_run_context(run, sessions))
    return WireCell(code, evidence, "wire")


def _called(run: HarnessRun, server: str, tool: str) -> Callable[[dict], tuple[Code, str]]:
    def verdict(session: dict) -> tuple[Code, str]:
        calls = _tool_calls(run.records((session.get("captures") or {}).get(server)), tool)
        if calls:
            return "y", _where(session, server, calls[0])
        if _permission_denied(run, session, tool):
            return "n", f"{_where(session, server)}: no tools/call {tool}; held, {session.get('output')} carries a permission event"
        return "n", f"{_where(session, server)}: no tools/call {tool}"
    return verdict


def _h_calls_read_only_tool(run: HarnessRun, revision: str | None = None) -> WireCell:
    # Unattended, so the cell asks whether the call gets through at all; whether it gets
    # through without a prompt is h-read-only-runs-without-prompt, in default mode.
    return _count_cell(run, "read_only-unattended", _called(run, "def", "search_records"), revision)


def _h_destructive_blocked(run: HarnessRun, revision: str | None = None) -> WireCell:
    def verdict(session: dict) -> tuple[Code, str]:
        calls = _tool_calls(run.records((session.get("captures") or {}).get("def")), "delete_account")
        if calls:
            return "n", f"reached the server: {_where(session, 'def', calls[0])}"
        if _permission_denied(run, session, "delete_account"):
            return "y", f"held: {session.get('output')} carries a permission event, no tools/call reached def"
        return "u", "no call and no permission event: held and never tried look the same"
    return _count_cell(run, "destructive", verdict, revision)


def _h_read_only_without_prompt(run: HarnessRun, revision: str | None = None) -> WireCell:
    def verdict(session: dict) -> tuple[Code, str]:
        calls = _tool_calls(run.records((session.get("captures") or {}).get("def")), "search_records")
        if _permission_denied(run, session, "search_records"):
            return "n", f"{session.get('output')}: permission event for search_records"
        if calls:
            return "y", _where(session, "def", calls[0])
        return "u", "the tool was never called"
    return _count_cell(run, "read_only", verdict, revision)


def _h_destructive_runs_unattended(run: HarnessRun, revision: str | None = None) -> WireCell:
    return _count_cell(run, "destructive-unattended", _called(run, "def", "delete_account"), revision)


def _h_reads_resource(run: HarnessRun, revision: str | None = None) -> WireCell:
    def verdict(session: dict) -> tuple[Code, str]:
        reads = [r for r in _c2s(run.records((session.get("captures") or {}).get("def")), "resources/read")
                 if (r["message"].get("params") or {}).get("uri") == "file:///fixtures/notes.txt"]
        return ("y", _where(session, "def", reads[0])) if reads else ("n", f"{_where(session, 'def')}: no resources/read of notes.txt")
    return _count_cell(run, "resource", verdict, revision)


def _h_expands_prompt(run: HarnessRun, revision: str | None = None) -> WireCell:
    def verdict(session: dict) -> tuple[Code, str]:
        gets = [r for r in _c2s(run.records((session.get("captures") or {}).get("def")), "prompts/get")
                if (r["message"].get("params") or {}).get("name") == "summarize_records"]
        right = [r for r in gets if ((r["message"]["params"].get("arguments") or {}).get("status")) == "active"]
        if right:
            return "y", _where(session, "def", right[0])
        if gets:
            # Issued, so the prompt is expanded, but with arguments a server cannot use:
            # partial. n is kept for never issuing prompts/get at all.
            sent = json.dumps(gets[0]["message"]["params"].get("arguments") or {}, sort_keys=True)
            return "a", f"{_where(session, 'def', gets[0])} sent arguments {sent}, not status 'active'"
        return "n", f"{_where(session, 'def')}: no prompts/get summarize_records"
    return _count_cell(run, "prompt", verdict, revision)


MRTR_REVISION = "2026-07-28"


def _negotiated_modern(records: list[dict]) -> bool:
    """Whether the server settled on 2026-07-28 or later with this client."""
    return any((r.get("negotiated_revision") or "") >= MRTR_REVISION for r in records if r.get("type") == "negotiation")


def _input_requested(records: list[dict], tool: str, method: str) -> tuple[Code, str]:
    """MRTR: the tools/call result carries inputRequests; the answer is a retried tools/call
    whose params.inputResponses holds one. Nothing is sent server-to-client."""
    calls = _tool_calls(records, tool)
    asked = None
    for call in calls:
        answer = _answer_to(records, call)
        result = (answer or {}).get("message", {}).get("result") or {}
        if result.get("resultType") == "input_required" and any(
            (request or {}).get("method") == method for request in (result.get("inputRequests") or {}).values()
        ):
            asked = (call, answer)
            break
    if asked is None:
        return "u", f"no {tool} result carried an input request for {method}"
    retries = [c for c in calls if c["seq"] > asked[0]["seq"] and (c["message"].get("params") or {}).get("inputResponses")]
    if not retries:
        return "n", f"input request {method} (MRTR) in {asked[1]['seq']} never answered: no tools/call retried {tool} with inputResponses"
    responses = retries[0]["message"]["params"]["inputResponses"]
    if any(isinstance(v, dict) and ("error" in v or "code" in v) for v in responses.values()):
        return "n", f"retried with an error in inputResponses (MRTR) seq {retries[0]['seq']}"
    return "y", f"answered by MRTR: tools/call {tool} with inputResponses seq {retries[0]['seq']}"


def _answered(run: HarnessRun, method: str, tool: str) -> Callable[[dict], tuple[Code, str]]:
    def verdict(session: dict) -> tuple[Code, str]:
        records = run.records((session.get("captures") or {}).get("cf"))
        if _negotiated_modern(records):
            code, why = _input_requested(records, tool, method)
            return code, f"{_where(session, 'cf')}: {why}"
        asks = _s2c(records, method, "request")
        if not asks:
            return "u", f"{_where(session, 'cf')}: the tool was never called, so nothing was asked"
        answer = _answer_to(records, asks[0])
        if answer is not None and answer["kind"] == "response":
            return "y", _where(session, "cf", answer)
        return "n", f"{_where(session, 'cf', asks[0])}: answered with an error or not at all"
    return verdict


def _h_answers_elicitation(run: HarnessRun, revision: str | None = None) -> WireCell:
    return _count_cell(run, "elicit", _answered(run, "elicitation/create", "ask_elicitation_form"), revision)


def _h_answers_sampling(run: HarnessRun, revision: str | None = None) -> WireCell:
    return _count_cell(run, "sampling", _answered(run, "sampling/createMessage", "ask_sampling"), revision)


def _h_tolerates_progress(run: HarnessRun, revision: str | None = None) -> WireCell:
    def verdict(session: dict) -> tuple[Code, str]:
        records = run.records((session.get("captures") or {}).get("util"))
        calls = _tool_calls(records, "slow_with_progress")
        if not calls:
            return "u", f"{_where(session, 'util')}: slow_with_progress never called"
        progress = _s2c(records, "notifications/progress", "notification")
        cancelled = _c2s(records, "notifications/cancelled", "notification")
        answer = _answer_to(records, calls[0])
        if cancelled:
            return ("a" if progress else "n"), f"{_where(session, 'util', cancelled[0])} after {len(progress)} progress"
        if answer is not None and answer["kind"] == "response":
            return "y", f"{_where(session, 'util', answer)} after {len(progress)} progress notifications"
        return "n", f"{_where(session, 'util', calls[0])}: no result"
    return _count_cell(run, "slow", verdict, revision)


def _h_relist_next_turn(run: HarnessRun, revision: str | None = None) -> WireCell:
    sessions, missing = _level_2_runs(run, "relist", revision)
    if missing:
        return missing
    fired, reacted, evidence = _list_changed_reaction(run, sessions, None)
    # The wire decides the code. With a model stub, whether the added tool then reached
    # the model is visible too, and is said beside it.
    for session in sessions:
        after = [r for r in run.records(session.get("model")) if r.get("request") and r.get("tool_results_seen", 0) >= 1]
        if after:
            offered = [n for n in after[0].get("offered") or [] if n.endswith("SENTINEL_ADDED_TOOL")]
            evidence.append(f"run {session.get('run')}: {session['model']} request {after[0]['request']} "
                            + ("offered the model " + offered[0] if offered else "did not offer the model the added tool"))
    if not fired:
        return WireCell("u", ["add_tool was never called, so ntf never fired list_changed", _run_context(run, sessions)], "wire")
    return WireCell(_by_fraction(reacted, fired), [*evidence, _run_context(run, sessions)], "wire")


def _h_error_surfaced(run: HarnessRun, revision: str | None = None) -> WireCell:
    sessions, missing = _level_2_runs(run, "error", revision)
    if missing:
        return missing
    outcomes, evidence = [], []
    for session in sessions:
        calls = _tool_calls(run.records((session.get("captures") or {}).get("def")), "delete_account")
        if not calls:
            evidence.append(f"run {session.get('run')}: never called")
            continue
        outcome = "retried" if len(calls) > 1 else "reported"
        outcomes.append(outcome)
        # With a model stub, what the harness handed the model is the part it controls;
        # the final text is the stub's. Without one, quote the harness's final text.
        handed = next((r["last_tool_result"] for r in run.records(session.get("model")) if r.get("last_tool_result")), None)
        if handed is not None:
            flag = f"is_error {handed['is_error']}" if "is_error" in handed else "no error flag"
            said = f"handed the model {flag}: {handed.get('text', '')[:200]!r}"
        else:
            text = next((e.get("text", "") for e in reversed(_output_events(run, session, "result"))), "")
            said = f"said {text[:200]!r}"
        evidence.append(f"run {session.get('run')}: {outcome} ({len(calls)} calls); {said}")
    evidence.append(_run_context(run, sessions))
    if not outcomes:
        return WireCell("u", evidence, "wire")
    return WireCell("y", evidence, "wire", ", ".join(f"{outcomes.count(o)}x {o}" for o in sorted(set(outcomes))))


# -- transports -----------------------------------------------------------------
#
# Level 1 starts the harness once over stdio and once over Streamable HTTP. The
# transport cells read one session each by what they ask. These three answer what the
# harness does on the wire, which can differ by transport: Claude Code picks its
# revision per transport, and Cline's stdio client is a different code path from its
# HTTP one. So each is derived on every transport, and the published cell names both
# when they differ.
LEVEL_1_TRANSPORTS = {"stdio": "stdio", "http": "HTTP"}
HARNESS_TRANSPORT_RULES: dict[str, Callable[[HarnessRun, str, str | None], WireCell]] = {
    "h-negotiated-revision": _h_negotiated_revision,
    "h-declared-capabilities": _h_declared_capabilities,
    "h-answers-ping": _h_answers_ping,
}
CODE_WORDS = {"y": "yes", "a": "in part", "n": "no", "u": "not measured"}


def _transports(run: HarnessRun) -> list[str]:
    """The level-1 transports this run started the harness on, stdio first."""
    ids = {s["id"] for s in run.sessions("level_1")}
    return [t for t in LEVEL_1_TRANSPORTS if t in ids] or ["stdio"]


def transports_differ(per: dict[str, WireCell]) -> bool:
    return len({(c.code, c.value) for c in per.values()}) > 1


def _merge_transports(per: dict[str, WireCell]) -> WireCell:
    """One cell from the same cell on each transport, each transport's evidence labelled.

    Codes combine the way runs do. When the transports agree the value stands; when they
    differ the value names each, so the headline says both.
    """
    if len(per) == 1:
        return next(iter(per.values()))
    evidence = [f"{LEVEL_1_TRANSPORTS[t]}: {line}" for t, cell in per.items() for line in cell.evidence]
    first = next(iter(per.values()))
    if not transports_differ(per):
        return WireCell(first.code, evidence, "wire", first.value)
    value = None
    if any(c.value is not None for c in per.values()):
        value = " | ".join(f"{LEVEL_1_TRANSPORTS[t]}: {c.value if c.value is not None else CODE_WORDS[c.code]}"
                           for t, c in per.items())
    return WireCell(_combine(c.code for c in per.values()), evidence, "wire", value)


def _on_every_transport(cell_id: str) -> Callable[[HarnessRun, str | None], WireCell]:
    rule = HARNESS_TRANSPORT_RULES[cell_id]
    return lambda run, revision=None: _merge_transports({t: rule(run, t, revision) for t in _transports(run)})


# Every cell, one per fixture revision: both levels run once per SDK line
# (plan/harness_sessions.yaml fixture_lines), so every cell exists per revision.
HARNESS_REVISION_RULES: dict[str, Callable[[HarnessRun, str | None], WireCell]] = {
    "h-negotiated-revision": _on_every_transport("h-negotiated-revision"),
    "h-declared-capabilities": _on_every_transport("h-declared-capabilities"),
    "h-lists-tools-at-startup": _h_lists_tools_at_startup,
    "h-lists-prompts-at-startup": _h_lists_prompts_at_startup,
    "h-lists-resources-at-startup": _h_lists_resources_at_startup,
    "h-follows-pagination": _h_follows_pagination,
    "h-tool-count-limit": _h_tool_count_limit,
    "h-reacts-to-list-changed": _h_reacts_to_list_changed,
    "h-answers-ping": _on_every_transport("h-answers-ping"),
    "h-transport-streamable-http": lambda run, revision=None: _h_transport(run, "http", revision),
    "h-transport-stdio": lambda run, revision=None: _h_transport(run, "stdio", revision),
    "h-startup-timeout": _h_startup_timeout,
    "h-tools-offered-to-model": _h_tools_offered_to_model,
    "h-calls-read-only-tool": _h_calls_read_only_tool,
    "h-destructive-blocked-by-default": _h_destructive_blocked,
    "h-read-only-runs-without-prompt": _h_read_only_without_prompt,
    "h-destructive-runs-unattended": _h_destructive_runs_unattended,
    "h-reads-resource": _h_reads_resource,
    "h-expands-prompt": _h_expands_prompt,
    "h-answers-elicitation": _h_answers_elicitation,
    "h-answers-sampling": _h_answers_sampling,
    "h-tolerates-progress": _h_tolerates_progress,
    "h-relist-next-turn": _h_relist_next_turn,
    "h-error-surfaced": _h_error_surfaced,
}
HARNESS_LEVEL_2_CELLS = frozenset({
    "h-calls-read-only-tool", "h-destructive-blocked-by-default", "h-read-only-runs-without-prompt",
    "h-destructive-runs-unattended", "h-reads-resource", "h-expands-prompt", "h-answers-elicitation",
    "h-answers-sampling", "h-tolerates-progress", "h-relist-next-turn", "h-error-surfaced",
})


def _revision_cells(run: HarnessRun, cell_id: str,
                    rule: Callable[[HarnessRun, str | None], WireCell] | None = None) -> dict[str, WireCell]:
    rule = rule or HARNESS_REVISION_RULES.get(cell_id)
    revisions = run.revisions("level_2" if cell_id in HARNESS_LEVEL_2_CELLS else "level_1")
    if rule is None or not revisions:
        return {}
    return {revision: rule(run, revision) for revision in revisions}


def _over_revisions(run: HarnessRun, cell_id: str, rule: Callable[[HarnessRun, str | None], WireCell]) -> WireCell:
    """The per-revision cells combined, each revision's evidence labelled.

    Codes combine the way runs do (unanimous stands, a mix is a, u does not vote), and a
    value cell names each revision's value, so h-negotiated-revision reads as both
    negotiations side by side.
    """
    per = _revision_cells(run, cell_id, rule)
    if len(per) <= 1:
        return rule(run, None)
    evidence = [f"{revision} fixture: {line}" for revision, cell in per.items() for line in cell.evidence]
    values = [f"{revision} fixture: {cell.value}" for revision, cell in per.items() if cell.value is not None]
    return WireCell(_combine(c.code for c in per.values()), evidence, "wire", "; ".join(values) or None)


def _transport_rule(cell_id: str, transport: str) -> Callable[[HarnessRun, str | None], WireCell]:
    rule = HARNESS_TRANSPORT_RULES[cell_id]
    return lambda run, revision=None: rule(run, transport, revision)


def _fresh(run: HarnessRun, cell: WireCell, cell_id: str | None = None) -> WireCell:
    """The cell with the run's freshness joined to its evidence, or u if the run was not fresh.

    A level-1 cell also says when the stub held its reply (session.json level_1
    reply_delay_s, from the module's LEVEL1_REPLY_DELAY_S), since the capture then shows
    a turn that was kept open for the harness's background server connect.
    """
    fresh, line = run.freshness()
    if not fresh:
        return WireCell("u", [line], "wire")
    evidence = [*cell.evidence, line]
    if cell_id is not None and cell_id not in HARNESS_LEVEL_2_CELLS:
        for session_id in ("stdio", "http", "slow-start"):
            default_line = level_1_default_line(run, session_id)
            if default_line and default_line not in evidence:
                evidence.append(default_line)
        delay = (run.manifest.get("level_1") or {}).get("reply_delay_s")
        if delay:
            evidence.append(f"level 1: the stub held its reply {delay:g} s (LEVEL1_REPLY_DELAY_S in "
                            f"scripts/harnesses/{run.harness}.py), so the turn outlasted the background connect")
    return WireCell(cell.code, evidence, cell.instrument, cell.value)


def harness_revision_cells(run: HarnessRun, cell_id: str) -> dict[str, WireCell]:
    """``{revision: WireCell}`` for a cell whose level ran on more than one line; else {}.

    {} as well for a run that was not fresh: it publishes no cell, per revision or combined.
    """
    if not run.freshness()[0]:
        return {}
    return {revision: _fresh(run, cell, cell_id) for revision, cell in _revision_cells(run, cell_id).items()}


def harness_transport_cells(run: HarnessRun, cell_id: str) -> dict[str, dict]:
    """``{transport: {"cell": WireCell, "by_revision": {revision: WireCell}}}`` for a
    transport cell on a run with more than one level-1 transport; else {}.

    {} as well for a run that was not fresh, like harness_revision_cells.
    """
    transports = _transports(run)
    if cell_id not in HARNESS_TRANSPORT_RULES or len(transports) < 2 or not run.freshness()[0]:
        return {}
    split = {}
    for transport in transports:
        rule = _transport_rule(cell_id, transport)
        split[transport] = {
            "cell": _fresh(run, _over_revisions(run, cell_id, rule), cell_id),
            "by_revision": {r: _fresh(run, c, cell_id) for r, c in _revision_cells(run, cell_id, rule).items()},
        }
    return split


def _across_revisions(cell_id: str) -> Callable[[HarnessRun], WireCell]:
    """The published cell: the per-revision cells combined, each revision's evidence labelled.

    A transport cell is combined over revisions on each transport first, then over
    transports, so its headline reads transport by transport.
    """
    def combined(run: HarnessRun) -> WireCell:
        if cell_id in HARNESS_TRANSPORT_RULES:
            per = {t: _over_revisions(run, cell_id, _transport_rule(cell_id, t)) for t in _transports(run)}
            return _fresh(run, _merge_transports(per), cell_id)
        return _fresh(run, _over_revisions(run, cell_id, HARNESS_REVISION_RULES[cell_id]), cell_id)

    return combined


HARNESS_RULES: dict[str, Callable[[HarnessRun], WireCell]] = {
    cell_id: _across_revisions(cell_id) for cell_id in HARNESS_REVISION_RULES
}


def load_harness_runs(snapshot: Path) -> list[HarnessRun]:
    """Every harness's run: the stubbed one under wire/, or the live one under wire-live/
    for a harness that has no stub (HarnessRun.live). A harness with both is a mistake
    the validator reports; the live one is read, so the row never silently loses it."""
    runs: dict[str, HarnessRun] = {}
    for wire in ("wire", "wire-live"):
        for path in sorted((snapshot / "measurements" / wire).glob(f"*/{HARNESS_SESSION}")):
            run = HarnessRun(path)
            runs[run.harness] = run
    return [runs[harness] for harness in sorted(runs)]


def derive_harness_cells(snapshot: Path) -> dict[str, dict[str, WireCell]]:
    """``{cell_id: {harness_id: WireCell}}`` for every harness with a session.json."""
    cells: dict[str, dict[str, WireCell]] = {}
    for run in load_harness_runs(snapshot):
        for cell_id, rule in HARNESS_RULES.items():
            cells.setdefault(cell_id, {})[run.harness] = rule(run)
    return cells


HARNESS_ROW_SCHEMA = "mcp-mirror/harness-capability@1"
CONTEXT_NOTE = "context: "


def publish_harness_rows(snapshot: Path, surface: dict, harnesses: dict) -> list[dict]:
    """One published row per surface feature, cells derived, notes taken from evidence.

    A harness cell's note is its evidence: the capture path and the message ids, so a
    maintainer can open the JSONL and see exactly what their harness sent. No code is
    typed by hand, which is why a harness row needs no overlay.

    The one hand-written part is context the wire cannot carry: a harness's
    ``cell_notes`` in harnesses.json (the mode the cell was measured in, or an upstream
    change since the measured release). A context note is appended to the evidence and
    numbered even on a y cell; it never changes the code.
    """
    derived = derive_harness_cells(snapshot)
    runs = {run.harness: run for run in load_harness_runs(snapshot)}
    agents = harnesses["agents"]
    rows = []
    for feature in surface["features"]:
        row_cells = derived.get(feature["id"], {})
        stats, values, evidence, notes_by_num, by_revision, by_transport = {}, {}, {}, {}, {}, {}
        number = 0
        for agent_id in reading_order(agents):
            agent = agents[agent_id]
            cell = row_cells.get(agent_id) or WireCell("u", ["no wire capture for this harness"], "wire")
            context = (agent.get("cell_notes") or {}).get(feature["id"])
            if context:
                cell = WireCell(cell.code, [*cell.evidence, f"{CONTEXT_NOTE}{context}"], cell.instrument, cell.value)
            code = cell.code
            if (code != "y" or context) and cell.evidence:
                number += 1
                notes_by_num[str(number)] = "; ".join(cell.evidence)
                code = f"{code} #{number}"
            stats[agent_id] = {agent["current_version"]: code}
            evidence[agent_id] = cell.evidence
            if cell.value is not None:
                values[agent_id] = cell.value
            if agent_id in runs:
                per = harness_revision_cells(runs[agent_id], feature["id"])
                if per:
                    by_revision[agent_id] = {revision: c.as_dict() for revision, c in per.items()}
                split = harness_transport_cells(runs[agent_id], feature["id"])
                if split:
                    by_transport[agent_id] = {
                        t: {**part["cell"].as_dict(),
                            "by_revision": {r: c.as_dict() for r, c in part["by_revision"].items()}}
                        for t, part in split.items()
                    }
        stats = {a: stats[a] for a in agents}
        reproduce = sorted({runs[a].manifest.get("command", "") for a in agents if a in runs} - {""})
        # A run that opened no session at any level has no measurement date (HarnessRun.blocked).
        measured_on = {a: runs[a].manifest.get("captured_at", "")[:10]
                       for a in agents if a in runs and not runs[a].blocked()}
        measured_on = {a: stamp for a, stamp in measured_on.items() if stamp}
        dates = list(measured_on.values())
        rows.append({
            "schema": HARNESS_ROW_SCHEMA,
            "id": feature["id"],
            "area": feature["area"],
            "level": feature["level"],
            "title": feature["title"],
            "question": feature.get("question", ""),
            "description": feature.get("description", ""),
            "observe": feature["observe"],
            "value_cell": feature["id"] in HARNESS_VALUE_CELLS,
            **({"labels": feature["labels"]} if feature.get("labels") else {}),
            "stats": stats,
            "values": values,
            "notes_by_num": notes_by_num,
            "evidence": evidence,
            # The same cell against each fixture revision, so the 2025-11-25 and
            # 2026-07-28 answers can be read apart.
            "by_revision": by_revision,
            # The same cell on each level-1 transport (stdio, http), for the cells a
            # harness can answer differently per transport (HARNESS_TRANSPORT_RULES).
            "by_transport": by_transport,
            "instruments": ["wire"],
            # One command per harness: joining them would make a row's command grow with
            # every harness added. `reproduce` keeps the first for older readers.
            "reproduce": reproduce[0] if reproduce else "",
            "reproduce_by_agent": {
                a: runs[a].manifest.get("command", "") for a in agents
                if a in runs and runs[a].manifest.get("command")
            },
            "measured": {"run_date": max(dates) if dates else None, "measured_on": measured_on},
        })
    return rows
