"""Scan-derived definition cells when an adapter drops a tool outright. No server or
framework needed."""

from __future__ import annotations

from mcp_mirror.cells import SCAN_RULES, derive_scan_cell, dropped_tools
from mcp_mirror.diff import diff_tool
from mcp_mirror.models import ToolRep

UNION = ToolRep(
    name="union",
    description="Has the only oneOf.",
    params={
        "type": "object",
        "properties": {"mode": {"oneOf": [{"const": "a"}, {"const": "b"}], "description": "Mode."}},
    },
    annotations={"title": "Union", "idempotentHint": True},
    origin="source",
)
PLAIN = ToolRep(
    name="plain",
    description="Survives.",
    params={"type": "object", "properties": {"q": {"type": "string", "description": "Query."}}},
    annotations={"title": "Plain"},
    origin="source",
)


def _code(capability: str, rendered: list[ToolRep]) -> tuple[str, list[str]]:
    by_name = {tool.name: tool for tool in rendered}
    diffs = [
        d.model_dump()
        for source in (UNION, PLAIN)
        for d in diff_tool(source, by_name.get(source.name))
    ]
    cell = derive_scan_cell(SCAN_RULES[capability], [UNION.model_dump(), PLAIN.model_dump()], diffs)
    return cell.code, cell.evidence


def _kept(tool: ToolRep) -> ToolRep:
    return tool.model_copy(update={"origin": "fw"})


def test_a_dropped_tool_is_recognised_from_its_lost_name() -> None:
    diffs = [d.model_dump() for d in diff_tool(UNION, None)]
    assert dropped_tools(diffs) == {"union"}
    renamed = [d.model_dump() for d in diff_tool(PLAIN, _kept(PLAIN).model_copy(update={"name": "p"}))]
    assert dropped_tools(renamed) == set()


def test_keyword_only_on_a_dropped_tool_reads_n() -> None:
    code, evidence = _code("one-of-any-of", [_kept(PLAIN)])
    assert code == "n"
    assert any(e.startswith("union: not rendered") for e in evidence)
    assert _code("idempotent-hint", [_kept(PLAIN)])[0] == "n"


def test_keyword_surviving_elsewhere_reads_a() -> None:
    for capability in ("title-annotation", "description-fidelity", "property-descriptions"):
        assert _code(capability, [_kept(PLAIN)])[0] == "a", capability


def test_dropped_tool_name_is_counted_once() -> None:
    code, evidence = _code("tool-name", [_kept(PLAIN)])
    assert code == "a"
    assert evidence == ["union: not rendered, so all 1 counted places are lost"]


def test_nothing_dropped_still_reads_y() -> None:
    assert _code("one-of-any-of", [_kept(UNION), _kept(PLAIN)])[0] == "y"


def test_keyword_count_is_one_place_per_keyword() -> None:
    tool = {
        "params": {
            "type": "object",
            "properties": {
                # A property named like a keyword is not that keyword.
                "format": {"type": "string", "format": "uri"},
                "pick": {"oneOf": [{"const": "a"}, {"const": "b"}]},
            },
            "required": ["format", "pick"],
        }
    }
    assert SCAN_RULES["format"].count(tool) == 1
    assert SCAN_RULES["one-of-any-of"].count(tool) == 1
    assert SCAN_RULES["required"].count(tool) == 1


def test_every_place_lost_reads_n_beside_a_same_named_property() -> None:
    source = ToolRep(
        name="t",
        params={
            "type": "object",
            "properties": {
                "config": {"type": "object", "properties": {"format": {"type": "string"}}},
                "email": {"type": "string", "format": "email"},
            },
        },
        origin="source",
    )
    rendered = source.model_copy(
        update={
            "origin": "fw",
            "params": {**source.params, "properties": {**source.params["properties"], "email": {"type": "string"}}},
        }
    )
    diffs = [d.model_dump() for d in diff_tool(source, rendered)]
    assert derive_scan_cell(SCAN_RULES["format"], [source.model_dump()], diffs).code == "n"
