"""outputSchema in the differ and the scan rule, and the extension rows read from the
probe's recorded handshakes. No server or framework needed."""

from __future__ import annotations

import json
from pathlib import Path

from mcp_mirror.cells import SCAN_RULES, _extension_cells, derive_probe_cells, derive_scan_cell
from mcp_mirror.diff import diff_tool
from mcp_mirror.models import ToolRep

SCHEMA = {
    "type": "object",
    "properties": {"count": {"type": "integer"}, "status": {"enum": ["a", "b"]}},
    "required": ["count"],
}


def _source() -> ToolRep:
    return ToolRep(name="t", params={"type": "object"}, output_schema=SCHEMA, origin="source")


def _rendered(**kwargs) -> ToolRep:
    return ToolRep(name="t", params={"type": "object"}, origin="fw", **kwargs)


def _schema_diffs(rendered: ToolRep) -> list:
    return [d for d in diff_tool(_source(), rendered) if d.path == "outputSchema"]


def test_identical_output_schema_is_faithful() -> None:
    assert _schema_diffs(_rendered(output_schema=dict(SCHEMA))) == []


def test_added_dialect_is_additive_not_a_change() -> None:
    (diff,) = _schema_diffs(_rendered(output_schema={**SCHEMA, "$schema": "https://json-schema.org/draft/2020-12/schema"}))
    assert diff.category == "additive"
    assert "$schema" in diff.detail


def test_changed_keyword_is_transformative() -> None:
    changed = {**SCHEMA, "required": []}
    (diff,) = _schema_diffs(_rendered(output_schema=changed))
    assert diff.category == "transformative"


def test_retained_off_definition_is_transformative() -> None:
    (diff,) = _schema_diffs(_rendered(framework_metadata={"outputSchema": SCHEMA}))
    assert diff.category == "transformative"
    assert "retained" in diff.detail


def test_dropped_output_schema_is_lossy() -> None:
    (diff,) = _schema_diffs(_rendered())
    assert diff.category == "lossy"


def test_output_schema_keywords_do_not_leak_into_parameter_rows() -> None:
    diffs = [d.model_dump() for d in diff_tool(_source(), _rendered())]
    source_tools = [_source().model_dump()]
    for capability in ("enum", "required"):
        assert derive_scan_cell(SCAN_RULES[capability], source_tools, diffs).code == "y"


def test_output_schema_rule_codes() -> None:
    rule = SCAN_RULES["tool-output-schema"]
    source_tools = [_source().model_dump()]
    code = lambda rendered: derive_scan_cell(  # noqa: E731
        rule, source_tools, [d.model_dump() for d in diff_tool(_source(), rendered)]
    ).code
    assert code(_rendered(output_schema={**SCHEMA, "$schema": "x"})) == "y"
    assert code(_rendered(framework_metadata={"outputSchema": SCHEMA})) == "a"
    assert code(_rendered()) == "n"
    # A scan of a fixture that declared no output schema measured nothing.
    assert derive_scan_cell(rule, [ToolRep(name="t", origin="source").model_dump()], []).code == "u"


def _handshake(client: dict | None, server: dict | None) -> dict:
    return {"via": "initialize", "client_capabilities": client, "server_capabilities": server}


UI = "io.modelcontextprotocol/ui"


def test_client_declaring_the_offered_extension_negotiates_it() -> None:
    cells = _extension_cells({"negotiations": [_handshake({"extensions": {UI: {}}}, {"extensions": {UI: {}}})]})
    assert cells["extension-negotiation"].code == "y"
    assert cells["extension-apps"].code == "y"


def test_client_declaring_nothing_negotiates_nothing() -> None:
    cells = _extension_cells({"negotiations": [_handshake({"sampling": {}}, {"extensions": {UI: {}}})]})
    assert cells["extension-negotiation"].code == "n"
    assert cells["extension-apps"].code == "n"


def test_client_declaring_other_extensions_is_partial() -> None:
    cells = _extension_cells({"negotiations": [_handshake({"extensions": {"x.example/other": {}}}, {"extensions": {UI: {}}})]})
    assert cells["extension-negotiation"].code == "a"
    assert cells["extension-apps"].code == "n"


def test_no_offer_or_no_handshake_is_unmeasured() -> None:
    assert _extension_cells({"negotiations": [_handshake({}, {})]})["extension-negotiation"].code == "u"
    missing = _extension_cells({"negotiations": []})
    assert {cell.code for cell in missing.values()} == {"u"}


def test_offer_seen_on_another_session_counts() -> None:
    # A client that never sends server/discover sees no offer on its own session.
    cells = _extension_cells({"negotiations": [_handshake({"extensions": {UI: {}}}, None)]}, {UI})
    assert cells["extension-negotiation"].code == "y"


def test_offer_seen_in_another_probe_file_counts(tmp_path: Path) -> None:
    # A framework alone in its own probe file, opening without server/discover, still
    # gets the fixture's offer from the other captures in the snapshot.
    probe = tmp_path / "measurements" / "probe"
    probe.mkdir(parents=True)
    for name, framework, handshake in (
        ("all-frameworks", "asks", _handshake({"extensions": {UI: {}}}, {"extensions": {UI: {}}})),
        ("alone", "alone", _handshake({"extensions": {UI: {}}}, None)),
    ):
        capture = {"frameworks": {framework: {"ok": True, "probe": {}, "wire": {"negotiations": [handshake]}}}}
        (probe / f"{name}.json").write_text(json.dumps(capture))
    cells = derive_probe_cells(tmp_path)
    assert cells["extension-negotiation"]["alone"].code == "y"
    assert cells["extension-negotiation"]["asks"].code == "y"
