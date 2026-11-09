"""Conformance-suite checks derive cells the way cells.CONFORMANCE_RULES says."""

from __future__ import annotations

import json
from pathlib import Path

from mcp_mirror.cells import derive_conformance_cells


def _check(check_id: str, status: str = "SUCCESS") -> dict:
    return {"id": check_id, "status": status}


def _wire(status_code: int, *, minted: bool = False, body: str | None = None) -> dict:
    details: dict = {"method": "POST", "path": "/mcp", "statusCode": status_code}
    if minted:
        details["sessionIdIssued"] = True
    if body is not None:
        details["body"] = body
    return {"id": "outgoing-response", "status": "INFO", "details": details}


def _write(snapshot: Path, renderer: str, suites: dict) -> None:
    path = snapshot / "measurements" / "conformance" / renderer / "checks.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"renderer": renderer, "suites": suites}))


def test_rules_read_scenarios_and_carry_their_names(tmp_path: Path) -> None:
    passing_auth = [_check("prm-pathbased-requested"), _check("client-registration"), _check("token-request")]
    no_dcr = [_check("prm-pathbased-requested"), _check("client-registration", "FAILURE"), _check("token-request")]
    _write(tmp_path, "langchain", {
        "auth": {"scenarios": {"auth/metadata-default": {"checks": passing_auth}}},
        "core": {"scenarios": {
            "auth/metadata-default": {"checks": passing_auth},
            "initialize": {"checks": [_check("mcp-client-initialization")]},
            "tools_call": {"checks": [_check("tool-add-numbers")]},
            "sse-retry": {"checks": [_check("client-sse-last-event-id", "FAILURE")]},
            "elicitation-sep1034-client-defaults": {"checks": [
                _wire(200, minted=True), _wire(200), _wire(202),
                # A second connection's pre-initialize probe owes no session ID.
                {**_wire(400, body="Invalid or missing session ID"),
                 "details": {"mcpMethod": "server/discover", "statusCode": 400, "body": "Invalid or missing session ID"}},
            ]},
        }},
    })
    _write(tmp_path, "mastra", {
        "auth": {"scenarios": {"auth/metadata-default": {"checks": no_dcr}}},
        "core": {"scenarios": {
            "initialize": {"checks": []},
            "elicitation-sep1034-client-defaults": {"checks": [
                _wire(200, minted=True), _wire(400, body='"Invalid or missing session ID"'),
            ]},
        }},
    })

    cells = derive_conformance_cells(tmp_path)

    assert cells["authorization-oauth"]["langchain"].code == "y"
    assert cells["authorization-oauth"]["mastra"].code == "a"
    assert any(e.startswith("auth/metadata-default:") for e in cells["authorization-oauth"]["mastra"].evidence)
    # Discovery is judged on the discovery checks only, so a registration miss is not one.
    assert cells["authorization-resource-metadata"]["mastra"].code == "y"
    assert cells["initialize-handshake"]["langchain"].code == "y"
    assert cells["initialize-handshake"]["mastra"].code == "u"  # no graded checks is no evidence
    assert cells["transport-streamable-http"]["langchain"].code == "y"
    assert cells["sse-resumability"]["langchain"].code == "n"
    assert cells["session-identifier"]["langchain"].code == "y"
    assert cells["session-identifier"]["mastra"].code == "n"
