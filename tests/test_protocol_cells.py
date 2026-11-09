"""Protocol-run captures derive the base-protocol cells the way cells.PROTOCOL_RULES says."""

from __future__ import annotations

import json
from pathlib import Path

from mcp_mirror.cells import _reproduce_commands, derive_protocol_cells

META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
    "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "0"},
}
SCHEMA_2020 = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "$defs": {"range": {"type": "object", "properties": {"low": {"type": "number"}}}},
    "properties": {
        "point": {"type": "array", "prefixItems": [{"type": "number"}], "items": False},
        "window": {"$ref": "#/$defs/range"},
    },
}


class Capture:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, **record) -> dict:
        record.setdefault("type", "message")
        record["seq"] = len(self.records)
        record.setdefault("elapsed", record["seq"] / 100)
        self.records.append(record)
        return record

    def request(self, method: str, params: dict | None = None, *, id=1, direction="client_to_server",
                origin="client", **extra) -> dict:
        return self.add(direction=direction, origin=origin, kind="request", method=method, id=id,
                        message={"method": method, "params": params or {}}, **extra)

    def answer(self, request: dict, result: dict | None = None, kind: str = "response") -> dict:
        direction = "server_to_client" if request["direction"] == "client_to_server" else "client_to_server"
        return self.add(direction=direction, origin="client" if direction == "client_to_server" else "server",
                        kind=kind, method=request["method"], id=request["id"], message={"result": result or {}})


def _snapshot(tmp_path: Path, spec: str, features: list[str]) -> Path:
    (tmp_path / "frameworks.json").write_text(json.dumps({"mcp_spec": spec}))
    (tmp_path / "surface.json").write_text(json.dumps({"features": [{"id": f} for f in features]}))
    return tmp_path


def _write(snapshot: Path, renderer: str, captures: dict[str, Capture], transports: dict) -> None:
    directory = snapshot / "measurements" / "protocol" / renderer
    directory.mkdir(parents=True)
    for name, capture in captures.items():
        (directory / name).write_text("".join(json.dumps(r) + "\n" for r in capture.records))
    (directory / "run.json").write_text(json.dumps({"renderer": renderer, "transports": transports}))


def _handshake(capture: Capture, *, answer_ping: bool = True) -> None:
    capture.request("initialize", id=0)
    capture.add(type="negotiation", via="initialize", requested_revision="2025-11-25",
                negotiated_revision="2025-11-25")
    capture.add(direction="client_to_server", origin="client", kind="notification",
                method="notifications/initialized", id=None, message={})
    ping = capture.request("ping", id="mcp-mirror-ping-1", direction="server_to_client", origin="recorder")
    capture.request("tools/list", id=1)
    if answer_ping:
        capture.answer(ping)


def test_legacy_rows_read_the_handshake_ping_and_rendered_schemas(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, "2025-11-25", ["initialize-handshake", "ping", "json-schema-dialect"])
    good = Capture()
    _handshake(good)
    good.add(direction="server_to_client", origin="server", kind="response", method="tools/list", id=1,
             message={"result": {"tools": [{"name": "dialect_2020_12", "inputSchema": SCHEMA_2020}]}})
    # A refused discovery probe before the real session is not a missing handshake.
    probe = Capture()
    probe.request("server/discover", id="server-discover-probe-1")
    inlined = json.loads(json.dumps(SCHEMA_2020))
    inlined["properties"]["window"] = inlined.pop("$defs")["range"]
    _write(snapshot, "mastra", {"stdio-00.jsonl": probe, "stdio-01.jsonl": good},
           {"stdio": {"ok": True, "tools": [{"name": "src_dialect_2020_12", "params": inlined}]}})

    silent = Capture()
    _handshake(silent, answer_ping=False)
    silent.add(direction="server_to_client", origin="server", kind="response", method="tools/list", id=1,
               message={"result": {"tools": [{"name": "dialect_2020_12", "inputSchema": SCHEMA_2020}]}})
    flattened = {"type": "object", "properties": {"point": {"type": "array", "items": {}}}}
    _write(snapshot, "crewai", {"stdio-00.jsonl": silent},
           {"stdio": {"ok": True, "omitted": ["dialect_draft_07"], "full_fixture_error": "boom",
                      "tools": [{"name": "dialect_2020_12", "params": flattened}]}})

    cells = derive_protocol_cells(snapshot)

    assert cells["initialize-handshake"]["mastra"].code == "y"
    assert "server/discover" in cells["initialize-handshake"]["mastra"].evidence[-1]
    assert cells["ping"]["mastra"].code == "y"
    assert cells["ping"]["crewai"].code == "n"
    # The 2020-12 tool survived with its $ref resolved; the draft-07 one was never offered.
    assert cells["json-schema-dialect"]["mastra"].code == "a"
    assert "resolving 1 $ref inline" in cells["json-schema-dialect"]["mastra"].evidence[0]
    assert "dialect_draft_07 not offered" in cells["json-schema-dialect"]["mastra"].evidence[1]
    assert cells["json-schema-dialect"]["crewai"].code == "n"
    assert any("while dialect_draft_07 was listed (boom)" in e for e in cells["json-schema-dialect"]["crewai"].evidence)
    assert "per-request-metadata" not in cells


def test_modern_rows_read_meta_discovery_result_type_and_headers(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, "2026-07-28",
                         ["per-request-metadata", "server-discover", "result-type", "x-mcp-header"])
    stdio = Capture()
    discover = stdio.request("server/discover", {"_meta": META}, id=0)
    stdio.add(type="negotiation", via="server/discover", supported_versions=["2026-07-28"])
    stdio.answer(discover)
    stdio.request("tools/list", {"_meta": META}, id=1)
    stdio.request("tools/call", {"name": "needs_input", "_meta": {}}, id=2)
    http = Capture()
    http.request("tools/call", {"name": "routed_lookup", "arguments": {"region": "eu"}, "_meta": META}, id=3,
                 http={"headers": {"mcp-param-region": "eu"}})
    _write(snapshot, "openai_agents", {"stdio-00.jsonl": stdio, "http.jsonl": http}, {
        "stdio": {"ok": True, "results": [
            {"tool": "needs_input", "ok": True, "rep": {"text": "done: SENTINEL_INPUT_ANSWERED {}"}},
            {"tool": "needs_input", "ok": True, "rep": {"text": "the lookup needs input"}},
        ]},
        "http": {"ok": True, "tools": [{"name": "routed_lookup"}], "results": [
            {"tool": "needs_input", "ok": False, "error": "input required: confirm"},
        ]},
    })

    cells = derive_protocol_cells(snapshot)
    row = {capability: cells[capability]["openai-agents"] for capability in cells}

    assert row["per-request-metadata"].code == "a"
    assert "3 of 4 requests" in row["per-request-metadata"].evidence[0]
    assert row["server-discover"].code == "y"
    assert row["result-type"].code == "a"
    assert row["x-mcp-header"].code == "y"
    assert "initialize-handshake" not in cells


def test_discovery_over_its_own_connection_counts_for_the_next_session(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, "2026-07-28", ["server-discover"])
    probe = Capture()
    probe.request("server/discover", {"_meta": META}, id=0)
    probe.add(type="negotiation", via="server/discover", supported_versions=["2026-07-28"])
    work = Capture()
    work.request("tools/list", {"_meta": META}, id=1)
    undiscovered = Capture()
    undiscovered.request("tools/list", {"_meta": META}, id=1)
    _write(snapshot, "mastra", {"stdio-00.jsonl": probe, "stdio-01.jsonl": work, "stdio-02.jsonl": undiscovered},
           {"stdio": {"ok": True}})

    cell = derive_protocol_cells(snapshot)["server-discover"]["mastra"]

    assert cell.code == "a"
    assert cell.evidence[0] == "1 of 2 working sessions used a discovered version"
    assert "discovered over stdio-00.jsonl" in cell.evidence[1]


def test_each_renderer_publishes_its_own_reproduce_command(tmp_path: Path) -> None:
    # One command naming every renderer would change whenever a framework is added, so
    # every branch's reproduce check would rerun all of them.
    snapshot = _snapshot(tmp_path, "2025-11-25", ["ping"])
    for renderer in ("mastra", "agno"):
        _write(snapshot, renderer, {}, {"stdio": {"ok": True}})
        run = snapshot / "measurements" / "protocol" / renderer / "run.json"
        run.write_text(json.dumps({**json.loads(run.read_text()), "spec_version": "2025-11-25"}))

    command = _reproduce_commands(snapshot, tmp_path)["protocol"]

    assert command.split(" && ") == [
        "uv run python scripts/measure_protocol.py --spec-version 2025-11-25 --renderers agno",
        "uv run python scripts/measure_protocol.py --spec-version 2025-11-25 --renderers mastra",
    ]
