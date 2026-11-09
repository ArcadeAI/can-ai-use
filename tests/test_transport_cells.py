"""Transport captures derive the transport rows the way cells.derive_transport_cells says,
and the legacy SSE bridge serves a real client."""

from __future__ import annotations

import http.client
import json
import re
import signal
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from mcp_mirror.cells import _merge_instruments, Cell, derive_snapshot, derive_transport_cells
from mcp_mirror.recorder import read_capture
from mcp_mirror.source import parse_server

ROOT = Path(__file__).resolve().parents[1]


def _message(session: str, method: str, *, kind: str = "request", revision: str | None = "2025-11-25",
             headers: dict | None = None) -> dict:
    return {"type": "message", "session": session, "direction": "client_to_server", "kind": kind,
            "method": method, "revision": revision, "http": {"method": "POST", "headers": headers or {}}}


def _answers(text: str = "ok") -> dict:
    return {"tools": [{"name": "t", "description": "d", "params": {}, "annotations": {}, "icons": []}],
            "results": [{"tool": "t", "boundary": None, "ok": True, "rep": {"text": text}}]}


def _write(snapshot: Path, frameworks: dict, wires: dict[str, list[dict]]) -> None:
    base = snapshot / "measurements" / "transports"
    (base / "wire").mkdir(parents=True)
    for name, records in wires.items():
        (base / "wire" / name).write_text("".join(json.dumps(r) + "\n" for r in records))
    (base / "all-frameworks.json").write_text(json.dumps({
        "schema": "mcp-mirror/transports@1", "captured_at": "2026-09-30T00:00:00Z",
        "fixtures": {"results": "fixtures/results_server.py"}, "command": "uv run python scripts/measure_transports.py",
        "frameworks": frameworks,
    }))


def _run(ok: bool = True, capture: str = "", answers: dict | None = None, error: str = "") -> dict:
    entry = {"ok": ok, "capture": capture, "fixture": "fixtures/results_server.py"}
    if ok:
        entry["answers"] = answers or _answers()
    else:
        entry["error"] = error
    return entry


def test_http_matching_stdio_with_sessions_carried_is_y(tmp_path: Path) -> None:
    wire = [
        _message("stateless", "server/discover", revision="2026-07-28"),
        _message("s1", "initialize", revision=None),
        _message("s1", "notifications/initialized", kind="notification"),
        _message("s1", "tools/list"),
    ]
    _write(tmp_path, {"openai_agents": {"transports": {
        "stdio": {"results": _run()},
        "http": {"results": _run(capture="wire/a-http.jsonl")},
        "sse": {"results": _run(False, error="ConnectError: refused")},
    }}}, {"a-http.jsonl": wire})
    cells = derive_transport_cells(tmp_path)
    assert cells["transport-stdio"]["openai-agents"].code == "y"
    assert cells["transport-streamable-http"]["openai-agents"].code == "y"
    assert cells["transport-http-sse"]["openai-agents"].code == "n"
    assert "no session (ConnectError" in cells["transport-http-sse"]["openai-agents"].evidence[0]
    session = cells["session-identifier"]["openai-agents"]
    assert session.code == "y"
    assert "2 requests after initialize" in session.evidence[0]


def test_answers_that_differ_from_stdio_mark_the_transport_down(tmp_path: Path) -> None:
    _write(tmp_path, {"pydantic_ai": {"transports": {
        "stdio": {"results": _run()},
        "http": {"results": _run(answers=_answers("changed"))},
    }}}, {})
    cell = derive_transport_cells(tmp_path)["transport-streamable-http"]["pydantic-ai"]
    assert cell.code == "a"
    assert "1 of 2 answers differ from stdio (call t@default)" in cell.evidence[0]


def test_a_request_without_the_minted_session_id_counts_against_it(tmp_path: Path) -> None:
    wire = [
        _message("s1", "initialize", revision=None),
        _message("stateless", "tools/list"),
        _message("s1", "tools/call"),
    ]
    _write(tmp_path, {"crewai": {"transports": {
        "stdio": {"results": _run()}, "http": {"results": _run(capture="wire/c.jsonl")},
    }}}, {"c.jsonl": wire})
    cell = derive_transport_cells(tmp_path)["session-identifier"]["crewai"]
    assert cell.code == "a"
    assert "1 of 2 requests after initialize lacked" in cell.evidence[0]


def test_stateless_revision_requires_method_and_name_headers(tmp_path: Path) -> None:
    modern = "2026-07-28"
    wire = [
        {"type": "negotiation", "session": "stateless", "via": "request-meta", "negotiated_revision": modern},
        _message("stateless", "tools/list", revision=modern, headers={"mcp-method": "tools/list"}),
        _message("stateless", "tools/call", revision=modern, headers={"mcp-method": "tools/call"}),
    ]
    _write(tmp_path, {"mastra": {"transports": {
        "stdio": {"results": _run()}, "http": {"results": _run(capture="wire/m.jsonl")},
    }}}, {"m.jsonl": wire})
    cells = derive_transport_cells(tmp_path)
    cell = cells["transport-streamable-http"]["mastra"]
    assert cell.code == "a"
    assert any("tools/call without Mcp-Name" in line for line in cell.evidence)
    # No session was minted, so the stateless capture says nothing about session IDs.
    assert "session-identifier" not in cells


def test_two_instruments_on_one_cell_combine_and_name_both() -> None:
    into = {"session-identifier": {"langchain": Cell("y", ["conformance says"], "conformance")}}
    _merge_instruments(into, {"session-identifier": {"langchain": Cell("a", ["recorder says"], "transports")},
                              "transport-stdio": {"langchain": Cell("y", [], "transports")}})
    merged = into["session-identifier"]["langchain"]
    assert (merged.code, merged.instrument) == ("a", "conformance+transports")
    assert merged.evidence == ["conformance says", "recorder says"]
    assert into["transport-stdio"]["langchain"].code == "y"


def test_committed_snapshots_derive_every_transport_row() -> None:
    for snapshot, features in (
        (ROOT / "data", ("transport-stdio", "transport-streamable-http", "transport-http-sse",
                         "session-identifier", "sse-resumability")),
        (ROOT / "data" / "specs" / "2026-07-28", ("transport-stdio", "transport-streamable-http")),
    ):
        if not (snapshot / "measurements" / "transports").exists():
            continue
        cells = derive_snapshot(snapshot)
        for feature in features:
            assert cells.get(feature), f"{snapshot.name}: {feature} derives no cells"


def test_sse_prefix_selects_the_legacy_transport() -> None:
    handle = parse_server("sse+http://127.0.0.1:9/sse")
    assert (handle.transport, handle.url) == ("sse", "http://127.0.0.1:9/sse")
    assert parse_server("http://127.0.0.1:9/mcp").transport == "http"


CLIENT = r"""
import json, sys, anyio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    async with sse_client(sys.argv[1]) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = await s.list_tools()
            result = await s.call_tool("search_records", {"query": "x"})
            print(json.dumps({"tools": len(tools.tools), "text": result.content[0].text}))

anyio.run(main)
"""


def test_legacy_sse_bridge_serves_and_records_a_session(tmp_path: Path) -> None:
    capture = tmp_path / "capture.jsonl"
    bridge = subprocess.Popen(
        [sys.executable, "-m", "mcp_mirror.legacy_sse", f"{sys.executable} fixtures/definitions_server.py",
         "--out", str(capture)],
        cwd=ROOT, stderr=subprocess.PIPE, text=True,
    )
    try:
        url = None
        assert bridge.stderr is not None
        for _ in range(10):
            found = re.search(r"listening on (\S+)", bridge.stderr.readline())
            if found:
                url = found.group(1)
                break
        assert url and url.endswith("/sse")
        client = subprocess.run([sys.executable, "-c", CLIENT, url], capture_output=True, text=True, timeout=60, cwd=ROOT)
        assert client.returncode == 0, client.stderr[-800:]
        assert json.loads(client.stdout)["tools"] > 0
    finally:
        bridge.send_signal(signal.SIGTERM)
        bridge.wait(timeout=15)
    records = read_capture(capture)
    assert records[0]["transport"] == "sse"
    methods = [r["method"] for r in records if r["type"] == "message" and r["direction"] == "client_to_server"]
    assert methods[:1] == ["initialize"] and "tools/call" in methods
    assert any(r["type"] == "http" and r["event"] == "get_opened" for r in records)
    assert records[-1]["type"] == "end"



def test_the_legacy_sse_bridge_keeps_the_body_of_an_h2c_upgrade_post(tmp_path: Path) -> None:
    # The JDK HttpClient offers h2c on every plain-http request; httptools then drops the body.
    pytest.importorskip("httptools")
    capture = tmp_path / "capture.jsonl"
    bridge = subprocess.Popen(
        [sys.executable, "-m", "mcp_mirror.legacy_sse", f"{sys.executable} fixtures/definitions_server.py",
         "--out", str(capture)],
        cwd=ROOT, stderr=subprocess.PIPE, text=True,
    )
    try:
        url = None
        assert bridge.stderr is not None
        for _ in range(10):
            found = re.search(r"listening on (\S+)", bridge.stderr.readline())
            if found:
                url = found.group(1)
                break
        assert url
        target = urlsplit(url)
        stream = http.client.HTTPConnection(target.hostname, target.port, timeout=30)
        stream.request("GET", target.path, headers={"Accept": "text/event-stream"})
        events = stream.getresponse()
        endpoint = None
        while endpoint is None:
            line = events.fp.readline().decode()
            if line.startswith("data: "):
                endpoint = line[len("data: "):].strip()
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "jdk", "version": "0"}}})
        post = http.client.HTTPConnection(target.hostname, target.port, timeout=30)
        post.request("POST", endpoint, body=body, headers={
            "Connection": "Upgrade, HTTP2-Settings", "Upgrade": "h2c",
            "HTTP2-Settings": "AAEAAEAAAAIAAAAAAAMAAAAAAAQBAAAAAAUAAEAAAAYABgAA",
            "Content-Type": "application/json"})
        status = post.getresponse().status
        post.close()
        stream.close()
    finally:
        bridge.send_signal(signal.SIGTERM)
        bridge.wait(timeout=15)
    records = read_capture(capture)
    assert status == 202
    assert not [r for r in records if r["type"] == "http" and r["event"] == "parse_error"]
    methods = [r["method"] for r in records if r["type"] == "message" and r["direction"] == "client_to_server"]
    assert methods == ["initialize"]

def test_ids_an_adapter_mints_per_block_are_not_a_transport_difference(tmp_path: Path) -> None:
    def minted(uuid: str) -> dict:
        answers = _answers()
        answers["results"][0]["rep"]["blocks"] = [{"type": "text", "text": "ok", "id": f"lc_{uuid}"}]
        return answers

    _write(tmp_path, {"langchain": {"transports": {
        "stdio": {"results": _run(answers=minted("d6c0a05c-d5df-43ab-b342-eee1e83ac714"))},
        "http": {"results": _run(answers=minted("c1a7d87e-8b80-43f6-821d-c3e86c33cb5c"))},
    }}}, {})
    assert derive_transport_cells(tmp_path)["transport-streamable-http"]["langchain"].code == "y"
