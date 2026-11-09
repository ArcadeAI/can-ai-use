"""The recorder captures a real client session on both Python SDK lines.

The recorder always runs under the project interpreter. The fixture and the client
each run on the 1.x SDK (project venv) or the 2.x SDK (``.mcp-mirror/fixture-2026-07-28``,
built by ``scripts/measure_results.py``; cases that need it are skipped when it is
absent). Every case drives initialize (or its 2026-07-28 replacement), tools/list and
one tools/call through the recorder, over stdio and over streamable HTTP, and checks
the capture against what the client itself reports it negotiated.
"""

from __future__ import annotations

import http.client
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from mcp_mirror.recorder import PING_ID, default_capture_path, messages, read_capture

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "fixtures/definitions_server.py"
SDK1_PYTHON = sys.executable
SDK2_PYTHON = ROOT / ".mcp-mirror" / "fixture-2026-07-28" / "bin" / "python"
needs_sdk2 = pytest.mark.skipif(not SDK2_PYTHON.exists(), reason="2.x fixture env not built")

SDK1_CLIENT = r"""
import json, sys, anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

target, argv = sys.argv[1], json.loads(sys.argv[2])

async def session(read, write):
    async with ClientSession(read, write) as s:
        init = await s.initialize()
        tools = await s.list_tools()
        result = await s.call_tool("search_records", {"query": "x"})
        await anyio.sleep(0.3)  # let the injected ping round-trip before closing
        return {"version": init.protocolVersion, "tools": len(tools.tools), "text": result.content[0].text}

async def main():
    if target == "stdio":
        params = StdioServerParameters(command=argv[0], args=argv[1:])
        async with stdio_client(params) as (read, write):
            out = await session(read, write)
    else:
        async with streamablehttp_client(target) as (read, write, _):
            out = await session(read, write)
    print(json.dumps(out))

anyio.run(main)
"""

SDK2_CLIENT = r"""
import json, sys, warnings, anyio
warnings.simplefilter("ignore")
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

target, argv, mode = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]

async def main():
    server = stdio_client(StdioServerParameters(command=argv[0], args=argv[1:])) if target == "stdio" else target
    async with Client(server, mode=mode) as client:
        tools = await client.list_tools()
        result = await client.call_tool("search_records", {"query": "x"})
        await anyio.sleep(0.3)
        print(json.dumps({"version": client.protocol_version, "tools": len(tools.tools),
                          "text": result.content[0].text}))

anyio.run(main)
"""

# (client line, client mode, fixture line, how the revision is established)
CASES = [
    pytest.param("sdk1", None, "sdk1", "initialize", id="sdk1-client"),
    pytest.param("sdk2", "legacy", "sdk2", "initialize", id="sdk2-client-legacy", marks=needs_sdk2),
    pytest.param("sdk2", "auto", "sdk1", "initialize", id="sdk2-client-auto-falls-back", marks=needs_sdk2),
    pytest.param("sdk2", "auto", "sdk2", "server/discover", id="sdk2-client-auto-modern", marks=needs_sdk2),
]


def _python(line: str) -> str:
    return SDK1_PYTHON if line == "sdk1" else str(SDK2_PYTHON)


def _recorder_argv(fixture_line: str, out: Path, *extra: str) -> list[str]:
    return [SDK1_PYTHON, "-m", "mcp_mirror.cli", "record", f"{_python(fixture_line)} {FIXTURE}",
            "--out", str(out), "--ping", *extra]


def _drive(client_line: str, mode: str | None, target: str, recorder: list[str]) -> dict:
    script = SDK1_CLIENT if client_line == "sdk1" else SDK2_CLIENT
    args = [_python(client_line), "-c", script, target, json.dumps(recorder)]
    if mode:
        args.append(mode)
    proc = subprocess.run(args, capture_output=True, text=True, timeout=120, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _run_stdio(client_line, mode, fixture_line, out: Path) -> dict:
    return _drive(client_line, mode, "stdio", _recorder_argv(fixture_line, out))


def _run_http(client_line, mode, fixture_line, out: Path) -> dict:
    recorder = subprocess.Popen(_recorder_argv(fixture_line, out, "--transport", "http", "--port", "0"),
                                stderr=subprocess.PIPE, text=True, cwd=str(ROOT))
    try:
        url = None
        deadline = time.monotonic() + 30
        while url is None and time.monotonic() < deadline:
            line = recorder.stderr.readline()
            if not line:
                break
            if "listening on " in line:
                url = line.split("listening on ", 1)[1].strip()
        assert url, "recorder never announced its URL"
        return _drive(client_line, mode, url, [])
    finally:
        recorder.terminate()
        recorder.wait(timeout=15)
        recorder.stderr.close()


def _requests(records: list[dict]) -> list[str]:
    return [r["method"] for r in messages(records)
            if r["direction"] == "client_to_server" and r["kind"] == "request" and r["origin"] == "client"]


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize(("client_line", "mode", "fixture_line", "via"), CASES)
def test_capture_lists_the_session(tmp_path, transport, client_line, mode, fixture_line, via):
    out = tmp_path / "capture.jsonl"
    run = _run_stdio if transport == "stdio" else _run_http
    observed = run(client_line, mode, fixture_line, out)
    assert observed["tools"] == 5 and "SENTINEL_TEXT_BLOCK" in observed["text"]

    records = read_capture(out)
    assert records[0]["type"] == "capture" and records[0]["transport"] == transport
    assert records[-1]["type"] == "end"
    assert [r["seq"] for r in records] == list(range(len(records)))

    requests = _requests(records)
    assert via in requests
    assert requests.count("tools/list") == 1
    assert requests.count("tools/call") == 1
    assert requests.index(via) < requests.index("tools/list") < requests.index("tools/call")

    # The revision the capture says was negotiated is the one the client reports.
    negotiated = [r for r in records if r["type"] == "negotiation" and r["negotiated_revision"]]
    assert negotiated, "no negotiation record"
    final = negotiated[-1]
    assert final["negotiated_revision"] == observed["version"]
    assert final["via"] == ("initialize" if via == "initialize" else "request-meta")
    assert isinstance(final["client_capabilities"], dict)

    # Every tools/* message, both directions, carries that revision and the method.
    tool_messages = [r for r in messages(records) if r["method"] in ("tools/list", "tools/call")]
    assert {(r["direction"], r["method"]) for r in tool_messages} == {
        (direction, method)
        for direction in ("client_to_server", "server_to_client")
        for method in ("tools/list", "tools/call")
    }
    assert {r["revision"] for r in tool_messages} == {observed["version"]}
    call = next(r for r in tool_messages if r["direction"] == "client_to_server" and r["method"] == "tools/call")
    assert call["message"]["params"]["name"] == "search_records"

    if via == "initialize":
        # --ping injects one server-initiated ping after notifications/initialized;
        # the client's answer is recorded and never reaches the fixture.
        pings = [r for r in messages(records, "ping")]
        assert [(r["direction"], r["origin"], r["kind"]) for r in pings] == [
            ("server_to_client", "recorder", "request"),
            ("client_to_server", "client", "response"),
        ]
        assert pings[1]["id"] == PING_ID

    if transport == "http":
        posts = [r for r in messages(records) if r["direction"] == "client_to_server"]
        assert all(r["http"]["method"] == "POST" for r in posts)
        assert not any("authorization" in {k.lower() for k in r["http"]["headers"]} for r in posts)
        if via == "initialize":
            session = final["session"]
            assert session != "stateless"
            after = [r for r in posts if r["session"] == session and r["method"] != "initialize"]
            assert after and all(r["http"]["headers"].get("mcp-session-id") == session for r in after)


def test_default_capture_path_names_the_fixture():
    path = default_capture_path("claude_code", "python fixtures/results_server.py", "http")
    assert path.parts[:3] == ("measurements", "wire", "claude_code")
    assert path.name.endswith("-results_server-http.jsonl")


def test_record_refuses_a_url(tmp_path):
    proc = subprocess.run(
        [SDK1_PYTHON, "-m", "mcp_mirror.cli", "record", "http://127.0.0.1:1/mcp", "--out", str(tmp_path / "c.jsonl")],
        capture_output=True, text=True, timeout=60, cwd=str(ROOT),
    )
    assert proc.returncode == 2
    assert "not a URL" in " ".join(proc.stderr.split())


# What the JDK HttpClient puts on every plain-http request: an offer to upgrade to h2c.
H2C_UPGRADE = {
    "Connection": "Upgrade, HTTP2-Settings",
    "Upgrade": "h2c",
    "HTTP2-Settings": "AAEAAEAAAAIAAAAAAAMAAAAAAAQBAAAAAAUAAEAAAAYABgAA",
}


def test_an_h2c_upgrade_post_keeps_its_body(tmp_path):
    # httptools is the parser that drops the body; it comes in with the crewai extra.
    pytest.importorskip("httptools")
    out = tmp_path / "capture.jsonl"
    recorder = subprocess.Popen(_recorder_argv("sdk1", out, "--transport", "http", "--port", "0"),
                                stderr=subprocess.PIPE, text=True, cwd=str(ROOT))
    try:
        url = None
        deadline = time.monotonic() + 30
        while url is None and time.monotonic() < deadline:
            line = recorder.stderr.readline()
            if not line:
                break
            if "listening on " in line:
                url = line.split("listening on ", 1)[1].strip()
        assert url, "recorder never announced its URL"
        target = urlsplit(url)
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "jdk", "version": "0"}}})
        conn = http.client.HTTPConnection(target.hostname, target.port, timeout=30)
        conn.request("POST", target.path, body=body, headers={
            **H2C_UPGRADE, "Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
        status = conn.getresponse().status
        conn.close()
    finally:
        recorder.terminate()
        recorder.wait(timeout=15)
        recorder.stderr.close()

    records = read_capture(out)
    assert status == 200
    assert not [r for r in records if r["type"] == "http" and r["event"] == "parse_error"]
    assert _requests(records) == ["initialize"]
