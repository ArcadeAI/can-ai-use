"""canaiuse.com/mcp spoken by the official MCP Python client, against `wrangler dev`.

The Worker serves the built site in docs/ and the MCP endpoint at /mcp. The test starts
wrangler's local runtime (no Cloudflare account involved), drives initialize, tools/list
and one call of each tool through mcp's streamablehttp_client, and checks that the site's
own URLs still serve the files in docs/.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[1]
WRANGLER = ROOT / "node_modules" / ".bin" / "wrangler"

pytestmark = pytest.mark.skipif(not WRANGLER.exists(), reason="wrangler is not installed (npm ci)")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


@pytest.fixture(scope="module")
def site_url():
    port = _free_port()
    env = {**os.environ, "CI": "1", "WRANGLER_SEND_METRICS": "false", "WRANGLER_LOG": "error"}
    process = subprocess.Popen(
        [str(WRANGLER), "dev", "--port", str(port), "--ip", "127.0.0.1", "--show-interactive-dev-session=false"],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 90
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"wrangler dev exited early:\n{process.stdout.read()}")
            try:
                if _get(f"{url}/data/specs.json")[0] == 200:
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                process.kill()
                raise RuntimeError(f"wrangler dev did not come up:\n{process.stdout.read()}")
            time.sleep(0.5)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def test_the_site_still_serves_from_docs(site_url: str) -> None:
    index = (ROOT / "docs" / "index.html").read_bytes()
    # The site is hash-routed, so /#/cap/destructive-hint is / to the server; /harnesses is a
    # path the single-page-application fallback must answer with the index too.
    for path in ("/", "/harnesses", "/index.html"):
        status, body = _get(f"{site_url}{path}")
        assert status == 200, path
        assert body == index, path
    status, body = _get(f"{site_url}/data/specs.json")
    assert status == 200
    assert body == (ROOT / "docs" / "data" / "specs.json").read_bytes()


def test_get_on_the_endpoint_is_405_with_allow(site_url: str) -> None:
    request = urllib.request.Request(f"{site_url}/mcp", method="GET")
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(request, timeout=10)
    assert raised.value.code == 405
    assert raised.value.headers["Allow"] == "POST, OPTIONS"


def test_the_official_client_can_use_every_tool(site_url: str) -> None:
    async def drive() -> dict:
        async with streamable_http_client(f"{site_url}/mcp") as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                tools = (await session.list_tools()).tools
                calls = {
                    "list_snapshots": {},
                    "search_capabilities": {"query": "destructive"},
                    "lookup_capability": {"capability_id": "destructive-hint", "frameworks": ["langchain"]},
                    "inspect_tool_definition": {
                        "tool": {
                            "name": "delete_account",
                            "description": "Permanently deletes an account.",
                            "inputSchema": {"type": "object", "properties": {}, "required": []},
                            "annotations": {"destructiveHint": True},
                        },
                        "revision": "latest",
                    },
                    "inspect_tool_result": {"result": {"content": [{"type": "text", "text": "ok"}], "isError": True}},
                }
                results = {name: await session.call_tool(name, args) for name, args in calls.items()}
                failed = await session.call_tool("lookup_capability", {"capability_id": "no-such-id"})
                return {"initialized": initialized, "tools": tools, "results": results, "failed": failed}

    outcome = anyio.run(drive)

    assert outcome["initialized"].protocolVersion in ("2025-11-25", "2025-06-18")
    assert outcome["initialized"].serverInfo.name == "canaiuse"
    assert [tool.name for tool in outcome["tools"]] == [
        "list_snapshots",
        "search_capabilities",
        "lookup_capability",
        "inspect_tool_definition",
        "inspect_tool_result",
    ]

    for name, result in outcome["results"].items():
        assert not result.isError, name
        assert result.structuredContent is not None, name
        assert json.loads(result.content[0].text) == result.structuredContent, name

    results = outcome["results"]
    assert [row["revision"] for row in results["list_snapshots"].structuredContent["snapshots"]] == ["2026-07-28", "2025-11-25"]
    assert results["search_capabilities"].structuredContent["matches"][0]["id"] == "destructive-hint"
    assert [row["framework_id"] for row in results["lookup_capability"].structuredContent["observations"]] == ["langchain"]
    assert "destructive-hint" in results["inspect_tool_definition"].structuredContent["identified_capabilities"]
    assert results["inspect_tool_definition"].structuredContent["revision"] == "2026-07-28"
    assert "tool-execution-errors" in results["inspect_tool_result"].structuredContent["identified_capabilities"]

    assert outcome["failed"].isError
    assert "unknown capability 'no-such-id'" in outcome["failed"].content[0].text
