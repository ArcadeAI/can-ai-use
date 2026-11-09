"""The split fixtures answer what their docstrings promise, on both SDK lines.

Each server is driven by a real 1.x ``ClientSession`` over stdio. The server runs under
the project interpreter (1.x) and, when ``.mcp-mirror/fixture-2026-07-28`` has been
built, under the 2.x SDK too, which exercises the other half of ``fixtures/_dual.py``.
The 2026-07-28 wire itself (input requests, listen streams) is covered by
``test_modern_wire`` with the 2.x client, skipped when that interpreter is absent.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
SDK2_PYTHON = ROOT / ".mcp-mirror" / "fixture-2026-07-28" / "bin" / "python"

INTERPRETERS = [pytest.param(sys.executable, id="sdk1")] + (
    [pytest.param(str(SDK2_PYTHON), id="sdk2")]
    if SDK2_PYTHON.exists()
    else [pytest.param(str(SDK2_PYTHON), id="sdk2", marks=pytest.mark.skip("2.x fixture env not built"))]
)


async def _with_session(python: str, fixture: str, body, **client):
    params = StdioServerParameters(command=python, args=[str(FIXTURES / fixture)], cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30), **client) as session:
            init = await session.initialize()
            return await body(session, init)


def _run(python, fixture, body, **client):
    return anyio.run(lambda: _with_session(python, fixture, body, **client))


def _text(result) -> str:
    return "".join(block.text for block in result.content if getattr(block, "type", None) == "text")


def test_tricky_server_is_the_definitions_server():
    async def body(session, _init):
        return [tool.name for tool in (await session.list_tools()).tools]

    alias = _run(sys.executable, "tricky_server.py", body)
    definitions = _run(sys.executable, "definitions_server.py", body)
    assert alias == definitions
    assert "search_records" in alias


@pytest.mark.parametrize("python", INTERPRETERS)
def test_results_server_returns_one_kind_per_tool(python):
    expected = {
        "text_content": ["text"],
        "image_content": ["image"],
        "audio_content": ["audio"],
        "embedded_resource_content": ["resource"],
        "resource_link_content": ["resource_link"],
        "annotated_content": ["text"],
        "mixed_content": ["text", "image", "audio", "resource", "resource_link"],
        "text_and_structured": ["text"],
        "tool_error": ["text"],
    }

    async def body(session, _init):
        out = {}
        for tool in expected:
            result = await session.call_tool(tool, {})
            out[tool] = result
        return out

    results = _run(python, "results_server.py", body)
    for tool, kinds in expected.items():
        assert [block.type for block in results[tool].content] == kinds, tool
    assert results["annotated_content"].content[0].annotations.audience == ["user"]
    assert results["text_and_structured"].structuredContent is not None
    assert results["mixed_content"].structuredContent is not None
    assert results["text_content"].structuredContent is None
    assert results["tool_error"].isError is True


@pytest.mark.parametrize("python", INTERPRETERS)
def test_notify_server_fires_list_changed_and_updated(python):
    seen: list[str] = []

    async def on_message(message):
        if isinstance(message, types.ServerNotification):
            seen.append(message.root.method)

    async def body(session, init):
        assert init.capabilities.tools.listChanged
        assert init.capabilities.resources.subscribe
        before = {t.name for t in (await session.list_tools()).tools}
        await session.call_tool("add_tool", {})
        after = {t.name for t in (await session.list_tools()).tools}
        await session.call_tool("add_resource", {})
        await session.call_tool("add_prompt", {})
        await session.subscribe_resource("file:///fixtures/watched.txt")
        await session.call_tool("update_resource", {})
        read = await session.read_resource("file:///fixtures/watched.txt")
        await session.call_tool("remove_tool", {})
        await anyio.sleep(0.2)
        return before, after, read

    before, after, read = _run(python, "notify_server.py", body, message_handler=on_message)
    assert "SENTINEL_ADDED_TOOL" not in before
    assert "SENTINEL_ADDED_TOOL" in after
    assert read.contents[0].text == "SENTINEL_RESOURCE_VERSION 1"
    assert seen.count("notifications/tools/list_changed") == 2
    assert "notifications/resources/list_changed" in seen
    assert "notifications/prompts/list_changed" in seen
    assert "notifications/resources/updated" in seen


@pytest.mark.parametrize("python", INTERPRETERS)
def test_client_features_server_asks_the_client(python):
    asked: list[str] = []

    async def sampling(_context, params):
        asked.append("sampling")
        return types.CreateMessageResult(
            role="assistant", content=types.TextContent(type="text", text="SENTINEL_SAMPLED"), model="fixture"
        )

    async def elicitation(_context, params):
        asked.append(f"elicitation:{params.mode}")
        if params.mode == "url":
            return types.ElicitResult(action="accept")
        return types.ElicitResult(action="accept", content={"name": "SENTINEL_NAME"})

    async def roots(_context):
        asked.append("roots")
        return types.ListRootsResult(roots=[types.Root(uri="file:///SENTINEL_ROOT", name="root")])

    async def body(session, _init):
        return {
            tool: await session.call_tool(tool, {})
            for tool in ("ask_sampling", "ask_elicitation_form", "ask_elicitation_url", "list_client_roots")
        }

    results = _run(
        python,
        "client_features_server.py",
        body,
        sampling_callback=sampling,
        elicitation_callback=elicitation,
        list_roots_callback=roots,
    )
    assert asked == ["sampling", "elicitation:form", "elicitation:url", "roots"]
    assert "SENTINEL_SAMPLED" in _text(results["ask_sampling"])
    assert "SENTINEL_NAME" in _text(results["ask_elicitation_form"])
    assert '"action": "accept"' in _text(results["ask_elicitation_url"])
    assert "SENTINEL_ROOT" in _text(results["list_client_roots"])
    assert not any(result.isError for result in results.values())


@pytest.mark.parametrize("python", INTERPRETERS)
def test_client_features_server_reports_an_unanswered_request(python):
    async def body(session, _init):
        return await session.call_tool("list_client_roots", {})

    result = _run(python, "client_features_server.py", body)
    assert result.isError is True
    assert _text(result).startswith("SENTINEL_CLIENT_FEATURE_UNAVAILABLE")


@pytest.mark.parametrize("python", INTERPRETERS)
def test_utilities_server(python):
    progress: list[float] = []
    logs: list[str] = []

    async def on_progress(value, total, message):
        progress.append(value)

    async def on_log(params):
        logs.append(params.data["message"])

    async def body(session, init):
        assert init.capabilities.logging is not None
        assert init.capabilities.completions is not None
        pages, cursor, names = 0, None, []
        while True:
            listed = await session.list_tools(cursor=cursor) if cursor else await session.list_tools()
            pages += 1
            names += [tool.name for tool in listed.tools]
            cursor = listed.nextCursor
            if not cursor:
                break
        first_resources = await session.list_resources()
        first_prompts = await session.list_prompts()
        slow = await session.call_tool("slow_with_progress", {}, progress_callback=on_progress)
        await session.set_logging_level("warning")
        emitted = await session.call_tool("emit_logs", {})
        colour = await session.complete(
            types.PromptReference(type="ref/prompt", name="pick_colour"), {"name": "colour", "value": "c"}
        )
        record = await session.complete(
            types.ResourceTemplateReference(type="ref/resource", uri="file:///fixtures/records/{record_id}"),
            {"name": "record_id", "value": "rec_1"},
        )
        return pages, names, first_resources, first_prompts, slow, emitted, colour, record

    pages, names, resources, prompts, slow, emitted, colour, record = _run(
        python, "utilities_server.py", body, logging_callback=on_log
    )
    assert pages == 3 and names[-1] == "SENTINEL_LAST_TOOL" and len(names) == 60
    assert resources.nextCursor and len(resources.resources) == 20
    assert prompts.nextCursor and len(prompts.prompts) == 20
    assert progress == [1, 2, 3, 4, 5]
    assert "progress_sent=5" in _text(slow)
    assert logs == ["SENTINEL_LOG_WARNING", "SENTINEL_LOG_ERROR"]
    assert "SENTINEL_LOGS_SENT 2" in _text(emitted)
    assert colour.completion.values == ["crimson", "cyan", "cobalt"]
    assert record.completion.values == ["rec_1", "rec_10", "rec_11"]


@pytest.mark.parametrize("python", INTERPRETERS)
def test_utilities_server_stops_a_cancelled_call(python):
    """Raw JSON-RPC, because the 1.x ClientSession never sends notifications/cancelled."""

    proc = subprocess.Popen(
        [python, str(FIXTURES / "utilities_server.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=str(ROOT),
    )

    def send(message: dict) -> None:
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        proc.stdin.flush()

    try:
        send({"id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
        assert json.loads(proc.stdout.readline())["id"] == 1
        send({"method": "notifications/initialized"})
        send({"id": 2, "method": "tools/call", "params": {"name": "cancellable_wait", "arguments": {"seconds": 2}}})
        send({"method": "notifications/cancelled", "params": {"requestId": 2, "reason": "test"}})
        send({"id": 3, "method": "ping"})
        replies = []
        while True:
            reply = json.loads(proc.stdout.readline())
            replies.append(reply)
            if reply.get("id") == 3:
                break
        # Past the wait, so an uncancelled call would have answered by now.
        time.sleep(2.5)
        send({"id": 4, "method": "ping"})
        while (reply := json.loads(proc.stdout.readline())).get("id") != 4:
            replies.append(reply)
    finally:
        proc.kill()
    for reply in replies:
        # 1.x answers the cancelled id with an error; 2.x stays silent. Neither delivers the result.
        assert "SENTINEL_NOT_CANCELLED" not in json.dumps(reply)


MODERN_SCRIPT = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, sys.argv[1])
import anyio
from mcp import types
from mcp.client import Client
import notify_server, client_features_server, utilities_server

async def main():
    out = {}
    async with Client(notify_server.build_server()) as client:
        out["version"] = client.session.protocol_version if hasattr(client.session, "protocol_version") else None
        events = []
        async with client.listen(tools_list_changed=True, resource_subscriptions=[notify_server.WATCHED_URI]) as sub:
            r1 = await client.call_tool("add_tool", {})
            r2 = await client.call_tool("update_resource", {})
            with anyio.move_on_after(2):
                async for event in sub:
                    events.append(type(event).__name__)
                    if len(events) == 2:
                        break
        out["notify"] = {"events": events, "texts": [r1.content[0].text, r2.content[0].text],
                         "tools": [t.name for t in (await client.list_tools()).tools]}
    async def sampling(_ctx, params):
        return types.CreateMessageResult(role="assistant", content=types.TextContent(type="text", text="SENTINEL_SAMPLED"), model="m")
    async def roots(_ctx):
        return types.ListRootsResult(roots=[types.Root(uri="file:///SENTINEL_ROOT")])
    async with Client(client_features_server.build_server(), sampling_callback=sampling, list_roots_callback=roots) as client:
        out["sampling"] = (await client.call_tool("ask_sampling", {})).content[0].text
        out["roots"] = (await client.call_tool("list_client_roots", {})).content[0].text
    async with Client(utilities_server.build_server()) as client:
        listed = await client.list_tools()
        out["page"] = [len(listed.tools), listed.next_cursor]
    print(json.dumps(out))

anyio.run(main)
"""


@pytest.mark.skipif(not SDK2_PYTHON.exists(), reason="2.x fixture env not built")
def test_modern_wire():
    """At 2026-07-28: change events ride a listen stream, client requests become input requests."""

    proc = subprocess.run(
        [str(SDK2_PYTHON), "-c", MODERN_SCRIPT, str(FIXTURES)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    observed = json.loads(proc.stdout.strip().splitlines()[-1])
    assert observed["version"] == "2026-07-28"
    assert observed["notify"]["events"] == ["ToolsListChanged", "ResourceUpdated"]
    assert all("subscriptions/listen" in text for text in observed["notify"]["texts"])
    assert "SENTINEL_ADDED_TOOL" in observed["notify"]["tools"]
    assert "SENTINEL_SAMPLED" in observed["sampling"]
    assert "SENTINEL_ROOT" in observed["roots"]
    assert observed["page"] == [20, "20"]
