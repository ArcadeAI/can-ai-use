"""Client features: the fixture's tool loop and roots counter, and how captures derive cells."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from mcp_mirror import client_probe as C
from mcp_mirror.cells import derive_client_cells

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "client_features_server.py"


def _text(result) -> str:
    return "".join(block.text for block in result.content if getattr(block, "type", None) == "text")


def test_fixture_runs_the_sampling_tool_loop_and_counts_roots_changes():
    log = C.HandlerLog()

    async def body():
        params = StdioServerParameters(command=sys.executable, args=[str(FIXTURE)], cwd=str(ROOT))
        async with stdio_client(params) as (read, write):
            async with ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=30),
                sampling_capabilities=types.SamplingCapability(tools=types.SamplingToolsCapability()),
                **C.sdk_callbacks(log),
            ) as session:
                await session.initialize()
                loop = await session.call_tool("ask_sampling_tools", {})
                before = await session.call_tool("roots_changes_seen", {})
                await session.send_roots_list_changed()
                with anyio.fail_after(5):
                    while "SEEN 0" in _text(after := await session.call_tool("roots_changes_seen", {})):
                        await anyio.sleep(0.05)
                declared = await session.call_tool("client_capabilities", {})
                return loop, before, after, declared

    loop, before, after, declared = anyio.run(body)
    assert C.SAMPLED_AFTER_TOOL in _text(loop)
    assert [s["replied"] for s in log.seen if s["hook"] == "sampling_callback"] == ["toolUse", "endTurn"]
    assert _text(before) == "SENTINEL_ROOTS_CHANGES_SEEN 0"
    assert _text(after) == "SENTINEL_ROOTS_CHANGES_SEEN 1"
    assert json.loads(_text(declared).split(" ", 1)[1])["sampling"] == {"tools": {}}


def test_fixture_refuses_sampling_tools_to_a_client_that_did_not_declare_them():
    async def body():
        params = StdioServerParameters(command=sys.executable, args=[str(FIXTURE)], cwd=str(ROOT))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, **C.sdk_callbacks(C.HandlerLog())) as session:
                await session.initialize()
                return await session.call_tool("ask_sampling_tools", {})

    result = anyio.run(body)
    assert result.isError is True
    assert _text(result).startswith("SENTINEL_CLIENT_FEATURE_UNAVAILABLE ask_sampling_tools")


def _entry(feature: str, text: str, hook: str | None = None, route: str | None = None) -> dict:
    tool, expect = C.FEATURES[feature]
    return {"tool": tool, "hook": hook, "route": route, "expect": expect, "text": text, "handler_saw": []}


def _write(snapshot: Path, frameworks: dict) -> None:
    path = snapshot / "measurements" / "client" / "all-frameworks.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"frameworks": frameworks}))


def test_captures_derive_by_route_and_answer(tmp_path: Path) -> None:
    unanswered = "SENTINEL_CLIENT_FEATURE_UNAVAILABLE ask_sampling: McpError: Method not found"
    _write(tmp_path, {
        "langchain": {"ok": True, "negotiated_mcp_spec_version": "2025-11-25", "client_probe": {
            "sampling": _entry("sampling", "SENTINEL_CLIENT_ANSWERED ask_sampling: SENTINEL_SAMPLED", "session_kwargs", "passthrough"),
            "sampling.tools": _entry("sampling.tools", 'SENTINEL_CLIENT_ANSWERED ask_sampling_tools: SENTINEL_SAMPLED', "session_kwargs", "passthrough"),
            "elicitation.form": _entry("elicitation.form", 'SENTINEL_CLIENT_ANSWERED x: {"name": "SENTINEL_ELICITED"}', "Callbacks.on_elicitation", "adapter"),
            "roots.list_changed": _entry("roots.list_changed", "SENTINEL_ROOTS_CHANGES_SEEN 0"),
        }},
        "crewai": {"ok": True, "negotiated_mcp_spec_version": "2025-11-25", "client_probe": {
            "sampling": _entry("sampling", unanswered),
            "elicitation.url": _entry("elicitation.url", "some adapter error that hides the cause"),
        }},
        "pydantic_ai": {"ok": True, "negotiated_mcp_spec_version": "2025-11-25", "client_probe": {
            "roots.list_changed": _entry("roots.list_changed", '{"seen": 1}', "client.send_roots_list_changed", "underlying"),
            "roots.list": _entry("roots.list", "ValueError: boom", "MCPToolset(roots=...)", "adapter"),
        }},
    })

    cells = derive_client_cells(tmp_path)

    assert cells["sampling-create-message"]["langchain"].code == "a"
    # One turn answered, the loop's second turn never produced its text.
    assert cells["sampling-tools"]["langchain"].code == "a"
    assert "without SENTINEL_SAMPLED_AFTER_TOOL" in cells["sampling-tools"]["langchain"].evidence[0]
    assert cells["elicitation-form-mode"]["langchain"].code == "y"
    assert cells["roots-list-changed"]["langchain"].code == "n"
    assert cells["sampling-create-message"]["crewai"].code == "n"
    # No hook registered: unanswered whatever the adapter says.
    assert cells["elicitation-url-mode"]["crewai"].code == "n"
    assert cells["roots-list"]["crewai"].code == "u"
    assert cells["roots-list-changed"]["pydantic-ai"].code == "a"
    # A hook that raised for an unknown reason is not judged.
    assert cells["roots-list"]["pydantic-ai"].code == "u"
    assert "mrtr-input-required" not in cells


def test_modern_captures_derive_mrtr_and_drop_the_removed_notification(tmp_path: Path) -> None:
    answered_form = _entry("elicitation.form", '{"answered": "ask_elicitation_form", "response": {"content": {"name": "SENTINEL_ELICITED"}}}', "inputRequests", "adapter")
    answered_url = _entry("elicitation.url", '{"answered": "ask_elicitation_url", "response": {"action": "accept"}}', "inputRequests", "adapter")
    refused = "RuntimeError: Server returned InputRequiredResult; pass allow_input_required=True"
    _write(tmp_path, {
        "mastra": {"ok": True, "negotiated_mcp_spec_version": "2026-07-28", "client_probe": {
            "elicitation.form": answered_form,
            "elicitation.url": answered_url,
            "roots.list_changed": _entry("roots.list_changed", '{"seen": 0}'),
        }},
        "openai_agents": {"ok": True, "negotiated_mcp_spec_version": "2026-07-28", "client_probe": {
            "elicitation.form": _entry("elicitation.form", refused),
            "elicitation.url": _entry("elicitation.url", refused),
        }},
    })

    cells = derive_client_cells(tmp_path)

    assert cells["elicitation-url-mode"]["mastra"].code == "y"
    assert cells["mrtr-input-required"]["mastra"].code == "y"
    assert cells["mrtr-input-required"]["openai-agents"].code == "n"
    assert "roots-list-changed" not in cells
