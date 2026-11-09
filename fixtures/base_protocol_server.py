"""The base-protocol fixture: tools whose wire exchange answers the base-protocol rows.

Most base-protocol behaviour is visible in any session's capture (the handshake, the
per-request ``_meta``, whether the client answers a ping), so the tools here exist to
give the capture something specific to look at:

- ``dialect_2020_12``   declares ``$schema`` 2020-12 and uses keywords only that
                        dialect has (``$defs``, ``prefixItems``, ``dependentRequired``)
- ``dialect_draft_07``  declares ``$schema`` draft-07 and uses its spellings
                        (``definitions``, tuple ``items``, ``additionalItems``)
- ``routed_lookup``     marks a string parameter ``x-mcp-header: Region``, which a
                        2026-07-28 client over Streamable HTTP mirrors into an
                        ``Mcp-Param-Region`` header
- ``bad_header``        marks an object parameter ``x-mcp-header``, which the spec
                        forbids; a conforming client leaves the tool out
- ``needs_input``       at 2026-07-28 returns an ``InputRequiredResult`` until the
                        client retries with ``inputResponses``; earlier it answers

``--omit`` leaves tools out, so an adapter that cannot load one of them can still be
measured on the rest. Every tool that completes echoes its arguments as JSON text, so the result shows what
reached the server.

    python fixtures/base_protocol_server.py [--omit TOOL ...]
    mcp-mirror record "python fixtures/base_protocol_server.py" --ping
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mcp.types as types

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dual import build, run_stdio, text_result  # noqa: E402

DIALECT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
DIALECT_DRAFT_07 = "http://json-schema.org/draft-07/schema#"
HEADER_NAME = "Region"
INPUT_KEY = "confirm"
INPUT_MESSAGE = "SENTINEL_BASE_PROTOCOL_INPUT: confirm the lookup."

TOOLS: list[types.Tool] = [
    types.Tool(
        name="dialect_2020_12",
        description="Plot a point inside a window. The schema is JSON Schema 2020-12.",
        inputSchema={
            "$schema": DIALECT_2020_12,
            "type": "object",
            "$defs": {
                "range": {
                    "type": "object",
                    "properties": {"low": {"type": "number"}, "high": {"type": "number"}},
                    "required": ["low", "high"],
                }
            },
            "properties": {
                "point": {
                    "type": "array",
                    "description": "An [x, y] pair.",
                    "prefixItems": [{"type": "number"}, {"type": "number"}],
                    "items": False,
                },
                "window": {"$ref": "#/$defs/range"},
                "label": {"type": "string"},
                "units": {"type": "string"},
            },
            "dependentRequired": {"label": ["units"]},
            "required": ["point"],
        },
    ),
    types.Tool(
        name="dialect_draft_07",
        description="Plot a point inside a window. The schema is JSON Schema draft-07.",
        inputSchema={
            "$schema": DIALECT_DRAFT_07,
            "type": "object",
            "definitions": {
                "range": {
                    "type": "object",
                    "properties": {"low": {"type": "number"}, "high": {"type": "number"}},
                    "required": ["low", "high"],
                }
            },
            "properties": {
                "point": {
                    "type": "array",
                    "description": "An [x, y] pair.",
                    "items": [{"type": "number"}, {"type": "number"}],
                    "additionalItems": False,
                },
                "window": {"$ref": "#/definitions/range"},
            },
            "required": ["point"],
        },
    ),
    types.Tool(
        name="routed_lookup",
        description="Look up a record in one region. The region is routed as a header.",
        inputSchema={
            "type": "object",
            "properties": {
                "region": {"type": "string", "x-mcp-header": HEADER_NAME},
                "key": {"type": "string"},
            },
            "required": ["region", "key"],
        },
    ),
    types.Tool(
        name="bad_header",
        description="A tool whose x-mcp-header annotation sits on an object, which the spec forbids.",
        inputSchema={
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "x-mcp-header": "Filter",
                    "properties": {"field": {"type": "string"}},
                }
            },
        },
    ),
    types.Tool(
        name="needs_input",
        description="Look something up after the user confirms.",
        inputSchema={"type": "object", "properties": {}},
    ),
]

# What each measurement calls, with arguments valid under every dialect involved.
CALLS: list[dict] = [
    {"tool": "dialect_2020_12", "arguments": {"point": [1, 2], "window": {"low": 0, "high": 5}}},
    {"tool": "dialect_draft_07", "arguments": {"point": [1, 2], "window": {"low": 0, "high": 5}}},
    {"tool": "routed_lookup", "arguments": {"region": "eu-west-1", "key": "k1"}},
    {"tool": "needs_input", "arguments": {}},
]


def _input_request():
    return types.ElicitRequest(
        params=types.ElicitRequestFormParams(
            message=INPUT_MESSAGE,
            requested_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        )
    )


async def _call_tool(call):
    name = call.params.name
    if name not in {tool.name for tool in TOOLS}:
        return text_result(f"unknown tool: {name}", is_error=True)
    if name == "needs_input" and call.modern:
        responses = call.input_responses
        if INPUT_KEY not in responses:
            return types.InputRequiredResult(input_requests={INPUT_KEY: _input_request()})
        return text_result(f"SENTINEL_INPUT_ANSWERED {json.dumps(responses[INPUT_KEY], sort_keys=True)}")
    return text_result(f"SENTINEL_ARGUMENTS {name} {json.dumps(call.arguments, sort_keys=True)}")


async def _list_tools(_call) -> types.ListToolsResult:
    return types.ListToolsResult(tools=TOOLS)


def build_server():
    return build("base-protocol-mcp", {"tools/list": _list_tools, "tools/call": _call_tool})


if __name__ == "__main__":
    omitted = {sys.argv[i + 1] for i, arg in enumerate(sys.argv[:-1]) if arg == "--omit"}
    TOOLS[:] = [tool for tool in TOOLS if tool.name not in omitted]
    run_stdio(build_server)
