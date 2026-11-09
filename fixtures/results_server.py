"""The results fixture: one tool per content kind, so a kind is judged on its own.

``definitions_server.py`` returns every content block in one result. That observes
all of them at once, until an adapter raises on one: LangChain's audio conversion
aborts the whole call, so the image, the embedded resource, the resource link and the
annotations beside it were never seen. Each ``*_content`` tool here returns exactly one
block of one kind and nothing else, so a failure on one kind cannot hide another.

The mixed result, the tool-reported error and the text-plus-structured result are
kept as their own tools, because what an adapter does with a combination is a
separate finding from what it does with each part.

    python fixtures/results_server.py

Every tool takes no arguments, so a generated argument schema cannot fail the call
before the result is observed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mcp.types as types

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dual import build, run_stdio  # noqa: E402
from definitions_server import SENTINEL_STRUCTURED, TINY_PNG, TINY_WAV  # noqa: E402

NO_ARGUMENTS = {"type": "object", "properties": {}}


def _text() -> types.TextContent:
    return types.TextContent(type="text", text="SENTINEL_TEXT_BLOCK")


def _annotated_text() -> types.TextContent:
    # Audience and priority are the only content annotations the spec defines, and
    # they are what a client would use to decide what to show a user.
    return types.TextContent(
        type="text",
        text="SENTINEL_ANNOTATED_TEXT",
        annotations=types.Annotations(audience=["user"], priority=0.9),
    )


def _image() -> types.ImageContent:
    return types.ImageContent(type="image", data=TINY_PNG, mimeType="image/png")


def _audio() -> types.AudioContent:
    return types.AudioContent(type="audio", data=TINY_WAV, mimeType="audio/wav")


def _embedded() -> types.EmbeddedResource:
    return types.EmbeddedResource(
        type="resource",
        resource=types.TextResourceContents(
            uri="file:///fixtures/embedded.txt",
            mimeType="text/plain",
            text="SENTINEL_EMBEDDED_RESOURCE",
        ),
    )


def _link() -> types.ResourceLink:
    return types.ResourceLink(
        type="resource_link",
        uri="file:///fixtures/linked.txt",
        name="linked.txt",
        mimeType="text/plain",
        description="SENTINEL_RESOURCE_LINK",
    )


# name -> (description, content, structuredContent, isError)
RESULTS: dict[str, tuple[str, list, dict | None, bool]] = {
    "text_content": ("Return one text block and nothing else.", [_text()], None, False),
    "image_content": ("Return one image block and nothing else.", [_image()], None, False),
    "audio_content": ("Return one audio block and nothing else.", [_audio()], None, False),
    "embedded_resource_content": (
        "Return one embedded resource block and nothing else.",
        [_embedded()],
        None,
        False,
    ),
    "resource_link_content": (
        "Return one resource_link block and nothing else.",
        [_link()],
        None,
        False,
    ),
    "annotated_content": (
        "Return one text block carrying audience and priority annotations.",
        [_annotated_text()],
        None,
        False,
    ),
    "mixed_content": (
        "Return one block of every kind plus structuredContent in one result.",
        [
            types.TextContent(
                type="text",
                text="SENTINEL_TEXT_BLOCK",
                annotations=types.Annotations(audience=["user"], priority=0.9),
            ),
            _image(),
            _audio(),
            _embedded(),
            _link(),
        ],
        SENTINEL_STRUCTURED,
        False,
    ),
    "text_and_structured": (
        "Return one text block plus structuredContent, with no media.",
        [_text()],
        SENTINEL_STRUCTURED,
        False,
    ),
    "tool_error": (
        "Always fail, reporting the failure through isError rather than a protocol error.",
        [types.TextContent(type="text", text="SENTINEL_TOOL_ERROR: this tool always fails")],
        None,
        True,
    ),
}

TOOLS: list[types.Tool] = [
    types.Tool(name=name, description=description, inputSchema=NO_ARGUMENTS)
    for name, (description, _content, _structured, _error) in RESULTS.items()
]


async def _list_tools(_call) -> types.ListToolsResult:
    return types.ListToolsResult(tools=TOOLS)


async def _call_tool(call) -> types.CallToolResult:
    if call.params.name not in RESULTS:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"unknown tool: {call.params.name}")],
            isError=True,
        )
    _description, content, structured, is_error = RESULTS[call.params.name]
    return types.CallToolResult(content=content, structuredContent=structured, isError=is_error)


def build_server():
    return build("results-mcp", {"tools/list": _list_tools, "tools/call": _call_tool})


if __name__ == "__main__":
    run_stdio(build_server)
