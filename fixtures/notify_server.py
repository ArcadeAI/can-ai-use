"""The notifications fixture: tools that change the catalogue and say so.

Each mutating tool changes what a later list call returns and then fires the matching
notification, so a harness can check two things separately: whether the notification
reached the client, and whether the client refetched. Everything a mutation adds
carries a ``SENTINEL_`` name, so a refetched list is recognisable in a capture.

- ``add_tool`` / ``remove_tool`` -> ``notifications/tools/list_changed``
- ``add_resource``               -> ``notifications/resources/list_changed``
- ``update_resource``            -> ``notifications/resources/updated`` for a
  subscribed URI
- ``add_prompt``                 -> ``notifications/prompts/list_changed``

Handshake-era revisions deliver the notification on the session. At 2026-07-28 there
is no standing stream, so the same events are published to ``subscriptions/listen``
subscribers instead; an unsubscribed client sees nothing, which is the protocol
working as specified.

    python fixtures/notify_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mcp.types as types
from mcp.server.lowlevel import NotificationOptions

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dual import SDK2, build, run_stdio, text_result  # noqa: E402

NO_ARGUMENTS = {"type": "object", "properties": {}}
WATCHED_URI = "file:///fixtures/watched.txt"

MUTATORS: list[types.Tool] = [
    types.Tool(
        name="add_tool",
        description="Add SENTINEL_ADDED_TOOL to the tool list and fire tools/list_changed.",
        inputSchema=NO_ARGUMENTS,
    ),
    types.Tool(
        name="remove_tool",
        description="Remove SENTINEL_ADDED_TOOL from the tool list and fire tools/list_changed.",
        inputSchema=NO_ARGUMENTS,
    ),
    types.Tool(
        name="add_resource",
        description="Add a resource to the resource list and fire resources/list_changed.",
        inputSchema=NO_ARGUMENTS,
    ),
    types.Tool(
        name="update_resource",
        description=f"Change the text of {WATCHED_URI} and fire resources/updated for it.",
        inputSchema=NO_ARGUMENTS,
    ),
    types.Tool(
        name="add_prompt",
        description="Add a prompt to the prompt list and fire prompts/list_changed.",
        inputSchema=NO_ARGUMENTS,
    ),
]

ADDED_TOOL = types.Tool(
    name="SENTINEL_ADDED_TOOL",
    description="Present only after add_tool ran.",
    inputSchema=NO_ARGUMENTS,
)


class Catalogue:
    """Mutable server state, one per server instance."""

    def __init__(self) -> None:
        self.tool_added = False
        self.resources: list[types.Resource] = [
            types.Resource(uri=WATCHED_URI, name="watched.txt", mimeType="text/plain")
        ]
        self.prompts: list[types.Prompt] = []
        self.version = 0
        self.subscribed: set[str] = set()

    def tools(self) -> list[types.Tool]:
        return MUTATORS + ([ADDED_TOOL] if self.tool_added else [])


def build_server():
    state = Catalogue()
    bus = listen = None
    if SDK2:
        from mcp.server.subscriptions import InMemorySubscriptionBus, ListenHandler
        from mcp.shared.subscriptions import (
            PromptsListChanged,
            ResourcesListChanged,
            ResourceUpdated,
            ToolsListChanged,
        )

        bus = InMemorySubscriptionBus()
        listen = ListenHandler(bus)
        events = {
            "tools": ToolsListChanged,
            "resources": ResourcesListChanged,
            "prompts": PromptsListChanged,
        }

    async def announce(call, kind: str, uri: str | None = None) -> str:
        """Deliver one change on whichever channel the negotiated revision uses."""

        if call.modern:
            event = ResourceUpdated(uri=uri) if kind == "updated" else events[kind]()
            await bus.publish(event)
            return "published to subscriptions/listen"
        session = call.session
        if kind == "tools":
            await session.send_tool_list_changed()
        elif kind == "resources":
            await session.send_resource_list_changed()
        elif kind == "prompts":
            await session.send_prompt_list_changed()
        elif uri in state.subscribed:
            await session.send_resource_updated(uri)
        else:
            return "not sent: the client has not subscribed to this uri"
        return "sent on the session"

    async def list_tools(_call) -> types.ListToolsResult:
        return types.ListToolsResult(tools=state.tools())

    async def call_tool(call) -> types.CallToolResult:
        name = call.params.name
        if name == "add_tool":
            state.tool_added = True
            how = await announce(call, "tools")
        elif name == "remove_tool":
            state.tool_added = False
            how = await announce(call, "tools")
        elif name == "add_resource":
            index = len(state.resources)
            state.resources.append(
                types.Resource(
                    uri=f"file:///fixtures/added-{index}.txt",
                    name=f"SENTINEL_ADDED_RESOURCE_{index}",
                    mimeType="text/plain",
                )
            )
            how = await announce(call, "resources")
        elif name == "update_resource":
            state.version += 1
            how = await announce(call, "updated", WATCHED_URI)
        elif name == "add_prompt":
            index = len(state.prompts)
            state.prompts.append(types.Prompt(name=f"SENTINEL_ADDED_PROMPT_{index}"))
            how = await announce(call, "prompts")
        elif name == ADDED_TOOL.name and state.tool_added:
            return text_result("SENTINEL_ADDED_TOOL_CALLED")
        else:
            return text_result(f"unknown tool: {name}", is_error=True)
        return text_result(f"SENTINEL_MUTATED {name}: {how}")

    async def list_resources(_call) -> types.ListResourcesResult:
        return types.ListResourcesResult(resources=state.resources)

    async def read_resource(call) -> types.ReadResourceResult:
        uri = str(call.params.uri)
        text = f"SENTINEL_RESOURCE_VERSION {state.version}" if uri == WATCHED_URI else "SENTINEL_ADDED"
        return types.ReadResourceResult(
            contents=[types.TextResourceContents(uri=uri, mimeType="text/plain", text=text)]
        )

    async def subscribe(call):
        state.subscribed.add(str(call.params.uri))

    async def unsubscribe(call):
        state.subscribed.discard(str(call.params.uri))

    async def list_prompts(_call) -> types.ListPromptsResult:
        return types.ListPromptsResult(prompts=state.prompts)

    async def get_prompt(call) -> types.GetPromptResult:
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=f"SENTINEL_PROMPT {call.params.name}"),
                )
            ]
        )

    return build(
        "notify-mcp",
        {
            "tools/list": list_tools,
            "tools/call": call_tool,
            "resources/list": list_resources,
            "resources/read": read_resource,
            "resources/subscribe": subscribe,
            "resources/unsubscribe": unsubscribe,
            "prompts/list": list_prompts,
            "prompts/get": get_prompt,
        },
        listen=listen,
    )


# Handshake-era servers advertise listChanged through the initialization options.
NOTIFICATIONS = NotificationOptions(prompts_changed=True, resources_changed=True, tools_changed=True)

if __name__ == "__main__":
    run_stdio(build_server, NOTIFICATIONS)
