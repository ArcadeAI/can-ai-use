"""The utilities fixture: pagination, progress, cancellation, logging and completion.

- Pagination: ``tools/list``, ``resources/list`` and ``prompts/list`` each hold more
  than one page (``PAGE_SIZE``) and hand out a ``nextCursor``. The last tool, resource
  and prompt are named ``SENTINEL_LAST_*``, so a client that stops at page one is
  visible as their absence.
- Progress: ``slow_with_progress`` sends ``PROGRESS_STEPS`` progress notifications
  against the request's ``progressToken`` before it returns.
- Cancellation: ``cancellable_wait`` sleeps for ``seconds`` (default 30) and only then
  returns ``SENTINEL_NOT_CANCELLED``; a client that cancels never sees that text.
- Logging: ``emit_logs`` sends one ``notifications/message`` at each of several levels,
  honouring the level set through ``logging/setLevel``.
- Completion: ``completion/complete`` answers for the ``pick_colour`` prompt's
  ``colour`` argument and for the ``record`` resource template's ``record_id``.
- Tasks (probe only): the server advertises task-augmented ``tools/call`` (the
  2025-11-25 ``tasks`` capability, the Tasks extension at 2026-07-28) and
  ``task_capable`` marks ``execution.taskSupport: optional``. It always answers
  synchronously; the probe records whether a client asks for a task at all.
- Caching (2026-07-28 only): ``tools/list`` pages carry ``ttlMs: CACHE_TTL_MS``
  with ``cacheScope: private``, so a client's re-fetch can be timed against it.

    python fixtures/utilities_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import anyio
import mcp.types as types

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dual import SDK2, build, page, run_stdio, text_result  # noqa: E402

PAGE_SIZE = 20
LISTED = 60  # three pages
PROGRESS_STEPS = 5
LOG_LEVELS = ["debug", "info", "notice", "warning", "error"]
SEVERITY = {level: rank for rank, level in enumerate(
    ["debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"]
)}
COLOURS = ["crimson", "cyan", "cobalt", "amber", "azure"]
RECORD_IDS = ["rec_1", "rec_2", "rec_10", "rec_11"]
CACHE_TTL_MS = 1500
TASKS_EXTENSION = "io.modelcontextprotocol/tasks"

NO_ARGUMENTS = {"type": "object", "properties": {}}

UTILITY_TOOLS: list[types.Tool] = [
    types.Tool(
        name="slow_with_progress",
        description=f"Take a moment and report {PROGRESS_STEPS} progress steps on the way.",
        inputSchema=NO_ARGUMENTS,
    ),
    types.Tool(
        name="cancellable_wait",
        description="Wait for a number of seconds, returning only if not cancelled first.",
        inputSchema={
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "How long to wait.", "default": 30}
            },
        },
    ),
    types.Tool(
        name="emit_logs",
        description="Send one log message at each of debug, info, notice, warning and error.",
        inputSchema=NO_ARGUMENTS,
    ),
    types.Tool(
        name="task_capable",
        description="Return at once; a client may ask for this call as a task.",
        inputSchema=NO_ARGUMENTS,
        execution=types.ToolExecution(taskSupport="optional"),
    ),
]


def _filler(index: int, last: bool) -> types.Tool:
    return types.Tool(
        name="SENTINEL_LAST_TOOL" if last else f"page_filler_{index:02d}",
        description=f"Filler tool {index} of {LISTED}, present to make the list span pages.",
        inputSchema=NO_ARGUMENTS,
    )


TOOLS: list[types.Tool] = UTILITY_TOOLS + [
    _filler(i, i == LISTED - len(UTILITY_TOOLS) - 1) for i in range(LISTED - len(UTILITY_TOOLS))
]

RESOURCES: list[types.Resource] = [
    types.Resource(
        uri=f"file:///fixtures/page/{i:02d}.txt",
        name="SENTINEL_LAST_RESOURCE" if i == LISTED - 1 else f"page_{i:02d}.txt",
        mimeType="text/plain",
    )
    for i in range(LISTED)
]

RESOURCE_TEMPLATES: list[types.ResourceTemplate] = [
    types.ResourceTemplate(
        uriTemplate="file:///fixtures/records/{record_id}",
        name="record",
        description="A record, with completion for record_id.",
        mimeType="application/json",
    )
]

PROMPTS: list[types.Prompt] = [
    types.Prompt(
        name="pick_colour",
        description="Pick a colour, with completion for the colour argument.",
        arguments=[types.PromptArgument(name="colour", description="A colour.", required=True)],
    )
] + [
    types.Prompt(name="SENTINEL_LAST_PROMPT" if i == LISTED - 2 else f"page_prompt_{i:02d}")
    for i in range(LISTED - 1)
]


def build_server():
    state = {"level": "debug"}

    async def list_tools(call) -> types.ListToolsResult:
        tools, cursor = page(TOOLS, call.cursor, PAGE_SIZE)
        return types.ListToolsResult(tools=tools, nextCursor=cursor)

    async def list_resources(call) -> types.ListResourcesResult:
        resources, cursor = page(RESOURCES, call.cursor, PAGE_SIZE)
        return types.ListResourcesResult(resources=resources, nextCursor=cursor)

    async def list_resource_templates(_call) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(resourceTemplates=RESOURCE_TEMPLATES)

    async def read_resource(call) -> types.ReadResourceResult:
        uri = str(call.params.uri)
        return types.ReadResourceResult(
            contents=[types.TextResourceContents(uri=uri, mimeType="text/plain", text=f"SENTINEL_PAGE {uri}")]
        )

    async def list_prompts(call) -> types.ListPromptsResult:
        prompts, cursor = page(PROMPTS, call.cursor, PAGE_SIZE)
        return types.ListPromptsResult(prompts=prompts, nextCursor=cursor)

    async def get_prompt(call) -> types.GetPromptResult:
        colour = (call.params.arguments or {}).get("colour", "<unset>")
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=f"SENTINEL_PROMPT {call.params.name} colour={colour}"),
                )
            ]
        )

    async def complete(call) -> types.CompleteResult:
        ref = call.params.ref
        argument = call.params.argument
        pool: list[str] = []
        if getattr(ref, "type", None) == "ref/prompt" and argument.name == "colour":
            pool = COLOURS
        elif getattr(ref, "type", None) == "ref/resource" and argument.name == "record_id":
            pool = RECORD_IDS
        values = [value for value in pool if value.startswith(argument.value or "")]
        return types.CompleteResult(
            completion=types.Completion(values=values, total=len(values), hasMore=False)
        )

    async def set_level(call):
        state["level"] = call.params.level

    async def call_tool(call) -> types.CallToolResult:
        name = call.params.name
        session = call.session
        if name == "slow_with_progress":
            token = call.progress_token
            for step in range(1, PROGRESS_STEPS + 1):
                await anyio.sleep(0.05)
                if token is not None:
                    await session.send_progress_notification(
                        token,
                        step,
                        total=PROGRESS_STEPS,
                        message=f"SENTINEL_PROGRESS {step}/{PROGRESS_STEPS}",
                        related_request_id=call.request_id,
                    )
            sent = PROGRESS_STEPS if token is not None else 0
            return text_result(f"SENTINEL_SLOW_DONE progress_sent={sent}")
        if name == "cancellable_wait":
            await anyio.sleep(float(call.arguments.get("seconds", 30)))
            return text_result("SENTINEL_NOT_CANCELLED")
        if name == "emit_logs":
            sent = 0
            for level in LOG_LEVELS:
                if SEVERITY[level] < SEVERITY[state["level"]]:
                    continue
                await session.send_log_message(
                    level,
                    {"message": f"SENTINEL_LOG_{level.upper()}"},
                    logger="utilities-fixture",
                    related_request_id=call.request_id,
                )
                sent += 1
            return text_result(f"SENTINEL_LOGS_SENT {sent} at level>={state['level']}")
        if name == "task_capable":
            requested = getattr(call.params, "task", None) is not None
            return text_result(f"SENTINEL_TASK_CAPABLE task_requested={requested}")
        if any(tool.name == name for tool in TOOLS):
            return text_result(f"SENTINEL_FILLER {name}")
        return text_result(f"unknown tool: {name}", is_error=True)

    server = build(
        "utilities-mcp",
        {
            "tools/list": list_tools,
            "tools/call": call_tool,
            "resources/list": list_resources,
            "resources/templates/list": list_resource_templates,
            "resources/read": read_resource,
            "prompts/list": list_prompts,
            "prompts/get": get_prompt,
            "completion/complete": complete,
            "logging/setLevel": set_level,
        },
    )
    _advertise_tasks_and_caching(server)
    return server


def _advertise_tasks_and_caching(server) -> None:
    if SDK2:
        from mcp.server.caching import CacheHint

        server.cache_hints["tools/list"] = CacheHint(ttl_ms=CACHE_TTL_MS, scope="private")
        server.extensions[TASKS_EXTENSION] = {}
        return
    original = server.get_capabilities

    def get_capabilities(*args, **kwargs):
        capabilities = original(*args, **kwargs)
        capabilities.tasks = types.ServerTasksCapability(
            requests=types.ServerTasksRequestsCapability(
                tools=types.TasksToolsCapability(call=types.TasksCallCapability())
            )
        )
        return capabilities

    server.get_capabilities = get_capabilities  # type: ignore[method-assign]


if __name__ == "__main__":
    run_stdio(build_server)
