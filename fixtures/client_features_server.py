"""The client-features fixture: tools whose handlers ask the client for something.

Sampling, elicitation and roots are requests the server sends to the client, so an
adapter only supports them if it answers. Each tool here makes exactly one such
request and returns what came back, so the tool result is the evidence:

- ``ask_sampling``          -> ``sampling/createMessage``
- ``ask_sampling_tools``    -> ``sampling/createMessage`` with ``tools`` and ``toolChoice``;
  if the answer is a ``tool_use``, a second request carrying the ``tool_result``
- ``ask_elicitation_form``  -> ``elicitation/create`` in form mode
- ``ask_elicitation_url``   -> ``elicitation/create`` in url mode
- ``list_client_roots``     -> ``roots/list``

Two more tools report what the server saw rather than asking for anything:

- ``roots_changes_seen``    how many ``notifications/roots/list_changed`` arrived on
  this session, so a client that changes its roots can prove it told the server
- ``client_capabilities``   the capabilities the client declared, so a missing answer
  can be told apart from an undeclared one

A client that does not answer (no capability declared, or the request raised) gets
an ``isError`` result starting ``SENTINEL_CLIENT_FEATURE_UNAVAILABLE`` rather than a
protocol error, so the failure is itself an observation.

Handshake-era revisions send the request on the session. At 2026-07-28 the server
cannot send requests; the tool returns an ``InputRequiredResult`` carrying the same
request, and answers when the client retries with ``inputResponses``.

    python fixtures/client_features_server.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mcp.types as types

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dual import build, run_stdio, text_result  # noqa: E402

NO_ARGUMENTS = {"type": "object", "properties": {}}

SAMPLING_PROMPT = "SENTINEL_SAMPLING_PROMPT: reply with one word."
FORM_MESSAGE = "SENTINEL_ELICIT_FORM: what should the record be called?"
FORM_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string", "description": "Record name."}},
    "required": ["name"],
}
URL_MESSAGE = "SENTINEL_ELICIT_URL: finish signing in at the link."
URL = "https://example.com/fixtures/elicit"
ELICITATION_ID = "SENTINEL_ELICITATION_ID"
TOOL_RESULT = "SENTINEL_TOOL_RESULT"
AFTER_TOOL = "ask_sampling_tools_after_tool"
SAMPLING_TOOL = types.Tool(
    name="sentinel_lookup",
    description="SENTINEL_SAMPLING_TOOL: look a word up.",
    inputSchema={"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]},
)

# notifications/roots/list_changed received. One stdio process serves one session, and
# 1.x gives a notification handler no session to key by. Handshake-era only: 2026-07-28
# removed the notification with the rest of the roots machinery.
ROOTS_CHANGES = {"seen": 0}

TOOLS: list[types.Tool] = [
    types.Tool(
        name="ask_sampling",
        description="Ask the client to sample a model once and return what it produced.",
        inputSchema=NO_ARGUMENTS,
    ),
    types.Tool(
        name="ask_sampling_tools",
        description="Ask the client to sample a model once with a request-scoped tool it may call.",
        inputSchema=NO_ARGUMENTS,
    ),
    types.Tool(
        name="ask_elicitation_form",
        description="Ask the user for a record name through a form elicitation.",
        inputSchema=NO_ARGUMENTS,
    ),
    types.Tool(
        name="ask_elicitation_url",
        description="Send the user to a URL through a url-mode elicitation.",
        inputSchema=NO_ARGUMENTS,
    ),
    types.Tool(
        name="list_client_roots",
        description="Ask the client for its roots and return them.",
        inputSchema=NO_ARGUMENTS,
    ),
    types.Tool(
        name="roots_changes_seen",
        description="Report how many roots/list_changed notifications this session has received.",
        inputSchema=NO_ARGUMENTS,
    ),
    types.Tool(
        name="client_capabilities",
        description="Report the capabilities the client declared.",
        inputSchema=NO_ARGUMENTS,
    ),
]


def _sampling_messages() -> list:
    return [types.SamplingMessage(role="user", content=types.TextContent(type="text", text=SAMPLING_PROMPT))]


def _dump(value) -> dict:
    return value.model_dump(mode="json", by_alias=True, exclude_none=True) if hasattr(value, "model_dump") else value


def _tool_use(response) -> dict | None:
    """The first ``tool_use`` block of a sampling answer, in wire form, if it asked for one."""

    payload = _dump(response) or {}
    content = payload.get("content")
    for block in content if isinstance(content, list) else [content]:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return block
    return None


def _after_tool_messages(use: dict) -> list:
    """The conversation for the loop's second turn: the model's tool_use, then our result."""

    return _sampling_messages() + [
        types.SamplingMessage.model_validate({"role": "assistant", "content": [use]}),
        types.SamplingMessage.model_validate({
            "role": "user",
            "content": [{"type": "tool_result", "toolUseId": use["id"], "content": [{"type": "text", "text": TOOL_RESULT}]}],
        }),
    ]


def _sampling_tools_request(messages: list, mode: str):
    return types.CreateMessageRequest(
        params=types.CreateMessageRequestParams(
            messages=messages, maxTokens=64, tools=[SAMPLING_TOOL], toolChoice=types.ToolChoice(mode=mode)
        )
    )


async def _sampling_tool_loop(session):
    """Run the tool loop: offer a tool, and if the answer uses it, send back its result.

    A client that answers the first turn with plain text never ran the loop; the answer
    is returned as it came so that is visible.
    """

    # The SDK refuses before sending when the client did not declare sampling.tools;
    # that refusal is the observation for a client that cannot take tools.
    first = await session.create_message(
        _sampling_messages(), max_tokens=64, tools=[SAMPLING_TOOL], tool_choice=types.ToolChoice(mode="auto")
    )
    use = _tool_use(first)
    if use is None:
        return first
    second = await session.create_message(
        _after_tool_messages(use), max_tokens=64, tools=[SAMPLING_TOOL], tool_choice=types.ToolChoice(mode="none")
    )
    return {"rounds": [_dump(first), _dump(second)]}


def _embedded_request(name: str):
    """The 2026-07-28 form of the request a tool makes: embedded, not sent."""

    if name == "ask_sampling":
        return types.CreateMessageRequest(
            params=types.CreateMessageRequestParams(messages=_sampling_messages(), maxTokens=64)
        )
    if name == "ask_sampling_tools":
        return _sampling_tools_request(_sampling_messages(), "auto")
    if name == "ask_elicitation_form":
        return types.ElicitRequest(
            params=types.ElicitRequestFormParams(message=FORM_MESSAGE, requestedSchema=FORM_SCHEMA)
        )
    if name == "ask_elicitation_url":
        return types.ElicitRequest(
            params=types.ElicitRequestURLParams(
                message=URL_MESSAGE, url=URL, elicitationId=ELICITATION_ID
            )
        )
    return types.ListRootsRequest()


async def _ask_session(session, name: str):
    """The handshake-era form: a request on the session, awaited."""

    if name == "ask_sampling":
        return await session.create_message(_sampling_messages(), max_tokens=64)
    if name == "ask_sampling_tools":
        return await _sampling_tool_loop(session)
    if name == "ask_elicitation_form":
        # Positional: the keyword is requestedSchema on 1.x and requested_schema on 2.x.
        return await session.elicit_form(FORM_MESSAGE, FORM_SCHEMA)
    if name == "ask_elicitation_url":
        return await session.elicit_url(URL_MESSAGE, URL, ELICITATION_ID)
    return await session.list_roots()


def _answer(name: str, response) -> types.CallToolResult:
    payload = _dump(response)
    # The structured copy names the tool too, for adapters that hand back only that.
    return text_result(
        f"SENTINEL_CLIENT_ANSWERED {name}: {json.dumps(payload, sort_keys=True)}",
        structured={"answered": name, "response": payload},
    )


def _declared_capabilities(session) -> dict:
    declared = getattr(session, "client_capabilities", None)
    if declared is None:
        params = getattr(session, "client_params", None)
        declared = getattr(params, "capabilities", None)
    if declared is None:
        return {}
    return declared.model_dump(mode="json", by_alias=True, exclude_none=True)


async def _call_tool(call):
    name = call.params.name
    if name not in {tool.name for tool in TOOLS}:
        return text_result(f"unknown tool: {name}", is_error=True)
    if name == "roots_changes_seen":
        seen = ROOTS_CHANGES["seen"]
        return text_result(f"SENTINEL_ROOTS_CHANGES_SEEN {seen}", structured={"seen": seen})
    if name == "client_capabilities":
        declared = _declared_capabilities(call.session)
        return text_result(f"SENTINEL_CLIENT_CAPABILITIES {json.dumps(declared, sort_keys=True)}", structured=declared)
    if call.modern:
        responses = call.input_responses
        if name == "ask_sampling_tools":
            return _modern_tool_loop(call, responses)
        if name in responses:
            return _answer(name, responses[name])
        return types.InputRequiredResult(input_requests={name: _embedded_request(name)})
    try:
        response = await _ask_session(call.session, name)
    except Exception as exc:  # noqa: BLE001 - an unanswered request is the finding
        return text_result(
            f"SENTINEL_CLIENT_FEATURE_UNAVAILABLE {name}: {type(exc).__name__}: {exc}", is_error=True
        )
    return _answer(name, response)


def _modern_tool_loop(call, responses: dict):
    """The tool loop over MRTR: the first turn's answer rides back in ``requestState``
    because a retry carries only the responses to the latest input requests."""

    name = "ask_sampling_tools"
    if AFTER_TOOL in responses:
        first = json.loads(getattr(call.params, "request_state", None) or "null")
        return _answer(name, {"rounds": [first, responses[AFTER_TOOL]]})
    if name not in responses:
        return types.InputRequiredResult(input_requests={name: _embedded_request(name)})
    use = _tool_use(responses[name])
    if use is None:
        return _answer(name, responses[name])
    return types.InputRequiredResult(
        input_requests={AFTER_TOOL: _sampling_tools_request(_after_tool_messages(use), "none")},
        request_state=json.dumps(responses[name], sort_keys=True),
    )


async def _list_tools(_call) -> types.ListToolsResult:
    return types.ListToolsResult(tools=TOOLS)


def build_server():
    server = build("client-features-mcp", {"tools/list": _list_tools, "tools/call": _call_tool})
    notification_handlers = getattr(server, "notification_handlers", None)
    if isinstance(notification_handlers, dict):
        # 1.x keys notification handlers by type; 2.x takes on_roots_list_changed,
        # which this fixture does not need because the notification is gone there.

        async def _roots_changed(_notification) -> None:
            ROOTS_CHANGES["seen"] += 1

        notification_handlers[types.RootsListChangedNotification] = _roots_changed
    return server


if __name__ == "__main__":
    run_stdio(build_server)
