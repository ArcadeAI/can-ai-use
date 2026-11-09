"""The shared shape of a client-features probe.

Sampling, elicitation and roots run the wrong way round: the server asks, the client
answers. ``fixtures/client_features_server.py`` has one tool per request, and the tool
result is what the server got back. So the probe registers whatever hook the adapter
offers, calls each tool through the adapter, and records the tool's text. The cell
generator reads that text; nothing here decides a support code.

Every feature entry records::

    {"tool", "hook", "route", "expect", "text", "is_error", "handler_saw"}

``hook`` names the API the probe registered (``None`` when the adapter offers none).
``route`` says whose API it is:

``adapter``
    a parameter or method on the adapter object itself
``passthrough``
    keyword arguments the adapter forwards unread to the MCP SDK's ``ClientSession``
``underlying``
    a method on the client library the adapter wraps, reached through a public
    attribute of the adapter

``handler_saw`` lists what reached the registered handler, so a round trip that
completed with a request the adapter altered on the way in can be told apart.
"""

from __future__ import annotations

import json
from typing import Any, Callable

SAMPLED = "SENTINEL_SAMPLED"
SAMPLED_AFTER_TOOL = "SENTINEL_SAMPLED_AFTER_TOOL"
TOOL_USE_ID = "SENTINEL_TOOL_USE_ID"
ELICITED = "SENTINEL_ELICITED"
ROOT_URI = "file:///tmp/mcp-mirror-root"
CHANGED_ROOT_URI = "file:///tmp/mcp-mirror-root-changed"

# feature -> (fixture tool, text the server echoes only if our handler answered)
FEATURES: dict[str, tuple[str, str]] = {
    "sampling": ("ask_sampling", SAMPLED),
    # Only the loop's second turn produces this, so a single-shot answer does not match.
    "sampling.tools": ("ask_sampling_tools", SAMPLED_AFTER_TOOL),
    "elicitation.form": ("ask_elicitation_form", ELICITED),
    "elicitation.url": ("ask_elicitation_url", '"action": "accept"'),
    "roots.list": ("list_client_roots", ROOT_URI),
    "roots.list_changed": ("roots_changes_seen", "SENTINEL_ROOTS_CHANGES_SEEN"),
}
CAPABILITIES_TOOL = "client_capabilities"
TEXT_LIMIT = 800


class HandlerLog:
    """What each registered handler was handed, in call order."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def note(self, hook: str, **facts: Any) -> None:
        self.seen.append({"hook": hook, **facts})

    def drain(self) -> list[dict[str, Any]]:
        seen, self.seen = self.seen, []
        return seen


def _get(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def rep_text(rep: Any) -> tuple[str, bool | None]:
    """Everything textual a ResultRep carries, and its error flag.

    An adapter that prefers ``structuredContent`` hands back the payload without the
    sentinel line, so the text blocks and the structured payload (as sorted JSON) are
    both kept; the cell generator looks for the answer in either.
    """

    parts: list[str] = []
    blocks = _get(rep, "blocks") or []
    parts.extend(str(b.get("text")) for b in blocks if isinstance(b, dict) and b.get("text"))
    structured = _get(rep, "structured_content")
    if structured is not None:
        parts.append(json.dumps(structured, sort_keys=True, default=str))
    if not parts and _get(rep, "text"):
        parts.append(str(_get(rep, "text")))
    if not parts and _get(rep, "raw") is not None:
        parts.append(str(_get(rep, "raw")))
    return "\n".join(parts)[:TEXT_LIMIT], _get(rep, "is_error")


def record(
    feature: str,
    hook: str | None,
    route: str | None,
    rep: Any = None,
    *,
    error: str | None = None,
    handler_saw: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tool, expect = FEATURES[feature]
    entry: dict[str, Any] = {"tool": tool, "hook": hook, "route": route, "expect": expect}
    if error is not None:
        entry.update({"text": error[:TEXT_LIMIT], "is_error": True, "probe_error": True})
    else:
        text, is_error = rep_text(rep)
        entry.update({"text": text, "is_error": is_error})
    entry["handler_saw"] = handler_saw or []
    return entry


def without_hooks(call: Callable[[str], Any]) -> dict[str, Any]:
    """The probe for an adapter that offers no client-feature hook: call every tool.

    ``call(tool)`` returns a ResultRep through the adapter's ordinary tool path.
    """

    out: dict[str, Any] = {}
    for feature, (tool, _expect) in FEATURES.items():
        out[feature] = attempt(feature, None, None, lambda tool=tool: call(tool))
    out["capabilities"] = capabilities(lambda: call(CAPABILITIES_TOOL))
    return out


def attempt(
    feature: str,
    hook: str | None,
    route: str | None,
    fn: Callable[[], Any],
    log: HandlerLog | None = None,
) -> dict[str, Any]:
    try:
        rep = fn()
    except Exception as exc:  # noqa: BLE001 - the failure is the observation
        return record(feature, hook, route, error=f"{type(exc).__name__}: {exc}", handler_saw=log.drain() if log else None)
    return record(feature, hook, route, rep, handler_saw=log.drain() if log else None)


async def attempt_async(
    feature: str,
    hook: str | None,
    route: str | None,
    fn: Callable[[], Any],
    log: HandlerLog | None = None,
) -> dict[str, Any]:
    try:
        rep = await fn()
    except Exception as exc:  # noqa: BLE001 - the failure is the observation
        return record(feature, hook, route, error=f"{type(exc).__name__}: {exc}", handler_saw=log.drain() if log else None)
    return record(feature, hook, route, rep, handler_saw=log.drain() if log else None)


def capabilities(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        text, _ = rep_text(fn())
    except Exception as exc:  # noqa: BLE001
        return {"text": f"{type(exc).__name__}: {exc}"[:TEXT_LIMIT]}
    return {"text": text}


def _blocks(message: Any) -> list:
    content = _get(message, "content")
    return content if isinstance(content, list) else [content]


def sampling_reply(messages: list, tools: list | None) -> dict[str, Any]:
    """The stub model's answer, in wire form.

    Offered tools and no tool result yet: call the first tool. Otherwise answer in
    text, saying whether a tool result came back, so the fixture's echo shows how far
    the loop got.
    """

    after_tool = any(_get(b, "type") == "tool_result" for m in messages or [] for b in _blocks(m))
    if tools and not after_tool:
        return {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": TOOL_USE_ID, "name": _get(tools[0], "name"), "input": {"word": "mirror"}}],
            "model": "mcp-mirror-stub",
            "stopReason": "toolUse",
        }
    return {
        "role": "assistant",
        "content": {"type": "text", "text": SAMPLED_AFTER_TOOL if after_tool else SAMPLED},
        "model": "mcp-mirror-stub",
        "stopReason": "endTurn",
    }


def sampling_result(types: Any, reply: dict[str, Any]) -> Any:
    """``reply`` as the installed SDK's result type (the tools variant for a tool_use)."""

    if isinstance(reply["content"], list):
        return types.CreateMessageResultWithTools.model_validate(reply)
    return types.CreateMessageResult.model_validate(reply)


# ---------------------------------------------------------------------------
# Handlers in the MCP SDK's own callback shapes, for adapters that forward them
# ---------------------------------------------------------------------------


def sdk_callbacks(log: HandlerLog) -> dict[str, Callable]:
    """``ClientSession`` keyword callbacks answering with the sentinels.

    Written against the installed ``mcp`` package so the same code runs on 1.x and
    2.x; the result types are imported lazily for that reason.
    """

    import mcp.types as types

    async def sampling_callback(_context, params):
        reply = sampling_reply(_get(params, "messages"), _get(params, "tools"))
        log.note("sampling_callback", tools=bool(_get(params, "tools")), replied=reply["stopReason"])
        return sampling_result(types, reply)

    async def elicitation_callback(_context, params):
        mode = _get(params, "mode") or ("url" if _get(params, "url") else "form")
        log.note("elicitation_callback", mode=mode)
        if mode == "url":
            return types.ElicitResult(action="accept")
        return types.ElicitResult(action="accept", content={"name": ELICITED})

    async def list_roots_callback(_context):
        log.note("list_roots_callback")
        return types.ListRootsResult(roots=[types.Root(uri=ROOT_URI, name="mcp-mirror")])

    return {
        "sampling_callback": sampling_callback,
        "elicitation_callback": elicitation_callback,
        "list_roots_callback": list_roots_callback,
    }
