"""The shared shape of a resources-and-prompts probe.

Every adapter that can reach resources or prompts answers the same seven questions
against the fixture, and records what came back in one normalized form so the cell
generator can compare it with what the server sent. An adapter with no route to a
surface reports it as ``unsupported``; one whose route raised reports ``error``. Both
are results, not failures of the probe.

Keys of the returned mapping::

    resources.list       {"ok", "entries": [{"uri", "name", "title", "description", "mimeType"}]}
    resources.read.text  {"ok", "uri", "mimeType", "text"}
    resources.read.blob  {"ok", "uri", "mimeType", "blob_sha256", "blob_bytes"}
    resources.templates  {"ok", "entries": [{"uriTemplate", "name", "description", "mimeType"}]}
    prompts.list         {"ok", "entries": [{"name", "title", "description",
                                             "arguments": [{"name", "description", "required"}]}]}
    prompts.get          {"ok", "description", "messages": [{"role", "type", "text"}]}

plus ``"error": str`` or ``"unsupported": str`` instead of ``ok: True``.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any, Callable

FEATURES = (
    "resources.list",
    "resources.read.text",
    "resources.read.blob",
    "resources.templates",
    "prompts.list",
    "prompts.get",
)


def unsupported(reason: str) -> dict[str, Any]:
    return {feature: {"ok": False, "unsupported": reason} for feature in FEATURES}


def attempt(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"ok": True, **fn()}
    except NotImplementedError as exc:
        return {"ok": False, "unsupported": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the failure is the observation
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _get(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def resource_entry(item: Any) -> dict[str, Any]:
    return {
        "uri": _s(_get(item, "uri")),
        "name": _get(item, "name"),
        "title": _get(item, "title"),
        "description": _get(item, "description"),
        "mimeType": _get(item, "mimeType", "mime_type"),
    }


def template_entry(item: Any) -> dict[str, Any]:
    return {
        "uriTemplate": _get(item, "uriTemplate", "uri_template"),
        "name": _get(item, "name"),
        "description": _get(item, "description"),
        "mimeType": _get(item, "mimeType", "mime_type"),
    }


def prompt_entry(item: Any) -> dict[str, Any]:
    arguments = _get(item, "arguments") or []
    return {
        "name": _get(item, "name"),
        "title": _get(item, "title"),
        "description": _get(item, "description"),
        "arguments": [
            {
                "name": _get(a, "name"),
                "description": _get(a, "description"),
                "required": _get(a, "required"),
            }
            for a in arguments
        ],
    }


def text_contents(uri: str | None, mime: str | None, text: str | None) -> dict[str, Any]:
    return {"uri": _s(uri), "mimeType": mime, "text": text}


def blob_contents(uri: str | None, mime: str | None, data: bytes | str | None) -> dict[str, Any]:
    raw: bytes | None
    if isinstance(data, str):
        try:
            raw = base64.b64decode(data, validate=True)
        except Exception:  # noqa: BLE001
            raw = data.encode()
    else:
        raw = data
    return {
        "uri": _s(uri),
        "mimeType": mime,
        "blob_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
        "blob_bytes": len(raw) if raw is not None else None,
    }


def message_entry(role: Any, content: Any) -> dict[str, Any]:
    # Adapters wrap the MCP content block in their own part type (Pydantic AI's
    # UserPromptPart carries `.content`, which may itself be a str or a list). Unwrap
    # until a string or a block with `text` appears.
    for _ in range(3):
        if isinstance(content, str) or _get(content, "text") is not None:
            break
        if isinstance(content, list) and content:
            content = content[0]
            continue
        inner = _get(content, "content")
        if inner is None:
            break
        content = inner
    text = content if isinstance(content, str) else _get(content, "text")
    kind = "text" if isinstance(content, str) else _get(content, "type")
    if hasattr(role, "value"):
        role = role.value
    return {"role": str(role) if role is not None else None, "type": kind, "text": text}


def _s(value: Any) -> str | None:
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# Server utilities probe (scripts/probe_utilities.py, fixtures/utilities_server.py)
# ---------------------------------------------------------------------------
#
# Each question runs against its own recorded fixture process, so the wire capture of
# one question never mixes with another. ``probe_utilities(servers)`` receives one
# ``ServerHandle`` per key and answers what the adapter handed its caller:
#
#     tools.list    {"ok", "names": [str]}                  the adapter's own tool loading
#     progress      {"ok", "events": [{"progress", "total", "message"}], "text"}
#     logging       {"ok", "messages": [{"level", "data"}], "text"}
#     set_level     {"ok", "level", "text"}                 emit_logs text after the set
#     completion    {"ok", "prompt": [str], "resource": [str]}
#     cancellation  {"ok", "mechanism", "abandoned_after_s", "outcome"}
#     tasks         {"ok", "text"}                          calling task_capable
#     caching       {"ok", "marks": [{"call", "started", "finished", "count"}]}
#
# The wire decides what the server saw; these answers decide what reached the caller.

UTILITY_FEATURES = (
    "tools.list",
    "progress",
    "logging",
    "set_level",
    "completion",
    "cancellation",
    "tasks",
    "caching",
)

# What every adapter is asked, so the answers compare.
SET_LEVEL = "warning"
COMPLETION_PROMPT = ("pick_colour", "colour", "c")
COMPLETION_RESOURCE = ("file:///fixtures/records/{record_id}", "record_id", "rec_1")
CANCEL_AFTER_S = 1.0
CANCEL_WAIT_S = 4
CACHE_PAUSE_S = 2.0  # past the fixture's CACHE_TTL_MS


def unsupported_utilities(reason: str, keys: tuple[str, ...] = UTILITY_FEATURES) -> dict[str, Any]:
    return {feature: {"ok": False, "unsupported": reason} for feature in keys}


async def attempt_async(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"ok": True, **(await fn())}
    except NotImplementedError as exc:
        return {"ok": False, "unsupported": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the failure is the observation
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def wall_clock() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def result_text(result: Any) -> str | None:
    """Text of the first text block in a tool result, whatever wraps it."""
    if isinstance(result, str):
        return result
    content = _get(result, "content")
    if isinstance(content, list):
        for block in content:
            text = _get(block, "text")
            if text is not None:
                return str(text)
    if isinstance(result, (list, tuple)):
        for block in result:
            text = block if isinstance(block, str) else _get(block, "text")
            if text is not None:
                return str(text)
    return None if result is None else str(result)
