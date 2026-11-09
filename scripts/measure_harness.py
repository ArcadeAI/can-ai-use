"""Run a harness headless against the recording server and capture the wire.

    uv run python scripts/measure_harness.py claude_code             # levels 1 and 2, scripted stub
    uv run python scripts/measure_harness.py claude_code --level 1   # level 1 only
    uv run python scripts/measure_harness.py claude_code --live      # level 2 against a real model
    uv run python scripts/measure_harness.py --write-surface         # data/harnesses/surface.json

A harness is measured on the wire only (plan/harness_surface.yaml). Each session mounts
the five fixtures through `mcp-mirror record` as servers def, res, ntf, cf and util,
starts the harness with its documented one-shot invocation (plan/harness_sessions.yaml),
and keeps one JSONL capture per server, a reduced copy of the harness's own
machine-readable output, and what the harness sent its model. Cells are derived from
those files by cells.py.

Everything harness-specific lives in scripts/harnesses/<harness>.py (binary, config,
argv per approval mode, stub environment, output reducer, prompt invocation); this file
has no per-harness table. Adding a harness is adding that module (start from
scripts/harnesses/_template.py) and a block in plan/harness_sessions.yaml.

A harness module with a STUB_KIND sends its model calls to a local stub (Anthropic
Messages, OpenAI Responses/Chat Completions, or Gemini) under a placeholder key, so no
level needs a credential and everything reruns from a clean checkout. Both levels run
once per fixture line (MCP SDK 1.x and 2.x). At level 1 the stub only says "ready". At
level 2 it is scripted per session: it asks for the session's tools in order, then ends
the turn, so the cell measures the harness, not a model. Each level-2 session runs
`runs` times per line and the cell publishes the count.

Every session runs fresh, and the shared code here enforces it so no harness module can
skip it (see "Freshness" below): a new temp HOME with every config-dir variable inside
it, proxies pointed at a local sink that refuses and logs, the harness's own `mcp list`
before the run, and the tools offered to the model after it. Each session directory
gets a fresh.json with the four results, and cells.py publishes no cell from a run in
which any session failed one.

--live runs against a real model instead. The operator wraps the whole measurement once:
`op run --env-file .env.harness -- uv run python scripts/measure_locked.py -- uv run
python scripts/measure_harness.py <id> --live`. The operator creates that file once with
1Password secret references, this script never opens it, and op asks 1Password once for
the run (approval is per process, so one op per session meant one prompt per session).
This script takes the file's variables out of its own environment at startup
(live_secrets: never printed or written, and gone from os.environ so no probe, install
or recorder inherits them) and gives them to the harness's process alone, on top of the
same isolated environment a stubbed session gets. A live run is written to measurements/wire-live/<harness>/. For a harness
with a STUB_KIND it measures level 2 only and is scratch: the stubbed run is the one
published. For a harness with no STUB_KIND it is the only measurement there is, so it
runs both levels and is published, with the row marked measured_live (the model the
service used, the hosts the sink forwarded to) so a reader knows the cells came from a
live service and can vary between runs. The sink forwards CONNECT only to the hosts the
module's LIVE_EGRESS_ALLOWED names, only in a live run, and records each one.

A live session is also contained (LiveSandbox): a real model with a real key runs shell
commands, and the first live Cursor run read outside its throwaway home by absolute path.
So the harness starts under a sandbox-exec profile that denies reading, writing and
executing anything but the session's home and workdir, its capture directory, the
fixtures and their interpreters, the harness's own install and the system's binaries,
and lets TCP leave only to the session's own loopback ports (the sink, the recorders),
so every request goes through the sink and no other local service is reachable; signals
and process information stop at the session's own processes. A probe in
the same profile must show each denial before the session starts, the profile's hash is
in fresh.json (checks.sandbox), and without sandbox-exec a live session does not start.
Every session, live or stubbed, runs in its own process group: at the timeout the whole
group is killed, and whatever outlives the harness, timeout or not, is killed and fails
the session (checks.process_group).

Output: data/harnesses/measurements/wire/<harness>/session.json and one directory per
session. A run replaces the previous one for that harness, and updates the harness's row
in data/harnesses/harnesses.json from the inventory and the version the binary reports.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp_mirror import cells  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import harnesses  # noqa: E402
from harnesses import HarnessError, clip  # noqa: E402

SNAPSHOT = ROOT / "data" / "harnesses"
INVENTORY = ROOT / "plan" / "inventory.yaml"
SESSIONS = ROOT / "plan" / "harness_sessions.yaml"
SURFACE_PLAN = ROOT / "plan" / "harness_surface.yaml"
SESSION_FORMAT = "mcp-mirror.harness-session/1"
OUTPUT_FORMAT = "mcp-mirror.harness-output/1"
FRESH_FORMAT = "mcp-mirror.harness-fresh/1"
MODEL_FORMAT = "mcp-mirror.harness-model-requests/1"
# Variables that would point a harness at the operator's own account or at the session
# this script might be running inside. No stubbed session sees them; a live run gets its
# credentials from the env file's variables only (live_secrets). A harness module adds
# its own in SCRUBBED_ENV.
SCRUBBED_ENV = re.compile(r"^(ANTHROPIC_|CLAUDE_CODE_|CLAUDECODE$|OPENAI_|GEMINI_|GOOGLE_API_KEY$)")


def _scrubbed(harness: Any = None) -> dict[str, str]:
    extra = tuple(getattr(harness, "SCRUBBED_ENV", ()) or ())
    return {k: v for k, v in os.environ.items() if not SCRUBBED_ENV.match(k) and not k.startswith(extra)}


# ---------------------------------------------------------------------------
# Plan files
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def inventory_entry(harness_id: str) -> dict:
    for entry in load_yaml(INVENTORY).get("harnesses") or []:
        if entry.get("id") == harness_id:
            return entry
    raise HarnessError(f"{harness_id!r} is not a harness in {INVENTORY.relative_to(ROOT)}")


def session_plan() -> dict:
    return load_yaml(SESSIONS)


def harness_plan(plan: dict, harness_id: str) -> dict:
    configured = plan.get("harnesses") or {}
    if harness_id not in configured:
        raise HarnessError(
            f"{harness_id!r} has no block in {SESSIONS.relative_to(ROOT)} yet "
            f"(configured: {', '.join(sorted(configured))})"
        )
    return configured[harness_id] or {}


def surface_from_plan() -> dict:
    """data/harnesses/surface.json, generated from plan/harness_surface.yaml."""
    plan = load_yaml(SURFACE_PLAN)
    areas = {"level_1": "Startup (level 1)", "level_2": "Session (level 2)"}
    features = []
    for level, area in areas.items():
        block = plan[level]
        for cell in block["cells"] if isinstance(block, dict) else block:
            words = cell["id"].removeprefix("h-").replace("-", " ")
            features.append({
                "id": cell["id"],
                "area": area,
                "level": 1 if level == "level_1" else 2,
                "title": words[0].upper() + words[1:],
                "question": cell.get("question", ""),
                "description": cell.get("description", ""),
                "observe": " ".join(cell["observe"].split()),
                "value_cell": cell["id"] in cells.HARNESS_VALUE_CELLS,
                **({"labels": cell["labels"]} if cell.get("labels") else {}),
            })
    return {
        "schema": "mcp-mirror/harness-surface@1",
        "source": str(SURFACE_PLAN.relative_to(ROOT)),
        "note": (
            "What the harness page measures. Generated from plan/harness_surface.yaml by "
            "scripts/measure_harness.py --write-surface; edit the plan, not this file."
        ),
        "fixtures": plan["fixtures"],
        "prompts": plan["level_2"]["prompts"],
        "approval_modes": plan["level_2"]["approval_modes"],
        "notes": plan.get("notes", []),
        "areas": list(areas.values()),
        "features": features,
    }


def write_surface() -> Path:
    path = SNAPSHOT / "surface.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(surface_from_plan(), indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# Model stubs
# ---------------------------------------------------------------------------
#
# A stub stands in for the model API a harness calls. It is scripted per session: each
# step of the script is one tool_use turn, chosen by how many tool results the harness
# has sent back so far, and once the script is spent the stub ends the turn with a short
# text. Level 1's script is empty, so the stub only says "ready". Every request is kept
# in memory, normalised to the tools the harness offered, and reduced to model.jsonl
# after the session (reduce_model_log): a cell reads what the model was shown, never the
# harness's system prompt.

HINT_KEYS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")


class StubBrain:
    """The script and the request log one stub shares across its handler threads."""

    def __init__(self, script: list[dict] | None = None, text: str = "ready", reply_delay_s: float = 0.0) -> None:
        self.script = list(script or [])
        self.text = text
        # Seconds the stub holds every reply (level_1_reply_delay): a harness that connects
        # its servers in the background needs a turn that outlasts the connect.
        self.reply_delay_s = float(reply_delay_s or 0.0)
        self.log: list[dict] = []
        self._lock = threading.Lock()
        # Set by the session: takes the isolated home's paths out of every request before
        # anything is clipped, so a quoted path never lands half-scrubbed in model.jsonl.
        self.scrub: Any = lambda text: text

    def decide(self, api: str, path: str, raw: bytes, request: dict, offered: list[dict], results: int,
               last_result: dict | None = None) -> dict:
        """The next turn: {"tool": name, "arguments": {...}} or {"text": ...}."""
        if self.reply_delay_s > 0:
            time.sleep(self.reply_delay_s)
        step = self.script[results] if results < len(self.script) else None
        names = [tool["name"] for tool in offered]
        reply: dict
        if step is None:
            reply = {"text": self.text}
        else:
            name = step.get("builtin") or cells.offered_name(names, step["server"], step["tool"])
            if name and name in names:
                reply = {"tool": name, "arguments": step.get("arguments") or {}, "step": results + 1}
            else:
                wanted = step.get("builtin") or f"{step['server']} {step['tool']}"
                reply = {"text": f"{wanted} is not among the tools offered", "step": results + 1, "not_offered": wanted}
        text = raw.decode("utf-8", "replace")
        with self._lock:
            self.log.append({
                "api": api,
                "path": path.split("?")[0],
                "model": request.get("model"),
                "stream": bool(request.get("stream")) or "stream" in path.lower(),
                "tool_results_seen": results,
                # What the harness handed back for the latest tool call: how an MCP result,
                # or an isError result, reached the model.
                "last_tool_result": last_result,
                "tools": offered,
                # Hint key names anywhere in the request, including the system prompt: how
                # an annotation would reach the model if not as a field of the tool.
                "hint_keys_in_request": [key for key in HINT_KEYS if key in text],
                "reply": reply,
            })
        return reply


class _StubHandler(BaseHTTPRequestHandler):
    brain: StubBrain  # set per stub on a subclass
    api = ""

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        raw = self.brain.scrub(self.rfile.read(int(self.headers.get("content-length") or 0)).decode("utf-8", "replace")).encode()
        try:
            request = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            request = {}
        self.answer(raw, request if isinstance(request, dict) else {})

    def do_GET(self) -> None:  # noqa: N802
        # Model listings (OpenAI /v1/models, Gemini /v1beta/models) answer with one stub model.
        if self.path.rstrip("/").endswith("models"):
            self._json({"object": "list", "data": [{"id": "stub", "object": "model"}], "models": [{"name": "models/stub"}]})
            return
        self.send_response(404)
        self.end_headers()

    def answer(self, raw: bytes, request: dict) -> None:
        raise NotImplementedError

    def _json(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse(self, events: list[tuple[str | None, dict | str]]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for name, data in events:
            head = f"event: {name}\n" if name else ""
            body = data if isinstance(data, str) else json.dumps(data)
            self.wfile.write(f"{head}data: {body}\n\n".encode())

    def log_message(self, *_args) -> None:
        pass


def _result(content: Any, is_error: Any = None) -> dict:
    result = {"text": clip(content, 300)}
    if is_error is not None:
        result["is_error"] = bool(is_error)
    return result


def _tool(name: Any, description: Any, schema: Any, extra: dict) -> dict:
    tool = {"name": str(name or ""), "description": description, "input_schema": schema}
    keys = {k: v for k, v in extra.items() if k not in ("name", "description")}
    if keys:
        tool["extra"] = keys
    return tool


class _AnthropicStub(_StubHandler):
    """Anthropic Messages API: POST /v1/messages (and count_tokens), streaming or not."""

    api = "anthropic-messages"

    def answer(self, raw: bytes, request: dict) -> None:
        if "count_tokens" in self.path:
            self._json({"input_tokens": 1})
            return
        offered = [
            _tool(t.get("name"), t.get("description"), t.get("input_schema"),
                  {k: v for k, v in t.items() if k not in ("input_schema", "cache_control")})
            for t in request.get("tools") or [] if isinstance(t, dict)
        ]
        blocks = [
            block for m in request.get("messages") or [] if isinstance(m.get("content"), list)
            for block in m["content"] if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        results = len(blocks)
        last = _result(blocks[-1].get("content"), blocks[-1].get("is_error", False)) if blocks else None
        reply = self.brain.decide(self.api, self.path, raw, request, offered, results, last)
        if "tool" in reply:
            block = {"type": "tool_use", "id": f"toolu_mcp_mirror_{results + 1}", "name": reply["tool"], "input": reply["arguments"]}
            stop = "tool_use"
        else:
            block, stop = {"type": "text", "text": reply["text"]}, "end_turn"
        message = {
            "id": f"msg_mcp_mirror_{len(self.brain.log)}", "type": "message", "role": "assistant",
            "model": request.get("model", "stub"), "content": [], "stop_reason": None,
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if not request.get("stream"):
            self._json({**message, "content": [block], "stop_reason": stop})
            return
        if block["type"] == "text":
            start, delta = {"type": "text", "text": ""}, {"type": "text_delta", "text": block["text"]}
        else:
            start = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        self._sse([
            ("message_start", {"type": "message_start", "message": message}),
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": start}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                               "usage": {"output_tokens": 1}}),
            ("message_stop", {"type": "message_stop"}),
        ])


class _OpenAIStub(_StubHandler):
    """OpenAI Responses (POST /v1/responses) and Chat Completions (POST /v1/chat/completions)."""

    api = "openai"

    def answer(self, raw: bytes, request: dict) -> None:
        if self.path.split("?")[0].rstrip("/").endswith("/responses"):
            self._responses(raw, request)
        else:
            self._chat(raw, request)

    def _chat(self, raw: bytes, request: dict) -> None:
        offered = []
        for t in request.get("tools") or []:
            function = (t or {}).get("function") or {}
            offered.append(_tool(function.get("name"), function.get("description"), function.get("parameters"),
                                 {**{k: v for k, v in function.items() if k != "parameters"},
                                  **{k: v for k, v in t.items() if k not in ("function", "type")}}))
        answers = [m for m in request.get("messages") or [] if isinstance(m, dict) and m.get("role") == "tool"]
        results = len(answers)
        last = _result(answers[-1].get("content")) if answers else None
        reply = self.brain.decide("openai-chat", self.path, raw, request, offered, results, last)
        call_id = f"call_mcp_mirror_{results + 1}"
        base = {"id": f"chatcmpl-mcp-mirror-{len(self.brain.log)}", "created": int(time.time()),
                "model": request.get("model", "stub")}
        if "tool" in reply:
            call = {"id": call_id, "type": "function",
                    "function": {"name": reply["tool"], "arguments": json.dumps(reply["arguments"])}}
            message, finish = {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": reply["text"]}, "stop"
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        if not request.get("stream"):
            self._json({**base, "object": "chat.completion", "usage": usage,
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
            return
        chunk = {**base, "object": "chat.completion.chunk"}
        delta: dict = {"role": "assistant"}
        if "tool" in reply:
            delta["tool_calls"] = [{"index": 0, **message["tool_calls"][0]}]
        else:
            delta["content"] = reply["text"]
        self._sse([
            (None, {**chunk, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}),
            (None, {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage}),
            (None, "[DONE]"),
        ])

    def _responses(self, raw: bytes, request: dict) -> None:
        # A Responses `namespace` tool groups functions under one name (a harness may send
        # one per MCP server). Each function is offered as <namespace>__<function>, and a
        # call to one names the function and its namespace.
        offered, namespaces = [], {}
        for t in request.get("tools") or []:
            if not isinstance(t, dict):
                continue
            if t.get("type") == "namespace":
                for inner in t.get("tools") or []:
                    if not isinstance(inner, dict):
                        continue
                    name = f"{t.get('name')}__{inner.get('name')}"
                    namespaces[name] = (t.get("name"), inner.get("name"))
                    offered.append(_tool(name, inner.get("description"), inner.get("parameters"),
                                         {**{k: v for k, v in inner.items() if k != "parameters"},
                                          "namespace": t.get("name")}))
                continue
            offered.append(_tool(t.get("name"), t.get("description"), t.get("parameters"),
                                 {k: v for k, v in t.items() if k != "parameters"}))
        items = request.get("input") if isinstance(request.get("input"), list) else []
        outputs = [item for item in items if isinstance(item, dict) and item.get("type") == "function_call_output"]
        results = len(outputs)
        last = _result(outputs[-1].get("output")) if outputs else None
        reply = self.brain.decide("openai-responses", self.path, raw, request, offered, results, last)
        if "tool" in reply:
            item = {"type": "function_call", "id": f"fc_mcp_mirror_{results + 1}", "call_id": f"call_mcp_mirror_{results + 1}",
                    "name": reply["tool"], "arguments": json.dumps(reply["arguments"]), "status": "completed"}
            if reply["tool"] in namespaces:
                item["namespace"], item["name"] = namespaces[reply["tool"]]
        else:
            item = {"type": "message", "id": f"msg_mcp_mirror_{len(self.brain.log)}", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text", "text": reply["text"], "annotations": []}]}
        response = {
            "id": f"resp_mcp_mirror_{len(self.brain.log)}", "object": "response", "created_at": int(time.time()),
            "status": "completed", "model": request.get("model", "stub"), "output": [item],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        if not request.get("stream"):
            self._json(response)
            return
        events: list[tuple[str | None, dict | str]] = [
            ("response.created", {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}}),
            ("response.output_item.added", {"type": "response.output_item.added", "output_index": 0,
                                            "item": {**item, "status": "in_progress"}}),
        ]
        if "tool" in reply:
            events.append(("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done", "output_index": 0, "item_id": item["id"],
                "arguments": item["arguments"]}))
        else:
            events.append(("response.output_text.delta", {
                "type": "response.output_text.delta", "output_index": 0, "content_index": 0,
                "item_id": item["id"], "delta": reply["text"]}))
        events += [
            ("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": item}),
            ("response.completed", {"type": "response.completed", "response": response}),
        ]
        self._sse(events)


class _GeminiStub(_StubHandler):
    """Gemini generateContent and streamGenerateContent (?alt=sse), with functionCall parts."""

    api = "gemini"

    def answer(self, raw: bytes, request: dict) -> None:
        if ":countTokens" in self.path:
            self._json({"totalTokens": 1})
            return
        offered = []
        for group in request.get("tools") or []:
            for declaration in (group or {}).get("functionDeclarations") or []:
                schema = declaration.get("parametersJsonSchema", declaration.get("parameters"))
                offered.append(_tool(declaration.get("name"), declaration.get("description"), schema,
                                     {k: v for k, v in declaration.items() if k not in ("parameters", "parametersJsonSchema")}))
        responses = [
            part["functionResponse"] for content in request.get("contents") or [] if isinstance(content, dict)
            for part in content.get("parts") or [] if isinstance(part, dict) and "functionResponse" in part
        ]
        results = len(responses)
        last = _result((responses[-1] or {}).get("response")) if responses else None
        reply = self.brain.decide(self.api, self.path, raw, request, offered, results, last)
        part = ({"functionCall": {"name": reply["tool"], "args": reply["arguments"]}}
                if "tool" in reply else {"text": reply["text"]})
        body = {
            "candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            "modelVersion": "stub",
        }
        if "streamGenerateContent" in self.path:
            self._sse([(None, body)])
        else:
            self._json(body)


# The handler per stub kind. Which variables point a harness at it is the harness
# module's env(); harnesses.stub_env has the conventional ones.
STUBS: dict[str, type[_StubHandler]] = {"anthropic": _AnthropicStub, "openai": _OpenAIStub, "gemini": _GeminiStub}


LEVEL_1_UNATTENDED_SUFFIX = "-unattended"


def level_1_unattended(harness: Any) -> dict[str, str]:
    """``{"unattended": <why>}`` when the module declares LEVEL1_UNATTENDED, else {}.

    A harness that starts no project MCP server headless until a switch approves them
    (Cursor's --approve-mcps) measures level 1 twice: the plan's sessions as they are,
    the documented default, and once more in the inventory's unattended mode, with
    ``-unattended`` on the session id, the way level 2 records read_only-unattended.
    The reason is published in session.json and read into every level-1 cell's evidence."""
    reason = getattr(harness, "LEVEL1_UNATTENDED", None)
    if reason is None:
        return {}
    if not isinstance(reason, str) or not reason.strip():
        raise HarnessError(f"LEVEL1_UNATTENDED must be a non-empty string naming the switch, not {reason!r}")
    return {"unattended": reason}


def level_1_modes(harness: Any, session_id: str) -> list[tuple[str | None, str]]:
    """``[(mode, session id)]`` to run for one planned level-1 session: the default, then
    the unattended pass when the module declares LEVEL1_UNATTENDED."""
    modes: list[tuple[str | None, str]] = [(None, session_id)]
    if level_1_unattended(harness):
        modes.append(("unattended", session_id + LEVEL_1_UNATTENDED_SUFFIX))
    return modes


def level_1_record(harness: Any, session_id: str, revision: str, mode: str | None, record: dict) -> dict:
    """One level-1 session for the manifest; ``mode`` and its switches only when the pass
    was not the default, the way level-2 sessions carry mode_args."""
    extra = {"mode": mode, "mode_args": list(harness.mode_args(mode))} if mode else {}
    return {"id": session_id, "level": 1, "revision": revision, **extra, **record}


def level_1_reply_delay(harness: Any) -> float:
    """Seconds the level-1 stub holds its reply, from the module's LEVEL1_REPLY_DELAY_S.

    0 unless the module declares it. A harness that connects its MCP servers in the
    background and does not wait for them before the first turn (Pi since 0.99.2) ends
    the session before any server has answered when the stub replies at once; the delay
    keeps the turn open long enough for the connect, so the capture shows the harness's
    MCP behaviour instead of the stub's speed. It is recorded in session.json and on
    every level-1 cell's evidence. Level 2 never uses it.
    """
    delay = getattr(harness, "LEVEL1_REPLY_DELAY_S", 0) or 0
    if not isinstance(delay, (int, float)) or delay < 0:
        raise HarnessError(f"LEVEL1_REPLY_DELAY_S must be a non-negative number, not {delay!r}")
    return float(delay)


class ModelStub:
    def __init__(self, kind: str, script: list[dict] | None = None, text: str = "ready",
                 reply_delay_s: float = 0.0) -> None:
        base = STUBS[kind]
        self.kind = kind
        self.brain = StubBrain(script, text, reply_delay_s)
        handler = type(base.__name__, (base,), {"brain": self.brain})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "ModelStub":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()


def _listed_tools(captures: dict[str, Path]) -> dict[str, list[str]]:
    """Tool names each server answered in tools/list, from the recorder captures."""
    listed: dict[str, list[str]] = {}
    for server, path in captures.items():
        names: list[str] = []
        if path.exists():
            for line in path.read_text().splitlines():
                record = json.loads(line) if line.strip() else {}
                if record.get("type") == "message" and record.get("method") == "tools/list" and record.get("kind") == "response":
                    names += [t.get("name") for t in (record["message"].get("result") or {}).get("tools") or []]
        listed[server] = names
    return listed


def reduce_model_log(log: list[dict], listed: dict[str, list[str]]) -> list[dict]:
    """Keep full definitions only for tools a fixture listed, and each set once per change.

    A harness's own tools (Bash, Read, ...) keep only their names: what they say is the
    harness's business, and what a cell compares is what reached the model of the
    fixture's tools. A request offering exactly what the previous one did says so instead
    of repeating it.
    """
    reduced: list[dict] = []
    previous: list[dict] | None = None
    for index, entry in enumerate(log, 1):
        names = [tool["name"] for tool in entry["tools"]]
        mapped = {
            name: (server, tool)
            for server, tools in listed.items() for tool in tools
            if (name := cells.offered_name(names, server, tool))
        }
        fixture_tools = [
            {**tool, "server": mapped[tool["name"]][0], "fixture_tool": mapped[tool["name"]][1]}
            for tool in entry["tools"] if tool["name"] in mapped
        ]
        record = {k: v for k, v in entry.items() if k != "tools"} | {"request": index}
        record["offered"] = names
        if fixture_tools == previous:
            record["fixture_tools"] = "same as the previous request"
        else:
            record["fixture_tools"] = fixture_tools
            previous = fixture_tools
        reduced.append(record)
    return reduced


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------
#
# The operator's own harness configs carry real MCP servers with real auth. A session
# must never see them, so each one runs in an isolated home, and four checks prove it:
#
#   home            HOME and every config-dir variable point inside a new temp dir, and
#                   nothing from the real home is copied in.
#   servers_listed  the harness's own `mcp list` there reports exactly the five fixtures.
#   tools_offered   every MCP tool the model was offered belongs to one of the five.
#   egress          every attempt to reach a host other than 127.0.0.1 went to the sink,
#                   was refused, and is recorded; one not allowlisted fails the session.

# Variables pointed inside the isolated home, relative to it. A module adds its own in
# CONFIG_DIR_ENV; each gets a directory named after the variable.
ISOLATED_DIRS = {
    "HOME": ".",
    "XDG_CONFIG_HOME": ".config",
    "XDG_DATA_HOME": ".local/share",
    "XDG_CACHE_HOME": ".cache",
    "XDG_STATE_HOME": ".local/state",
    "CODEX_HOME": ".codex",
    "CLAUDE_CONFIG_DIR": ".claude",
    "GEMINI_CLI_HOME": "gemini-home",
}
# All that is taken from the real environment: where binaries are, and the locale. A
# module adds what its harness cannot run without in PASSED_ENV.
PASSED_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "TZ", "USER", "LOGNAME", "SHELL")
PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
LOCAL_HOSTS = "127.0.0.1,localhost"


class _SinkHandler(BaseHTTPRequestHandler):
    """A proxy that forwards nothing: each request is logged by host and refused. A live
    run names the hosts the harness's service lives on (LIVE_EGRESS_ALLOWED); a CONNECT
    to one of those is tunnelled and logged as forwarded, everything else stays refused."""

    attempts: list[dict]  # set per sink on a subclass
    forward: frozenset[str] = frozenset()

    def do_CONNECT(self) -> None:
        host, _, port = self.path.rpartition(":")
        host = host.strip("[]").lower()
        if host not in self.forward or not port.isdigit():
            self._refuse()
            return
        self.attempts.append({"host": host, "port": int(port), "method": "CONNECT", "forwarded": True})
        try:
            upstream = socket.create_connection((host, int(port)), timeout=30)
        except OSError as exc:
            self.send_response(502, f"mcp-mirror egress sink: {exc}")
            self.send_header("content-length", "0")
            self.send_header("connection", "close")
            self.end_headers()
            self.close_connection = True
            return
        self.send_response(200, "Connection established")
        self.end_headers()
        self.wfile.flush()
        self.close_connection = True
        _tunnel(self.connection, upstream)

    def _refuse(self) -> None:
        if self.command == "CONNECT":
            host, _, port = self.path.rpartition(":")
        else:
            target = self.path if "://" in self.path else f"http://{self.headers.get('host', '')}{self.path}"
            from urllib.parse import urlsplit

            parts = urlsplit(target)
            host, port = parts.hostname or "", str(parts.port or (443 if parts.scheme == "https" else 80))
        self.attempts.append({"host": host.strip("[]").lower(), "port": int(port) if port.isdigit() else None,
                              "method": self.command})
        self.send_response(403, "mcp-mirror egress sink: refused")
        self.send_header("content-length", "0")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _refuse

    def log_message(self, *_args) -> None:
        pass


def _tunnel(client: socket.socket, upstream: socket.socket) -> None:
    """Bytes both ways until either side closes; the TLS inside is opaque to the sink."""

    def copy(source: socket.socket, target: socket.socket) -> None:
        try:
            while chunk := source.recv(65536):
                target.sendall(chunk)
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                target.shutdown(socket.SHUT_WR)

    back = threading.Thread(target=copy, args=(upstream, client), daemon=True)
    back.start()
    copy(client, upstream)
    back.join()
    upstream.close()


class EgressSink:
    """The local sink every proxy variable points at, for one session.

    ``forward`` is ``{host: reason}``: the hosts a live run may reach, from the module's
    LIVE_EGRESS_ALLOWED. Empty, as in every stubbed session, and nothing is forwarded.
    """

    def __init__(self, forward: dict[str, str] | None = None) -> None:
        self.attempts: list[dict] = []
        self.forward = {host.lower(): reason for host, reason in (forward or {}).items()}
        handler = type("_Sink", (_SinkHandler,), {"attempts": self.attempts, "forward": frozenset(self.forward)})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "EgressSink":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    def env(self) -> dict[str, str]:
        env = {name: self.url for name in PROXY_ENV} | {name.lower(): self.url for name in PROXY_ENV}
        return env | {"NO_PROXY": LOCAL_HOSTS, "no_proxy": LOCAL_HOSTS}

    def check(self, harness: Any) -> dict:
        allowed = dict(getattr(harness, "EGRESS_ALLOWED", {}) or {}) | self.forward
        attempts = [{**a, "allowed": allowed.get(a["host"])} for a in self.attempts]
        # One line per host, method and port, with how often: the log is committed.
        counted: dict[tuple, dict] = {}
        for attempt in attempts:
            key = (attempt["host"], attempt["port"], attempt["method"])
            counted.setdefault(key, {**attempt, "count": 0})["count"] += 1
        refused = sorted({a["host"] for a in attempts if not a["allowed"]})
        forwarded = sorted({a["host"] for a in attempts if a.get("forwarded")})
        return {"passed": not refused, "attempts": sorted(counted.values(), key=lambda a: (a["host"], a["method"])),
                "not_allowed": refused, **({"forwarded": forwarded} if self.forward else {})}


# A module can declare that the proxy variables must be withheld on a transport,
# WITHHOLD_PROXY_ENV = {"http": "<why>"}, when setting them breaks the harness there
# (Gemini CLI's HTTP transport stops after initialize whenever NO_PROXY is set). The
# sink then cannot see what the harness tries to reach, so egress is held by another
# route: the run is started under a macOS sandbox-exec profile that denies every
# outbound connection except loopback, and the system resolver's socket, and a probe
# inside the same profile must show both denied before the session counts as fresh.
# The trade-off, written into fresh.json: nothing can leave the machine, but a refused
# attempt is not listed by host the way the sink lists it.
SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
(deny network-outbound (remote unix-socket (path-literal "/private/var/run/mDNSResponder")))
"""
# TEST-NET-1 (RFC 5737): never routed, so the probe sends nothing even unsandboxed.
SANDBOX_PROBE = """
import errno, socket
s = socket.socket(); s.settimeout(3)
try:
    s.connect(("192.0.2.1", 443)); tcp = "connected"
except OSError as e:
    tcp = errno.errorcode.get(e.errno, str(e.errno))
try:
    socket.getaddrinfo("example.com", 443); dns = "resolved"
except OSError:
    dns = "failed"
print(tcp, dns)
"""
SANDBOX_TRADEOFF = (
    "The proxy variables were withheld on this transport, so the egress sink could not see what the "
    "harness tried to reach. Egress was held instead by a sandbox-exec profile allowing outbound "
    "connections to loopback only, with the system resolver denied, and verified by a probe in the same "
    "profile. Nothing could leave the machine, but refused attempts are not listed by host."
)


def withheld_proxy(harness: Any, transport: str) -> str | None:
    """The module's reason for withholding proxy variables on this transport, if any."""
    return (getattr(harness, "WITHHOLD_PROXY_ENV", {}) or {}).get(transport)


class LoopbackSandbox:
    """sandbox-exec with SANDBOX_PROFILE, for a session whose proxy variables are withheld."""

    def __init__(self, root: Path, reason: str) -> None:
        self.reason = reason
        self.profile = root / "loopback-only.sb"
        self.profile.write_text(SANDBOX_PROFILE)

    def prefix(self) -> list[str]:
        return ["sandbox-exec", "-f", str(self.profile)]

    def probe(self) -> dict:
        if not shutil.which("sandbox-exec"):
            return {"tcp": "no sandbox-exec on this machine", "dns": None}
        try:
            done = subprocess.run([*self.prefix(), sys.executable, "-c", SANDBOX_PROBE],
                                  capture_output=True, text=True, timeout=30)
            tcp, _, dns = done.stdout.strip().partition(" ")
        except (OSError, subprocess.TimeoutExpired) as exc:
            tcp, dns = f"probe did not run: {exc}", None
        return {"tcp": tcp, "dns": dns}

    def check(self, sink: "EgressSink", harness: Any, probe: dict) -> dict:
        """The sink still hears the pre-run listing, which keeps its proxy variables."""
        listed = sink.check(harness)
        held = probe.get("tcp") == "EPERM" and probe.get("dns") == "failed"
        return {**listed, "passed": listed["passed"] and held, "route": "sandbox",
                "proxy_env": f"withheld on this transport: {self.reason}",
                "sandbox": {"profile": "sandbox-exec: outbound to loopback only, system resolver denied",
                            "probe": f"connect 192.0.2.1:443 -> {probe.get('tcp')}; resolve example.com -> {probe.get('dns')}"},
                "tradeoff": SANDBOX_TRADEOFF}


# A live session runs a real model with a real key, and a real model runs shell commands:
# in the first live Cursor run the model read outside its throwaway home by absolute
# path (the operator's repository worktree, ~/.local/share/cursor-agent, the operator's
# own cursor-agent binary). So a live harness is started under a second sandbox-exec
# profile, over the files this time. Nothing is readable, writable or executable except
# the session's isolated home and workdir, its capture directory, the fixtures and their
# interpreters, the harness's own install, and the system's binaries and libraries
# (/usr, /bin, /System, ...). Names can still be stat'ed, which path resolution needs;
# contents cannot be read and directories cannot be listed. Outbound TCP reaches
# loopback only, so every request goes through the sink and is listed by host; the
# system resolver stays reachable. Later rules win, so each allow narrows a deny above it.
# The root directory itself is readable (every process reads it at start, and aborts
# without it); no other directory outside the allowed paths can be listed. Loopback is
# open only to the session's own ports (the egress sink, the HTTP recorders), rendered
# per session, so another local service (a database, a dev server, Docker's API) is as
# unreachable as the internet. Signals reach only the session's own processes, and the
# process table shows only them: pid info, open files and argv of every other process
# (ps, lsof, pgrep, kill -0 on the operator's processes) are denied. The argv read
# (sysctl KERN_PROCARGS2) is held only when the sysctl denial and the process-info
# denial stand together; with the process-info filter alone it still answers for
# every process of the same user, and the session's own argv stays readable.
LIVE_SANDBOX_PROFILE = """(version 1)
(allow default)
(deny file-read* file-write*)
(allow file-read-metadata)
(allow file-read* (literal "/"))
(allow file-read* (literal "{root}"))
(allow file-read* {system})
(allow file-read* {read})
(allow file-read* file-write* {write})
(allow file-write* (subpath "/dev"))
(deny process-exec)
(allow process-exec {system_exec})
(allow process-exec {exec})
(deny process-exec {desktop_exec})
(deny network-outbound)
(allow network-outbound {ports})
(allow network-outbound (remote unix-socket (subpath "{home}")))
(deny mach-lookup)
(deny appleevent-send)
(deny user-preference-read user-preference-write)
(deny signal)
(allow signal (target self) (target same-sandbox))
(deny process-info*)
(allow process-info* (target self) (target same-sandbox))
(deny sysctl-read (sysctl-name-prefix "kern.procargs"))
"""
LIVE_SANDBOX_SYSTEM = ("/usr", "/bin", "/sbin", "/System", "/Library", "/private/etc", "/dev",
                       "/private/var/db", "/private/var/select", "/private/var/run")
LIVE_SANDBOX_SYSTEM_EXEC = ("/usr/bin", "/usr/sbin", "/usr/lib", "/usr/libexec", "/bin", "/sbin", "/System")
# System binaries a live model must not run even though they are under /usr/bin: each
# reaches the operator's desktop through Mach/XPC (the clipboard, LaunchServices opening a
# URL in the browser outside the egress sink, Apple Events to other apps, keychain
# dialogs, the real preference domains, the screen). The Mach, Apple Event and
# preference denials above close the services themselves; this closes the front doors.
LIVE_SANDBOX_DESKTOP_EXEC = ("/usr/bin/open", "/usr/bin/osascript", "/usr/bin/security", "/usr/bin/pbcopy",
                             "/usr/bin/pbpaste", "/usr/bin/defaults", "/usr/sbin/screencapture")
# Mach services a live harness may look up: none. The pinned cursor-agent, with a
# placeholder key and nothing forwarded, runs to its API call and stops with mach-lookup
# denied outright, and the fixtures and recorder run under the same denial in the
# contained tests. (sandbox-exec's denial log is not readable without privileges on
# this machine and its trace directive writes nothing, so the list was derived by
# running under the denial, not from a log.) A service a future build needs goes here
# with its reason, and the profile text changes with it, so the published hash does too.
LIVE_SANDBOX_MACH_ALLOWED: tuple[str, ...] = ()
# Run inside the profile before the session: each denial, and that the allowed home still
# works, so a broken profile is caught rather than a session that silently did nothing.
LIVE_SANDBOX_PROBE = """
import ctypes, errno, json, os, socket, subprocess, sys
canary, binary, inside, outside_sock, inside_sock = sys.argv[1:6]
desktop = json.loads(sys.argv[6])
sink_port, other_port, outside_pid = (int(a) for a in sys.argv[7:10])
def code(exc):
    return errno.errorcode.get(exc.errno, str(exc.errno)) if getattr(exc, "errno", None) else type(exc).__name__
def attempt(action):
    try:
        return action()
    except OSError as exc:
        return code(exc)
def tcp(address):
    s = socket.socket(); s.settimeout(3)
    return attempt(lambda: s.connect(address) or "connected")
def unix(path):
    s = socket.socket(socket.AF_UNIX); s.settimeout(3)
    return attempt(lambda: s.connect(path) or "connected")
out = {
    "read_outside": attempt(lambda: open(canary).read() and "read"),
    "list_outside": attempt(lambda: os.listdir(os.path.dirname(canary)) and "listed"),
    "write_outside": attempt(lambda: open(canary + ".w", "w").write("x") and "written"),
    "exec_outside": attempt(lambda: subprocess.run([binary, "-c", "true"]).returncode == 0 and "ran"),
    "read_inside": attempt(lambda: open(inside).read() and "read"),
    "write_inside": attempt(lambda: open(inside + ".w", "w").write("x") and "written"),
    "tcp_outside": tcp(("192.0.2.1", 443)),
    "tcp_sink": tcp(("127.0.0.1", sink_port)),
    "tcp_other_local": tcp(("127.0.0.1", other_port)),
    "unix_outside": unix(outside_sock),
    "signal_outside": attempt(lambda: os.kill(outside_pid, 0) or "signalled"),
}
child = subprocess.Popen(["/bin/sleep", "30"])
child.terminate()
out["signal_child"] = "killed" if child.wait(timeout=10) < 0 else "survived"
lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
def argv_of(pid):
    buf = ctypes.create_string_buffer(1 << 20); size = ctypes.c_size_t(1 << 20)
    rc = lib.sysctl((ctypes.c_int * 3)(1, 49, pid), 3, buf, ctypes.byref(size), None, 0)  # KERN_PROCARGS2
    return "read" if rc == 0 and size.value else "denied"
def path_of(pid):
    buf = ctypes.create_string_buffer(4096)
    return "read" if lib.proc_pidpath(pid, buf, 4096) > 0 else "denied"
out["argv_outside"] = argv_of(outside_pid)
out["argv_self"] = argv_of(os.getpid())
out["pidinfo_outside"] = path_of(outside_pid)
out["pidinfo_self"] = path_of(os.getpid())
listener = socket.socket(socket.AF_UNIX); listener.bind(inside_sock); listener.listen(1)
out["unix_inside"] = unix(inside_sock)
for path in desktop:
    out["exec_" + os.path.basename(path)] = attempt(
        lambda: subprocess.run([path, "--version"], capture_output=True).returncode >= 0 and "ran")
# The services behind those binaries, asked for directly: the pasteboard's Mach port,
# and the operator's own Finder preferences through CoreFoundation.
port = ctypes.c_uint32()
rc = lib.bootstrap_look_up(ctypes.c_uint32.in_dll(lib, "bootstrap_port"), b"com.apple.pasteboard.1", ctypes.byref(port))
out["mach_pasteboard"] = "denied" if rc else "looked up"
cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
cf.CFStringCreateWithCString.restype = ctypes.c_void_p
cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
cf.CFPreferencesCopyKeyList.restype = ctypes.c_void_p
cf.CFPreferencesCopyKeyList.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
cfstr = lambda text: cf.CFStringCreateWithCString(None, text, 0x08000100)
keys = cf.CFPreferencesCopyKeyList(cfstr(b"com.apple.finder"), cfstr(b"kCFPreferencesCurrentUser"), cfstr(b"kCFPreferencesAnyHost"))
out["prefs_read"] = "read" if keys else "denied"
print(json.dumps(out))
"""
LIVE_SANDBOX_HELD = {"read_outside": "EPERM", "list_outside": "EPERM", "write_outside": "EPERM",
                     "exec_outside": "EPERM", "read_inside": "read", "write_inside": "written", "tcp_outside": "EPERM",
                     "tcp_sink": "connected", "tcp_other_local": "EPERM",
                     "unix_outside": "EPERM", "unix_inside": "connected", "signal_outside": "EPERM",
                     "signal_child": "killed", "argv_outside": "denied", "argv_self": "read",
                     "pidinfo_outside": "denied", "pidinfo_self": "read",
                     **{"exec_" + Path(p).name: "EPERM" for p in LIVE_SANDBOX_DESKTOP_EXEC},
                     "mach_pasteboard": "denied", "prefs_read": "denied"}


def _interpreter_roots(executable: Path) -> list[Path]:
    """The directories an interpreter needs: its own prefix (a venv) and, from the venv's
    pyvenv.cfg, the installation it was made from (uv's managed CPython under ~/.local)."""
    prefix = Path(executable).absolute().parent.parent  # the venv, before the symlink is followed
    roots = [prefix, Path(executable).resolve().parent.parent]
    cfg = prefix / "pyvenv.cfg"
    if cfg.is_file():
        for line in cfg.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "home" and value.strip():
                home = Path(value.strip()).resolve()
                roots.append(home.parent if home.name == "bin" else home)
    return roots


class LiveSandbox:
    """sandbox-exec with LIVE_SANDBOX_PROFILE around a live session's harness.

    ``read`` is readable, ``write`` readable and writable, ``exec`` executable; every path
    is resolved, as the kernel sees it. Unix sockets connect only under the session's
    home (Docker's and other apps' sockets sit under /var/run and /tmp by well-known
    paths); Mach lookups, Apple Events and the user's preference domains are denied, and
    the desktop-facing system binaries (LIVE_SANDBOX_DESKTOP_EXEC) cannot be run. Loopback
    TCP reaches only ``ports`` (the egress sink and the session's recorders); signals and
    process information stop at the session's own processes. The repository root itself is readable as a
    literal (the entry names of that one directory, nothing under it): the recorder
    starts each fixture with ``cd`` into it, and getcwd needs the directory, or the
    fixture's Python aborts on import. The profile text is hashed as a template, before
    the paths go in, so the hash names the rules and the check lists the paths.
    """

    PROFILE_SHA256 = hashlib.sha256(LIVE_SANDBOX_PROFILE.encode()).hexdigest()

    def __init__(self, home: "IsolatedHome", session_dir: Path, harness: Any, python: str,
                 ports: Iterable[int]) -> None:
        self.home = home
        self.ports = sorted({int(p) for p in ports})
        if not self.ports:
            raise HarnessError("a live session needs at least one loopback port to allow: the egress sink's")
        self.tmp = home.root / "tmp"
        self.tmp.mkdir(exist_ok=True)
        binary = harnesses.resolve_binary(getattr(harness, "BINARY", ""))
        read = [ROOT / "fixtures", ROOT / "src", *_interpreter_roots(Path(sys.executable))]
        if "/" in python:
            read += _interpreter_roots(ROOT / python if not python.startswith("/") else Path(python))
        if binary is not None:
            read.append(binary.resolve().parent)
        self.read = _unique_resolved(read)
        self.write = _unique_resolved([home.root, session_dir])
        self.exec = _unique_resolved([p for p in self.read if p not in (ROOT / "fixtures", ROOT / "src")] + [home.root])
        self.profile = home.root / "live-contained.sb"
        self.profile.write_text(self.render())

    def render(self) -> str:
        def rules(paths: Iterable[Path | str], kind: str = "subpath") -> str:
            return " ".join(f'({kind} "{p}")' for p in paths)
        return LIVE_SANDBOX_PROFILE.format(
            root=ROOT.resolve(), home=self.home.root.resolve(), system=rules(LIVE_SANDBOX_SYSTEM),
            read=rules(self.read), write=rules(self.write), system_exec=rules(LIVE_SANDBOX_SYSTEM_EXEC),
            exec=rules(self.exec), desktop_exec=rules(LIVE_SANDBOX_DESKTOP_EXEC, "literal"),
            ports=" ".join(f'(remote ip "localhost:{port}")' for port in self.ports))

    def prefix(self) -> list[str]:
        return ["sandbox-exec", "-f", str(self.profile)]

    def env(self) -> dict[str, str]:
        """The harness's temp dir inside the home: /var/folders is not writable in here."""
        return {"TMPDIR": str(self.tmp)}

    def probe(self) -> dict:
        """LIVE_SANDBOX_PROBE's result, against a canary and a binary outside every allowed path."""
        if not shutil.which("sandbox-exec"):
            return {"error": "no sandbox-exec on this machine"}
        outside = Path(tempfile.mkdtemp(prefix="mcp-mirror-canary-")).resolve()
        inside = self.home.root / "probe-inside.txt"
        inside_sock = self.home.root / "probe.sock"
        listener = socket.socket(socket.AF_UNIX)
        other = socket.socket()  # a local service the session has no business with
        try:
            other.bind(("127.0.0.1", 0))
            other.listen(1)
            (outside / "canary.txt").write_text("canary\n")
            shutil.copy("/bin/sh", outside / "sh")
            inside.write_text("inside\n")
            listener.bind(str(outside / "probe.sock"))  # a socket outside, listening, as another app's would
            listener.listen(1)
            done = subprocess.run([*self.prefix(), sys.executable, "-I", "-c", LIVE_SANDBOX_PROBE,
                                   str(outside / "canary.txt"), str(outside / "sh"), str(inside),
                                   str(outside / "probe.sock"), str(inside_sock), json.dumps(LIVE_SANDBOX_DESKTOP_EXEC),
                                   str(self.ports[0]), str(other.getsockname()[1]), str(os.getpid())],
                                  capture_output=True, text=True, timeout=60)
            if done.returncode != 0:
                return {"error": f"probe exited {done.returncode}: {done.stderr.strip()[-200:]}"}
            return json.loads(done.stdout.strip().splitlines()[-1])
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
            return {"error": f"probe did not run: {exc}"}
        finally:
            listener.close()
            other.close()
            shutil.rmtree(outside, ignore_errors=True)
            inside.unlink(missing_ok=True)
            (self.home.root / "probe-inside.txt.w").unlink(missing_ok=True)
            inside_sock.unlink(missing_ok=True)

    def check(self, probe: dict) -> dict:
        held = all(probe.get(key) == want for key, want in LIVE_SANDBOX_HELD.items())

        def shown(path: Path) -> str:
            if path.is_relative_to(self.home.root):
                return "<session home>" + (f"/{path.relative_to(self.home.root)}" if path != self.home.root else "")
            return self.home.scrub(_shown_path(path))
        return {"passed": held, "route": "sandbox-exec", "profile_sha256": self.PROFILE_SHA256,
                "profile": "LIVE_SANDBOX_PROFILE in scripts/measure_harness.py, hashed before the paths go in",
                "read": [shown(p) for p in self.read], "write": [shown(p) for p in self.write],
                "exec": [shown(p) for p in self.exec], "root_names_only": ".", "system": list(LIVE_SANDBOX_SYSTEM),
                "system_exec": list(LIVE_SANDBOX_SYSTEM_EXEC), "desktop_exec_denied": list(LIVE_SANDBOX_DESKTOP_EXEC),
                "unix_sockets": "<session home>", "mach_lookup_allowed": list(LIVE_SANDBOX_MACH_ALLOWED),
                "appleevent_send": "denied", "user_preferences": "denied", "loopback_ports": list(self.ports),
                "signals": "<session processes>", "process_info": "<session processes>",
                "probe": probe, "expected": dict(LIVE_SANDBOX_HELD)}


def _loopback_port(url: str) -> int:
    port = urllib.parse.urlsplit(url).port
    if port is None:
        raise HarnessError(f"no port in {url}")
    return port


def _unique_resolved(paths: Iterable[Path]) -> list[Path]:
    """Resolved, in order, without duplicates or a path already inside a listed one."""
    seen: list[Path] = []
    for path in paths:
        resolved = Path(path).resolve()
        if not any(resolved.is_relative_to(root) for root in seen):
            seen.append(resolved)
    return seen


def _shown_path(path: Path) -> str:
    """A sandbox path as published: relative to the repository, under ~, or as it is."""
    if path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT)) or "."
    return str(path)


def _group_members(pgid: int) -> list[dict]:
    """Every process still in the group, by pid and command name (pgrep, outside any sandbox)."""
    try:
        done = subprocess.run(["pgrep", "-l", "-g", str(pgid)], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    members = []
    for line in done.stdout.splitlines():
        pid, _, name = line.strip().partition(" ")
        if pid.isdigit():
            members.append({"pid": int(pid), "command": name})
    return members


def _session_strays(sid: int, group: int) -> list[dict]:
    """Processes in the harness's session but outside its group: a child that called setpgid.

    The harness's session id is its pid (start_new_session), and only a member of the
    session can move itself out of the group with setpgid; one that calls setsid leaves
    the session and this cannot find it afterwards (its parent becomes launchd), which is
    the limit of a post-exit scan.
    """
    try:
        done = subprocess.run(["ps", "-axo", "pid=,pgid=,comm="], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    strays = []
    for line in done.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        pid, pgid, name = int(parts[0]), int(parts[1]), parts[2]
        if pgid == group or pid == os.getpid():
            continue
        with contextlib.suppress(ProcessLookupError, PermissionError):
            if os.getsid(pid) == sid:
                strays.append({"pid": pid, "command": name})
    return strays


def _reap_process_group(pgid: int, timed_out: bool, exited_at: float, grace: float = 5.0) -> dict:
    """What outlived the harness in its process group, and that it is gone.

    The harness is started in its own session, so its pid is the group id and every
    child it spawned is in the group. After it has exited (or was killed at the timeout)
    and the stdio recorders have finished, whatever is still in the group after ``grace``
    is killed and recorded under ``outlived`` with its pid, command and the seconds since
    the harness exited. The session passes when everything that outlived it was in that
    group (so inside the same sandbox) and the kill took it: it fails when anything
    survives the kill (``still_alive``) or a process of the same session is found outside
    the group (``escaped``, a child that called setpgid; it is killed too).

    Cursor's CLI (build 2026.10.12) leaves its ``index.js worker-server`` child running
    after it exits, in every session; the runner reaps it here, a few seconds after the
    exit, and the session passes with that recorded.
    """
    deadline = time.monotonic() + grace
    while (outlived := _group_members(pgid)) and time.monotonic() < deadline:
        time.sleep(0.2)
    after_exit = round(time.monotonic() - exited_at, 1)
    if outlived:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while _group_members(pgid) and time.monotonic() < deadline:
            time.sleep(0.2)
    remaining = _group_members(pgid)
    escaped = _session_strays(pgid, pgid)
    for stray in escaped:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(stray["pid"], signal.SIGKILL)
    return {"passed": not remaining and not escaped, "timed_out": timed_out,
            "outlived": [{**m, "after_exit_s": after_exit} for m in outlived],
            "still_alive": [dict(m) for m in remaining],
            "escaped": [dict(m) for m in escaped]}


# Where a live session's home goes. The default temp dir on macOS is ~70 characters
# deep (/private/var/folders/xx/.../T), and Cursor's CLI keeps its per-project state
# under <home>/.cursor/projects only while that path is at most 84 characters, falling
# back to the shared /tmp/.cursor otherwise (the bundle, build 2026.10.12): in the first
# live run every session's state landed there, outside the throwaway home. A home under
# /tmp is short enough to keep it inside, where the sandbox allows it and the session's
# removal takes it away.
SHORT_HOME = "/tmp"


class IsolatedHome:
    """A fresh temp dir standing in for the operator's home, deleted after the session."""

    def __init__(self, harness: Any, isolate: bool = True, base: str | None = None) -> None:
        self.harness = harness
        self.isolate = isolate  # False only in the test that shows the checks catch a leak
        # ``base`` is where the temp dir goes; a live session asks for a short one (SHORT_HOME).
        self.root = Path(tempfile.mkdtemp(prefix="mcp-mirror-home-", dir=base)).resolve()
        self.workdir = self.root / "work"
        self.workdir.mkdir()
        self.dirs = dict(ISOLATED_DIRS)
        for name in getattr(harness, "CONFIG_DIR_ENV", ()) or ():
            self.dirs.setdefault(name, f"config-dirs/{name.lower()}")
        for relative in self.dirs.values():
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        self.copied, self.refused = self._copy_home_files()

    def _copy_home_files(self) -> tuple[list[dict], list[str]]:
        """Copy what a module declares it needs from the real home, if allowlisted with a reason."""
        allowed = dict(getattr(self.harness, "HOME_FILES_ALLOWED", {}) or {})
        copied, refused = [], []
        for relative in getattr(self.harness, "HOME_FILES", ()) or ():
            if relative not in allowed:
                refused.append(relative)
                continue
            source, target = Path.home() / relative, self.root / relative
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            copied.append({"path": relative, "reason": allowed[relative], "present": source.is_file()})
        return copied, refused

    def env(self) -> dict[str, str]:
        """The harness's whole environment before its own env(): nothing else is inherited."""
        passed = PASSED_ENV + tuple(getattr(self.harness, "PASSED_ENV", ()) or ())
        env = {k: v for k, v in os.environ.items() if k in passed}
        if not self.isolate:
            return _scrubbed(self.harness)
        return env | {name: str((self.root / relative).resolve()) for name, relative in self.dirs.items()}

    def contains(self, path: Path) -> bool:
        return Path(path).resolve().is_relative_to(self.root)

    def check(self) -> dict:
        env = self.env()
        pointed = sorted(name for name in self.dirs if name in env and Path(env[name]).is_relative_to(self.root))
        passed = self.isolate and pointed == sorted(self.dirs) and not self.refused
        return {"passed": passed, "isolated": pointed, "copied": self.copied, "refused": self.refused}

    def scrub(self, text: str) -> str:
        """The isolated home as ~, in every form a harness quotes it: resolved or not
        (/private/var vs /var on macOS), and slugged into a directory name the way Claude
        Code names ~/.claude/projects/<cwd>. The real home goes the same way."""
        # Longest form first: replacing "/tmp/x/home" before "/private/tmp/x/home" would
        # leave a stray "/private" in front of the ~.
        roots = {str(self.root), str(self.root).removeprefix("/private")}
        roots |= {"/private" + r for r in list(roots) if not r.startswith("/private")}
        for root in sorted(roots, key=len, reverse=True):
            text = text.replace(root, "~")
        for root in sorted(roots, key=len, reverse=True):
            text = text.replace(re.sub(r"[^A-Za-z0-9]", "-", root), "-home")
        return _scrub_home(text)

    def remove(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _placeholders(servers: dict[str, dict]) -> dict[str, dict]:
    """The same five names, with nothing behind them: a list command that health-checks a
    server must not write the session's captures before the session starts."""
    return {
        name: ({"url": "http://127.0.0.1:9/mcp"} if "url" in server else {"command": "false", "args": []})
        for name, server in servers.items()
    }


def fixture_tool_names(fixtures: dict[str, str], python: str = "python") -> dict[str, list[str]]:
    """{server: [tool name, ...]} as each fixture answers tools/list, every page, under
    ``python`` (the fixture line's interpreter). For a harness whose approval switch
    names tools one by one (write_config's ``tools``); a tool a fixture adds later, such
    as ntf's SENTINEL_ADDED_TOOL, is not in it."""
    probe = (
        "import json,sys,anyio;from mcp import ClientSession,StdioServerParameters;"
        "from mcp.client.stdio import stdio_client\n"
        "async def m():\n"
        " async with stdio_client(StdioServerParameters(command=sys.argv[1],args=[sys.argv[2]])) as (r,w):\n"
        "  async with ClientSession(r,w) as s:\n"
        "   await s.initialize(); names=[]; cursor=None\n"
        "   while True:\n"
        "    page=await s.list_tools(cursor=cursor); names+=[t.name for t in page.tools]; cursor=page.nextCursor\n"
        "    if not cursor: break\n"
        " print(json.dumps(names))\n"
        "anyio.run(m)"
    )
    names: dict[str, list[str]] = {}
    for name, path in fixtures.items():
        out = subprocess.run([sys.executable, "-c", probe, python, path], cwd=ROOT, capture_output=True,
                             text=True, timeout=60, env=os.environ | _venv_path_env())
        if out.returncode != 0:
            raise HarnessError(f"listing the {name} fixture's tools failed: {out.stderr.strip()[-300:]}")
        names[name] = json.loads(out.stdout.strip().splitlines()[-1])
    return names


def _config_options(harness: Any, mode: str | None, tools: dict[str, list[str]] | None) -> dict:
    """The optional write_config keywords a module declares: ``mode`` (None at level 1,
    else "default" / "unattended") and ``tools`` (fixture_tool_names). A module that
    declares neither is called as write_config(servers, workdir, env)."""
    accepted = inspect.signature(harness.write_config).parameters
    return {key: value for key, value in (("mode", mode), ("tools", tools)) if key in accepted}


def check_servers_listed(harness: Any, env: dict[str, str], workdir: Path) -> dict:
    expected = sorted(cells.SERVERS)
    try:
        listed = harness.list_servers(env, workdir)
    except HarnessError as exc:
        return {"passed": False, "status": f"list command failed: {exc}", "expected": expected, "listed": None}
    if listed is None:
        return {"passed": True, "status": "not available", "expected": expected, "listed": None}
    listed = sorted(listed)
    return {"passed": listed == expected, "status": "listed", "expected": expected, "listed": listed,
            "unexpected": sorted(set(listed) - set(expected)), "missing": sorted(set(expected) - set(listed))}


def check_tools_offered(harness: Any, model_log: Path | None, output: Path) -> dict:
    """Every MCP tool offered to the model (model.jsonl), and every MCP server the harness
    reported at init (harness.jsonl), belongs to one of the five fixtures."""
    server_of = getattr(harness, "mcp_server_of", harnesses.mcp_server_of)
    offered: set[str] = set()
    if model_log is not None and model_log.exists():
        for line in model_log.read_text().splitlines()[1:]:
            offered.update(json.loads(line).get("offered") or [])
    reported: set[str] = set()
    for line in output.read_text().splitlines()[1:]:
        event = json.loads(line)
        if event.get("event") == "init":
            reported.update(s.get("name") if isinstance(s, dict) else str(s) for s in event.get("mcp_servers") or [])
            offered.update(event.get("mcp_tools") or [])
    mcp_tools = sorted(name for name in offered if server_of(name))
    foreign = sorted(name for name in mcp_tools if server_of(name) not in cells.SERVERS)
    foreign_servers = sorted(reported - set(cells.SERVERS))
    return {"passed": not foreign and not foreign_servers, "source": "model.jsonl" if model_log else "harness.jsonl",
            "mcp_tools": len(mcp_tools), "servers": sorted({server_of(n) for n in mcp_tools}),
            "foreign_tools": foreign, "foreign_servers": foreign_servers}


# A harness run is timing-sensitive: a fixture that answers initialize late looks like a
# harness that never listed its tools. Runs on one machine therefore take a machine-wide
# lock (shared across worktrees), and a session only starts when the machine is quiet.
MEASURE_LOCK = Path.home() / ".cache" / "mcp-mirror" / "harness-measure.lock"
FIXTURE_WAIT_SECONDS = 1.0  # wait until a fixture answers initialize this fast
FIXTURE_FAIL_SECONDS = 2.0  # a session that still starts slower publishes nothing
LOAD_WAIT_SECONDS = 600


LOCK_HELD_ENV = "MCP_MIRROR_MEASURE_LOCK_HELD"


@contextlib.contextmanager
def measurement_lock():
    import fcntl

    # A command run under scripts/measure_locked.py already holds this lock in its
    # parent; taking it again here would block on that parent forever.
    if os.environ.get(LOCK_HELD_ENV):
        yield
        return
    MEASURE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(MEASURE_LOCK, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"waiting for the machine-wide harness measurement lock ({MEASURE_LOCK}) ...", flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _fixture_initialize_seconds() -> float | None:
    """How long a fixture takes to answer initialize right now, through the project SDK.

    This, not the load average, is what corrupts a harness cell: a harness that gives a
    server a few seconds to start drops it when the machine is slow. Load on this machine
    is often I/O from other work, which a CPU-bound fixture barely notices."""
    probe = (
        "import time,anyio;from mcp import ClientSession,StdioServerParameters;"
        "from mcp.client.stdio import stdio_client\n"
        "async def m():\n"
        " t=time.monotonic()\n"
        " async with stdio_client(StdioServerParameters(command='python',args=['fixtures/definitions_server.py'])) as (r,w):\n"
        "  async with ClientSession(r,w) as s:\n"
        "   await s.initialize()\n"
        " print(time.monotonic()-t)\n"
        "anyio.run(m)"
    )
    try:
        out = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True,
                             timeout=60, env=os.environ | _venv_path_env())
        return float(out.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001 - a probe that cannot run reads as slow
        return None


def check_load() -> dict:
    """Wait until a fixture answers initialize quickly, then record the machine state."""
    cores = os.cpu_count() or 1
    if os.environ.get("MCP_MIRROR_LOAD_GATE") == "off":
        # Unit tests drive the runner against stubs; they set this, and the record says so.
        return {"passed": True, "skipped": "MCP_MIRROR_LOAD_GATE=off", "cores": cores}
    deadline = time.monotonic() + LOAD_WAIT_SECONDS
    waited = 0.0
    while True:
        latency = _fixture_initialize_seconds()
        if (latency is not None and latency <= FIXTURE_WAIT_SECONDS) or time.monotonic() >= deadline:
            break
        time.sleep(10)
        waited += 10
    return {
        "passed": latency is not None and latency <= FIXTURE_FAIL_SECONDS,
        "fixture_initialize_seconds": None if latency is None else round(latency, 2),
        "limit_seconds": FIXTURE_FAIL_SECONDS,
        "load_1m_per_core": round(os.getloadavg()[0] / cores, 2),
        "cores": cores,
        "waited_seconds": waited,
    }


def fresh_record(checks: dict[str, dict]) -> dict:
    return {"format": FRESH_FORMAT, "passed": all(c["passed"] for c in checks.values()), "checks": checks}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def _venv_path_env() -> dict[str, str]:
    """PATH with this interpreter's bin first, so a fixture's bare `python` has the SDK."""
    env = _scrubbed()
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    return env


def _fixture_command(path: str, delay: float | None, python: str = "python") -> str:
    command = f"{python} {path}"
    return f"sh -c 'sleep {delay:g}; exec {command}'" if delay else command


def _recorder_alive(capture: Path) -> bool:
    """Whether a `mcp-mirror record ... --out <capture>` process is still running."""

    found = subprocess.run(["pgrep", "-f", "--", f"--out {re.escape(str(capture))}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return found.returncode == 0


def _wait_for_recorders(captures: Iterable[Path], timeout: float = 30.0) -> dict:
    """Wait until no recorder process for any of ``captures`` is alive, at most
    ``timeout`` seconds. A freshness check: it passes when every recorder has exited.

    Stdio recorders are the harness's children, not this process's, so the runner
    cannot join them. Scrubbing a capture while its recorder still has it open rewrote
    the file shorter under the recorder's offset, and the end record then landed after
    a run of NUL bytes (Goose 1.54.0, two runs out of two)."""

    started = time.monotonic()
    pending = [Path(capture) for capture in captures]
    while pending and time.monotonic() - started < timeout:
        pending = [capture for capture in pending if _recorder_alive(capture)]
        if pending:
            time.sleep(0.2)
    return {"passed": not pending, "waited_seconds": round(time.monotonic() - started, 3),
            "still_running": [capture.name for capture in pending]}


def _refuse_padded(session_dir: Path) -> None:
    """Fail the measurement on a capture with a NUL byte: it was rewritten under a live
    writer, and no cell may be derived from it."""

    padded = cells.padded_captures(session_dir)
    if padded:
        names = ", ".join(path.name for path in padded)
        raise HarnessError(f"{session_dir.name}: {names} contains NUL bytes; a recorder wrote after the "
                           "capture was rewritten, so the session cannot be published. Measure again.")


def _record_argv(fixture: str, capture: Path, *extra: str) -> list[str]:
    return [sys.executable, "-m", "mcp_mirror.cli", "record", fixture, "--out", str(capture), "--ping", *extra]


class HttpRecorders:
    """One `mcp-mirror record --transport http` per fixture, owned by this process."""

    def __init__(self, fixtures: dict[str, str], captures: dict[str, Path]) -> None:
        self.processes: dict[str, subprocess.Popen] = {}
        self.urls: dict[str, str] = {}
        for name, fixture in fixtures.items():
            process = subprocess.Popen(
                _record_argv(fixture, captures[name], "--transport", "http", "--port", "0"),
                cwd=ROOT, env=_venv_path_env(), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            self.processes[name] = process
        for name, process in self.processes.items():
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                line = process.stderr.readline() if process.stderr else ""
                found = re.search(r"listening on (\S+)", line)
                if found:
                    self.urls[name] = found.group(1)
                    # Keep draining, or a chatty fixture fills the pipe and stalls.
                    threading.Thread(target=process.stderr.read, daemon=True).start()
                    break
                if not line and process.poll() is not None:
                    break
            if name not in self.urls:
                self.stop()
                raise HarnessError(f"the {name} recorder did not start listening")

    def stop(self) -> None:
        for process in self.processes.values():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in self.processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


def _scrub_home(text: str) -> str:
    """The operator's home directory as ~: a harness quotes paths it wrote to (Claude Code
    saves binary tool content under ~/.claude and tells the model where)."""
    return text.replace(str(Path.home()), "~")


def run_session(
    *, harness: Any, fixtures: dict[str, str], session_dir: Path, out_root: Path, prompt: str,
    max_turns: int, timeout: float, env_extra: dict[str, str], secrets: dict[str, str], mode: str | None,
    transport: str, delays: dict[str, float], python: str = "python", stub: ModelStub | None = None,
    isolate: bool = True, tools: dict[str, list[str]] | None = None, live: bool = False,
) -> dict:
    session_dir.mkdir(parents=True, exist_ok=True)
    captures = {name: session_dir / f"{name}.jsonl" for name in fixtures}
    commands = {name: _fixture_command(path, delays.get(name), python) for name, path in fixtures.items()}
    recorders: HttpRecorders | None = None
    home = IsolatedHome(harness, isolate, base=SHORT_HOME if live else None)
    workdir = home.workdir
    model_log: Path | None = None
    code, timed_out, duration, output, stderr_tail, shown = None, False, 0.0, session_dir / "harness.jsonl", [], []
    withheld = withheld_proxy(harness, transport)
    sandbox = LoopbackSandbox(home.root, withheld) if withheld else None
    if live and not shutil.which("sandbox-exec"):
        # Fail closed: a real model with a real key runs only contained (LiveSandbox).
        home.remove()
        raise HarnessError("a live session runs only under sandbox-exec, which this machine does not have")
    contained: LiveSandbox | None = None
    probe: dict = {}
    checks: dict = {}
    forward = live_egress_allowed(harness) if live else {}
    try:
        with EgressSink(forward) as sink:
            env = home.env() | sink.env()
            if stub is not None:
                stub.brain.scrub = home.scrub
            if transport == "http":
                recorders = HttpRecorders(commands, captures)
                servers = {name: {"url": url} for name, url in recorders.urls.items()}
            else:
                # The harness launches the recorder; `cd` because not every harness honours a
                # cwd key, and the fixture paths are relative to the repository.
                path_env = _venv_path_env()["PATH"]
                servers = {
                    name: {
                        "command": "sh",
                        "args": ["-c", f"cd {shlex.quote(str(ROOT))} && PATH={shlex.quote(path_env)} exec "
                                 + shlex.join(_record_argv(commands[name], captures[name]))],
                    }
                    for name in fixtures
                }
            checks = {"home": home.check(), "load": check_load()}
            options = _config_options(harness, mode, tools)
            listed_config = harness.write_config(_placeholders(servers), workdir, env, **options)
            checks["servers_listed"] = check_servers_listed(harness, env, workdir)
            config = harness.write_config(servers, workdir, env, **options)
            if isolate and not (home.contains(listed_config) and home.contains(config)):
                raise HarnessError(f"write_config wrote {config} outside the isolated home")
            argv = [str(part) for part in harness.argv(prompt, mode, config, max_turns)]
            shown = [_shown_part(part, config) for part in argv]
            env = env | env_extra
            run_prefix: list[str] = []
            if sandbox is not None:
                env = {k: v for k, v in env.items() if k not in sink.env()}
                run_prefix = sandbox.prefix()
                probe = sandbox.probe()
            events: list[dict] = []
            held = sandbox is None or sandbox.check(sink, harness, probe)["passed"]
            if live:
                # The live profile is rendered with this session's loopback ports (the
                # sink, the HTTP recorders) and probed before the harness starts; a probe
                # that shows any allowance leaves the session unstarted (checks.sandbox).
                ports = [_loopback_port(sink.url)] + [_loopback_port(u) for u in (recorders.urls.values() if recorders else [])]
                contained = LiveSandbox(home, session_dir, harness, python, ports)
                checks["sandbox"] = contained.check(contained.probe())
                held = held and checks["sandbox"]["passed"]
                run_prefix = contained.prefix() + run_prefix
                env = env | contained.env()
            if checks["home"]["passed"] and checks["servers_listed"]["passed"] and checks["load"]["passed"] and held:
                launch, launch_env = (live_launch(env, argv, secrets) if live else (run_prefix + argv, env))
                launch = run_prefix + launch
                started = time.monotonic()
                with (workdir / "stdout").open("w") as stdout, (workdir / "stderr").open("w") as stderr:
                    # Its own session, so its pid is a process group holding every child it
                    # spawns: the timeout kills the group, and _reap_process_group below
                    # kills and records whatever outlived it (Cursor's worker-server does,
                    # every session), failing the session only for what survives or escapes.
                    process = subprocess.Popen(launch, cwd=workdir, env=launch_env, stdin=subprocess.DEVNULL,
                                               stdout=stdout, stderr=stderr, start_new_session=True)
                    try:
                        code = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            code = process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            with contextlib.suppress(ProcessLookupError):
                                os.killpg(process.pid, signal.SIGKILL)
                            code = process.wait()
                exited_at = time.monotonic()
                duration = exited_at - started
                if recorders is not None:
                    recorders.stop()
                else:
                    # The harness launched the stdio recorders; each writes its end record
                    # once the fixture is down, after the harness has exited. Nothing may
                    # rewrite a capture while its recorder is alive.
                    checks["recorders"] = _wait_for_recorders(captures.values())
                checks["process_group"] = _reap_process_group(process.pid, timed_out, exited_at)
                events = harness.reduce_output(home.scrub((workdir / "stdout").read_text(errors="replace")))
                stderr_tail = (workdir / "stderr").read_text(errors="replace").strip().splitlines()[-3:]
            else:
                # A home that is not isolated, or a list with a server that is not a
                # fixture: starting the harness would connect it to the operator's servers.
                stderr_tail = ["not started: the pre-run freshness checks failed (fresh.json)"]
            scrub = home.scrub
            output.write_text(scrub("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in [{"format": OUTPUT_FORMAT}, *events])))
            if stub is not None:
                model_log = session_dir / "model.jsonl"
                header = {"format": MODEL_FORMAT, "stub": stub.kind, "script": stub.brain.script}
                requests = reduce_model_log(stub.brain.log, _listed_tools(captures))
                model_log.write_text(scrub("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in [header, *requests])))
            checks["tools_offered"] = check_tools_offered(harness, model_log, output)
        checks["egress"] = sandbox.check(sink, harness, probe) if sandbox is not None else sink.check(harness)
    finally:
        if recorders is not None:
            recorders.stop()
        else:
            _wait_for_recorders(captures.values())
        # Harnesses echo their working directory into requests (Goose sends
        # params._meta.agent-working-dir on every tools/call), so the recorder captures
        # get the same scrub as stdout and the stub log before the home is removed.
        for capture in session_dir.glob("*.jsonl"):
            try:
                text = capture.read_text()
                scrubbed = home.scrub(text)
                if scrubbed != text:
                    capture.write_text(scrubbed)
            except OSError:
                pass
        if live and secrets and checks:
            # Before the home goes: nothing the session wrote, in the capture or in its
            # home and workdir, may hold a value op injected (check_secrets_absent).
            checks["secrets"] = check_secrets_absent(secrets, [session_dir, home.root])
        home.remove()
    _refuse_padded(session_dir)
    fresh = fresh_record(checks)
    (session_dir / cells.HARNESS_FRESH).write_text(json.dumps(fresh, indent=2) + "\n")
    record = {
        "transport": transport,
        "prompt": prompt,
        "argv": shown,
        "exit_code": code,
        "timed_out": timed_out,
        "duration_s": round(duration, 2),
        "delay_s": delays or None,
        "fixtures": {name: commands[name] for name in fixtures},
        "captures": {name: str(path.relative_to(out_root)) for name, path in captures.items() if path.exists()},
        "output": str(output.relative_to(out_root)),
        "fresh": str((session_dir / cells.HARNESS_FRESH).relative_to(out_root)),
        "stderr_tail": [line for line in stderr_tail if str(Path.home()) not in line and str(home.root) not in line],
    }
    if model_log is not None:
        record["model"] = str(model_log.relative_to(out_root))
    return record


def _shown_part(part: str, config: Path) -> str:
    """One argv element as published: the config file as <config>, and any path inside the
    repository (a project-local harness binary, for instance) relative to it, so a capture
    never carries a machine-specific absolute path."""
    if part == str(config):
        return "<config>"
    try:
        candidate = Path(part)
        if candidate.is_absolute() and candidate.is_relative_to(ROOT):
            return str(candidate.relative_to(ROOT))
    except (OSError, ValueError):
        pass
    return part


def ensure_installed(harness: Any, entry: dict) -> str:
    """Install the harness project-locally if it is missing; the version install_check reports.

    A module declares ``INSTALL``, an argv run from the repository root that installs a
    pinned version under .mcp-mirror/harnesses/<id> (for example ``npm install --prefix
    .mcp-mirror/harnesses/codex @openai/codex@0.46.0``), and points ``BINARY`` at the
    repository-relative path it produces. Global installs are never run: they change the
    operator's machine and make the measured version depend on whatever was there.
    """
    from harnesses import resolve_binary, throwaway_home

    if resolve_binary(harness.BINARY) is None:
        install = getattr(harness, "INSTALL", None)
        if not install:
            raise HarnessError(
                f"{harness.BINARY} is not installed and the module declares no INSTALL; "
                f"add a project-local INSTALL argv (the inventory says {entry.get('install', '')!r}, "
                "which is not run because global installs are not allowed)"
            )
        print("installing: " + " ".join(install))
        # Some installs write to HOME (Kilo's postinstall creates ~/.cache/kilo and friends).
        with throwaway_home() as env:
            subprocess.run(install, cwd=ROOT, check=True, env=env)
        if resolve_binary(harness.BINARY) is None:
            raise HarnessError(f"{' '.join(install)} ran but {harness.BINARY} does not exist")
    return harness.install_check()


def live_egress_allowed(harness: Any) -> dict[str, str]:
    """``{host: reason}`` a live session may reach: the module's LIVE_EGRESS_ALLOWED."""
    return {host.lower(): reason for host, reason in (getattr(harness, "LIVE_EGRESS_ALLOWED", {}) or {}).items()}


def live_secrets(env_file: Path) -> dict[str, str]:
    """The values of the env file's variables, taken from this process's own environment
    and removed from it.

    The operator wraps the whole measurement once in `op run --env-file .env.harness --
    ...`, so op asks 1Password once and every session reuses what it injected. The values
    are never printed or written; they leave this dict only into the harness's process
    (live_launch) and the secret checker (check_secrets_absent). Popping them from
    os.environ means no other child (a fixture probe, a recorder, an install, `mcp
    list`, the sandbox probe) can inherit them.
    """
    names = env_file_variables(env_file)
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise HarnessError(
            f"{', '.join(missing)} is not set: wrap the whole run once, "
            f"`op run --env-file {env_file.name} -- uv run python scripts/measure_locked.py -- "
            "uv run python scripts/measure_harness.py <harness> --live`"
        )
    return {name: os.environ.pop(name) for name in names}


def live_launch(env: dict[str, str], argv: list[str], secrets: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """(the command to start, the environment to start it in) for a live session: the
    harness itself, in the isolated session ``env`` plus the env file's secrets and
    nothing else of the operator's. Only this process gets the secrets."""
    return list(argv), env | secrets


def observed_models(out_root: Path, sessions: list[dict]) -> list[str]:
    """The model names the harness's own output reported, over ``sessions``: the init
    events' model, then "routed to <model>" for each model a ``routed`` event named (a
    service's automatic choice, such as Cursor's Auto picking Grok)."""
    models: set[str] = set()
    routed: set[str] = set()
    for session in sessions:
        output = out_root / session["output"]
        if not output.is_file():
            continue
        for line in output.read_text().splitlines()[1:]:
            event = json.loads(line)
            if event.get("event") == "init" and event.get("model"):
                models.add(str(event["model"]))
            elif event.get("event") == "routed" and event.get("model"):
                routed.add(str(event["model"]))
    return sorted(models) + [f"routed to {model}" for model in sorted(routed)]


def live_blocker(plan: dict) -> str | None:
    env_file = ROOT / plan["level_2"]["env_file"]
    if not env_file.exists():
        return cells.LEVEL_2_SKIPPED
    missing = [name for name in env_file_variables(env_file) if not os.environ.get(name)]
    if missing:
        return (f"{', '.join(missing)} is not in the environment: start the whole run once under "
                f"`op run --env-file {env_file.name} -- ...`")
    return None


def session_env(harness: Any, stub: ModelStub | None) -> dict[str, str]:
    """The module's own variables for a session: its stub pointers and quiet switches with
    a stub, its quiet switches alone without one (a live session, where nothing is
    pointed at a stub but a switch such as Cursor's credential store still applies)."""
    return harness.env(stub.url if stub is not None else "")


def env_file_variables(env_file: Path) -> list[str]:
    """The names `op run --env-file` will set: the left side of each ``NAME=op://...`` line.
    Only names are read; the references stay where they are and the values never exist
    outside op's child."""
    names = []
    for line in env_file.read_text().splitlines():
        name, _, reference = line.strip().partition("=")
        if name and not name.startswith("#") and reference.startswith("op://"):
            names.append(name)
    return names


SECRET_CHECK = ROOT / "scripts" / "check_secret_absent.py"


def check_secrets_absent(secrets: dict[str, str], paths: list[Path]) -> dict:
    """Scan ``paths`` for the values of every variable the env file named: a live capture
    must not hold the key in any form.

    The checker (scripts/check_secret_absent.py) gets the values in its environment, from
    this process's live_secrets alone, and prints only the path and name of a match,
    never the value.
    """
    names = sorted(secrets)
    argv = [sys.executable, "-I", str(SECRET_CHECK)]
    for name in names:
        argv += ["--var", name]
    argv += ["--", *map(str, paths)]
    env = {k: v for k, v in os.environ.items() if k in PASSED_ENV} | secrets
    found = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    if found.returncode not in (0, 1):
        raise HarnessError(f"the secret check could not run (exit {found.returncode}): {found.stderr.strip()}")
    matches = [line for line in found.stdout.splitlines() if line.startswith("match: ")]
    return {"passed": found.returncode == 0 and not matches, "variables": names,
            "paths": [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else "<session home>" for p in paths],
            "matches": matches}


def fixture_interpreter(line: dict) -> tuple[str, str]:
    """(the fixture command's interpreter, relative to the repository; its MCP SDK version)."""
    if line["python"] == "venv":
        return "python", package_version("mcp")
    sys.path.insert(0, str(ROOT / "scripts"))
    from _fixture import fixture_python

    absolute = Path(fixture_python(str(line["python"])))
    shown = str(absolute.relative_to(ROOT)) if absolute.is_relative_to(ROOT) else str(absolute)
    found = subprocess.run([str(absolute), "-c", "import importlib.metadata as m; print(m.version('mcp'))"],
                           capture_output=True, text=True, timeout=60)
    return shown, found.stdout.strip() or "unknown"


def session_script(plan: dict, spec: dict, prompt_id: str) -> list[dict]:
    """What the stubbed model does in a session: the harness's override, else the plan's."""
    return list((spec.get("scripts") or {}).get(prompt_id, plan["level_2"]["scripts"].get(prompt_id) or []))


def session_prompt(plan: dict, harness: Any, prompt_id: str) -> tuple[str, str]:
    """(the text the harness is started with, which invocation that is)."""
    if prompt_id == "prompt":
        call = plan["level_2"]["prompt_call"]
        invocation = harness.prompt_invocation(call["server"], call["name"], dict(call.get("arguments") or {}))
        if invocation:
            return invocation, "prompt_invocation"
    return plan["level_2"]["prompts"][prompt_id], "prompt"


def measure(harness_id: str, levels: set[int], out_root: Path, runs: int | None = None, live: bool = False) -> Path:
    with measurement_lock():
        return _measure(harness_id, levels, out_root, runs, live)


def _measure(harness_id: str, levels: set[int], out_root: Path, runs: int | None = None, live: bool = False) -> Path:
    entry = inventory_entry(harness_id)
    plan = session_plan()
    spec = harness_plan(plan, harness_id)
    harness = harnesses.load(harness_id)
    version = ensure_installed(harness, entry)
    fixtures = dict(plan["servers"])
    stub_kind = harness.STUB_KIND
    lines = [{**line, "revision": str(line["revision"])} for line in plan["fixture_lines"]]
    interpreters = {line["revision"]: fixture_interpreter(line) for line in lines}
    wants_tools = "tools" in inspect.signature(harness.write_config).parameters
    tools = {revision: fixture_tool_names(fixtures, python) if wants_tools else None
             for revision, (python, _) in interpreters.items()}
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    command = f"uv run python scripts/measure_harness.py {harness_id}" + (
        " --live" if live else (" --level 1" if levels == {1} else ""))
    manifest: dict = {
        "format": SESSION_FORMAT,
        "harness": harness_id,
        "name": entry["name"],
        "version": version,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "recorder": f"mcp-mirror {package_version('mcp-mirror')}",
        "fixture_sdk": f"mcp {interpreters[lines[0]['revision']][1]}",
        "fixture_lines": [
            {"revision": line["revision"], "python": interpreters[line["revision"]][0],
             "sdk": f"mcp {interpreters[line['revision']][1]}"}
            for line in lines
        ],
        "servers": fixtures,
    }

    level_1 = plan["level_1"]
    level_1_sessions = []
    # One op run wraps the whole measurement: the secrets are read once here, out of this
    # process's environment, and reach only each session's harness process.
    # live_blocker is evaluated once: live_secrets takes the variables out of the
    # environment, so a second look would report them missing.
    blocked = live_blocker(plan) if live else None
    secrets = live_secrets(ROOT / plan["level_2"]["env_file"]) if live and not blocked else {}
    if live and stub_kind:
        manifest["level_1"] = {"status": "skipped", "reason": "a live run measures level 2 only", "sessions": []}
    elif live and blocked:
        manifest["level_1"] = {"status": "skipped", "reason": blocked, "sessions": []}
    elif live:
        # No stub to answer this harness, so the live service answers level 1 too.
        for line in lines:
            revision, python = line["revision"], interpreters[line["revision"]][0]
            for session in level_1["sessions"]:
                for mode, session_id in level_1_modes(harness, session["id"]):
                    print(f"level 1 {revision} {session_id} (live) ...", flush=True)
                    record = run_session(
                        harness=harness, fixtures=fixtures, session_dir=out_root / f"level1-{revision}-{session_id}",
                        out_root=out_root, prompt=level_1["prompt"], max_turns=level_1["max_turns"],
                        timeout=level_1["timeout_s"], env_extra=session_env(harness, None), secrets=secrets,
                        mode=mode, transport=session["transport"], delays=session.get("delay_s") or {},
                        python=python, stub=None, tools=tools[revision], live=True,
                    )
                    level_1_sessions.append(level_1_record(harness, session_id, revision, mode, record))
        manifest["level_1"] = {
            "status": "ran", "model": f"live, under op run --env-file {plan['level_2']['env_file']}",
            "revisions": [line["revision"] for line in lines], "sessions": level_1_sessions,
            **level_1_unattended(harness),
        }
    elif stub_kind:
        reply_delay = level_1_reply_delay(harness)
        for line in lines:
            revision, python = line["revision"], interpreters[line["revision"]][0]
            for session in level_1["sessions"]:
                for mode, session_id in level_1_modes(harness, session["id"]):
                    print(f"level 1 {revision} {session_id} ...", flush=True)
                    with ModelStub(stub_kind, reply_delay_s=reply_delay) as stub:
                        record = run_session(
                            harness=harness, fixtures=fixtures, session_dir=out_root / f"level1-{revision}-{session_id}",
                            out_root=out_root, prompt=level_1["prompt"], max_turns=level_1["max_turns"],
                            timeout=level_1["timeout_s"], env_extra=session_env(harness, stub), secrets={}, mode=mode,
                            transport=session["transport"], delays=session.get("delay_s") or {}, python=python,
                            stub=stub, tools=tools[revision],
                        )
                    if reply_delay:
                        record["reply_delay_s"] = reply_delay
                    level_1_sessions.append(level_1_record(harness, session_id, revision, mode, record))
        manifest["level_1"] = {
            "status": "ran", "model": f"stub:{stub_kind}", "revisions": [line["revision"] for line in lines],
            "sessions": level_1_sessions, **level_1_unattended(harness),
        }
        if reply_delay:
            manifest["level_1"]["reply_delay_s"] = reply_delay
    else:
        manifest["level_1"] = {
            "status": "skipped",
            "reason": f"scripts/harnesses/{harness_id}.py has no STUB_KIND; level 1 needs one",
            "sessions": [],
        }

    level_2 = plan["level_2"]
    blocker = None if 2 in levels else "level 2 not requested (--level 1)"
    if live:
        blocker = blocker or blocked
    elif not stub_kind:
        blocker = blocker or f"scripts/harnesses/{harness_id}.py has no STUB_KIND, so its model API cannot be scripted"
    if blocker:
        manifest["level_2"] = {"status": "skipped", "reason": blocker, "sessions": []}
    else:
        count = runs or level_2["runs"]
        sessions = []
        # Like level 1, once per fixture line: at 2026-07-28 elicitation and sampling are
        # MRTR input requests on the tools/call result, not requests on the session.
        for line in lines:
            revision, python = line["revision"], interpreters[line["revision"]][0]
            for session in level_2["sessions"]:
                prompt, invocation = session_prompt(plan, harness, session["prompt"])
                script = session_script(plan, spec, session["prompt"])
                for run in range(1, count + 1):
                    print(f"level 2 {revision} {session['id']} run {run} ...", flush=True)
                    stub = None if live else ModelStub(stub_kind, script, text="done")
                    with stub or contextlib.nullcontext():
                        record = run_session(
                            harness=harness, fixtures=fixtures,
                            session_dir=out_root / f"level2-{revision}-{session['id']}-{run}",
                            out_root=out_root, prompt=prompt, max_turns=level_2["max_turns"],
                            timeout=level_2["timeout_s"], env_extra=session_env(harness, stub),
                            secrets=secrets, mode=session["mode"], transport="stdio", delays={}, python=python, stub=stub,
                            tools=tools[revision], live=live,
                        )
                    sessions.append({
                        "id": session["id"], "level": 2, "run": run, "mode": session["mode"], "revision": revision,
                        "invocation": invocation, "mode_args": list(harness.mode_args(session["mode"])), **record,
                    })
        manifest["level_2"] = {
            "status": "ran",
            "model": f"live, under op run --env-file {level_2['env_file']}" if live else f"stub:{stub_kind} (scripted)",
            "runs": count,
            "revisions": [line["revision"] for line in lines],
            "sessions": sessions,
        }

    sessions = level_1_sessions + manifest["level_2"]["sessions"]
    manifest["fresh"] = fresh_summary(out_root, sessions)
    models: list[str] = []
    if live:
        # What makes a live run readable later: the file the secrets came from (never its
        # contents), the hosts the sink forwarded to, and the model the service reported.
        models = observed_models(out_root, sessions)
        manifest["live"] = {"env_file": plan["level_2"]["env_file"], "egress_allowed": live_egress_allowed(harness),
                            "egress_attempted": egress_attempted(out_root, sessions), "models": models}
    path = out_root / cells.HARNESS_SESSION
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not live or not stub_kind:
        # A live run of a stubbed harness is scratch; of a harness with no stub, it is the
        # row's only measurement.
        update_harness_row(harness_id, entry, version, manifest["captured_at"][:10], out_root=out_root,
                           live=live, live_service=getattr(harness, "LIVE_SERVICE", None), models=models)
    if live and sessions:
        # Once more over everything this run wrote, manifest and row included: the key op
        # injected must appear in no file that could be committed.
        checked = check_secrets_absent(secrets, [out_root, SNAPSHOT / "harnesses.json"])
        manifest["live"]["secrets_absent"] = checked
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        if not checked["passed"]:
            raise HarnessError("a secret from the env file appears in the run's files; do not commit: "
                               + "; ".join(checked["matches"]))
    return path


def egress_attempted(out_root: Path, sessions: list[dict]) -> list[dict]:
    """Every host the sessions' sinks saw a request for, summed over the run: host, port,
    method, how often, and whether the sink forwarded it (LIVE_EGRESS_ALLOWED) or refused it."""
    counted: dict[tuple, dict] = {}
    for session in sessions:
        record = json.loads((out_root / session["fresh"]).read_text())
        for attempt in record["checks"].get("egress", {}).get("attempts", []):
            key = (attempt["host"], attempt["port"], attempt["method"])
            line = counted.setdefault(key, {"host": attempt["host"], "port": attempt["port"], "method": attempt["method"],
                                            "count": 0, "forwarded": bool(attempt.get("forwarded")), "sessions": 0})
            line["count"] += attempt.get("count", 1)
            line["sessions"] += 1
    return sorted(counted.values(), key=lambda a: (a["host"], a["port"], a["method"]))


def fresh_summary(out_root: Path, sessions: list[dict]) -> dict:
    """Every session's fresh.json in one place: which passed, and which check failed where."""
    failed = []
    for session in sessions:
        record = json.loads((out_root / session["fresh"]).read_text())
        if not record["passed"]:
            failed.append({"fresh": session["fresh"],
                           "checks": sorted(k for k, c in record["checks"].items() if not c["passed"])})
    return {"passed": not failed, "sessions": len(sessions), "failed": failed}


def update_harness_row(harness_id: str, entry: dict, version: str, date: str, *, out_root: Path | None = None,
                       live: bool = False, live_service: str | None = None, models: list[str] | None = None) -> None:
    """The harness's row in harnesses.json: inventory facts plus what this run observed.

    A live row names its capture under wire-live and carries ``measured_live`` with the
    service it ran against and the model that service reported, which the agent page shows;
    a stubbed row carries neither.
    """
    path = SNAPSHOT / "harnesses.json"
    wire = "wire-live" if live else "wire"
    if out_root is not None and out_root.is_relative_to(SNAPSHOT / "measurements"):
        wire = out_root.relative_to(SNAPSHOT / "measurements").parts[0]
    document = json.loads(path.read_text())
    document["updated"] = date
    previous = document["agents"].get(harness_id) or {}
    row = {
        "name": entry["name"],
        "abbr": entry.get("abbr", entry["name"]),
        "vendor": entry.get("vendor", ""),
        "type": "harness",
        "tier": entry.get("tier"),
        "install": entry.get("install", ""),
        # Where scripts/refresh_versions.py looks for releases after the measured one.
        **({"registry": entry["registry"], "package": entry["package"]} if entry.get("registry") else {}),
        "headless": entry.get("headless", ""),
        "mcp_config": entry.get("mcp_config", ""),
        "unattended": entry.get("unattended", ""),
        "capture": f"measurements/{wire}/{harness_id}/{cells.HARNESS_SESSION}",
        "measurement_status": "measured",
        **({"measured_live": True, "live_service": live_service or entry["name"], "model": ", ".join(models or [])}
           if live else {}),
        "current_version": version,
        "stable_version": version,
        "dev_version": None,
        "version_list": [{"version": version, "era": 0, "release_date": None}],
    }
    # A new row lands as preview (shown with ?preview=1). Publishing is a maintainer's edit
    # after review, never this script's, and a re-measure keeps whatever it was, including
    # no key at all, which is how a published row is written.
    existing = harness_id in document["agents"]
    visibility = previous.get("visibility") if existing else "preview"
    if visibility is not None:
        row["visibility"] = visibility
    # Context notes are a maintainer's, like visibility (cells.publish_harness_rows).
    if previous.get("cell_notes"):
        row["cell_notes"] = previous["cell_notes"]
    # An existing row keeps its place, which is the page's column order.
    document["agents"][harness_id] = row
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("harness", nargs="?", help="harness id from plan/inventory.yaml")
    parser.add_argument("--level", type=int, choices=(1, 2), help="run only up to this level (default: both)")
    parser.add_argument("--runs", type=int, help="level-2 runs per session (default from the plan)")
    parser.add_argument("--out", type=Path, help="output directory (default data/harnesses/measurements/wire/<harness>)")
    parser.add_argument("--live", action="store_true",
                        help="against a real model: start the whole run once under op run --env-file .env.harness (off by default; published "
                             "only for a harness with no STUB_KIND, as its measured_live row)")
    parser.add_argument("--write-surface", action="store_true", help="regenerate data/harnesses/surface.json")
    arguments = parser.parse_args()
    if arguments.write_surface:
        print(f"wrote {write_surface().relative_to(ROOT)}")
        if not arguments.harness:
            return 0
    if not arguments.harness:
        parser.error("name a harness, or pass --write-surface")
    levels = {1} if arguments.level == 1 else {1, 2}
    wire = "wire-live" if arguments.live else "wire"
    out_root = arguments.out or SNAPSHOT / "measurements" / wire / arguments.harness
    try:
        path = measure(arguments.harness, levels, out_root.resolve(), arguments.runs, arguments.live)
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    manifest = json.loads(path.read_text())
    print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    for level in ("level_1", "level_2"):
        block = manifest[level]
        print(f"  {level}: {block['status']}" + (f" ({block['reason']})" if block.get("reason") else ""))
    fresh = manifest["fresh"]
    print(f"  fresh: {'passed' if fresh['passed'] else 'FAILED'} in {fresh['sessions'] - len(fresh['failed'])}"
          f"/{fresh['sessions']} sessions")
    for failure in fresh["failed"]:
        print(f"    {failure['fresh']}: {', '.join(failure['checks'])}")
    run = cells.HarnessRun(path)
    for cell_id, rule in cells.HARNESS_RULES.items():
        cell = rule(run)
        value = f" = {cell.value}" if cell.value else ""
        print(f"  {cell_id:<34} {cell.code}{value}")
    return 0 if fresh["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
