"""DeepSeek Harness: `dsh --profile headless "<prompt>" --json`. https://github.com/deepseek-ai/deepseek-harness

Every flag, setting and variable below is from the dsh-v0.2.0-rc.2 tree of that repository:
apps/cli/README.md and apps/cli/reference/README.md (launcher, profiles, --patch),
packages/bundle/headless/README.md (the one-shot runner and its --json stream),
packages/mcp/mcp-client/README.md and docs/user/guide/mcp-memory.md (one
@deepseek-ai/dsh-mcp-client row per server), packages/llm/llm-deepseek/README.md (the
model endpoint) and packages/bundle/base/README.md (telemetry). The composed rows were
read with `dsh --profile headless --dump-default-config`.
"""

from __future__ import annotations

import json
from pathlib import Path

from harnesses import STUB_KEY, HarnessError, clip, resolve_binary, version_of

# Project-local and pinned, never global. The developer preview ships on npm as
# @deepseek-ai/dsh (README "Run from npm"); the package's repository is
# github.com/deepseek-ai/deepseek-harness.
INSTALL = ["npm", "install", "--prefix", ".mcp-mirror/harnesses/deepseek_harness", "@deepseek-ai/dsh@0.2.0-rc.2"]
BINARY = ".mcp-mirror/harnesses/deepseek_harness/node_modules/.bin/dsh"


def _binary() -> str:
    """The absolute path of BINARY; the published argv is made repo-relative again."""
    found = resolve_binary(BINARY)
    return str(found) if found else BINARY


# The shipped model route, llm-deepseek, speaks Anthropic Messages: "An explicit baseURL or
# $DEEPSEEK_BASE_URL supplies a Messages-compatible root", and requests go to
# <root>/v1/messages with the key from DEEPSEEK_API_KEY (llm-deepseek, "Endpoint and wire
# format"; llm-deepseek-api-key).
STUB_KIND = "anthropic"

# DeepSeek's own account variables. The shared list already drops ANTHROPIC_*, OPENAI_*,
# GEMINI_* and GOOGLE_API_KEY.
SCRUBBED_ENV = ("DEEPSEEK_", "DSH_")

# $DSH_HOME holds the profiles, the home patch layer, credentials and logs, and defaults to
# ~/.dsh (apps/cli/README.md "Profiles"; reference "Startup diagnostics").
CONFIG_DIR_ENV = ("DSH_HOME",)

EGRESS_ALLOWED: dict[str, str] = {}


def install_check() -> str:
    return version_of([_binary(), "--version"])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """A Cordis patch overlay, applied with --patch after the profile and home layers.

    One `insert` row per server with the generic @deepseek-ai/dsh-mcp-client entry fields
    (docs/user/guide/mcp-memory.md, "Bring another MCP server"). Its tools reach the model
    as mcp__<serverName>__<tool>, the default mcp_server_of.
    """
    rows = []
    for name, server in servers.items():
        if "url" in server:
            config = {"serverName": name, "transport": "streamable-http", "url": server["url"]}
        else:
            config = {"serverName": name, "transport": "stdio", "command": server["command"],
                      "args": list(server["args"])}
        rows.append({"id": f"mcp-{name}", "name": "@deepseek-ai/dsh-mcp-client", "config": config})
    patch = [
        {"insert": rows},
        # The default-on session-log contributor adds a `dsh_session_log` field, the run's
        # whole Session log, to every DeepSeek model request (session-log-deepseek README:
        # "`enabled: false` stops Session-log upload"). The request goes to the stub, but
        # the log would carry the isolated home's paths into model.jsonl.
        {"id": "session-log-deepseek", "config": {"enabled": False}},
    ]
    path = workdir / "cordis.yml"
    # JSON is YAML, and the patch carries no !!js expression.
    path.write_text(json.dumps(patch, indent=2))
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """dsh has no command that lists MCP servers: the launcher's modes are profile boot,
    plugin management and the three config dumps (apps/cli/reference/README.md), and a dump
    shows configured rows without connecting to them."""
    return None


def mode_args(mode: str | None) -> list[str]:
    """Nothing at level 1. Level 2 is not measured: dsh documents no switch that approves
    held calls headless. Headless mounts no approval answerer, so an `ask` request resolves
    `unavailable` and fails closed, and DSH_PERMISSION_MODE=danger-full-access pins the
    approval policy to `never`, which rejects every request (dsh-user-approval README;
    the `approval` row of the headless profile)."""
    if mode is None:
        return []
    raise HarnessError(f"dsh documents no headless approval switch for the {mode!r} mode; measure --level 1")


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # Launcher flags come first; the first token it does not recognise starts the headless
    # app's arguments (the task positional and --json). dsh has no turn limit: one
    # invocation is one task, run to quiescence.
    return [_binary(), "--profile", "headless", "--patch", str(config), prompt, "--json", *mode_args(mode)]


def env(stub_url: str) -> dict[str, str]:
    return {
        "DEEPSEEK_BASE_URL": stub_url,
        "DEEPSEEK_API_KEY": STUB_KEY,
        # OTel session upload defaults to FEEDBACK_ONLY; DISABLED prevents capture
        # (packages/bundle/base/README.md).
        "DSH_TELEMETRY_MODE": "DISABLED",
    }


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """"MCP prompt templates are unsupported" (dsh-mcp-client README, Summary)."""
    return None


def reduce_output(text: str) -> list[dict]:
    """The --json stream: session, status, thinking, text, tool_call, tool_result, then
    final, or error for a failure outside a turn (dsh-headless README, "Machine-readable
    output"; src/json-stream.ts)."""
    events: list[dict] = []
    turns = 0
    reason = None
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "status" and item.get("phase") == "turn_end":
            turns += 1
            # The turn/end reason object, {"kind": "completed"} or aborted or error.
            reason = item.get("reason")
            if isinstance(reason, dict):
                reason = reason.get("kind")
        elif kind == "tool_call":
            events.append({"event": "tool_use", "tool": item.get("tool"), "input": item.get("input")})
        elif kind == "tool_result":
            events.append({"event": "tool_result", "is_error": item.get("status") == "error",
                           "text": clip(item.get("result") or "")})
        elif kind == "text":
            events.append({"event": "text", "text": clip(item.get("text") or "")})
        elif kind == "final":
            # Exit 0 only for a `completed` turn end (README, "Exit mapping").
            events.append({"event": "result", "subtype": reason, "is_error": reason != "completed",
                           "num_turns": turns, "text": clip(item.get("text") or "")})
        elif kind == "error":
            events.append({"event": "result", "subtype": "error", "is_error": True, "num_turns": turns,
                           "text": clip(item.get("message") or "")})
    return events
