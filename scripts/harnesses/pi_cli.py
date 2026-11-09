"""Pi: `pi --mode json -p "<prompt>" --no-session`. https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/json.md

Pi does not ask for approval before a tool call (docs/security.md: "it does not ask for
approval before every tool call"), so it has no approval switch and no machine-readable
"held" event; a permission gate is something an extension adds. Level 2 runs both
modes with the same argv, and h-destructive-blocked-by-default = n is the documented
behaviour. Its MCP docs are docs/mcp.md in the same package.

MCP tools are reached the way Pi's default exposure (`codemode`, docs/mcp.md#exposure)
presents them: through the `codemode` tool, whose script calls
`tools.mcp__<server>__<tool>(args)` (docs/cli.md, "How codemode works"). The level-2
scripts in plan/harness_sessions.yaml do exactly that. `exposure: direct` would declare
them to the model one by one; it is not the default, so the cells do not use it.
"""

from __future__ import annotations

import json
from pathlib import Path

from harnesses import clip, list_command, resolve_binary, stub_env, version_of

# Project-local and pinned; the inventory's `npm i -g` is never run.
INSTALL = ["npm", "install", "--prefix", ".mcp-mirror/harnesses/pi_cli", "@earendil-works/pi-coding-agent@1.2.0"]
BINARY = ".mcp-mirror/harnesses/pi_cli/node_modules/.bin/pi"
STUB_KIND = "anthropic"
# The model the stub stands in for: any id in Pi's bundled anthropic catalog works, the
# stub answers whatever is asked.
MODEL = "anthropic/claude-sonnet-4-5"
# Pi reads its user config (mcp.json, models.json, settings.json, trust.json, mcp.log)
# from PI_CODING_AGENT_DIR, default ~/.pi/agent (docs/environment-variables.md).
CONFIG_DIR_ENV = ("PI_CODING_AGENT_DIR",)
# PI_OFFLINE=1 turns off startup network operations (the pi.dev model-catalog overlay,
# the update check) and PI_TELEMETRY=0 install telemetry; both are documented in
# `pi --help` and docs/environment-variables.md. Neither touches MCP.
QUIET = {"PI_OFFLINE": "1", "PI_TELEMETRY": "0"}
# Since 0.99.2 (CHANGELOG #10212) Pi connects every enabled server in the background and
# the first prompt "waits up to 10 seconds only for servers with `direct` tools"
# (docs/mcp.md, "Startup"). The default exposure is `codemode`, so with the stub answering
# at once the level-1 turn ends about 1 s in, before any server has answered initialize,
# and the capture shows nothing of Pi's MCP client. Holding the stub's reply keeps the
# turn open for as long as Pi's documented first-prompt wait, 10 s: Pi then connects all
# five servers (initialize, initialized, tools/list and ping at about 0.5 s on stdio and
# 3.5 s on HTTP in a probe), and the slow-start session's util, held 8 s, is listed at
# about 8.2 s. At 5 s the turn ended before util answered: Pi no longer blocks the first
# prompt on a slow codemode server. Level 1 only; the runner records it in session.json
# and on each level-1 cell's evidence.
LEVEL1_REPLY_DELAY_S = 10

# Pi does not read ANTHROPIC_BASE_URL, and models.json does not interpolate `baseUrl`, so
# the stub URL reaches the anthropic provider through the documented provider API
# (docs/custom-provider.md: "Registering only baseUrl or headers for an existing
# provider preserves its built-in models"). It is written to <agent-dir>/extensions/, the
# user extension directory Pi discovers (docs/extensions.md), so no temp path is in argv.
STUB_EXTENSION = """\
export default function (pi) {
  pi.registerProvider("anthropic", { baseUrl: process.env.MCP_MIRROR_STUB_URL });
}
"""


def _binary() -> str:
    return str(resolve_binary(BINARY) or BINARY)


def install_check() -> str:
    return version_of([_binary(), "--version"])


def _agent_dir(env: dict[str, str]) -> Path:
    return Path(env["PI_CODING_AGENT_DIR"])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """The global <agent-dir>/mcp.json, which needs no project trust (docs/mcp.md).

    Every server gets `exposure: codemode` written out, which is Pi's default: tools are
    callable from `codemode` scripts and listed in that tool's description, not declared
    to the model one by one. Writing the default keeps the measured setting visible.
    """
    agent = _agent_dir(env)
    agent.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, server in servers.items():
        if "url" in server:
            entries[name] = {"type": "http", "url": server["url"], "exposure": "codemode"}
        else:
            entries[name] = {"command": server["command"], "args": server["args"], "exposure": "codemode"}
    (agent / "extensions").mkdir(exist_ok=True)
    (agent / "extensions" / "mcp-mirror-stub.js").write_text(STUB_EXTENSION)
    path = agent / "mcp.json"
    path.write_text(json.dumps({"mcpServers": entries}, indent=2))
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """`pi mcp list --json`: every server in mcp.json, connected or not (docs/cli.md)."""
    text = list_command([_binary(), "mcp", "list", "--json"], env | QUIET, workdir)
    try:
        listed = json.loads(text)
    except json.JSONDecodeError:
        return set()
    servers = listed.get("servers", listed) if isinstance(listed, dict) else listed
    if isinstance(servers, dict):
        return set(servers)
    return {s.get("name") for s in servers or [] if isinstance(s, dict) and s.get("name")}


def mode_args(mode: str | None) -> list[str]:
    """Pi has no approval switch: every tool call runs unless an extension gates it
    (docs/security.md, docs/mcp.md#permissions). The default is already unattended."""
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # --mode json -p: one JSONL event stream, then exit (docs/json.md). --no-session: no
    # session file. mcp.json is read from the agent dir, so `config` is not a flag. Pi has
    # no turn limit flag. --no-skills, --no-prompt-templates, --no-themes and
    # --no-context-files keep discovered resources out; --no-extensions is not used
    # because MCP support is itself a built-in extension (docs/mcp.md), and it would also
    # drop the stub provider extension.
    return [
        _binary(), "--mode", "json", "-p", prompt, "--no-session", "--model", MODEL, "--no-skills",
        "--no-prompt-templates", "--no-themes", "--no-context-files", *mode_args(mode),
    ]


def env(stub_url: str) -> dict[str, str]:
    return stub_env(STUB_KIND, stub_url) | QUIET | {"MCP_MIRROR_STUB_URL": stub_url}


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """Pi's MCP support covers tools and resources; docs/mcp.md names no way to expand
    an MCP prompt."""
    return None


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text")


def reduce_output(text: str) -> list[dict]:
    """Keep what a cell reads from docs/json.md's stream; drop system prompts, tool lists
    and usage."""
    events: list[dict] = []
    for line in text.split("\n"):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "session":
            events.append({"event": "init", "harness_version": None, "permission_mode": None,
                           "mcp_servers": None, "mcp_tools": None})
        elif kind == "tool_execution_start":
            events.append({"event": "tool_use", "tool": item.get("toolName"), "input": item.get("args")})
        elif kind == "tool_execution_end":
            events.append({"event": "tool_result", "is_error": bool(item.get("isError")),
                           "text": clip(_text((item.get("result") or {}).get("content")))})
        elif kind == "message_end" and (item.get("message") or {}).get("role") == "assistant":
            message = item["message"]
            body = _text(message.get("content"))
            if body:
                events.append({"event": "text", "text": clip(body)})
            if message.get("stopReason") == "error":
                events.append({"event": "text", "text": clip(f"error: {message.get('errorMessage')}")})
        elif kind == "agent_end":
            messages = [m for m in item.get("messages") or [] if isinstance(m, dict)]
            last = next((m for m in reversed(messages) if m.get("role") == "assistant"), {})
            events.append({"event": "result", "subtype": last.get("stopReason"),
                           "is_error": last.get("stopReason") == "error",
                           "num_turns": sum(1 for m in messages if m.get("role") == "assistant"),
                           "text": clip(_text(last.get("content")))})
    return events
