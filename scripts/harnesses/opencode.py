"""OpenCode: `opencode run "<prompt>" --format json`. https://opencode.ai/docs/cli/#run"""

from __future__ import annotations

import json
import re
from pathlib import Path

from harnesses import clip, list_command, resolve_binary, stub_env, version_of

# Installed project-locally at a pinned version by measure_harness.py when missing.
# https://opencode.ai/docs/#install (npm package opencode-ai)
INSTALL = ["npm", "install", "--prefix", ".mcp-mirror/harnesses/opencode", "opencode-ai@1.19.0"]
BINARY = ".mcp-mirror/harnesses/opencode/node_modules/.bin/opencode"


def _binary() -> str:
    """The absolute path of BINARY; the published argv is made repo-relative again."""
    found = resolve_binary(BINARY)
    return str(found) if found else BINARY


# OpenCode's built-in anthropic provider, pointed at the stub through options.baseURL in
# opencode.json. https://opencode.ai/docs/providers/#base-url
STUB_KIND = "anthropic"
# A model defined in the config, so nothing needs the models.dev catalogue.
MODEL = "anthropic/stub"

# OpenCode's own real-account variables. https://opencode.ai/docs/cli/#environment-variables
SCRUBBED_ENV = ("OPENCODE_",)

# Documented switches that turn off what OpenCode fetches on its own: the update check,
# the models.dev catalogue, and the default plugins (which it would npm-install).
# https://opencode.ai/docs/cli/#environment-variables
QUIET = {
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_DISABLE_MODELS_FETCH": "1",
    "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
    # No CLAUDE.md or ~/.claude skills folded into the prompt (the home is empty anyway).
    "OPENCODE_DISABLE_CLAUDE_CODE": "1",
    "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
}

# OpenCode installs @opencode-ai/plugin into its config dir in the background on every
# start (for local plugins and custom tools). No documented switch turns it off; the
# refused GET is logged as "background dependency install failed" and the run goes on.
EGRESS_ALLOWED = {
    "registry.npmjs.org": "background install of @opencode-ai/plugin into the config dir on start; "
                          "no documented off switch, and a refusal only logs a warning",
}

# opencode names an MCP tool <server>_<tool>; its own tools carry no server prefix.
# https://opencode.ai/docs/mcp-servers/
BUILTIN = {"bash", "edit", "glob", "grep", "list", "patch", "read", "skill", "task", "todoread", "todowrite",
           "webfetch", "websearch", "write", "question", "lsp", "codesearch", "apply_patch", "invalid", "batch",
           "multiedit", "plan_enter", "plan_exit",
           # Built-in resource tools, offered when a connected server has resources.
           "list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource"}


def install_check() -> str:
    return version_of([_binary(), "--version"])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """opencode.json in the workdir: the project config OpenCode loads from its cwd.

    Local servers are `type: local` with a command array; HTTP servers `type: remote`.
    The timeout stays at the documented default (5000 ms), which h-startup-timeout
    measures. https://opencode.ai/docs/mcp-servers/
    """
    mcp = {}
    for name, server in servers.items():
        if "url" in server:
            mcp[name] = {"type": "remote", "url": server["url"], "oauth": False}
        else:
            mcp[name] = {"type": "local", "command": [server["command"], *server["args"]]}
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": MODEL,
        # The title generator uses small_model (a real Haiku id by default); keep it on the stub.
        "small_model": MODEL,
        "autoupdate": False,
        "share": "disabled",
        "provider": {
            "anthropic": {
                "options": {"baseURL": "{env:ANTHROPIC_BASE_URL}/v1", "apiKey": "{env:ANTHROPIC_API_KEY}"},
                "models": {"stub": {"name": "stub", "tool_call": True, "limit": {"context": 200000, "output": 8192}}},
            }
        },
        "mcp": mcp,
    }
    path = workdir / "opencode.json"
    path.write_text(json.dumps(config, indent=2))
    return path


LISTED = re.compile(r"^\s*[●○✓✗•*-]?\s*[✓✗○●]?\s*([A-Za-z0-9_.-]+)\s+(?:\S*\s*)?(connected|failed|disabled|pending|needs)", re.I)


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """`opencode mcp list`: every server in the merged config with its status."""
    text = list_command([_binary(), "mcp", "list"], env | QUIET, workdir)
    return {m.group(1) for line in _plain(text).splitlines() if (m := LISTED.match(line))}


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def mode_args(mode: str | None) -> list[str]:
    if mode == "unattended":
        # "auto-approve permissions that are not explicitly denied (dangerous!)"
        # https://opencode.ai/docs/cli/#run
        return ["--auto"]
    # "default": no flag. OpenCode's documented default is that most permissions,
    # MCP tools included, are "allow". https://opencode.ai/docs/permissions/
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # --format json: raw JSON events, one per line. --pure: no external plugins.
    # OpenCode has no turn limit flag; the stub's script ends the turn.
    # A prompt_invocation start text ("/<command> <args>") runs as `--command <command>`
    # with the rest as the message: "the command to run, use message for args" (run --help).
    if prompt.startswith("/"):
        command, _, rest = prompt[1:].partition(" ")
        message = [rest, "--command", command] if rest else ["--command", command]
    else:
        message = [prompt]
    return [_binary(), "run", *message, "--format", "json", "--pure", "--model", MODEL, *mode_args(mode)]


def env(stub_url: str) -> dict[str, str]:
    return stub_env(STUB_KIND, stub_url) | QUIET


def mcp_server_of(tool_name: str) -> str | None:
    if tool_name in BUILTIN or "_" not in tool_name:
        return None
    return tool_name.split("_", 1)[0]


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """OpenCode lists each MCP prompt as a command named <server>:<prompt>, its arguments
    positional ($1, $2, ...), and `opencode run --command` runs one. Read from the 1.18.33
    source (unchanged at 1.19.0): the docs do not say how an MCP prompt is named."""
    return " ".join([f"/{server}:{name}", *(str(v) for v in arguments.values())])


# How `opencode run` ends a tool call it held: without --auto every permission ask is
# answered "reject", and the tool part ends in error with one of these messages.
HELD = ("The user rejected permission to use this specific tool call",
        "The user has specified a rule which prevents you from using this specific tool call")


def reduce_output(text: str) -> list[dict]:
    """`--format json` events: step_start, tool_use, text, step_finish and error, each with a
    `part`. Keep what a cell reads; drop ids, timings, tokens and attachments."""
    events: list[dict] = []
    turns = 0
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        kind, part = item.get("type"), item.get("part") or {}
        if kind == "tool_use":
            state = part.get("state") or {}
            tool, arguments = part.get("tool"), state.get("input")
            events.append({"event": "tool_use", "tool": tool, "input": arguments})
            error = state.get("status") == "error"
            message = str(state.get("error") or "") if error else state.get("output")
            if error and message.startswith(HELD):
                events.append({"event": "permission_denied", "tool": tool, "input": arguments})
            events.append({"event": "tool_result", "is_error": error, "text": clip(message or "")})
        elif kind == "text":
            events.append({"event": "text", "text": clip(part.get("text") or "")})
        elif kind == "step_finish":
            turns += 1
            if part.get("reason") not in ("tool-calls", "tool_calls"):
                last = next((e["text"] for e in reversed(events) if e["event"] == "text"), "")
                events.append({"event": "result", "subtype": part.get("reason"), "is_error": False,
                               "num_turns": turns, "text": last})
        elif kind == "error":
            error = item.get("error") or {}
            message = (error.get("data") or {}).get("message") or error.get("name") or ""
            events.append({"event": "result", "subtype": "error", "is_error": True, "num_turns": turns,
                           "text": clip(message)})
    return events
