"""Claude Code: `claude -p` with stream-json output. https://code.claude.com/docs/en/headless"""

from __future__ import annotations

import json
from pathlib import Path

from harnesses import clip, list_command, names_before_colon, resolve_binary, stub_env, version_of

# Installed project-locally at the measured version, never globally, so the version under
# test does not depend on what the operator has on PATH.
# https://code.claude.com/docs/en/setup
INSTALL = ["npm", "install", "--prefix", ".mcp-mirror/harnesses/claude_code", "@anthropic-ai/claude-code@2.1.296"]
BINARY = ".mcp-mirror/harnesses/claude_code/node_modules/.bin/claude"
STUB_KIND = "anthropic"
# Without it Claude Code reaches api.anthropic.com on its own (telemetry, error reporting,
# the update check) even with ANTHROPIC_BASE_URL on the stub, and the egress sink fails
# the run. It turns off exactly that traffic and nothing a cell reads.
# https://code.claude.com/docs/en/settings#environment-variables
QUIET = {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}


def _binary() -> str:
    """Absolute, because the harness runs with the temp workdir as its cwd."""
    found = resolve_binary(BINARY)
    return str(found) if found else BINARY


def install_check() -> str:
    return version_of([_binary(), "--version"])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """--mcp-config JSON: stdio servers as command/args, HTTP servers as type http.

    Named .mcp.json so `claude mcp list` sees it as the project scope; the run reads it
    through --strict-mcp-config --mcp-config, which ignores every other scope.
    """
    path = workdir / ".mcp.json"
    entries = {}
    for name, server in servers.items():
        if "url" in server:
            entries[name] = {"type": "http", "url": server["url"]}
        else:
            entries[name] = {"command": server["command"], "args": server["args"]}
    path.write_text(json.dumps({"mcpServers": entries}, indent=2))
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """`claude mcp list`: user, local and project scopes. An unapproved project server is
    listed as pending and not connected to. https://code.claude.com/docs/en/mcp"""
    return names_before_colon(list_command([_binary(), "mcp", "list"], env | QUIET, workdir))


def mode_args(mode: str | None) -> list[str]:
    if mode == "default":
        # The documented built-in start for `claude -p` is `default` (Manual), pinned here
        # because a fresh home is a "first session after an install", which can start in
        # auto before feature flags arrive; auto's classifier is a model call the stub
        # cannot answer, so it fails closed and a cell would measure that, not a prompt.
        # https://code.claude.com/docs/en/permission-modes#which-mode-a-session-starts-in
        # With no permission host, whatever would prompt is denied and listed in the result
        # event's permission_denials, the machine-readable "held".
        # https://code.claude.com/docs/en/headless#turn-off-permission-prompts-in-unattended-runs
        return ["--permission-mode", "default"]
    if mode == "unattended":
        return ["--allowedTools", "mcp__def__*", "mcp__res__*", "mcp__ntf__*", "mcp__cf__*", "mcp__util__*"]
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # --bare: no hooks, plugins, CLAUDE.md or keychain; auth is strictly ANTHROPIC_API_KEY.
    # --setting-sources project in an empty working directory: the documented defaults,
    # not whatever the operator's user settings say.
    return [
        _binary(), "-p", prompt, "--output-format", "stream-json", "--verbose", "--bare",
        "--max-turns", str(max_turns), "--strict-mcp-config", "--mcp-config", str(config),
        "--setting-sources", "project", "--no-session-persistence", *mode_args(mode),
    ]


def env(stub_url: str) -> dict[str, str]:
    return stub_env(STUB_KIND, stub_url) | QUIET


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """MCP prompts are user slash commands in Claude Code; the model has no tool that
    expands one. https://code.claude.com/docs/en/mcp#use-mcp-prompts-as-commands"""
    return " ".join([f"/mcp__{server}__{name}", *(str(v) for v in arguments.values())])


def reduce_output(text: str) -> list[dict]:
    """Keep what a cell reads; drop the rest (slash-command lists, usage, paths)."""
    events: list[dict] = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = item.get("type")
        if kind == "system" and item.get("subtype") == "init":
            events.append({
                "event": "init",
                "harness_version": item.get("claude_code_version"),
                "permission_mode": item.get("permissionMode"),
                "mcp_servers": item.get("mcp_servers"),
                "mcp_tools": sorted(t for t in item.get("tools") or [] if t.startswith("mcp__")),
            })
        elif kind in ("assistant", "user"):
            for block in (item.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    events.append({"event": "tool_use", "tool": block.get("name"), "input": block.get("input")})
                elif block.get("type") == "tool_result":
                    events.append({"event": "tool_result", "is_error": bool(block.get("is_error")),
                                   "text": clip(block.get("content"))})
                elif block.get("type") == "text" and kind == "assistant":
                    events.append({"event": "text", "text": clip(block.get("text", ""))})
        elif kind == "result":
            for denial in item.get("permission_denials") or []:
                events.append({"event": "permission_denied", "tool": denial.get("tool_name"),
                               "input": denial.get("tool_input")})
            events.append({"event": "result", "subtype": item.get("subtype"), "is_error": item.get("is_error"),
                           "num_turns": item.get("num_turns"), "text": clip(item.get("result") or "")})
    return events
