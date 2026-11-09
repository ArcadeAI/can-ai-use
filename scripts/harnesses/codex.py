"""Codex CLI: `codex exec --json`. https://learn.chatgpt.com/docs/non-interactive-mode

Configuration keys are from https://learn.chatgpt.com/docs/config-file/config-reference.
`codex exec` 0.162 (as 0.159) has no -a/--ask-for-approval flag (the inventory's `-a never` is from
the interactive CLI); the approval policy is the `approval_policy` key, set with -c.
"""

from __future__ import annotations

import json
from pathlib import Path

from harnesses import clip, list_command, resolve_binary, stub_env, version_of

VERSION = "0.162.1"
INSTALL = ["npm", "install", "--prefix", ".mcp-mirror/harnesses/codex", f"@openai/codex@{VERSION}"]
BINARY = ".mcp-mirror/harnesses/codex/node_modules/.bin/codex"
STUB_KIND = "openai"
# CODEX_API_KEY is the key `codex exec` reads before OPENAI_API_KEY; never a real one here.
SCRUBBED_ENV = ("CODEX_",)

# Written into config.toml: the documented switches for traffic Codex sends on its own
# (analytics, the update check, feedback uploads). With them alone a session still made
# CONNECTs to github.com, api.github.com and chatgpt.com; turning off the plugins feature
# (the plugin marketplace sync) removed all of them. None of this is anything a cell reads.
QUIET_CONFIG = """\
check_for_update_on_startup = false

[analytics]
enabled = false

[feedback]
enabled = false

[features]
plugins = false
"""


def _binary() -> str:
    found = resolve_binary(BINARY)
    return str(found) if found else BINARY


def install_check() -> str:
    return version_of([_binary(), "--version"])


def _toml(value: object) -> str:
    return json.dumps(value)  # a JSON string or list of strings is valid TOML


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """$CODEX_HOME/config.toml, the only file Codex reads MCP servers from: one
    [mcp_servers.<name>] table each, stdio as command/args, HTTP as url."""
    path = Path(env["CODEX_HOME"]) / "config.toml"
    lines = [QUIET_CONFIG]
    for name, server in servers.items():
        lines.append(f"[mcp_servers.{name}]")
        if "url" in server:
            lines.append(f"url = {_toml(server['url'])}")
        else:
            lines.append(f"command = {_toml(server['command'])}")
            lines.append(f"args = {_toml(server['args'])}")
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """`codex mcp list --json`: every configured server; a disabled one is not connected to."""
    text = list_command([_binary(), "mcp", "list", "--json"], env, workdir)
    try:
        listed = json.loads(text)
    except json.JSONDecodeError:
        return set()
    return {item["name"] for item in listed if isinstance(item, dict) and item.get("enabled")}


# Level 2 only. mcp_optional_startup_grace_ms (default 1000) is how long Codex waits for
# MCP servers while it builds the first tool catalog; the fixtures can take longer than
# that to answer initialize on a loaded machine, and a server that misses it is not in
# the first model request. Level 1 keeps the default and measures exactly that; level 2
# waits, so its cells measure approvals and the wire, not the fixtures' cold start.
LEVEL_2_GRACE = ["-c", "mcp_optional_startup_grace_ms=30000"]


def mode_args(mode: str | None) -> list[str]:
    if mode == "default":
        # `codex exec` out of the box: its approval policy is already never (there is
        # nobody to ask), so a call that needs approval fails with "MCP tool call requires
        # approval, but approval policy is never" in the JSONL stream; that is the held
        # event. MCP tools keep Codex's default approval mode.
        return list(LEVEL_2_GRACE)
    if mode == "unattended":
        # The inventory's unattended switch: approval_policy never plus every fixture
        # server's tools approved with default_tools_approval_mode.
        return [*LEVEL_2_GRACE, "-c", 'approval_policy="never"',
                *(arg for name in ("def", "res", "ntf", "cf", "util")
                  for arg in ("-c", f'mcp_servers.{name}.default_tools_approval_mode="approve"'))]
    return []


# Codex reads no base-URL variable, so the stub is a model_providers entry set with -c.
# env() is called with the stub's URL before argv() for the same session; this keeps it.
_STUB_URL: list[str] = []


def _provider_args() -> list[str]:
    if not _STUB_URL:
        return []
    base = f"{_STUB_URL[-1]}/v1"
    return [
        "-c", 'model_provider="stub"',
        "-c", 'model_providers.stub.name="mcp-mirror stub"',
        "-c", f"model_providers.stub.base_url={_toml(base)}",
        "-c", 'model_providers.stub.env_key="OPENAI_API_KEY"',
        "-c", 'model_providers.stub.wire_api="responses"',
    ]


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # --skip-git-repo-check: the workdir is an empty temp dir, not a repository.
    # --ephemeral: no session files. --ignore-rules: no operator execpolicy rules.
    # -s read-only: the model's shell is not what is measured. The config is read from
    # $CODEX_HOME; Codex has no turn limit flag.
    return [
        _binary(), "exec", "--json", "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
        "--color", "never", "-s", "read-only", "-m", "stub", *_provider_args(), *mode_args(mode), prompt,
    ]


def env(stub_url: str) -> dict[str, str]:
    _STUB_URL.append(stub_url)
    return stub_env(STUB_KIND, stub_url)


def mcp_server_of(tool_name: str) -> str | None:
    """Codex offers an MCP tool as mcp__<server>__<tool>."""
    import harnesses

    return harnesses.mcp_server_of(tool_name)


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """Codex exec has no user-side way to expand an MCP prompt."""
    return None


def reduce_output(text: str) -> list[dict]:
    events: list[dict] = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        body = item.get("item") if isinstance(item.get("item"), dict) else {}
        if kind == "item.completed" and body.get("type") == "mcp_tool_call":
            tool = f"mcp__{body.get('server')}__{body.get('tool')}"
            events.append({"event": "tool_use", "tool": tool, "input": body.get("arguments")})
            error = body.get("error")
            message = (error or {}).get("message") if isinstance(error, dict) else error
            if message and "requires approval" in str(message):
                # Codex refused the call before sending it: the machine-readable "held".
                events.append({"event": "permission_denied", "tool": tool, "input": body.get("arguments")})
            if error is not None or body.get("status") == "failed":
                events.append({"event": "tool_result", "is_error": True, "text": clip(message or body.get("result"))})
            else:
                result = body.get("result") or {}
                events.append({"event": "tool_result", "is_error": bool(result.get("is_error") if isinstance(result, dict) else False),
                               "text": clip(result)})
        elif kind == "item.completed" and body.get("type") == "agent_message":
            events.append({"event": "text", "text": clip(body.get("text") or "")})
        elif kind in ("turn.completed", "turn.failed"):
            events.append({"event": "result", "subtype": kind, "is_error": kind == "turn.failed", "num_turns": None,
                           "text": clip((item.get("error") or {}).get("message") or "")})
    return events
