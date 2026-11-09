"""Kiro CLI: `kiro-cli chat --no-interactive --output-format stream-json`. https://kiro.dev/docs/cli/headless/

Neither level runs, so every cell is u. Kiro CLI 2.26.0 cannot be measured fresh, for
two reasons observed on 2026-09-29:

1. Its model API cannot be stubbed. Model calls are AWS CodeWhisperer ConverseStream
   requests to runtime.us-east-1.kiro.dev and q.us-east-1.amazonaws.com, which are not
   Anthropic Messages, OpenAI or Gemini. The CLI settings reference documents no base
   URL or endpoint override (https://kiro.dev/docs/cli/reference/settings/).
2. MCP is off until the Kiro management API answers. Headless auth is KIRO_API_KEY only
   (Pro tiers and up, https://kiro.dev/docs/cli/headless/). Without an answer from
   management.<region>.kiro.dev the CLI logs "Failed to get governance config from API
   — MCP disabled, web tools disabled" and never starts a configured server. So a
   level-1 run with the egress sink in place would record an empty wire: a false n, not
   a measurement. A reachable, authenticated account breaks freshness.

Everything below is from the docs and the 2.26.0 binary, so the module is ready if
either reason goes away. Until then STUB_KIND stays None.
"""

from __future__ import annotations

import json
from pathlib import Path

from harnesses import clip, resolve_binary, version_of

# Project-local: the pinned 2.26.0 DMG from the channel the official installer reads
# (https://kiro.dev/docs/cli/, `curl -fsSL https://cli.kiro.dev/install | bash`), checked
# against the sha256 in that release's manifest.json, and its app bundle copied here with
# ditto, as the installer does, but never into /Applications.
VERSION = "2.26.0"
DMG_SHA256 = "7a45800fbe5853089f367d1e9bc9d5c674ca1b5ba755c23772dce6dea493008d"
PREFIX = ".mcp-mirror/harnesses/kiro_cli"
INSTALL = ["sh", "-c", (
    "set -eu; d=$(mktemp -d); trap 'hdiutil detach \"$d/mnt\" -quiet 2>/dev/null || true; rm -rf \"$d\"' EXIT; "
    f"curl -fsSL -o \"$d/kiro.dmg\" \"https://prod.download.cli.kiro.dev/stable/{VERSION}/Kiro%20CLI.dmg\"; "
    f"echo \"{DMG_SHA256}  $d/kiro.dmg\" | shasum -a 256 -c -; "
    "hdiutil attach \"$d/kiro.dmg\" -nobrowse -readonly -mountpoint \"$d/mnt\" >/dev/null; "
    f"mkdir -p {PREFIX}; rm -rf \"{PREFIX}/Kiro CLI.app\"; ditto \"$d/mnt/Kiro CLI.app\" \"{PREFIX}/Kiro CLI.app\""
)]
# The chat binary, not the `kiro-cli` wrapper: `kiro-cli chat` re-executes kiro-cli-chat
# from $HOME/.local/bin, which the isolated HOME does not have ("error: failed to launch
# ~/.local/bin/kiro-cli-chat"). It takes the same `chat` arguments.
BINARY = f"{PREFIX}/Kiro CLI.app/Contents/MacOS/kiro-cli-chat"
# None: see the module docstring. There is no stub for ConverseStream.
STUB_KIND: str | None = None
# The one line the site shows for an agent with no measurement, next to "Not measured yet".
UNMEASURED = "Model calls go to AWS CodeWhisperer, which cannot be scripted, and MCP stays off until Kiro's management API answers."
# The headless credential. The runner never sets it; listed so a real key cannot leak in.
SCRUBBED_ENV: tuple[str, ...] = ("KIRO_", "AWS_")
# Kiro reads ~/.kiro unless KIRO_HOME says otherwise; point it inside the isolated home too.
CONFIG_DIR_ENV: tuple[str, ...] = ("KIRO_HOME",)


def _binary() -> str:
    return str(resolve_binary(BINARY) or BINARY)


def install_check() -> str:
    return version_of([_binary(), "--version"])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """Workspace MCP config, .kiro/settings/mcp.json in the cwd. It takes precedence over
    the user-level file, and the isolated HOME has none. https://kiro.dev/docs/mcp/configuration/"""
    path = workdir / ".kiro" / "settings" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = {
        name: ({"url": server["url"]} if "url" in server else {"command": server["command"], "args": server["args"]})
        for name, server in servers.items()
    }
    path.write_text(json.dumps({"mcpServers": entries}, indent=2))
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """None. `kiro-cli-chat mcp list` (2.26.0) prints to stderr, not stdout, and in the
    isolated home it listed a stdio server but left out a `url` one, so it cannot confirm
    the five names."""
    return None


def mode_args(mode: str | None) -> list[str]:
    if mode == "default":
        # No trust flag. Headless has no one to approve a call, and the inventory records
        # that an ask is treated as a deny. https://kiro.dev/docs/cli/headless/
        return []
    if mode == "unattended":
        # "Auto-approve all tool calls without prompting." https://kiro.dev/docs/cli/headless/
        return ["--trust-all-tools"]
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # stream-json implies --no-interactive; both are spelled out as the docs show them.
    # --require-mcp-startup exits 3 if a configured server fails to start, so a dropped
    # server cannot pass silently. Kiro has no turn limit flag. The config is found from
    # the cwd (write_config), so it is not passed here.
    return [
        _binary(), "chat", "--no-interactive", "--output-format", "stream-json", "--require-mcp-startup",
        *mode_args(mode), prompt,
    ]


def env(stub_url: str) -> dict[str, str]:
    # Nothing points Kiro's model calls anywhere else; see the module docstring.
    return {}


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    return None


def reduce_output(text: str) -> list[dict]:
    """ACP events as JSON Lines. Only the types a 2.26.0 run has emitted are mapped
    (runStarted, runError). No run has reached a tool call, so tool events are not
    guessed at."""
    events: list[dict] = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind, data = item.get("type"), item.get("data") or {}
        if kind == "runStarted":
            events.append({"event": "init", "harness_version": None, "permission_mode": None,
                           "mcp_servers": None, "mcp_tools": [], "engine": data.get("engine")})
        elif kind == "runError":
            events.append({"event": "result", "subtype": data.get("stage"), "is_error": True, "num_turns": None,
                           "text": clip(data.get("message") or "")})
    return events
