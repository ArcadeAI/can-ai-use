"""Cline CLI: `cline --json "<prompt>"`, NDJSON output. https://www.npmjs.com/package/cline (README, "Headless mode")

Every flag below is from `cline --help` and the README shipped in the npm package
(checked against 3.0.66, re-checked at 3.0.71). The MCP settings path is not in either: the README only
documents the `cline mcp install` wizard, which needs a TTY. The path is the CLI's
exported CLINE_MCP_SETTINGS_FILE_NAME under its data dir,
`<CLINE_DATA_DIR or ~/.cline/data>/settings/cline_mcp_settings.json`, the same file
the VS Code extension documents at https://docs.cline.bot/mcp/configuring-mcp-servers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from harnesses import HarnessError, clip, resolve_binary, stub_env, version_of

# Project-local, pinned: the inventory's `npm i -g cline` would install globally.
INSTALL = ["npm", "install", "--prefix", ".mcp-mirror/harnesses/cline_cli", "cline@3.0.71"]
BINARY = ".mcp-mirror/harnesses/cline_cli/node_modules/.bin/cline"
# Cline takes a base URL only for its OpenAI providers (see argv), so the stub speaks
# OpenAI Responses, which is what openai-native calls.
STUB_KIND = "openai"
# CLINE_API_KEY is the Cline account key the default provider (`-P cline`) would use.
SCRUBBED_ENV: tuple[str, ...] = ("CLINE_",)
# Cline's config and data dirs (README, "Environment variables"); each is pointed inside
# the isolated home, so the settings file below is the only MCP config it can read.
CONFIG_DIR_ENV: tuple[str, ...] = ("CLINE_DIR", "CLINE_DATA_DIR")
# Cline polls PostHog feature flags at data.cline.bot on every run. telemetryOptOut
# (below) does not stop it; only IS_TEST / E2E_TEST do, and those are test switches that
# change more than telemetry. The sink refuses it, so nothing leaves the machine.
EGRESS_ALLOWED: dict[str, str] = {
    "data.cline.bot": "PostHog feature-flag poll; not covered by telemetryOptOut",
}
# Provider and model for `-P`/`-m`; the stub answers any model. The README's own
# `cline auth --provider openai-native ... --modelid gpt-5 --baseurl` example.
PROVIDER = "openai-native"
MODEL = "gpt-5"
QUIET = {
    # README: "Set to 0/false to disable runtime file logging". Nothing a cell reads.
    "CLINE_LOG_ENABLED": "0",
}


# data/settings/global-settings.json, keys from the CLI's settings schema. Without
# autoUpdateEnabled false Cline reaches registry.npmjs.org (its update check) on its own;
# telemetryOptOut keeps telemetry off (the feature-flag poll above ignores it).
GLOBAL_SETTINGS = {"telemetryOptOut": True, "autoUpdateEnabled": False}


def _binary() -> str:
    found = resolve_binary(BINARY)
    if found is None:
        raise HarnessError(f"{BINARY} is not installed")
    return str(found)


def install_check() -> str:
    return version_of([_binary(), "--version"])


# Cline gives a stdio server a hard-coded connect-attempt timeout to answer initialize
# (3s up to 3.0.68, 10s since 3.0.69 per its release note), then kills it and retries
# once in Content-Length framing, which an MCP stdio server does not speak, so the server
# is dropped. The fixtures take 1-3s to start on a busy machine, so with the 3s default
# the cells would measure host load; 60s is kept at 3.0.71 so both measurements match. A server's
# documented `timeout` (seconds) also sets that connect attempt; Cline's own error text
# says to raise it ("Increase the "timeout" field (in seconds) for this server in
# cline_mcp_settings.json"). Every fixture gets 60s except one started late on purpose
# (h-startup-timeout's util), which keeps the shipped default so that cell measures it.
SERVER_TIMEOUT_S = 60


def _started_late(server: dict) -> bool:
    """The runner starts a delayed fixture as `sh -c 'sleep N; exec ...'` (slow-start)."""
    return any("sleep " in str(arg) for arg in server.get("args") or [])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """cline_mcp_settings.json: stdio servers as command/args, HTTP as type streamableHttp.

    The schema accepts type stdio / sse / streamableHttp; a URL entry without a type is
    read as sse, so HTTP servers name streamableHttp explicitly.
    """
    path = Path(env["CLINE_DATA_DIR"]) / "settings" / "cline_mcp_settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, server in servers.items():
        if "url" in server:
            entries[name] = {"type": "streamableHttp", "url": server["url"]}
        else:
            entries[name] = {"command": server["command"], "args": server["args"]}
        if not _started_late(server):
            entries[name]["timeout"] = SERVER_TIMEOUT_S
    path.write_text(json.dumps({"mcpServers": entries}, indent=2))
    (path.parent / "global-settings.json").write_text(json.dumps(GLOBAL_SETTINGS, indent=2))
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """Cline has no non-interactive list: `cline mcp` has only install/uninstall, and
    `cline config` and `cline mcp` open a TUI ("interactive mode requires a TTY"). Since
    3.0.69 `cline config --json mcp` prints the configured servers as JSON without a TTY;
    reading it here is a separate change."""
    return None


def mode_args(mode: str | None) -> list[str]:
    """README, "Tool approval": "Tool calls are auto-approved by default." So the default
    mode passes nothing, and the unattended switch is the documented flag set to the same
    state. `--auto-approve false` (a call needing approval is denied without a TTY) is not
    the default, so no session measures it."""
    if mode == "unattended":
        return ["--auto-approve", "true"]
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # Cline takes a model base URL only through `cline auth --baseurl`, and only for the
    # OpenAI providers (README, "Quick provider setup"); the CLI ignores OPENAI_BASE_URL.
    # So the session first saves the stub as the openai-native provider in the isolated
    # data dir, then execs the prompt. The auth line reads the stub env, so no key or URL
    # is in the published argv.
    # --json: NDJSON events, non-interactive. -P/-m pin the run to that provider, so the
    # default `cline` provider (an OAuth account) is never chosen. The cwd is the empty
    # workdir, which is also where tools run. --data-dir is not passed: it turns on
    # sandbox mode, which keeps its own state; CLINE_DATA_DIR already points inside the
    # isolated home. Cline has no turn limit; --retries caps consecutive mistakes only.
    # The binary is sh's $0 and the prompt its $1, each its own argv element, so the
    # published command carries the repo-relative binary path and no machine path.
    run = f'"$0" --json -P {PROVIDER} -m {MODEL} {" ".join(mode_args(mode))}'.strip()
    script = (f'"$0" auth -p {PROVIDER} -k "$OPENAI_API_KEY" -m {MODEL} -b "$OPENAI_BASE_URL" >/dev/null'
              f' && exec {run} "$1"')
    return ["sh", "-c", script, _binary(), prompt]


def env(stub_url: str) -> dict[str, str]:
    return stub_env(STUB_KIND, stub_url) | QUIET


def mcp_server_of(tool_name: str) -> str | None:
    """Cline offers an MCP tool to the model as <server>__<tool>; its own tools have no
    double underscore (read_files, team_spawn_teammate)."""
    found = re.match(r"^(.+?)__.", tool_name)
    return found.group(1) if found else None


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    return None


def reduce_output(text: str) -> list[dict]:
    """--json NDJSON: agent_event content_start/content_end per text or tool block, and a
    closing run_result. Cline prints no init event with its tool list, and no usage,
    model info or hook ids are kept."""
    events: list[dict] = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        event = item.get("event") if isinstance(item.get("event"), dict) else {}
        if kind == "agent_event" and event.get("contentType") == "tool":
            if event.get("type") == "content_start":
                events.append({"event": "tool_use", "tool": event.get("toolName"), "input": event.get("input")})
            elif event.get("type") == "content_end":
                output = event.get("output")
                failed = bool(event.get("error")) or (isinstance(output, dict) and bool(output.get("isError")))
                if _denied(event):
                    events.append({"event": "permission_denied", "tool": event.get("toolName"),
                                   "input": event.get("input")})
                events.append({"event": "tool_result", "is_error": failed,
                               "text": clip(event.get("error") or output)})
        elif kind == "agent_event" and event.get("type") == "content_end" and event.get("contentType") == "text":
            events.append({"event": "text", "text": clip(event.get("text") or "")})
        elif kind == "run_result":
            events.append({"event": "result", "subtype": item.get("finishReason"),
                           "is_error": item.get("finishReason") == "error", "num_turns": item.get("iterations"),
                           "text": clip(item.get("text") or "")})
    return events


def _denied(event: dict) -> bool:
    """A tool block Cline closed because approval was needed and there is no TTY
    (`--auto-approve false`): its error reads 'Tool "<name>" requires approval in a TTY
    session -- NOT a tool or system failure. ...' (measured, 3.0.66)."""
    return "requires approval" in str(event.get("error") or "")
