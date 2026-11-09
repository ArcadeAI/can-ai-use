"""Cursor agent CLI: `agent -p` with stream-json output. https://cursor.com/docs/cli/headless

Installed project-locally, never globally, at a pinned build: INSTALL downloads the
tarball Cursor's installer (curl https://cursor.com/install -fsS | bash) fetches for that
build into .mcp-mirror/harnesses/cursor_agent/<build>/, without the installer's ~/.local
symlinks. BINARY is Cursor's `cursor-agent` there, never `agent` on PATH: on this machine
that name is Grok's CLI.

The CLI leaves its `index.js worker-server` child running after it exits; the runner
reaps it a few seconds later and records it (fresh.json, checks.process_group.outlived).

No stub can answer it, so STUB_KIND is None. `agent -p` first exchanges CURSOR_API_KEY
at <endpoint>/auth/exchange_user_api_key and fails with "The provided API key is
invalid" before it reads .cursor/mcp.json, so with a placeholder no fixture sees an
initialize. The model call that follows goes to Cursor's own agent service, not to an
Anthropic, OpenAI or Gemini API: `--endpoint` / CURSOR_API_ENDPOINT only moves that
service (default https://api2.cursor.sh), and the docs name no model base URL or
bring-your-own key for the CLI.

So this row is the one measured live: `op run --env-file .env.harness -- uv run python
scripts/measure_locked.py -- uv run python scripts/measure_harness.py cursor_agent
--live` runs both levels with one 1Password approval for the run, with a real key that
reaches only the CLI's process, in the same throwaway home every stubbed session gets
(measure_harness.live_secrets). The sink
forwards CONNECT to LIVE_EGRESS_ALLOWED and nothing else (the CLI honours HTTPS_PROXY:
with a placeholder key it sends one CONNECT api2.cursor.sh:443 and, refused, stops). The
row is published as measured_live, with the model the stream-json init event reports,
and the agent page says that the result came from a live service.

Every session, live or stubbed, runs with AGENT_CLI_CREDENTIAL_STORE=memory (QUIET). The
CLI's default store is the macOS keychain: at every start it persists CURSOR_API_KEY
there under the account `cursor-user`, and in a throwaway home with no login keychain
macOS raises a "Keychain Not Found" dialog per session. The switch is not in the docs or
--help; it is read in the bundle's src/utils/credential-store.ts (build 2026.10.12:
`memory`, `file` for ~/.cursor/auth.json, otherwise the keychain), so it is a bundle-only
switch and may move with a build. After every live session, and once more over the whole
run, the runner checks that the key's value appears in no file of the capture or the
session homes (measure_harness.check_secrets_absent).

A live session is contained (measure_harness.LiveSandbox). The first live run (34 of 78
sessions, stopped) showed why: in default mode the CLI runs what it classes as simple
shell commands without asking (reads; a change to a file is proposed, not applied,
without --force), and the live model used them to read outside its throwaway home by
absolute path: the operator's repository worktree, the operator's
~/.local/share/cursor-agent, and the operator's own cursor-agent, which it ran. Reads
only, nothing written, but every one of them outside the session. Under the profile
nothing outside the session's home, its capture directory, the fixtures and their
interpreters and this install is readable, listable, writable or executable, and TCP
leaves to the sink alone. Two more things that run showed, which the notes beside the
cells should say: headless, the CLI declines an elicitation request itself rather than
handing it to the model, so the elicitation cells read the CLI; and the CLI keeps its
per-project state under <home>/.cursor/projects only while that path is at most 84
characters, falling back to the shared /tmp/.cursor otherwise, which is why a live
session's home is made under /tmp (measure_harness.SHORT_HOME).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from harnesses import clip, list_command, resolve_binary, version_of

# The build https://cursor.com/install pinned on 2026-09-29. The URL, the os/arch names
# and --strip-components=1 are the installer's own.
BUILD = "2026.10.12-d47ca18"
PREFIX = f".mcp-mirror/harnesses/cursor_agent/{BUILD}"
INSTALL = ["bash", "-c", (
    "set -eo pipefail; "
    'case "$(uname -s)" in Darwin) os=darwin;; Linux) os=linux;; *) exit 1;; esac; '
    'case "$(uname -m)" in x86_64|amd64) arch=x64;; arm64|aarch64) arch=arm64;; *) exit 1;; esac; '
    f"mkdir -p {PREFIX}; "
    f"curl -fsSL https://downloads.cursor.com/lab/{BUILD}/$os/$arch/agent-cli-package.tar.gz "
    f"| tar --strip-components=1 -xzf - -C {PREFIX}"
)]
BINARY = f"{PREFIX}/cursor-agent"

STUB_KIND: str | None = None
# The one line the site shows for an agent with no measurement, next to "Not measured yet".
UNMEASURED = "The CLI only talks to Cursor's own agent service, so it cannot be pointed at the scripted model; it is measured live, under op run, when a key is available."  # noqa: E501

# A real Cursor account; a stubbed session must never carry one. A live session gets
# CURSOR_API_KEY from the env file's variable alone, read once from the runner's
# environment under the operator's single op run (measure_harness.live_secrets).
SCRUBBED_ENV: tuple[str, ...] = ("CURSOR_",)
# What a live session may reach, and why. The auth exchange and the agent service share
# the one endpoint (`--endpoint`, default https://api2.cursor.sh); every other host stays
# refused and logged, as in a stubbed session.
LIVE_EGRESS_ALLOWED: dict[str, str] = {
    "api2.cursor.sh": "Cursor's agent service: the API key exchange and every model call go here",
    "agentn.us.api5.cursor.sh": "Cursor's agent service for the account's region: the CLI opened it beside api2 "
                                "in the first live run, where the sink refused it",
}
# Attempted but never forwarded: the sink refuses these and the session still counts as
# fresh. The CLI asks downloads.cursor.com for a newer build at start; refused, it runs
# the pinned one.
EGRESS_ALLOWED: dict[str, str] = {
    "downloads.cursor.com": "the CLI's update check; refused, so the measured build stays the pinned one",
    "registry.npmjs.org": "the bundle's local TypeScript LSP provider (@anysphere/local-lsp) runs "
                          "`npx -y typescript-language-server --stdio` when npx is on PATH, and npx "
                          "fetches the package from the registry; no switch in --help, the docs or the "
                          "bundle's env vars disables it, so it is refused and the CLI runs without a "
                          "language server, which touches no MCP cell",
    "o4504648565915648.ingest.us.sentry.io": "the bundle's Sentry error telemetry (the DSN is compiled into the "
                                             "bundle, and no switch in --help, the docs or the bundle's env vars "
                                             "turns it off); refused, so nothing about the session leaves",
    "github.com": "reached for by the CLI itself in the first live run, not by a model command (the bundle "
                  "resolves github.com for its repository and plugin features); refused, and no cell reads it",
}
# Level 1 runs twice (measure_harness.level_1_modes): the documented default, where
# `agent -p` starts no project MCP server headless (the first live run: no fixture saw an
# initialize, the init event reported mcp_servers null and 0 MCP tools), and once more
# with mode_args("unattended"), whose --approve-mcps is the documented switch:
# "Automatically approve all MCP servers" (cursor-agent --help; cursor.com/docs/cli/
# reference/parameters). Level-1 cells read the approved session and say so beside the
# default's count. Level 2's default mode carries --approve-mcps alone (mode_args), so
# its cells read the tool-call default and not the server-approval gate.
LEVEL1_UNATTENDED = ("--approve-mcps, the documented switch that approves project MCP servers headless; "
                     "the default run starts none")
# Shown on the agent page and column tooltips: "measured live against Cursor's service".
LIVE_SERVICE = "Cursor"
# Keep the key out of the operator's keychain (module docstring): bundle-only switch,
# src/utils/credential-store.ts at build 2026.10.12. Set for every session, live or not.
QUIET: dict[str, str] = {"AGENT_CLI_CREDENTIAL_STORE": "memory"}

def _binary() -> str:
    """Absolute, as argv[0]; the runner publishes it repo-relative."""
    return str(resolve_binary(BINARY) or BINARY)


LISTED = re.compile(r"^([A-Za-z0-9_.-]+): \S")
ROUTED = re.compile(r"^\s*>\s*Auto routed to ([^\n]+)")


def install_check() -> str:
    """`agent --version` prints a date build, e.g. 2026.09.28-64d2043."""
    return version_of([_binary(), "--version"])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """.cursor/mcp.json in the workdir, the project config the CLI shares with the IDE.

    Stdio servers are command/args, HTTP servers url. https://cursor.com/docs/context/mcp
    """
    path = workdir / ".cursor" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = {
        name: ({"url": server["url"]} if "url" in server else {"command": server["command"], "args": server["args"]})
        for name, server in servers.items()
    }
    path.write_text(json.dumps({"mcpServers": entries}, indent=2))
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """`agent mcp list`: `<name>: <status>` per configured server. It runs with no account;
    an unapproved project server is listed as "not loaded (needs approval)".
    https://cursor.com/docs/cli/mcp"""
    text = list_command([_binary(), "mcp", "list"], env | QUIET, workdir)
    return {m.group(1) for line in re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).splitlines()
            if (m := LISTED.match(line))}


def mode_args(mode: str | None) -> list[str]:
    if mode == "default":
        # Level 2's default: --approve-mcps alone, so the project servers start (headless,
        # none does without it: LEVEL1_UNATTENDED) and the tool-call approval stays the
        # documented default, where a change is proposed, not applied, without --force.
        # With neither switch a "blocked" cell would only say that no server ran.
        # https://cursor.com/docs/cli/headless
        return ["--approve-mcps"]
    if mode == "unattended":
        # The inventory's unattended switch: --force runs every tool call unless denied,
        # --approve-mcps skips the per-server approval. https://cursor.com/docs/cli/headless
        return ["--force", "--approve-mcps"]
    # Level 1's default pass: no switch at all, the documented default.
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # -p with stream-json: one JSON event per line. --trust: the temp workdir is not a
    # trusted workspace, and headless there is no prompt to trust it. The CLI has no turn
    # limit flag. https://cursor.com/docs/cli/headless
    return [_binary(), "-p", prompt, "--output-format", "stream-json", "--trust", *mode_args(mode)]


def env(stub_url: str) -> dict[str, str]:
    """Nothing points Cursor's model calls at a stub (see the module docstring); the
    credential-store switch applies to every session, live ones included."""
    return dict(QUIET)


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """The CLI docs name no user-side way to expand an MCP prompt."""
    return None


def reduce_output(text: str) -> list[dict]:
    """stream-json: system/init, assistant, tool_call started/completed, result.
    https://cursor.com/docs/cli/reference/output-format"""
    events: list[dict] = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = item.get("type")
        if kind == "system" and item.get("subtype") == "init":
            # `model` is the one the service answered with; the published row carries it.
            events.append({"event": "init", "harness_version": None, "model": item.get("model"),
                           "permission_mode": item.get("permissionMode"), "mcp_servers": None, "mcp_tools": None})
        elif kind == "assistant":
            for block in (item.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    # With the Auto model the reply opens with "> Auto routed to <model>":
                    # the model the service picked, which the row's model field records
                    # beside the init event's "Auto Balance".
                    if (routed := ROUTED.match(text)) is not None:
                        events.append({"event": "routed", "model": routed.group(1).strip()})
                    events.append({"event": "text", "text": clip(text)})
        elif kind == "tool_call":
            # tool_call holds one key: the built-in's kind (readToolCall, ...) or `function`
            # with a name for other tools. The docs show no MCP example; untested (no stub).
            key, call = next(iter((item.get("tool_call") or {}).items()), (None, {}))
            call = call or {}
            if item.get("subtype") == "started":
                events.append({"event": "tool_use", "tool": call.get("name") or key,
                               "input": call.get("args") or call.get("arguments")})
            elif item.get("subtype") == "completed":
                result = call.get("result") or {}
                events.append({"event": "tool_result", "is_error": "error" in result or "rejected" in result,
                               "text": clip(result)})
        elif kind == "result":
            events.append({"event": "result", "subtype": item.get("subtype"), "is_error": item.get("is_error"),
                           "num_turns": None, "text": clip(item.get("result") or "")})
    return events
