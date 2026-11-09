"""Kilo CLI: `kilo run "<prompt>" --format json`. https://kilo.ai/docs/code-with-ai/platforms/cli

Kilo CLI 7.x is a fork of OpenCode (its source marks every change `kilocode_change`), so
its config, run events and permission model follow OpenCode's, under kilo names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from harnesses import clip, list_command, resolve_binary, stub_env, version_of

# Installed project-locally at a pinned version by measure_harness.py when missing.
# https://kilo.ai/docs/code-with-ai/platforms/cli (npm package @kilocode/cli; the
# platform binary comes in as the optional dependency @kilocode/cli-<os>-<arch>)
INSTALL = ["npm", "install", "--prefix", ".mcp-mirror/harnesses/kilo", "@kilocode/cli@7.8.8"]
BINARY = ".mcp-mirror/harnesses/kilo/node_modules/.bin/kilo"


def _binary() -> str:
    """The absolute path of BINARY; the published argv is made repo-relative again."""
    found = resolve_binary(BINARY)
    return str(found) if found else BINARY


# Kilo's built-in anthropic provider, pointed at the stub through options.baseURL in
# kilo.json ("Provider settings support baseURL and options").
# https://kilo.ai/docs/code-with-ai/platforms/cli
STUB_KIND = "anthropic"
# A model defined in the config, so nothing needs the models.dev catalogue.
MODEL = "anthropic/stub"

# Kilo's own real-account variables (KILO_API_KEY, KILO_ORG_ID, KILOCODE_<FIELD>), and the
# OPENCODE_ names the fork still reads. https://kilo.ai/docs/code-with-ai/platforms/cli
SCRUBBED_ENV = ("KILO_", "KILOCODE_", "OPENCODE_")

# Kilo's docs list no switch for what it fetches on its own. These are OpenCode's documented
# switches (https://opencode.ai/docs/cli/#environment-variables) under the KILO_ prefix the
# fork renamed them to, read from the 7.8.3 binary (7.8.8 yields no such strings; with them set,
# 72/72 sessions on 7.8.8 still attempted no egress): the update check, the models catalogue,
# the default plugins (which it would npm-install), and Kilo's own additions for telemetry,
# session ingest and presence.
QUIET = {
    "KILO_DISABLE_AUTOUPDATE": "1",
    "KILO_DISABLE_MODELS_FETCH": "1",
    "KILO_DISABLE_DEFAULT_PLUGINS": "1",
    # No CLAUDE.md or ~/.claude skills folded into the prompt (the home is empty anyway).
    "KILO_DISABLE_CLAUDE_CODE": "1",
    "KILO_DISABLE_LSP_DOWNLOAD": "1",
    "KILO_DISABLE_SHARE": "1",
    "KILO_DISABLE_SESSION_INGEST": "1",
    "KILO_DISABLE_PRESENCE": "1",
    "KILO_DISABLE_CODEBASE_INDEXING": "1",
    # PostHog product telemetry (us.i.posthog.com): on unless KILO_TELEMETRY_LEVEL is "all"
    # or unset with telemetry enabled; any other value turns it off. Not documented.
    "KILO_TELEMETRY_LEVEL": "off",
}

EGRESS_ALLOWED: dict[str, str] = {}

# "Each MCP tool's permission key is its namespaced name: {server}_{tool}"; Kilo's own tools
# carry no server prefix. https://kilo.ai/docs/automate/mcp/using-in-kilo-code
BUILTIN = {"bash", "edit", "glob", "grep", "list", "patch", "read", "skill", "task", "todoread", "todowrite",
           "webfetch", "websearch", "write", "question", "lsp", "codesearch", "apply_patch", "invalid", "batch",
           "multiedit", "plan_enter", "plan_exit",
           # Kilo's own additions, offered in every 7.8.3 session (several contain "_").
           "agent_manager_models", "background_process", "board_post", "board_read", "cancel_wakeup",
           "cron_create", "cron_delete", "cron_list", "goal", "goal_report", "kilo_local_recall", "link_pr",
           "schedule_wakeup",
           # Built-in resource tools, offered when a connected server has resources.
           "list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource"}


def install_check() -> str:
    return version_of([_binary(), "--version"])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """kilo.json in the isolated home's global config dir ("~/.config/kilo/kilo.json[c]
    (preferred)"). https://kilo.ai/docs/code-with-ai/platforms/cli

    Not the project ./kilo.json: in a project config Kilo 7.8.3 refuses `{env:...}`
    references ("blocked environment reference"), which the provider block needs, and
    then skips the whole file with only a log line.

    Local servers are `type: local` with a command array; HTTP servers `type: remote`.
    The timeout stays at the documented default (10 s local, 15 s remote), which
    h-startup-timeout measures. https://kilo.ai/docs/automate/mcp/using-in-kilo-code
    """
    mcp = {}
    for name, server in servers.items():
        if "url" in server:
            mcp[name] = {"type": "remote", "url": server["url"], "oauth": False}
        else:
            mcp[name] = {"type": "local", "command": [server["command"], *server["args"]]}
    config = {
        "model": MODEL,
        # The title generator uses small_model (a real Haiku id by default); keep it on the stub.
        "small_model": MODEL,
        "autoupdate": False,
        "share": "disabled",
        # Kilo's own gateway provider fetches its model list from api.kilo.ai on every start;
        # nothing here uses it. disabled_providers is OpenCode's documented key
        # (https://opencode.ai/docs/config/#disabled-providers), and Kilo keeps it.
        "disabled_providers": ["kilo"],
        "provider": {
            "anthropic": {
                "options": {"baseURL": "{env:ANTHROPIC_BASE_URL}/v1", "apiKey": "{env:ANTHROPIC_API_KEY}"},
                "models": {"stub": {"name": "stub", "tool_call": True, "limit": {"context": 200000, "output": 8192}}},
            }
        },
        "mcp": mcp,
    }
    path = Path(env["XDG_CONFIG_HOME"]) / "kilo" / "kilo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    return path


LISTED = re.compile(r"^\s*[●○✓✗•*-]?\s*[✓✗○●]?\s*([A-Za-z0-9_.-]+)\s+(?:\S*\s*)?(connected|failed|disabled|pending|needs)", re.I)


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """`kilo mcp list`: every server in the merged config with its status."""
    # The provider block reads {env:ANTHROPIC_BASE_URL}; the list runs before the stub
    # exists, so give it an inert URL rather than "/v1". Nothing calls the model.
    inert = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9", "ANTHROPIC_API_KEY": "placeholder"}
    text = list_command([_binary(), "mcp", "list"], env | QUIET | inert, workdir)
    return {m.group(1) for line in _plain(text).splitlines() if (m := LISTED.match(line))}


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def mode_args(mode: str | None) -> list[str]:
    if mode == "unattended":
        # "auto-approve permissions that are not explicitly denied (dangerous!)" (kilo run
        # --help, 7.8.3). https://kilo.ai/docs/code-with-ai/platforms/cli
        return ["--auto"]
    # "default": no flag. Kilo documents MCP tools as needing runtime approval unless a
    # `permission` rule allows them (https://kilo.ai/docs/automate/mcp/using-in-kilo-code);
    # `kilo run` without --auto answers every ask "reject" ("auto-rejecting").
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # --format json: raw JSON events, one per line. --pure: no external plugins.
    # Kilo has no turn limit flag; the stub's script ends the turn.
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
    """Kilo, like OpenCode, lists each MCP prompt as a command named <server>:<prompt>, its
    arguments positional ($1, $2, ...), and `kilo run --command` runs one. The docs do not
    say how an MCP prompt is named; the prompt session's capture shows whether it holds."""
    return " ".join([f"/{server}:{name}", *(str(v) for v in arguments.values())])


# How `kilo run` ends a tool call it held: without --auto every permission ask is
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
