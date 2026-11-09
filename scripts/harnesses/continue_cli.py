"""Continue CLI: `cn -p "<prompt>" --format json` with a local --config. https://docs.continue.dev/cli/headless-mode

Continue's headless JSON output is the final reply only (one JSON object, or the reply
wrapped as {"response", "status", "note"}); it carries no tool calls. So reduce_output
emits a single `result` event, and every tool call is read from the recorders and from
what the stub was sent back (model.jsonl), not from stdout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from harnesses import HarnessError, ROOT, clip, resolve_binary, version_of

# Project-local, pinned: the version measured is the version installed.
INSTALL = ["npm", "install", "--prefix", ".mcp-mirror/harnesses/continue_cli", "@continuedev/cli@1.5.47"]
BINARY = ".mcp-mirror/harnesses/continue_cli/node_modules/.bin/cn"

# The config names an `openai` provider whose apiBase is the stub.
# https://docs.continue.dev/customize/model-providers/top-level/openai
STUB_KIND = "openai"

# Continue reads its global dir (config.yaml, permissions.yaml, sessions, .env) from
# CONTINUE_GLOBAL_DIR, else ~/.continue. Pointing it inside the isolated home keeps the
# operator's saved config URI, permissions and login out.
CONFIG_DIR_ENV = ("CONTINUE_GLOBAL_DIR",)
# CONTINUE_API_KEY would log the CLI into the Continue hub.
SCRUBBED_ENV = ("CONTINUE_API_KEY", "CONTINUE_API_BASE", "ORGANIZATION_ID")

# Telemetry off (CONTINUE_METRICS_ENABLED=0 overrides everything else in
# src/telemetry/telemetryService.ts; CONTINUE_CLI_ENABLE_TELEMETRY is the documented
# name), and CONTINUE_CLI_AUTO_UPDATED set so the update check never runs `npm i -g`
# over the pinned install (src/services/UpdateService.ts skips auto-update when set).
# https://docs.continue.dev/cli/overview
QUIET = {
    "CONTINUE_CLI_ENABLE_TELEMETRY": "0",
    "CONTINUE_METRICS_ENABLED": "0",
    "CONTINUE_CLI_AUTO_UPDATED": "true",
}
# Secret names the config reads through ${{ secrets.NAME }}; Continue resolves them from
# the process environment first. https://docs.continue.dev/cli/configuration
BASE_SECRET, KEY_SECRET = "MCP_MIRROR_STUB_BASE", "MCP_MIRROR_STUB_KEY"
MODEL = "gpt-4o"

# Continue offers MCP tools under their bare MCP name, next to these built-ins
# (src/tools/builtInToolNames.ts in 1.5.47).
BUILT_IN = {
    "AskQuestion", "Edit", "Exit", "Fetch", "List", "MultiEdit", "Read", "ReportFailure", "Bash",
    "Search", "Status", "Subagent", "Skills", "UploadArtifact", "Diff", "Checklist", "Write",
    "CheckBackgroundJob",
}


def _binary() -> str:
    found = resolve_binary(BINARY)
    if found is None:
        raise HarnessError(f"{BINARY} is not installed; run {' '.join(INSTALL)}")
    return str(found)


def install_check() -> str:
    return version_of([_binary(), "--version"])


def _fixture_names() -> list[tuple[str, re.Pattern]]:
    """(server, pattern) for every identifier-like string literal in each fixture's source.

    Fixtures declare tools as name="...", as dict keys, or as f"page_filler_{index:02d}";
    an f-string's {...} matches any text. Broader than the tool list, never narrower, so a
    fixture tool is always attributed and only a name no fixture mentions is foreign.
    """
    import yaml

    plan = yaml.safe_load((ROOT / "plan" / "harness_sessions.yaml").read_text())
    names = []
    for server, path in plan["servers"].items():
        for literal in re.findall(r'(?<![A-Za-z0-9_"])f?"([A-Za-z_][A-Za-z0-9_]*(?:\{[^}"]*\}[A-Za-z0-9_]*)*)"', (ROOT / path).read_text()):
            pattern = "".join(".+" if part.startswith("{") else re.escape(part)
                              for part in re.split(r"(\{[^}]*\})", literal) if part)
            names.append((server, re.compile(pattern)))
    return names


_FIXTURE_NAMES: list[tuple[str, re.Pattern]] | None = None


def mcp_server_of(tool_name: str) -> str | None:
    """Continue does not prefix MCP tools with their server, so a bare name is matched
    against the fixtures' tool names. A name that is neither a built-in nor a fixture
    tool is reported as foreign, which fails tools_offered."""
    global _FIXTURE_NAMES
    if tool_name in BUILT_IN:
        return None
    if _FIXTURE_NAMES is None:
        _FIXTURE_NAMES = _fixture_names()
    return next((server for server, pattern in _FIXTURE_NAMES if pattern.fullmatch(tool_name)),
                f"not-a-fixture:{tool_name}")


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """A local config.yaml read through --config: one stub model and the five servers.

    mcpServers take `command`/`args` over stdio and `type: streamable-http` with `url`
    over HTTP. https://docs.continue.dev/reference#mcpservers
    """
    entries = []
    for name, server in servers.items():
        if "url" in server:
            entries.append({"name": name, "type": "streamable-http", "url": server["url"]})
        else:
            entries.append({"name": name, "command": server["command"], "args": server["args"]})
    config = {
        "name": "mcp-mirror",
        "version": "0.0.1",
        "schema": "v1",
        "models": [{
            "name": "stub", "provider": "openai", "model": MODEL, "roles": ["chat"],
            "apiBase": "${{ secrets.%s }}" % BASE_SECRET, "apiKey": "${{ secrets.%s }}" % KEY_SECRET,
        }],
        "mcpServers": entries,
    }
    path = workdir / "config.yaml"
    path.write_text(json.dumps(config, indent=2))  # JSON is YAML
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """cn has no `mcp list` command (`cn --help`, 1.5.47); the servers are only listed in
    the TUI's /mcp. The config is the one --config names, and tools_offered checks what
    reached the model."""
    return None


def mode_args(mode: str | None) -> list[str]:
    if mode == "default":
        # Headless defaults: `ask` tools are excluded, and MCP tools are not `ask` in
        # headless (src/permissions/defaultPolicies.ts). No flag.
        # https://docs.continue.dev/cli/tool-permissions
        return []
    if mode == "unattended":
        # "--auto: Allows everything — *: allow". https://docs.continue.dev/cli/tool-permissions
        return ["--auto"]
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # -p runs one task and exits; --format json is the machine-readable output; --config
    # with a file path takes precedence over any saved config or hub assistant.
    # cn has no turn limit flag. https://docs.continue.dev/cli/headless-mode
    return [_binary(), "-p", prompt, "--format", "json", "--config", str(config), *mode_args(mode)]


def env(stub_url: str) -> dict[str, str]:
    return {BASE_SECRET: f"{stub_url}/v1/", KEY_SECRET: "mcp-mirror-stub"} | QUIET


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """None: cn 1.5.47 lists MCP prompts (the TUI's /mcp shows them) but never calls
    prompts/get, and slash commands are handled only in the TUI, so -p sends a
    "/summarize_records" start text to the model as is (src/slashCommands.ts,
    src/ui/hooks/useChat.ts)."""
    return None


def reduce_output(text: str) -> list[dict]:
    """The one JSON object --format json prints; keep its reply."""
    events: list[dict] = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        is_error = item.get("status") == "error"
        reply = item.get("response", item.get("message", item))
        events.append({"event": "result", "subtype": item.get("status"), "is_error": is_error,
                       "num_turns": None, "text": clip(reply)})
    return events
