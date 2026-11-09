"""OpenHands CLI: `openhands --headless --json --override-with-envs -t <prompt>`. https://docs.openhands.dev/openhands/usage/cli/headless

The MCP servers go in ~/.openhands/mcp.json (inside the isolated home), the file
`openhands mcp add` writes and `openhands mcp list` reads.
https://docs.openhands.dev/openhands/usage/cli/mcp-servers

OpenHands mounts the servers through one fastmcp client, which offers an MCP tool to the
model as `<server>_<tool>` when more than one server is configured (`def_search_records`),
and under its bare name when only one is; next to its own tools (terminal, file_editor,
task_tracker, task, finish, think). Every tool without `readOnlyHint: true` gets a
required `security_risk` argument added to its schema (openhands-sdk tool/tool.py
`_get_tool_schema`), and a call without one is refused before it reaches the server
("Failed to provide security_risk field"). Under mcp 2.x that is every MCP tool, since
the hint does not survive into the SDK (openhands/sdk/mcp/tool.py
`MCPToolDefinition.create` reads annotations back as defaults). So the session scripts
in plan/harness_sessions.yaml pass one wherever the harness asked for it.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from harnesses import ROOT, HarnessError, clip, list_command, resolve_binary, stub_env, version_of

# `pip install openhands` (the inventory's install), pinned and project-local: a uv venv
# under .mcp-mirror/harnesses/openhands. --exclude-newer pins the dependency resolution to
# the measurement date, so the SDK, fastmcp and mcp versions under test do not float.
# The CLI needs Python 3.12 (`Requires-Python ==3.12.*`).
# https://docs.openhands.dev/openhands/usage/cli/installation
_DIR = ".mcp-mirror/harnesses/openhands"
VERSION = "1.16.0"
INSTALL = ["sh", "-c",
           f"uv venv --quiet --python 3.12 {_DIR}/venv && uv pip install --quiet --python {_DIR}/venv/bin/python"
           f" --exclude-newer 2026-10-01 openhands=={VERSION}"]
BINARY = f"{_DIR}/venv/bin/openhands"
# The CLI's model is a LiteLLM model string; `openai/<name>` with a base URL is Chat
# Completions at <base>/chat/completions.
STUB_KIND = "openai"
# LLM_* and OPENHANDS_* could hand the run a real model, a persistence dir or OpenHands
# Cloud from the operator's shell; LITELLM_*, LMNR_* and EXTENSIONS_REF change what the
# SDK loads or reports.
SCRUBBED_ENV = ("LLM_", "OPENHANDS_", "PERSISTENCE_DIR", "LITELLM_", "LMNR_", "EXTENSIONS_REF")
# OpenHands has no switch for these two (openhands_cli 1.16.0, openhands-sdk 1.21.0). The
# sink refuses both and the run goes on without them.
EGRESS_ALLOWED = {
    "github.com": ("the agent context always loads public skills (load_public_skills=True in "
                   "openhands_cli/stores/agent_store.py), a git clone of github.com/OpenHands/extensions "
                   "into ~/.openhands/cache; no setting turns it off. Refused, so no skill is loaded"),
    "pypi.org": ("the splash screen's update check (openhands_cli/version_check.py, "
                 "https://pypi.org/pypi/openhands/json), run in headless mode too; no setting turns it off"),
}
# LiteLLM fetches its model cost map from raw.githubusercontent.com at import unless told
# to use the copy it ships (litellm/litellm_core_utils/get_model_cost_map.py).
QUIET = {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "OPENHANDS_SUPPRESS_BANNER": "1"}

# The CLI's own tools (openhands_cli/utils.py get_default_cli_tools, and the agent's
# built-ins finish and think).
OWN_TOOLS = {"terminal", "file_editor", "task_tracker", "task", "delegate", "finish", "think", "browser_use"}


def _binary() -> str:
    found = resolve_binary(BINARY)
    if found is None:
        raise HarnessError(f"{BINARY} does not exist (run INSTALL)")
    return str(found)


def install_check() -> str:
    # Identity, not only a version: the CLI must report itself and have --headless --json.
    version = version_of([_binary(), "--version"])
    help_text = subprocess.run([_binary(), "--help"], capture_output=True, text=True, timeout=60,
                               env={"OPENHANDS_SUPPRESS_BANNER": "1", "PATH": "/usr/bin:/bin"}).stdout
    if "--headless" not in help_text or "--json" not in help_text:
        raise HarnessError(f"{BINARY} is not the OpenHands CLI (no --headless --json)")
    if version != VERSION:
        raise HarnessError(f"{BINARY} is {version}, INSTALL pins {VERSION}")
    return version


def _config_path(env: dict[str, str]) -> Path:
    """~/.openhands/mcp.json (openhands_cli/locations.py: persistence dir, MCP_CONFIG_FILE)."""
    return Path(env["HOME"]) / ".openhands" / "mcp.json"


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """fastmcp MCPConfig: stdio as command/args, HTTP as url with transport http, which is
    what `openhands mcp add --transport http` writes (openhands_cli/mcp/mcp_utils.py)."""
    entries = {}
    for name, server in servers.items():
        if "url" in server:
            entries[name] = {"url": server["url"], "transport": "http", "enabled": True}
        else:
            entries[name] = {"command": server["command"], "args": server["args"], "enabled": True}
    path = _config_path(env)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": entries}, indent=2))
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """`openhands mcp list`: one `  • <name> [✓ enabled]` line per configured server."""
    text = list_command([_binary(), "mcp", "list"], env | QUIET, workdir)
    return {m.group(1) for line in text.splitlines() if (m := re.match(r"^\s*•\s+(\S+)\s+\[✓ enabled\]", line))}


def _fixture_names() -> list[tuple[str, re.Pattern]]:
    """(server, pattern) for every identifier-like string literal in each fixture's source.

    Broader than the tool list, never narrower, so a fixture tool is always attributed and
    only a name no fixture mentions is foreign. An f-string's {...} matches any text.
    """
    import yaml

    plan = yaml.safe_load((ROOT / "plan" / "harness_sessions.yaml").read_text())
    names = []
    for server, path in plan["servers"].items():
        for literal in re.findall(r'(?<![A-Za-z0-9_"])f?"([A-Za-z_][A-Za-z0-9_]*(?:\{[^}"]*\}[A-Za-z0-9_]*)*)"',
                                  (ROOT / path).read_text()):
            pattern = "".join(".+" if part.startswith("{") else re.escape(part)
                              for part in re.split(r"(\{[^}]*\})", literal) if part)
            names.append((server, re.compile(pattern)))
    return names


_FIXTURE_NAMES: list[tuple[str, re.Pattern]] | None = None


def mcp_server_of(tool_name: str) -> str | None:
    """fastmcp prefixes a tool with its server and `_` when several are mounted, so a name
    is attributed to the server it is prefixed with when the rest is one of that fixture's
    names; a bare name (one server mounted) is matched against every fixture. A name that
    is neither OpenHands' own nor a fixture tool is reported as foreign, which fails
    tools_offered."""
    global _FIXTURE_NAMES
    if tool_name in OWN_TOOLS:
        return None
    if _FIXTURE_NAMES is None:
        _FIXTURE_NAMES = _fixture_names()
    for server, pattern in _FIXTURE_NAMES:
        if tool_name.startswith(f"{server}_") and pattern.fullmatch(tool_name[len(server) + 1:]):
            return server
    return next((server for server, pattern in _FIXTURE_NAMES if pattern.fullmatch(tool_name)),
                f"not-a-fixture:{tool_name}")


def mode_args(mode: str | None) -> list[str]:
    """Headless mode always runs always-approve, and that cannot be changed: "Headless mode
    always runs in `always-approve` mode", and --llm-approve is not available there.
    https://docs.openhands.dev/openhands/usage/cli/headless
    So "default" sets nothing and "unattended" passes the documented --always-approve; the
    two are the same state."""
    if mode == "unattended":
        return ["--always-approve"]
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # --override-with-envs: take the model from LLM_MODEL / LLM_BASE_URL / LLM_API_KEY
    # instead of a saved ~/.openhands/agent_settings.json (there is none in a fresh home).
    # OpenHands has no turn-limit flag. The MCP config is the file write_config wrote.
    return [_binary(), "--headless", "--json", "--override-with-envs", "-t", prompt, *mode_args(mode)]


def env(stub_url: str) -> dict[str, str]:
    return {
        "LLM_MODEL": "openai/stub",
        "LLM_BASE_URL": f"{stub_url}/v1",
        "LLM_API_KEY": stub_env(STUB_KIND, stub_url)["OPENAI_API_KEY"],
    } | QUIET


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """None: neither the CLI nor the SDK has a way to expand an MCP prompt (no prompts/get
    anywhere in openhands_cli 1.16.0 or openhands-sdk 1.21.0)."""
    return None


def _arguments(call: dict) -> object:
    raw = call.get("arguments")
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return raw


def _content_text(content: object) -> str:
    if isinstance(content, list):
        return " ".join(str(c.get("text")) for c in content if isinstance(c, dict) and c.get("type") == "text")
    return "" if content is None else str(content)


def reduce_output(text: str) -> list[dict]:
    """--json streams one event per line (MessageEvent, ActionEvent, ObservationEvent,
    AgentErrorEvent, UserRejectObservation), between plain status lines. There is no init
    or closing event, so the result is synthesized from the last agent message."""
    events: list[dict] = []
    turns = 0
    finished = False
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            if line.strip() == "Agent finished":
                finished = True
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind == "ActionEvent":
            turns += 1
            call = item.get("tool_call") or {}
            events.append({"event": "tool_use", "tool": item.get("tool_name") or call.get("name"),
                           "input": _arguments(call)})
        elif kind == "ObservationEvent":
            observation = item.get("observation") or {}
            events.append({"event": "tool_result", "is_error": bool(observation.get("is_error")),
                           "text": clip(_content_text(observation.get("content")))})
        elif kind == "AgentErrorEvent":
            events.append({"event": "tool_result", "is_error": True, "text": clip(item.get("error") or "")})
        elif kind == "UserRejectObservation":
            events.append({"event": "permission_denied", "tool": item.get("tool_name"),
                           "input": None})
        elif kind == "MessageEvent" and item.get("source") == "agent":
            turns += 1
            message = item.get("llm_message") or {}
            events.append({"event": "text", "text": clip(_content_text(message.get("content")))})
    last = next((e["text"] for e in reversed(events) if e["event"] == "text"), "")
    events.append({"event": "result", "subtype": "success" if finished else "incomplete", "is_error": not finished,
                   "num_turns": turns, "text": last})
    return events
