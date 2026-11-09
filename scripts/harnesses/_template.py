"""<Harness name>: <one-line headless invocation>. <link to the harness's headless docs>

A starting point for a new harness. Copy this file to scripts/harnesses/<id>.py, where
<id> is the harness's id in plan/inventory.yaml, fill in every TODO, and add a
`<id>:` block under `harnesses:` in plan/harness_sessions.yaml (it may be `{}`). Then:

    uv run python scripts/measure_harness.py <id> --level 1   # startup only, fast
    uv run python scripts/measure_harness.py <id>             # both levels, both SDK lines

Every value here must come from the harness's own documentation, and each non-obvious
flag gets a comment that says why. Never put a key here: the stub accepts any key, and
a real one must not reach the harness (see SCRUBBED_ENV). Files starting with _ are
not loaded as harnesses. scripts/harnesses/__init__.py lists the interface in one screen.
"""

from __future__ import annotations

import json
from pathlib import Path

from harnesses import clip, list_command, names_before_colon, stub_env, version_of

# The executable the inventory's `install` command puts on PATH. If it is missing and
# that command is an `npm i -g` / `brew install` / `pip install` / `uv tool install`,
# measure_harness.py runs it first.
# Optional. Project-local install run from the repository root when BINARY is missing.
# Never a global install. Pin the version; BINARY then names the path it produces.
# INSTALL = ["npm", "install", "--prefix", ".mcp-mirror/harnesses/<id>", "<package>@<version>"]
# BINARY = ".mcp-mirror/harnesses/<id>/node_modules/.bin/<binary>"

BINARY = "TODO"

# Which model API the harness calls: "anthropic" (Messages), "openai" (Responses or Chat
# Completions) or "gemini" (generateContent). The stub answers it, so no level needs a
# credential. None only if the harness cannot be pointed at another base URL; then
# neither level runs and every cell is u with that reason.
STUB_KIND: str | None = "openai"

# Optional: extra environment-variable prefixes that would hand the harness a real
# account (its own API key or login token). The shared list already drops ANTHROPIC_*,
# CLAUDE_CODE_*, OPENAI_*, GEMINI_* and GOOGLE_API_KEY.
SCRUBBED_ENV: tuple[str, ...] = ()

# Every session runs in a fresh temp HOME (HOME, XDG_*, CODEX_HOME, CLAUDE_CONFIG_DIR and
# GEMINI_CLI_HOME point inside it), with proxies pointed at a sink that refuses and logs.
# Optional: other variables the harness reads a config dir from; each gets its own dir
# inside the isolated home.
CONFIG_DIR_ENV: tuple[str, ...] = ()
# Optional: hosts the harness contacts on its own (a version check, say), each with why.
# The sink still refuses them; listing one here only stops it failing the run.
EGRESS_ALLOWED: dict[str, str] = {}
# Nothing from the real home is copied in: the stub needs no login. A file listed in
# HOME_FILES fails freshness unless HOME_FILES_ALLOWED gives the reason. Keep both empty.
HOME_FILES: tuple[str, ...] = ()
HOME_FILES_ALLOWED: dict[str, str] = {}


def install_check() -> str:
    """The installed version, e.g. "1.2.3"; raise harnesses.HarnessError if it cannot run."""
    return version_of([BINARY, "--version"])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """Write the harness's MCP config and return its path; it must be inside the isolated home.

    ``servers`` is {name: {"command": str, "args": [str]}} for stdio sessions and
    {name: {"url": str}} for HTTP sessions, for the five names def, res, ntf, cf, util.
    ``workdir`` is the harness's cwd, an empty dir in the isolated home. ``env`` is the
    isolated environment: a harness that only reads a fixed user config writes it there,
    e.g. Path(env["CODEX_HOME"]) / "config.toml".

    If the documented unattended switch is in the MCP config (a per-tool allow list),
    add keyword-only ``mode=None`` and ``tools=None``: the runner then passes the
    session's mode (None at level 1) and {server: [tool name, ...]} as each fixture's
    tools/list answers it. A module that does not declare them is called without them.
    """
    path = workdir / "mcp.json"  # TODO: the harness's own file name and shape
    entries = {
        name: ({"url": server["url"]} if "url" in server else {"command": server["command"], "args": server["args"]})
        for name, server in servers.items()
    }
    path.write_text(json.dumps({"mcpServers": entries}, indent=2))
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """The server names the harness's own list command reports, run with ``env`` in ``workdir``.

    The runner calls it after write_config and before the session; it must be exactly
    def, res, ntf, cf and util. Return None if the harness has no list command.
    """
    return names_before_colon(list_command([BINARY, "mcp", "list"], env, workdir))  # TODO


def mode_args(mode: str | None) -> list[str]:
    """The documented approval switch for a level-2 mode; [] at level 1 (mode None).

    "default": the harness's out-of-the-box approval behaviour, run headless. It must
    make a held call machine-readable (reduce_output emits permission_denied), or
    h-destructive-blocked-by-default can only be u.
    "unattended": the documented switch that removes prompts (inventory `unattended`).
    These are published in each level-2 cell's evidence.
    """
    if mode == "default":
        return []  # TODO
    if mode == "unattended":
        return ["--TODO-yolo"]  # TODO
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    """The whole headless command for one session. Run from ``workdir``; stdin is closed.

    Include machine-readable output (stream-json / NDJSON) so reduce_output has events,
    a turn limit if the harness has one, and every flag that keeps the operator's own
    settings, plugins and memory out of the run.
    """
    return [BINARY, "TODO", prompt, "--mcp-config", str(config), *mode_args(mode)]


def env(stub_url: str) -> dict[str, str]:
    """Variables that send the harness's model calls to the stub at ``stub_url``."""
    return stub_env(STUB_KIND, stub_url)  # TODO: if the harness reads other variables, name them here


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """Start text that makes the harness expand an MCP prompt, e.g. a slash command.

    Return None if the harness has no user-side way to invoke one; the prompt session
    then starts with the plan's plain-language prompt and the model has to find it.
    """
    return None


def reduce_output(text: str) -> list[dict]:
    """The harness's stdout reduced to the shared event vocabulary (scripts/harnesses/__init__.py).

    Keep only what cells read. Drop paths, usage, the harness's own tool list, and
    anything the operator's account might put there; the result is committed.
    """
    events: list[dict] = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        # TODO: map the harness's own event types, for example:
        if item.get("type") == "tool_call":
            events.append({"event": "tool_use", "tool": item.get("name"), "input": item.get("arguments")})
        elif item.get("type") == "final":
            events.append({"event": "result", "subtype": None, "is_error": False, "num_turns": None,
                           "text": clip(item.get("text") or "")})
    return events
