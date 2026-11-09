"""Goose: `goose run -t <prompt> --output-format stream-json --no-session`. https://goose-docs.ai/docs/guides/running-tasks

The MCP servers ("extensions") go in Goose's own config.yaml, inside the isolated
GOOSE_PATH_ROOT, rather than on --with-extension / --with-streamable-http-extension:
Goose names a --with-streamable-http-extension after its URL (127_0_0_1_<port>_mcp), so
the fixture names def, res, ntf, cf and util survive only through config.yaml's `name`.
`goose info -v` then lists them, which is the freshness check's list command.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import yaml

from harnesses import HarnessError, clip, list_command, resolve_binary, stub_env, version_of

# The documented installer (the inventory's "install script"), pinned to v1.54.0 and pointed
# at a project-local bin dir. CONFIGURE=false skips `goose configure` and the offer to edit
# a shell rc file; stdin is closed so it never waits on a prompt. `brew install goose` is
# not this: that formula is pressly's database migration tool.
# https://goose-docs.ai/docs/getting-started/installation
_DIR = ".mcp-mirror/harnesses/goose"
VERSION = "1.54.0"
INSTALL = ["sh", "-c",
           f"mkdir -p {_DIR} && curl -fsSL -o {_DIR}/download_cli.sh "
           f"https://github.com/aaif-goose/goose/releases/download/v{VERSION}/download_cli.sh"
           f" && GOOSE_BIN_DIR={_DIR}/bin GOOSE_VERSION=v{VERSION} CONFIGURE=false bash {_DIR}/download_cli.sh </dev/null"]
BINARY = f"{_DIR}/bin/goose"
# Goose's openai provider speaks Chat Completions at $OPENAI_HOST/v1/chat/completions.
# https://goose-docs.ai/docs/getting-started/providers
STUB_KIND = "openai"
# GOOSE_* could select a real provider, mode or config root from the operator's shell.
SCRUBBED_ENV = ("GOOSE_",)
# Goose keeps config, sessions and logs under GOOSE_PATH_ROOT when it is set.
# https://goose-docs.ai/docs/guides/environment-variables
CONFIG_DIR_ENV = ("GOOSE_PATH_ROOT",)

# Since 1.53.0 the CLI refreshes its model catalogue from models.dev in a background task
# on every start (PR #12560: goose-cli main.rs calls model_catalog::initialize() before any
# subcommand, and model_catalog.rs spawns refresh_remote_catalog(CATALOG_URL) with no
# guard). The only switch is the compile-time Cargo feature `online-model-meta`, on in the
# released binary; `goose --help`, the environment-variables guide and the config guide
# name nothing that skips the fetch, and the binary carries no GOOSE_* name about it.
# This is not MCP traffic. The bundled catalogue is the fallback: the sink refuses the
# request, Goose logs a warning and the run goes on.
EGRESS_ALLOWED = {
    "models.dev": "the model catalogue refresh (GET /api.json) on every start; no documented off switch, "
                  "and a refusal falls back to the bundled catalogue with a warning",
}

# Goose's own tools are `<extension>__<tool>` too, from its bundled platform extensions
# (on by default, and left on: they are what a fresh Goose offers). None is an MCP server.
_OWN = re.compile(r"^(platform|developer|todo|chatrecall|extensionmanager|skills|subagent|summon|"
                  r"code_execution|apps|computercontroller|memory|tutorial|autovisualiser)__")


def _binary() -> str:
    found = resolve_binary(BINARY)
    if found is None:
        raise HarnessError(f"{BINARY} does not exist (run INSTALL)")
    return str(found)


def install_check() -> str:
    # Identity, not only a version: another `goose` (pressly's migration tool) has no `run`.
    version = version_of([_binary(), "--version"])
    help_text = subprocess.run([_binary(), "run", "--help"], capture_output=True, text=True, timeout=60).stdout
    if "--output-format" not in help_text:
        raise HarnessError(f"{BINARY} is not Block's Goose (no `goose run --output-format`)")
    if version != VERSION:
        raise HarnessError(f"{BINARY} is {version}, INSTALL pins {VERSION}")
    return version


def _config_path(env: dict[str, str]) -> Path:
    return Path(env["GOOSE_PATH_ROOT"]) / "config" / "config.yaml"


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """config.yaml `extensions:` entries, stdio as cmd/args and HTTP as streamable_http.
    https://goose-docs.ai/docs/guides/config-files"""
    extensions = {}
    for name, server in servers.items():
        if "url" in server:
            extensions[name] = {"enabled": True, "type": "streamable_http", "name": name, "uri": server["url"]}
        else:
            extensions[name] = {"enabled": True, "type": "stdio", "name": name, "cmd": server["command"],
                                "args": server["args"]}
    path = _config_path(env)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"extensions": extensions}, sort_keys=False))
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """The enabled non-bundled extensions `goose info -v` reports from config.yaml."""
    text = list_command([_binary(), "info", "-v"], env, workdir)
    _, found, block = text.partition("goose Configuration:\n")
    if not found:
        raise HarnessError("`goose info -v` printed no configuration block")
    try:
        extensions = (yaml.safe_load(block) or {}).get("extensions") or {}
    except yaml.YAMLError as exc:
        raise HarnessError(f"`goose info -v` configuration is not YAML: {exc}") from exc
    return {name for name, ext in extensions.items()
            if isinstance(ext, dict) and ext.get("enabled") and not ext.get("bundled") and ext.get("type") != "platform"}


def mcp_server_of(tool_name: str) -> str | None:
    """Goose names an extension's tool `<extension>__<tool>`."""
    if _OWN.match(tool_name):
        return None
    found = re.match(r"^(.+?)__.", tool_name)
    return found.group(1) if found else None


def mode_args(mode: str | None) -> list[str]:
    """Goose's approval mode is the GOOSE_MODE variable, which defaults to `auto` (every tool
    runs without asking). There is no flag; argv sets it through `env`.
    https://goose-docs.ai/docs/guides/goose-permissions"""
    if mode == "unattended":
        return ["GOOSE_MODE=auto"]
    # "default": nothing set, so Goose's out-of-the-box mode, which is auto.
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # --no-session: no session file, so nothing carries over between runs. The profile
    # Goose loads is the config.yaml above, in the isolated home.
    # https://goose-docs.ai/docs/guides/goose-cli-commands#run-options
    return [
        "env", *mode_args(mode), _binary(), "run", "-t", prompt, "--output-format", "stream-json",
        "--no-session", "--max-turns", str(max_turns),
    ]


def env(stub_url: str) -> dict[str, str]:
    return {
        "GOOSE_PROVIDER": "openai",
        "GOOSE_MODEL": "stub",
        # Goose appends v1/chat/completions itself (OPENAI_BASE_PATH's default).
        "OPENAI_HOST": stub_url,
        "OPENAI_API_KEY": stub_env(STUB_KIND, stub_url)["OPENAI_API_KEY"],
        # Secrets go to a file in the isolated home, not the operator's keychain.
        "GOOSE_DISABLE_KEYRING": "1",
        "GOOSE_TELEMETRY_ENABLED": "false",
    }


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """`/prompt <name> key=value` expands an MCP prompt in a Goose session.
    https://goose-docs.ai/docs/guides/goose-cli-commands#slash-commands"""
    return " ".join(["/prompt", name, *(f"{k}={v}" for k, v in arguments.items())])


def reduce_output(text: str) -> list[dict]:
    """Keep what a cell reads; drop images, usage and the startup banner."""
    events: list[dict] = []
    turns = 0
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "message":
            message = item.get("message") or {}
            if message.get("role") == "assistant":
                turns += 1
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "toolRequest":
                    call = (block.get("toolCall") or {}).get("value") or {}
                    events.append({"event": "tool_use", "tool": call.get("name"), "input": call.get("arguments")})
                elif btype == "toolResponse":
                    result = block.get("toolResult") or {}
                    value = result.get("value") or {}
                    is_error = result.get("status") == "error" or bool(value.get("isError"))
                    body = value.get("content") if isinstance(value, dict) else value
                    if result.get("status") == "error":
                        body = result.get("error")
                    texts = [c.get("text") for c in body or [] if isinstance(c, dict) and c.get("type") == "text"] \
                        if isinstance(body, list) else [body]
                    events.append({"event": "tool_result", "is_error": is_error, "text": clip(" ".join(map(str, texts)))})
                elif btype == "toolConfirmationRequest":
                    events.append({"event": "permission_denied", "tool": block.get("toolName"),
                                   "input": block.get("arguments")})
                elif btype == "text" and message.get("role") == "assistant":
                    events.append({"event": "text", "text": clip(block.get("text", ""))})
        elif kind == "error":
            events.append({"event": "result", "subtype": "error", "is_error": True, "num_turns": turns,
                           "text": clip(item.get("error") or "")})
        elif kind == "complete":
            events.append({"event": "result", "subtype": "success", "is_error": False, "num_turns": turns,
                           "text": next((e["text"] for e in reversed(events) if e["event"] == "text"), "")})
    return events
