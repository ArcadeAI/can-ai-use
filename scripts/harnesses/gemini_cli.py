"""Gemini CLI: `gemini -p` with stream-json output. https://geminicli.com/docs/cli/headless

Every flag and variable below is from the docs shipped in the pinned package
(bundle/docs/), which match geminicli.com/docs for 0.63.0.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from harnesses import ROOT, HarnessError, clip, stub_env, version_of

# Project-local and pinned, never global.
INSTALL = ["npm", "install", "--prefix", ".mcp-mirror/harnesses/gemini_cli", "@google/gemini-cli@0.63.0"]
BINARY = ".mcp-mirror/harnesses/gemini_cli/node_modules/.bin/gemini"
# The harness runs in a temp working directory, so it is started by absolute path; the
# runner publishes that path relative to the repository.
EXECUTABLE = str(ROOT / BINARY)
STUB_KIND = "gemini"
# GOOGLE_CLOUD_* / GOOGLE_APPLICATION_CREDENTIALS would switch auth to Vertex or Code
# Assist with the operator's account. The shared list already drops GEMINI_* and
# GOOGLE_API_KEY. https://geminicli.com/docs/reference/configuration#environment-variables-and-env-files
SCRUBBED_ENV = ("GOOGLE_CLOUD_", "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_GENAI_")
# stdio MCP servers connect only in a trusted folder, in `gemini mcp list` as in a run.
# This trusts the (empty, temp) working directory for the session, the documented switch
# for headless use; it is not an approval setting. https://geminicli.com/docs/cli/trusted-folders
TRUST = {"GEMINI_CLI_TRUST_WORKSPACE": "true"}
# Pinned so the default `auto` model router does not first ask a classifier model for a
# JSON routing decision, which the stub answers with text, so the router retries for
# about a minute before it falls back. A model-backed step the stub cannot answer is
# replaced by the documented built-in choice (the exception to "default means shipped
# default" in plan/harness_sessions.yaml). https://geminicli.com/docs/cli/model
MODEL = "gemini-2.5-flash"
# With NO_PROXY/no_proxy set, createUrlTransport (the same code in 0.62.0 and 0.63.0)
# sends every Streamable HTTP fetch through a bundled undici EnvHttpProxyAgent, and the
# handshake stops after initialize. The runner's egress sink sets NO_PROXY, so
# on HTTP the proxy variables are withheld and egress is held by the loopback sandbox
# instead (measure_harness.py WITHHOLD_PROXY_ENV).
WITHHOLD_PROXY_ENV = {
    "http": "Gemini CLI 0.63.0 stops the Streamable HTTP handshake after initialize whenever NO_PROXY is set",
}


def install_check() -> str:
    return version_of([EXECUTABLE, "--version"])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """User settings, $GEMINI_CLI_HOME/.gemini/settings.json: mcpServers plus the switches
    that keep the run on the stub and off the network. https://geminicli.com/docs/tools/mcp-server"""
    path = Path(env["GEMINI_CLI_HOME"]) / ".gemini" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, server in servers.items():
        if "url" in server:
            # Streamable HTTP. `httpUrl` means the same but is deprecated in favour of this.
            entries[name] = {"url": server["url"], "type": "http"}
        else:
            entries[name] = {"command": server["command"], "args": server["args"]}
    settings = {
        "mcpServers": entries,
        # GEMINI_API_KEY auth, which is the one GOOGLE_GEMINI_BASE_URL applies to.
        "security": {"auth": {"selectedType": "gemini-api-key"}},
        # Usage statistics (Clearcut), the update check and telemetry all leave the machine
        # on their own; these are their documented off switches.
        # https://geminicli.com/docs/reference/configuration#privacy
        "privacy": {"usageStatisticsEnabled": False},
        "general": {"enableAutoUpdate": False, "enableAutoUpdateNotification": False},
        "telemetry": {"enabled": False},
    }
    path.write_text(json.dumps(settings, indent=2))
    return path


LISTED = re.compile(r"^\s*[✓✗…○●]?\s*([A-Za-z0-9_.-]+): \S")


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """`gemini mcp list`, which prints `✓ <name>: <command> (stdio) - Connected` lines to
    stderr. https://geminicli.com/docs/tools/mcp-server#listing-servers-gemini-mcp-list"""
    try:
        done = subprocess.run([EXECUTABLE, "mcp", "list"], cwd=workdir, env=env | TRUST, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HarnessError(f"gemini mcp list did not run: {exc}") from exc
    return {m.group(1) for line in (done.stdout + done.stderr).splitlines() if (m := LISTED.match(line))}


def mode_args(mode: str | None) -> list[str]:
    if mode == "default":
        # The shipped default, so no flag: --approval-mode default.
        # Headless, a tool that would ask is denied, and a tool denied without an argument
        # pattern is not offered to the model at all. https://geminicli.com/docs/reference/policy-engine
        return []
    if mode == "unattended":
        return ["--approval-mode", "yolo"]
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # The config is found through GEMINI_CLI_HOME, not a flag. There is no turn-limit
    # flag (model.maxSessionTurns counts kept history, not turns); the stub ends the turn.
    return [EXECUTABLE, "-p", prompt, "--output-format", "stream-json", "--model", MODEL, *mode_args(mode)]


def env(stub_url: str) -> dict[str, str]:
    # GOOGLE_GEMINI_BASE_URL may be http only on 127.0.0.1 / localhost, which the stub is.
    return stub_env(STUB_KIND, stub_url) | TRUST


def mcp_server_of(tool_name: str) -> str | None:
    """MCP tools are offered as mcp_{server}_{tool}. https://geminicli.com/docs/tools/mcp-server"""
    found = re.match(r"^mcp_([^_]+)_.", tool_name)
    return found.group(1) if found else None


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """MCP prompts are slash commands named after the prompt, with --key="value" arguments.
    https://geminicli.com/docs/tools/mcp-server#mcp-prompts-as-slash-commands"""
    return " ".join([f"/{name}", *(f'--{k}="{v}"' for k, v in arguments.items())])


def reduce_output(text: str) -> list[dict]:
    """stream-json events: init, message, tool_use, tool_result, error, result."""
    events: list[dict] = []
    reply: list[str] = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = item.get("type")
        if kind == "init":
            events.append({"event": "init", "harness_version": None, "permission_mode": None, "model": item.get("model"),
                           "mcp_servers": None, "mcp_tools": []})
        elif kind == "message" and item.get("role") == "assistant":
            reply.append(item.get("content") or "")
        elif kind == "tool_use":
            if reply:
                events.append({"event": "text", "text": clip("".join(reply))})
                reply = []
            events.append({"event": "tool_use", "tool": item.get("tool_name"), "input": item.get("parameters")})
        elif kind == "tool_result":
            error = item.get("error") or {}
            events.append({"event": "tool_result", "is_error": item.get("status") != "success",
                           "text": clip(item.get("output") if item.get("output") is not None else error.get("message") or "")})
        elif kind == "error":
            events.append({"event": "text", "text": clip(f"[{item.get('severity') or 'error'}] {item.get('message') or ''}")})
        elif kind == "result":
            if reply:
                events.append({"event": "text", "text": clip("".join(reply))})
                reply = []
            stats = item.get("stats") or {}
            error = item.get("error") or {}
            events.append({"event": "result", "subtype": item.get("status"), "is_error": item.get("status") != "success",
                           "num_turns": None, "tool_calls": stats.get("tool_calls"),
                           "text": clip(error.get("message") or "")})
    if reply:
        events.append({"event": "text", "text": clip("".join(reply))})
    return events
