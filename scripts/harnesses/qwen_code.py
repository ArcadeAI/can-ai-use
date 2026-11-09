"""Qwen Code: `qwen -p` with stream-json output. https://qwenlm.github.io/qwen-code-docs/en/users/features/headless/

Every flag, setting and variable below is from the docs at qwenlm.github.io/qwen-code-docs
(headless, MCP and settings pages) and from `qwen --help` of the pinned package.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from harnesses import ROOT, HarnessError, clip, stub_env, version_of

# Project-local and pinned, never global.
INSTALL = ["npm", "install", "--prefix", ".mcp-mirror/harnesses/qwen_code", "@qwen-code/qwen-code@0.25.1"]
BINARY = ".mcp-mirror/harnesses/qwen_code/node_modules/.bin/qwen"
# The harness runs in a temp working directory, so it is started by absolute path; the
# runner publishes that path relative to the repository.
EXECUTABLE = str(ROOT / BINARY)
# Qwen Code's `openai` auth type speaks Chat Completions to OPENAI_BASE_URL with
# OPENAI_API_KEY and OPENAI_MODEL ("CI / Headless" in its own auth help).
STUB_KIND = "openai"
# Other provider keys Qwen Code reads on its own (DashScope, OpenRouter, its daemon
# token). The shared list already drops ANTHROPIC_*, OPENAI_*, GEMINI_* and GOOGLE_API_KEY.
SCRUBBED_ENV = ("DASHSCOPE_", "OPENROUTER_", "REQUESTY_")
# QWEN_HOME replaces ~/.qwen as the user settings dir (settings docs, "Settings File
# Locations"), so it gets its own dir inside the isolated home.
CONFIG_DIR_ENV = ("QWEN_HOME",)
# A placeholder model name: the stub answers any name. The docs make model.name required.
MODEL = "qwen3-coder-plus"


def install_check() -> str:
    return version_of([EXECUTABLE, "--version"])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """User settings, $QWEN_HOME/settings.json: mcpServers plus the switches that keep the
    run on the stub and off the network. https://qwenlm.github.io/qwen-code-docs/en/users/configuration/settings/"""
    path = Path(env["QWEN_HOME"]) / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, server in servers.items():
        if "url" in server:
            # `url` is SSE in Qwen Code; `httpUrl` is Streamable HTTP (MCP docs, settings table).
            entries[name] = {"httpUrl": server["url"]}
        else:
            entries[name] = {"command": server["command"], "args": server["args"]}
    settings = {
        "mcpServers": entries,
        "security": {"auth": {"selectedType": "openai"}},
        "model": {"name": MODEL},
        # Usage statistics, the update check and telemetry leave the machine on their own;
        # these are their documented off switches (settings docs, privacy and general).
        "privacy": {"usageStatisticsEnabled": False},
        "general": {"enableAutoUpdate": False},
        "telemetry": {"enabled": False},
    }
    path.write_text(json.dumps(settings, indent=2))
    return path


LISTED = re.compile(r"^\s*[✓✗…○●]?\s*([A-Za-z0-9_.-]+): \S")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """`qwen mcp list`, which prints `✓ <name>: <command> (stdio) - Connected` lines, with
    the mark coloured even when stdout is not a terminal."""
    try:
        done = subprocess.run([EXECUTABLE, "mcp", "list"], cwd=workdir, env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HarnessError(f"qwen mcp list did not run: {exc}") from exc
    return {m.group(1) for line in ANSI.sub("", done.stdout + done.stderr).splitlines() if (m := LISTED.match(line))}


def mode_args(mode: str | None) -> list[str]:
    if mode == "default":
        # With no flag, 0.25.1 starts in `auto` (init reports permission_mode "auto"): an LLM
        # classifier decides each call, and the stub cannot answer its respond_in_schema
        # request, so it fails closed and every call is declined. Per the exception to
        # "default means shipped default" in plan/harness_sessions.yaml, the default mode pins the
        # documented built-in `default` mode instead: "Require approval for file edits or
        # shell commands" (`qwen --help`, --approval-mode).
        return ["--approval-mode", "default"]
    if mode == "unattended":
        return ["--approval-mode", "yolo"]
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # The config is found through QWEN_HOME, not a flag. --max-session-turns is the
    # documented turn limit.
    return [EXECUTABLE, "-p", prompt, "--output-format", "stream-json", "--max-session-turns", str(max_turns),
            *mode_args(mode)]


# Since 0.25.0 the CLI refreshes its models.dev catalogue (GET /api.json into
# $QWEN_HOME/model-registry.json) in the background on every start. The settings docs
# ("Environment variables", qwenlm.github.io/qwen-code-docs/en/users/configuration/settings/)
# name QWEN_CODE_MODELS_DEV_REFRESH=off as the switch that stops that download "(air-gapped
# or proxy-restricted environments)" and keeps the bundled catalogue; it has to come from the
# shell environment, settings.json cannot set it. It is not MCP traffic.
QUIET = {"QWEN_CODE_MODELS_DEV_REFRESH": "off"}


def env(stub_url: str) -> dict[str, str]:
    return stub_env(STUB_KIND, stub_url) | {"OPENAI_MODEL": MODEL} | QUIET


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """MCP prompts are slash commands named after the prompt, with --key="value" arguments.
    https://qwenlm.github.io/qwen-code-docs/en/users/features/mcp/"""
    return " ".join([f"/{name}", *(f'--{k}="{v}"' for k, v in arguments.items())])


def reduce_output(text: str) -> list[dict]:
    """stream-json events: system, assistant, user, result (the headless docs)."""
    events: list[dict] = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = item.get("type")
        if kind == "system" and item.get("subtype") in ("init", "session_start"):
            events.append({
                "event": "init",
                "harness_version": item.get("qwen_code_version"),
                "permission_mode": item.get("permission_mode") or item.get("permissionMode"),
                "mcp_servers": item.get("mcp_servers"),
                "mcp_tools": sorted(t for t in item.get("tools") or [] if isinstance(t, str) and t.startswith("mcp__")),
            })
        elif kind in ("assistant", "user"):
            for block in (item.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    events.append({"event": "tool_use", "tool": block.get("name"), "input": block.get("input")})
                elif block.get("type") == "tool_result":
                    events.append({"event": "tool_result", "is_error": bool(block.get("is_error")),
                                   "text": clip(block.get("content"))})
                elif block.get("type") == "text" and kind == "assistant":
                    events.append({"event": "text", "text": clip(block.get("text", ""))})
        elif kind == "result":
            for denial in item.get("permission_denials") or []:
                events.append({"event": "permission_denied", "tool": denial.get("tool_name"),
                               "input": denial.get("tool_input")})
            events.append({"event": "result", "subtype": item.get("subtype"), "is_error": item.get("is_error"),
                           "num_turns": item.get("num_turns"), "text": clip(item.get("result") or "")})
    return events
