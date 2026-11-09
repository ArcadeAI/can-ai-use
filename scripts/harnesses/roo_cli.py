"""Roo Code CLI: `roo -p "<prompt>" --output-format stream-json --oneshot`. https://github.com/RooCodeInc/Roo-Code/tree/main/apps/cli

Every flag below is from `roo --help` and apps/cli/README.md at the cli-v0.1.17 tag. The
CLI runs the Roo Code VS Code extension bundle under a VS Code API shim, so MCP config,
tool names and approvals are the extension's (https://docs.roocode.com/features/mcp/using-mcp-in-roo).
"""

from __future__ import annotations

import json
from pathlib import Path

from harnesses import HarnessError, clip, resolve_binary, stub_env, version_of

# The installer (apps/cli/install.sh) downloads the release tarball, extracts it with
# --strip-components=1, runs `npm install --production` in it and keeps the bundled
# ripgrep, which npm install would replace. This is those steps into a project-local
# prefix, pinned to the release and its published sha256, without the installer's
# symlink into ~/.local/bin or its `roo --version` in the real home. macOS arm64 only:
# the release has no darwin-x64 build.
VERSION = "0.1.17"
TARBALL_SHA256 = "cf3c1612cd6cd1e50d3db005548a071a634f17a62ca8e8b1ead5a6e832763537"
PREFIX = f".mcp-mirror/harnesses/roo_cli/{VERSION}"
INSTALL = ["sh", "-c", (
    "set -eu; d=$(mktemp -d); trap 'rm -rf \"$d\"' EXIT; "
    f"curl -fsSL -o \"$d/roo.tgz\" https://github.com/RooCodeInc/Roo-Code/releases/download/cli-v{VERSION}/roo-cli-darwin-arm64.tar.gz; "
    f"echo \"{TARBALL_SHA256}  $d/roo.tgz\" | shasum -a 256 -c -; "
    f"mkdir -p {PREFIX}; tar -xzf \"$d/roo.tgz\" -C {PREFIX} --strip-components=1; "
    f"cp {PREFIX}/node_modules/@vscode/ripgrep/bin/rg \"$d/rg\"; "
    f"npm install --prefix {PREFIX} --omit=dev --silent; "
    f"mkdir -p {PREFIX}/node_modules/@vscode/ripgrep/bin; cp \"$d/rg\" {PREFIX}/node_modules/@vscode/ripgrep/bin/rg"
)]
BINARY = f"{PREFIX}/bin/roo"

# `--provider anthropic`. The CLI has no base-URL flag or setting; its Anthropic handler
# builds the client with `baseURL: anthropicBaseUrl || undefined`, so the Anthropic SDK's
# own ANTHROPIC_BASE_URL applies (src/api/providers/anthropic.ts).
STUB_KIND = "anthropic"
PROVIDER = "anthropic"
MODEL = "claude-sonnet-4-5"
# Roo Code Cloud's own account and the other providers' keys the CLI reads
# (README, "Environment Variables").
SCRUBBED_ENV: tuple[str, ...] = ("ROO_", "OPENROUTER_", "VERCEL_AI_GATEWAY_")
REFRESH = ("the public model-list fetch the extension starts 2 s after activation "
           "(initializeModelCacheRefresh, src/api/providers/fetchers/modelCache.ts) for openrouter and "
           "vercel-ai-gateway whatever the provider; no setting turns it off, and a refusal falls back to "
           "the empty cache")
EGRESS_ALLOWED = {
    "openrouter.ai": REFRESH,
    "ai-gateway.vercel.sh": REFRESH,
}


def _binary() -> str:
    found = resolve_binary(BINARY)
    if found is None:
        raise HarnessError(f"{BINARY} is not installed")
    return str(found)


def install_check() -> str:
    return version_of([_binary(), "--version"])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str], *, mode: str | None = None,
                 tools: dict[str, list[str]] | None = None) -> Path:
    """The project config `.roo/mcp.json` in the workspace (the cwd): "Project-level
    configuration ... .roo/mcp.json" (docs, "Editing MCP Settings Files"). HTTP servers are
    type streamable-http.

    Unattended lists every fixture tool in its server's `alwaysAllow` ("An array of tool
    names from this server to automatically approve"). Roo approves an MCP tool call only
    when the tool is listed there (isMcpToolAlwaysAllowed); the default writes none, which
    is Roo's out-of-box config."""
    entries = {}
    for name, server in servers.items():
        if "url" in server:
            entries[name] = {"type": "streamable-http", "url": server["url"]}
        else:
            entries[name] = {"command": server["command"], "args": server["args"]}
        if mode == "unattended":
            entries[name]["alwaysAllow"] = list((tools or {}).get(name) or [])
    path = workdir / ".roo" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": entries}, indent=2))
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """`roo list` lists commands, modes, models and sessions; nothing lists MCP servers."""
    return None


def mode_args(mode: str | None) -> list[str]:
    """No flag: the CLI's only approval flag is `--require-approval` (prompts on). Without it
    the CLI sets alwaysAllowMcp, which approves resources but an MCP tool only if the tool
    is in its server's `alwaysAllow`, so unattended is in write_config. In print mode the
    CLI answers no MCP ask (handleMcpApproval returns without a response)."""
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # -p: print mode, non-interactive. --output-format stream-json: NDJSON events.
    # --oneshot: exit when the task completes. --ephemeral: temporary storage, so no task
    # history persists. --exit-on-error: an API error ends the run instead of retrying.
    # --consecutive-mistake-limit 0 would disable the guard; it stays at the default.
    # Roo has no turn limit flag; the stub's script ends the turn. The prompt goes last:
    # options after it are not read (0.1.17 then falls back to the openrouter provider).
    return [_binary(), "--provider", PROVIDER, "--model", MODEL, "--output-format", "stream-json",
            "--oneshot", "--ephemeral", "--exit-on-error", *mode_args(mode), "-p", prompt]


def env(stub_url: str) -> dict[str, str]:
    return stub_env(STUB_KIND, stub_url)


def mcp_server_of(tool_name: str) -> str | None:
    """Roo offers an MCP tool to the model as mcp--<server>--<tool> (MCP_TOOL_PREFIX "mcp",
    MCP_TOOL_SEPARATOR "--" in the extension bundle)."""
    if not tool_name or not tool_name.startswith("mcp--"):
        return None
    server, _, rest = tool_name[len("mcp--"):].partition("--")
    return server if rest else None


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    return None


def _mcp_use(use: dict) -> tuple[str | None, dict | None]:
    """An MCP tool_use carries the ask's JSON as input.raw: {"type": "use_mcp_tool",
    "serverName", "toolName", "arguments": "<json>"} or {"type": "access_mcp_resource",
    "serverName", "uri"}. Named as offered: mcp--<server>--<tool>."""
    try:
        ask = json.loads((use.get("input") or {}).get("raw") or "")
    except (TypeError, json.JSONDecodeError):
        return use.get("name"), use.get("input")
    if not isinstance(ask, dict):
        return use.get("name"), use.get("input")
    if ask.get("type") == "access_mcp_resource":
        return "access_mcp_resource", {"server_name": ask.get("serverName"), "uri": ask.get("uri")}
    try:
        arguments = json.loads(ask.get("arguments") or "{}")
    except (TypeError, json.JSONDecodeError):
        arguments = ask.get("arguments")
    return f"mcp--{ask.get('serverName')}--{ask.get('toolName')}", arguments


def reduce_output(text: str) -> list[dict]:
    """`--output-format stream-json` events (RooCliStreamEvent, packages/types/src/cli.ts).
    Keep what a cell reads; drop ids, cost and the echoed prompt.

    - An MCP result is a tool_result whose output starts "Error:\n" when the result was
      isError (useMcpToolTool). A call the client itself failed (a result its schema
      rejects) is an `error` event right after the tool_use, so that is the tool's result.
    - Held: in print mode the CLI answers no MCP ask (handleMcpApproval), so a tool call
      Roo does not auto-approve waits until the run is stopped. The stream then ends on
      a tool_use with no result after it: that is the permission event.
    """
    events: list[dict] = []
    pending: dict | None = None
    turns = 0
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "tool_use":
            if not item.get("done"):
                # A partial ask streamed before the complete one with the same id
                # (access_mcp_resource with an empty serverName and uri).
                continue
            use = item.get("tool_use") or {}
            tool, arguments = _mcp_use(use) if item.get("subtype") == "mcp" else (use.get("name"), use.get("input"))
            pending = {"event": "tool_use", "tool": tool, "input": arguments}
            events.append(pending)
        elif kind == "tool_result":
            result = item.get("tool_result") or {}
            output = str(result.get("output") or "")
            error = bool(result.get("error")) or (item.get("subtype") == "mcp" and output.startswith("Error:\n"))
            events.append({"event": "tool_result", "is_error": error, "text": clip(str(result.get("error") or output))})
            pending = None
        elif kind == "error":
            if pending is not None:
                events.append({"event": "tool_result", "is_error": True, "text": clip(item.get("content") or "")})
                pending = None
            else:
                events.append({"event": "text", "text": clip(item.get("content") or "")})
        elif kind == "assistant" and item.get("done"):
            turns += 1
            pending = None
            if item.get("subtype") == "mistake_limit_reached":
                events.append({"event": "result", "subtype": "mistake_limit_reached", "is_error": True,
                               "num_turns": turns, "text": clip(item.get("content") or "")})
            else:
                events.append({"event": "text", "text": clip(item.get("content") or "")})
        elif kind == "result":
            pending = None
            events.append({"event": "result", "subtype": "success" if item.get("success") else "error",
                           "is_error": not item.get("success"), "num_turns": turns,
                           "text": clip(item.get("content") or "")})
    if pending is not None and str(pending.get("tool") or "").startswith(("mcp--", "access_mcp_resource")):
        events.append({"event": "permission_denied", "tool": pending["tool"], "input": pending["input"]})
    return events
