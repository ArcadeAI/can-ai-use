"""Charm Crush: `crush run -q "<prompt>"`, plain text on stdout. https://github.com/charmbracelet/crush#readme"""

from __future__ import annotations

import shlex
from pathlib import Path

from harnesses import clip, resolve_binary, stub_env, version_of

# Installed project-locally at a pinned version by measure_harness.py when missing. The
# npm package is GoReleaser's wrapper around the release binary; its postinstall
# downloads the checksummed archive into bin/. BINARY is that Go binary itself, so a run
# never goes through node. https://github.com/charmbracelet/crush#installation
INSTALL = ["npm", "install", "--prefix", ".mcp-mirror/harnesses/crush", "@charmland/crush@0.98.1"]
BINARY = ".mcp-mirror/harnesses/crush/node_modules/@charmland/crush/bin/crush"


def _binary() -> str:
    """The absolute path of BINARY; the published argv is made repo-relative again."""
    found = resolve_binary(BINARY)
    return str(found) if found else BINARY


# A custom provider of type anthropic, pointed at the stub by `provider add --base-url`
# in the crushrc. https://github.com/charmbracelet/crush/blob/main/docs/config/README.md#provider-add
STUB_KIND = "anthropic"
MODEL = "stub/stub"

# Crush's own variables (CRUSH_HYPER_API_KEY is a Charm account key).
SCRUBBED_ENV = ("CRUSH_", "HYPER_")
# Where Crush reads its global config and data from, beyond XDG_*. Each gets its own
# dir in the isolated home. https://github.com/charmbracelet/crush#configuration
CONFIG_DIR_ENV = ("CRUSH_GLOBAL_CONFIG", "CRUSH_GLOBAL_DATA")

# Documented switches for what Crush sends on its own: usage metrics
# (https://github.com/charmbracelet/crush#metrics) and the provider catalogue refresh
# from catwalk.charm.sh (https://github.com/charmbracelet/crush#disabling-automatic-provider-updates).
QUIET = {
    "CRUSH_DISABLE_METRICS": "1",
    "DO_NOT_TRACK": "1",
    "CRUSH_DISABLE_PROVIDER_AUTO_UPDATE": "1",
}

# The release check runs on every start (internal/app/app.go, go app.checkForUpdates, in
# 0.98.1) with no option or variable that turns it off. The sink refuses it and the
# check gives up silently.
EGRESS_ALLOWED = {
    "api.github.com": "the release check (GET /repos/charmbracelet/crush/releases/latest) on every start; "
                      "no documented off switch, and a refusal is silently ignored",
}

# Crush's own tools carry no mcp_ prefix; an MCP tool is mcp_<server>_<tool>
# (internal/agent/tools/mcp-tools.go). The two resource tools are built in.
RESOURCE_TOOLS = ("list_mcp_resources", "read_mcp_resource")


def install_check() -> str:
    return version_of([_binary(), "--version"])


def write_config(servers: dict[str, dict], workdir: Path, env: dict[str, str]) -> Path:
    """.crushrc in the workdir, the first place Crush reads config from.

    It is Bash with Crush's builtins: the stub provider and model, both model slots on
    it, and one `mcp add` per server, each at the default startup timeout, which
    h-startup-timeout measures. The provider's URL and key are read from the stub env
    when Crush runs it. https://github.com/charmbracelet/crush/blob/main/docs/config/README.md
    """
    lines = [
        'provider add stub --type anthropic --base-url "$ANTHROPIC_BASE_URL" --api-key "$ANTHROPIC_API_KEY"',
        # Image-capable, as the models Crush ships with are: a model without it gets
        # "This model (stub) does not support image data." in place of any tool result that
        # carries an image block (internal/agent/tools/mcp-tools.go).
        f"model add {MODEL} --name stub --context-window 200000 --default-max-tokens 8192 --supports-images true",
        f"model large {MODEL}",
        f"model small {MODEL}",
        # No built-in catalogue providers: only the stub is configured.
        "option default-providers false",
        # The crush-config skill and any skill dir are left out of the system prompt.
        "option disable-skill crush-config",
    ]
    for name, server in servers.items():
        if "url" in server:
            lines.append(f"mcp add {shlex.quote(name)} --type http --url {shlex.quote(server['url'])}")
        else:
            args = " ".join(f"--args {shlex.quote(arg)}" for arg in server["args"])
            lines.append(f"mcp add {shlex.quote(name)} --command {shlex.quote(server['command'])} {args}".rstrip())
    path = workdir / ".crushrc"
    path.write_text("\n".join(lines) + "\n")
    return path


def list_servers(env: dict[str, str], workdir: Path) -> set[str] | None:
    """None: Crush 0.98.1 has no command that lists MCP servers (`crush --help`); `mcp`
    exists only as a crushrc builtin. tools_offered checks what reached the model."""
    return None


def mode_args(mode: str | None) -> list[str]:
    # `crush run` approves every permission request of its own session
    # (internal/app/app.go, RunNonInteractive: app.Permissions.AutoApproveSession, 0.98.1).
    # `--yolo` is a flag of the interactive root command only; `crush run --yolo` fails
    # with "Unknown flag: --yolo". So both modes run with no flag.
    return []


def argv(prompt: str, mode: str | None, config: Path, max_turns: int) -> list[str]:
    # -q hides the spinner. Crush has no turn limit flag; the stub's script ends the turn.
    return [_binary(), "run", "-q", "--model", MODEL, *mode_args(mode), prompt]


def env(stub_url: str) -> dict[str, str]:
    return stub_env(STUB_KIND, stub_url) | QUIET


def mcp_server_of(tool_name: str) -> str | None:
    if not tool_name.startswith("mcp_") or tool_name in RESOURCE_TOOLS:
        return None
    server, _, rest = tool_name[len("mcp_"):].partition("_")
    return server if rest else None


def prompt_invocation(server: str, name: str, arguments: dict) -> str | None:
    """None: MCP prompts are commands in Crush's TUI (internal/commands/commands.go);
    `crush run` sends its text to the model as is."""
    return None


def reduce_output(text: str) -> list[dict]:
    """`crush run` prints only the assistant's text, with no events, so the one event is
    the result. Tool calls and permission requests are read from the recorder and the
    stub's model.jsonl, never from stdout."""
    stripped = text.strip()
    if not stripped:
        return []
    return [{"event": "result", "subtype": None, "is_error": None, "num_turns": None, "text": clip(stripped)}]
