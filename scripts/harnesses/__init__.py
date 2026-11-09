"""One module per harness: everything measure_harness.py needs to know about one harness.

scripts/measure_harness.py owns what every harness shares (fixtures, recorders, model
stubs, sessions, the manifest). A harness module owns the rest: how to find and version
the binary, how to write its MCP config, how to start it headless in each approval
mode, where its model calls go, and how to read its machine-readable output. The module
is found by id, so adding a harness is adding scripts/harnesses/<id>.py (start from
_template.py) and a block under `harnesses:` in plan/harness_sessions.yaml.

The interface (checked by ``load``; _template.py shows every entry in use):

    BINARY: str                                   # the executable the inventory's install puts on PATH
    STUB_KIND: str | None                         # "anthropic", "openai" or "gemini"; None = no stub
    install_check() -> str                        # the installed version, or raise HarnessError
    write_config(servers, workdir, env) -> Path   # the MCP config file, written inside the isolated home;
                                                  # may also declare keyword-only mode= and tools=, below
    list_servers(env, workdir) -> set[str] | None # the harness's own `mcp list`; None = it has none
    mode_args(mode) -> list[str]                  # the documented approval switch, "default" / "unattended"
    argv(prompt, mode, config, max_turns) -> list[str]   # the full headless command; mode None at level 1
    env(stub_url) -> dict[str, str]               # what points the harness's model calls at the stub
    reduce_output(text) -> list[dict]             # stdout reduced to the event vocabulary below
    prompt_invocation(server, name, arguments) -> str | None   # start text that expands an MCP prompt

Optional:

    SCRUBBED_ENV: tuple[str, ...]      # env prefixes that would hand the harness a real account
    CONFIG_DIR_ENV: tuple[str, ...]    # extra variables naming a config dir; each is pointed inside the isolated home
    PASSED_ENV: tuple[str, ...]        # extra variables passed through from the real environment
    EGRESS_ALLOWED: dict[str, str]     # host -> why the harness may try to reach it (still refused, still recorded)
    HOME_FILES: tuple[str, ...]        # files under the real home the harness needs copied in (none should)
    HOME_FILES_ALLOWED: dict[str, str] # path -> why; a HOME_FILES entry not here fails freshness
    mcp_server_of(tool_name) -> str | None   # the MCP server an offered tool belongs to; default reads mcp__<server>__
    write_config(..., *, mode=None, tools=None)   # for an approval switch that lives in the MCP config:
                                       # mode as in mode_args (None at level 1), tools {server: [name, ...]}
                                       # from each fixture's tools/list; passed only if declared

Every session runs fresh (scripts/measure_harness.py, "Freshness"): a new temp HOME with
the XDG, CODEX_HOME, CLAUDE_CONFIG_DIR and GEMINI_CLI_HOME dirs inside it, only PATH and
locale from the real environment, and every proxy variable pointed at a local sink that
refuses and logs. ``env`` in write_config and list_servers is that environment.

Output events are dicts with an ``event`` key. Cells read only these:

    {"event": "init", "harness_version", "permission_mode", "mcp_servers", "mcp_tools"}
    {"event": "tool_use", "tool", "input"}
    {"event": "tool_result", "is_error", "text"}
    {"event": "text", "text"}
    {"event": "permission_denied", "tool", "input"}   # the machine-readable "held"
    {"event": "result", "subtype", "is_error", "num_turns", "text"}
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
STUB_KEY = "mcp-mirror-stub"  # a placeholder, not a credential: the stubs accept anything
STUB_KINDS = ("anthropic", "openai", "gemini")

REQUIRED = {
    "BINARY": str,
    "STUB_KIND": (str, type(None)),
    "install_check": callable,
    "write_config": callable,
    "list_servers": callable,
    "mode_args": callable,
    "argv": callable,
    "env": callable,
    "reduce_output": callable,
    "prompt_invocation": callable,
}


class HarnessError(RuntimeError):
    pass


ROOT = HERE.parent.parent


def resolve_binary(binary: str) -> Path | None:
    """Where ``BINARY`` lives: a path relative to the repository when it contains a slash
    (a project-local install under .mcp-mirror/harnesses/), otherwise a PATH lookup."""
    import shutil

    if "/" in binary:
        candidate = (ROOT / binary) if not binary.startswith("/") else Path(binary)
        return candidate if candidate.exists() else None
    found = shutil.which(binary)
    return Path(found) if found else None


def available() -> list[str]:
    """Every harness id with a module; files starting with _ are not harnesses."""
    return sorted(p.stem for p in HERE.glob("*.py") if not p.stem.startswith("_"))


def load(harness_id: str) -> ModuleType:
    """The module for ``harness_id``, checked against the interface."""
    if not re.fullmatch(r"[a-z0-9_]+", harness_id) or not (HERE / f"{harness_id}.py").exists():
        raise HarnessError(
            f"{harness_id!r} has no module in scripts/harnesses/ yet (copy _template.py; "
            f"available: {', '.join(available()) or 'none'})"
        )
    module = importlib.import_module(f"{__name__}.{harness_id}")
    problems = interface_problems(module)
    if problems:
        raise HarnessError(f"scripts/harnesses/{harness_id}.py: {'; '.join(problems)}")
    return module


def interface_problems(module: ModuleType) -> list[str]:
    problems = []
    for name, kind in REQUIRED.items():
        if not hasattr(module, name):
            problems.append(f"missing {name}")
        elif kind is callable:
            if not callable(getattr(module, name)):
                problems.append(f"{name} is not callable")
        elif not isinstance(getattr(module, name), kind):
            problems.append(f"{name} has the wrong type")
    install = getattr(module, "INSTALL", None)
    if install is not None and not (isinstance(install, list) and all(isinstance(a, str) for a in install)):
        problems.append("INSTALL must be an argv list of strings")
    kind = getattr(module, "STUB_KIND", None)
    if kind is not None and kind not in STUB_KINDS:
        problems.append(f"STUB_KIND {kind!r} is not one of {', '.join(STUB_KINDS)}")
    for name in ("SCRUBBED_ENV", "CONFIG_DIR_ENV", "PASSED_ENV", "HOME_FILES"):
        if not isinstance(getattr(module, name, ()), tuple):
            problems.append(f"{name} must be a tuple")
    for name in ("EGRESS_ALLOWED", "LIVE_EGRESS_ALLOWED", "HOME_FILES_ALLOWED"):
        allowed = getattr(module, name, {})
        if not isinstance(allowed, dict) or not all(isinstance(v, str) and v.strip() for v in allowed.values()):
            problems.append(f"{name} must map each entry to a reason")
    return problems


# -- helpers a module may use ------------------------------------------------------


def clip(text: Any, limit: int = 500) -> str:
    """Text as the output events keep it: JSON if it is not a string, cut at ``limit``."""
    text = text if isinstance(text, str) else json.dumps(text)
    return text if len(text) <= limit else text[:limit] + "..."


def stub_env(kind: str, url: str) -> dict[str, str]:
    """The conventional variables for each stub kind, under the placeholder key.

    A module whose harness reads other variables writes its own ``env``.
    """
    return {
        "anthropic": {"ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": STUB_KEY},
        "openai": {"OPENAI_BASE_URL": f"{url}/v1", "OPENAI_API_KEY": STUB_KEY},
        "gemini": {"GOOGLE_GEMINI_BASE_URL": url, "GEMINI_API_KEY": STUB_KEY},
    }[kind]


@contextlib.contextmanager
def throwaway_home():
    """The real environment with HOME and the XDG dirs moved into a temp dir.

    Even `--version` starts some harnesses, and some create config, data and cache
    directories under HOME on startup; Kilo's npm postinstall does the same at install
    time. Anything that runs a harness outside a measured session (its install, its
    version check) runs in here, so it never writes into the operator's own home. npm
    keeps the real download cache, which holds packages, not configuration."""
    with tempfile.TemporaryDirectory(prefix="mcp-mirror-home-") as home:
        env = {**os.environ, "HOME": home, "USERPROFILE": home,
               "XDG_CONFIG_HOME": f"{home}/.config", "XDG_DATA_HOME": f"{home}/.local/share",
               "XDG_CACHE_HOME": f"{home}/.cache", "XDG_STATE_HOME": f"{home}/.local/state"}
        env.setdefault("npm_config_cache", str(Path.home() / ".npm"))
        yield env


def version_of(argv: list[str]) -> str:
    """The first x.y.z the command prints, for ``install_check``."""
    import shutil
    import subprocess

    if not shutil.which(argv[0]):
        raise HarnessError(f"{argv[0]} is not on PATH")
    with throwaway_home() as env:
        try:
            text = subprocess.run(argv, capture_output=True, text=True, timeout=60, env=env).stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HarnessError(f"cannot read the version: {exc}") from exc
    found = re.search(r"\d+\.\d+\.\d+[\w.+-]*", text)
    if not found:
        raise HarnessError(f"{' '.join(argv)} printed no version: {text.strip()[:120]!r}")
    return found.group(0)


MCP_TOOL = re.compile(r"^mcp__(.+?)__.")


def mcp_server_of(tool_name: str) -> str | None:
    """The server a ``mcp__<server>__<tool>`` name belongs to; None for a harness's own tool."""
    found = MCP_TOOL.match(tool_name)
    return found.group(1) if found else None


def list_command(argv: list[str], env: dict[str, str], workdir: Path) -> str:
    """Run a harness's `mcp list` in the isolated environment; its stdout, for ``list_servers``."""
    import subprocess

    try:
        done = subprocess.run(argv, cwd=workdir, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HarnessError(f"{' '.join(argv)} did not run: {exc}") from exc
    return done.stdout


def names_before_colon(text: str) -> set[str]:
    """Server names from a `<name>: <command or url> - <status>` listing, one per line."""
    return {m.group(1) for line in text.splitlines() if (m := re.match(r"^([A-Za-z0-9_.-]+): \S", line))}
