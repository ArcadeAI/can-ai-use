"""Execute every reproduce command the site publishes, from a clean checkout.

Each measured capability publishes one command that is supposed to re-derive its row.
Nothing checked that the commands ran, so they rotted in two directions at once: they
named a package that was never uploaded to PyPI, and they reached for a virtualenv that
existed on one laptop. Both failures are invisible from inside a developer environment
that already has everything, which is why this runs somewhere that has nothing.

    uv run python scripts/check_reproduce_commands.py

Tracked files are copied to a temporary directory, so the run sees the working tree as
a fresh clone would see it: no .venv, no node_modules, no fixture interpreter, no
uncommitted scratch files. A command passes only if it exits 0 and prints none of the
markers below, because several of these commands report a dead renderer on stdout and
still exit 0.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILT_DATA = ROOT / "data" / "fulldata"

# Exiting 0 is not enough. A renderer that cannot start is reported in the body of the
# output by the measurement scripts, and uv reports an unresolvable package the same
# way, so the text has to be read as well as the status.
FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "ModuleNotFoundError",
    "command not found",
    "was not found in the package registry",
    "FAILED:",
    "adapter error:",
    "no frameworks scanned",
)

# Commands that cannot be judged here, with the reason. These are counted and reported
# separately rather than folded into the pass total.
UNRUNNABLE: dict[str, str] = {}

# Steps a fresh clone needs before any published command can run. Mastra used to need
# a local `npm ci` here; the managed runner now installs its Node packages itself, and
# running without that step is what proves it.
SETUP: tuple[list[str], ...] = ()


def _built_files(ref: str | None) -> dict[str, str]:
    """The built data files, read from the working tree or from a git ref."""

    if ref is None:
        return {path.name: path.read_text() for path in BUILT_DATA.glob("*.json")}
    rel = BUILT_DATA.relative_to(ROOT).as_posix()
    names = subprocess.run(["git", "ls-tree", "--name-only", f"{ref}:{rel}"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    return {name: subprocess.run(["git", "show", f"{ref}:{rel}/{name}"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout
            for name in names if name.endswith(".json")}


def split_command(command: str) -> list[str]:
    """The independent commands a published reproduce line chains with ``&&``.

    A row re-derived from several capture files publishes one command per file joined
    by ``&&``, so adding a framework changes every such line. Checking the parts lets a
    branch run only the parts it added, and runs a part shared by many rows once."""
    return [part.strip() for part in command.split(" && ") if part.strip()]


def published_commands(ref: str | None = None) -> dict[str, list[str]]:
    """Map each published reproduce command to the cells that publish it."""

    files = _built_files(ref)
    commands: dict[str, list[str]] = {}
    for name in sorted(n for n in files if n.startswith("data-2")):
        path = Path(name)
        document = json.loads(files[name])
        spec = document.get("mcp_spec", path.stem)
        for capability_id, capability in document["data"].items():
            command = (capability.get("reproduce") or "").strip()
            if command:
                for part in split_command(command):
                    commands.setdefault(part, []).append(f"{spec}/{capability_id}")
    # The harness table publishes its own commands; level 1 needs no credential, so
    # they run here like any other.
    if "harnesses.json" in files:
        for capability_id, row in json.loads(files["harnesses.json"])["data"].items():
            per_agent = row.get("reproduce_by_agent") or {}
            for command in (per_agent.values() if per_agent else [row.get("reproduce") or ""]):
                command = command.strip()
                if not command:
                    continue
                # A full harness run is three runs of every level-1 and level-2 session,
                # tens of minutes each. This check proves the published command runs from
                # a clean checkout, so it runs one level-1 pass of it.
                if "--level" not in command:
                    command += " --level 1 --runs 1"
                commands.setdefault(command, []).append(f"harnesses/{capability_id}")
    return commands


def clean_checkout(destination: Path) -> None:
    """Copy every tracked file, which is what a fresh clone of this tree would hold."""

    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for name in listing.stdout.split("\0"):
        source = ROOT / name
        # A tracked path can be absent from the working tree: `vite build` empties
        # docs/ and replaces hashed bundles, so the previous ones are staged deletions
        # until the next commit. Copy what is on disk.
        if not name or not source.is_file():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def bare_environment() -> dict[str, str]:
    """Drop the caller's virtualenv, so a published command cannot borrow it.

    Running this through `uv run` puts the project interpreter on PATH, which is how a
    bare `python scripts/...` command passed for months without being installable by
    anyone who had not already activated the environment.
    """

    environment = dict(os.environ)
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("PYTHONPATH", None)
    kept = [
        entry
        for entry in environment.get("PATH", "").split(os.pathsep)
        if entry and not Path(entry).is_relative_to(ROOT)
    ]
    environment["PATH"] = os.pathsep.join(kept)
    return environment


def run(command: str, cwd: Path, timeout: int) -> tuple[str, str, float]:
    """Run one published command and classify it as pass, fail, or skip."""

    if command in UNRUNNABLE:
        return "skip", UNRUNNABLE[command], 0.0

    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=bare_environment(),
        )
    except subprocess.TimeoutExpired:
        return "fail", f"timed out after {timeout}s", time.monotonic() - started

    elapsed = time.monotonic() - started
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        tail = output.strip().splitlines()[-1:] or ["no output"]
        return "fail", f"exit {proc.returncode}: {tail[0][:160]}", elapsed
    for marker in FAILURE_MARKERS:
        if marker in output:
            return "fail", f"exit 0 but output contains {marker!r}", elapsed
    return "pass", "", elapsed


MARKER = ".mcp-mirror-reproduce-pid"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove(path: Path) -> None:
    """Delete a checkout, including the read-only trees Go writes into its module cache.

    ``rmtree(ignore_errors=True)`` skips those silently and takes the marker with it,
    so the 544 MB cache it left behind was invisible to the next run's orphan sweep."""

    def make_writable(function, name, _info):
        os.chmod(os.path.dirname(name), stat.S_IRWXU)
        if os.path.lexists(name) and not os.path.islink(name):
            os.chmod(name, stat.S_IRWXU)
        function(name)

    if path.exists():
        shutil.rmtree(path, onerror=make_writable)


def remove_orphaned_checkouts() -> list[Path]:
    """Delete clean checkouts left by earlier runs whose process is gone.

    A run cleans up after itself, but one killed outright (SIGKILL, out of memory)
    cannot, and each checkout holds a virtualenv and runner caches. On one machine
    these piled up to 34 GB and filled the disk."""

    removed = []
    for marker in Path(tempfile.gettempdir()).glob(f"*/{MARKER}"):
        try:
            pid = int(marker.read_text().strip())
        except (OSError, ValueError):
            continue
        if not _alive(pid):
            _remove(marker.parent)
            removed.append(marker.parent)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=1800, help="per-command seconds")
    parser.add_argument("--no-setup", action="store_true", help="skip the setup steps")
    parser.add_argument("--keep", action="store_true", help="keep the temporary checkout")
    parser.add_argument("--changed-from", metavar="REF",
                        help="run only commands that are new or changed since REF (e.g. origin/main); "
                             "a branch checks its own commands, the full run happens after merge")
    parser.add_argument("--match", action="append", default=[], metavar="TEXT",
                        help="run only commands containing TEXT (repeatable)")
    arguments = parser.parse_args()

    commands = published_commands()
    if arguments.changed_from:
        before = published_commands(arguments.changed_from)
        commands = {c: ids for c, ids in commands.items() if c not in before}
    if arguments.match:
        commands = {c: ids for c, ids in commands.items() if any(t in c for t in arguments.match)}
    cells = sum(len(ids) for ids in commands.values())
    print(f"{cells} published reproduce commands, {len(commands)} distinct")
    if not commands:
        print("nothing to run")
        return 0

    orphans = remove_orphaned_checkouts()
    if orphans:
        print(f"removed {len(orphans)} checkout(s) left by killed runs")
    workspace = Path(tempfile.mkdtemp())
    (workspace / MARKER).write_text(str(os.getpid()))
    # SIGTERM and SIGHUP become an ordinary exit, so the finally below still runs.
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda number, _frame: (_ for _ in ()).throw(SystemExit(128 + number)))
    try:
        return _run_in(workspace, commands, cells, arguments)
    finally:
        if arguments.keep:
            print(f"\nkept {workspace / 'mcp-mirror'}")
        else:
            _remove(workspace)


def _run_in(workspace: Path, commands: dict[str, list[str]], cells: int, arguments) -> int:
    checkout = workspace / "mcp-mirror"
    checkout.mkdir()
    print(f"clean checkout: {checkout}")
    clean_checkout(checkout)
    # A Node install cached by an earlier run would hide a broken install path.
    os.environ["MCP_MIRROR_CACHE_DIR"] = str(workspace / "cache")

    if not arguments.no_setup:
        for step in SETUP:
            print(f"setup: {' '.join(step)}")
            setup = subprocess.run(step, cwd=checkout, capture_output=True, text=True)
            if setup.returncode != 0:
                print(f"  setup failed: {setup.stderr.strip().splitlines()[-1:]}")

    tally = {"pass": 0, "fail": 0, "skip": 0}
    failures = []
    for command, ids in sorted(commands.items(), key=lambda item: -len(item[1])):
        state, detail, elapsed = run(command, checkout, arguments.timeout)
        tally[state] += len(ids)
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[state]
        print(f"\n{mark}  {len(ids)} cell(s), {elapsed:.0f}s")
        print(f"      {command}")
        if detail:
            print(f"      {detail}")
        if state == "fail":
            failures.append((command, detail, ids))

    print(
        f"\n{tally['pass']}/{cells} published commands run clean, "
        f"{tally['fail']} fail, {tally['skip']} not judged here"
    )
    for command, detail, ids in failures:
        print(f"  {len(ids)} cell(s): {command}\n    {detail}")
    return 1 if tally["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
