"""Replace home-directory paths in committed data with `~`.

Measurements record interpreter paths and tracebacks, which carry the operator's home
directory (/Users/<name>/... or /home/<name>/...). They say nothing about the framework
being measured, so every writer's output is scrubbed before commit. A path inside the
checkout that ran the measurement then names a workspace directory, which says nothing
either, so it becomes `<checkout>`:

    uv run python scripts/scrub_home_paths.py          # rewrite in place
    uv run python scripts/scrub_home_paths.py --check  # exit 1 if any remain
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = re.compile(r"(?:/Users|/home)/[A-Za-z0-9._-]+(?=/)")
# A worktree created for one measuring task, or the tail of one where a stderr tail starts
# mid-path: the IDE's ~/orca/workspaces/<repo>/<worktree>, or an agent's
# <checkout>/.claude/worktrees/<name> under any checkout. The path may end there, as a
# harness's working directory or an `ls` argument does.
WORKTREE = re.compile(
    r"[A-Za-z0-9._~-]*/orca/workspaces/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?![A-Za-z0-9._-])"
    r"|(?:[A-Za-z0-9._~/-]*/)?\.claude/worktrees/[A-Za-z0-9._-]+(?![A-Za-z0-9._-])"
)
# The directory holding those worktrees, named on its own when an agent searches above
# its checkout, and the IDE's own bin directory on a recorded PATH.
WORKSPACE = re.compile(r"[A-Za-z0-9._~-]*/orca/workspaces/[A-Za-z0-9._-]+(?![A-Za-z0-9._-])")
IDE_BIN = re.compile(r"/Applications/Orca\.app/Contents/Resources/bin")
SCOPES = ("data", "plan")
SUFFIXES = {".json", ".jsonl", ".md", ".yaml", ".yml", ".txt", ".log"}


def files() -> list[Path]:
    return [p for scope in SCOPES for p in (ROOT / scope).rglob("*")
            if p.is_file() and p.suffix in SUFFIXES]


def scrub(text: str) -> str:
    text = WORKTREE.sub("<checkout>", HOME.sub("~", text))
    return IDE_BIN.sub("<ide>/bin", WORKSPACE.sub("<workspace>", text))


def main(argv: list[str]) -> int:
    check = "--check" in argv
    dirty = []
    for path in files():
        text = path.read_text(errors="ignore")
        clean = scrub(text)
        if clean != text:
            dirty.append(path.relative_to(ROOT))
            if not check:
                path.write_text(clean)
    for path in dirty:
        print(("home path in " if check else "scrubbed ") + str(path))
    return 1 if check and dirty else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
