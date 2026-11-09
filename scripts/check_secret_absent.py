"""Fail if a secret from the environment appears in any file under the given paths.

    op run --env-file .env.harness -- python -I scripts/check_secret_absent.py \
        --var CURSOR_API_KEY -- data/harnesses/measurements/wire-live/cursor_agent

The runner starts this after each live session with the values it took from its own
environment (measure_harness.live_secrets), so the value reaches this process's
environment and nowhere else. It is never printed: a match reports the
file's path and the variable's name only. Exit 0 when no file holds a value, 1 on a
match, 2 when a named variable is unset or empty (then nothing was checked).

Searched as the raw bytes and as the value's base64 and percent-encoded forms, the
shapes a CLI stores a key in.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import urllib.parse
from pathlib import Path


def forms(value: str) -> set[bytes]:
    raw = value.encode()
    return {raw, base64.b64encode(raw), urllib.parse.quote(value, safe="").encode()}


def files_under(paths: list[str]):
    for given in paths:
        path = Path(given)
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from (p for p in sorted(path.rglob("*")) if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--var", action="append", default=[], help="environment variable holding a secret (repeatable)")
    parser.add_argument("paths", nargs="+", help="files or directories to search")
    arguments = parser.parse_args()
    needles: dict[str, set[bytes]] = {}
    for name in arguments.var:
        value = os.environ.get(name, "")
        if len(value) < 8:
            print(f"{name} is not set (or shorter than 8 characters); nothing checked", file=sys.stderr)
            return 2
        needles[name] = forms(value)
    matches = 0
    for path in files_under(arguments.paths):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        for name, shapes in needles.items():
            if any(shape in data for shape in shapes):
                print(f"match: {path} holds {name}")
                matches += 1
    return 1 if matches else 0


if __name__ == "__main__":
    sys.exit(main())
