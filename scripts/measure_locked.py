"""Run a measurement command under the machine-wide measurement lock and load gate.

Harness sessions already take the lock inside measure_harness.py. Framework and feature
instruments (scan, result boundaries, probes, conformance) have timing-sensitive cells too,
so any worker measuring on this machine wraps the command:

    uv run python scripts/measure_locked.py -- uv run python scripts/measure_result_boundaries.py ...

Only one wrapped command runs at a time. It starts once a fixture answers initialize
quickly, and it refuses to start when the machine stays too slow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_harness  # noqa: E402


def main(argv: list[str]) -> int:
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    with measure_harness.measurement_lock():
        load = measure_harness.check_load()
        print(f"load gate: {json.dumps(load)}", flush=True)
        if not load["passed"]:
            print("machine too slow for a measurement; not starting", file=sys.stderr)
            return 3
        return subprocess.call(argv, env={**os.environ, measure_harness.LOCK_HELD_ENV: str(os.getpid())})


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
