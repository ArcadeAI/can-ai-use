"""Alias for ``definitions_server.py``, kept so every published command still runs.

The fixture was split by instrument (definitions, results, notifications, client
features, utilities). This name predates the split and appears in reproduce commands
and measurement artifacts, so it keeps serving exactly what it always served.

    python fixtures/tricky_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from definitions_server import *  # noqa: E402,F401,F403
from definitions_server import _main, build_server  # noqa: E402,F401

if __name__ == "__main__":
    import anyio

    anyio.run(_main)
