"""Build (once) and return an interpreter that serves the fixture at a given revision.

The fixture must speak the revision under test, and 2026-07-28 needs the 2.x Python
SDK while the project venv is pinned to 1.x. Every measurement script takes
``--spec-version`` and asks here for the matching interpreter, so a published
reproduce command never names a path that only exists on one machine.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp_mirror.runner_manifests import managed_source_profile  # noqa: E402


def fixture_python(spec_version: str) -> str:
    venv = ROOT / ".mcp-mirror" / f"fixture-{spec_version}"
    executable = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if executable.exists():
        return str(executable)
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("building the fixture environment requires uv: https://docs.astral.sh/uv/")
    profile = managed_source_profile(spec_version)
    print(f"building the {spec_version} fixture environment in {venv} ...", file=sys.stderr)
    subprocess.run([uv, "venv", "--python", str(profile["python"]), str(venv)], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(executable), *profile["packages"]], check=True)
    return str(executable)
