"""Shared helpers for renderers."""

from __future__ import annotations

from importlib import metadata


def dist_version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except Exception:
        return "unknown"


def contain_crewai_storage() -> None:
    """Keep CrewAI's local storage out of the operator's home.

    crewai_core.paths stores data in appdirs.user_data_dir(CREWAI_STORAGE_DIR or the
    current directory's name), so every run in a checkout named X created
    ~/Library/Application Support/X (or ~/.local/share/X). An absolute
    CREWAI_STORAGE_DIR is used as the directory itself, so point it at a temp dir.
    """
    import os
    import tempfile
    from pathlib import Path

    os.environ.setdefault("CREWAI_STORAGE_DIR", str(Path(tempfile.gettempdir()) / "mcp-mirror-crewai"))
