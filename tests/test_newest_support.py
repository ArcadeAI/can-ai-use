import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import newest_support  # noqa: E402


def test_each_framework_is_read_at_the_newest_revision_it_negotiates() -> None:
    chosen, cells, area = newest_support.newest_rows()
    assert len(chosen) == 31
    # A framework moves back to 2025-11-25 only when every 2026-07-28 cell is x.
    assert "x" not in {c for row in cells.values() for c in row.values()}
    assert set(cells) == set(area)


def test_area_table_and_feature_counts_print(capsys) -> None:
    assert newest_support.main([]) == 0
    assert newest_support.main(["structured-content"]) == 0
    out = capsys.readouterr().out
    assert "frameworks:" in out and "structured-content: y=" in out
