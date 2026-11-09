"""Support counts at each framework's newest MCP revision.

A framework is read at the newest spec revision it can negotiate: the 2026-07-28 row
unless every cell there is `x` (the adapter cannot negotiate that revision), in which
case its 2025-11-25 row. Within a row, the last listed version is used.

    uv run python scripts/newest_support.py                # per-area share of cells dropped
    uv run python scripts/newest_support.py structured-content tool-execution-errors

The area table counts only cells coded y, a or n. The share is n over those cells,
pooled across the area's features. The per-feature view prints every code.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = ["2026-07-28", "2025-11-25"]  # newest first
MEASURED = {"y", "a", "n"}


def load(revision: str) -> dict:
    return json.loads((ROOT / "data" / "fulldata" / f"data-{revision}.json").read_text())


def code(versions: dict[str, str]) -> str:
    return list(versions.values())[-1].split()[0]


def newest_rows() -> tuple[dict[str, str], dict[str, dict[str, str]], dict[str, str]]:
    """Return (framework -> revision, feature -> {framework: code}, feature -> area)."""
    data = {rev: load(rev)["data"] for rev in REVISIONS}
    frameworks = sorted({fw for rev in REVISIONS for cap in data[rev].values() for fw in cap["stats"]})
    chosen = {}
    for fw in frameworks:
        for rev in REVISIONS:
            codes = [code(cap["stats"][fw]) for cap in data[rev].values() if fw in cap["stats"]]
            if codes and set(codes) != {"x"}:
                chosen[fw] = rev
                break
    cells: dict[str, dict[str, str]] = defaultdict(dict)
    area: dict[str, str] = {}
    for fw, rev in chosen.items():
        for feature, cap in data[rev].items():
            area.setdefault(feature, cap.get("area") or "Unassigned")
            if fw in cap["stats"]:
                cells[feature][fw] = code(cap["stats"][fw])
    return chosen, cells, area


def main(argv: list[str]) -> int:
    chosen, cells, area = newest_rows()
    by_rev = Counter(chosen.values())
    print(f"{len(chosen)} frameworks: " + ", ".join(f"{n} at {rev}" for rev, n in sorted(by_rev.items(), reverse=True)))
    if argv:
        for feature in argv:
            counts = Counter(cells[feature].values())
            print(f"{feature}: " + " ".join(f"{k}={counts[k]}" for k in "yanxu" if counts[k]))
        return 0
    totals: dict[str, Counter] = defaultdict(Counter)
    for feature, row in cells.items():
        totals[area[feature]].update(c for c in row.values() if c in MEASURED)
    print(f"{'area':<18} {'dropped':>8} {'cells':>6}")
    for name, c in sorted(totals.items(), key=lambda kv: -kv[1]["n"] / max(1, sum(kv[1].values()))):
        n = sum(c.values())
        if n:
            print(f"{name:<18} {c['n'] / n:>7.0%} {n:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
