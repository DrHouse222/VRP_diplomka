#!/usr/bin/env python3
"""
Scan all experiment CSVs under exp_results/ and build one CSV with the best row
per problem variant (problem_type × bool_capacity), by highest pct_vs_nn.

Adds column source_file (path relative to exp_results/).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def _float(x: str | None) -> float:
    if x is None or x == "":
        return math.nan
    return float(x)


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("problem_type", "")), str(row.get("bool_capacity", "")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate best pct_vs_nn row per variant across exp_results CSVs.")
    parser.add_argument(
        "--exp_results",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "exp_results",
        help="Root exp_results directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: <exp_results>/best_per_variant.csv)",
    )
    args = parser.parse_args()
    root = args.exp_results.resolve()
    out_path = args.out or (root / "best_per_variant.csv")

    best: dict[tuple[str, str], tuple[float, dict[str, str], str]] = {}

    for path in sorted(root.rglob("*.csv")):
        # Do not ingest a previous aggregate output
        if path.name == "best_per_variant.csv":
            continue
        rel = str(path.relative_to(root))
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except OSError:
            continue
        if not rows:
            continue
        for row in rows:
            key = row_key(row)
            v = _float(row.get("pct_vs_nn"))
            if math.isnan(v):
                continue
            prev = best.get(key)
            if prev is None or v > prev[0] or (v == prev[0] and rel < prev[2]):
                best[key] = (v, {k: row.get(k, "") for k in row}, rel)

    if not best:
        raise SystemExit(f"No rows with finite pct_vs_nn found under {root}")

    # Stable order: by index if present, else by problem_type, bool_capacity
    def sort_key(item: tuple[tuple[str, str], Any]) -> tuple:
        k, (_, row, _) = item
        try:
            idx = int(row.get("index", -1))
        except ValueError:
            idx = -1
        return (idx, k[0], k[1])

    items = sorted(best.items(), key=sort_key)
    fieldnames = list(items[0][1][1].keys()) + ["source_file"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for _k, (_score, row, rel) in items:
            out_row = dict(row)
            out_row["source_file"] = rel
            w.writerow(out_row)

    print(f"Wrote {out_path} ({len(items)} rows)")


if __name__ == "__main__":
    main()
