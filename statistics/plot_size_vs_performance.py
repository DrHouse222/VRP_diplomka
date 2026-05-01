#!/usr/bin/env python3
"""
Scatter plot: best-individual tree size (parsed from best_expr in JSONL)
vs performance (pct_vs_nn from the matching CSV row).

JSONL and CSV are expected to have the same basename and aligned rows
(same problem_type × bool_capacity order), as produced by the node_size pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from node_size_labels import node_depth_label, node_depth_sort_tuple

FNAME_RE = re.compile(r"^exp_([A-Za-z])_(\d+)\.jsonl$", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

MIN_PCT_VS_NN = -30.0


def parse_jsonl_name(path: Path) -> tuple[str, int] | None:
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2))


def best_expr_size(best_expr: object) -> float:
    """Approximate GP tree size from expression string by counting node tokens."""
    if not isinstance(best_expr, str) or not best_expr.strip():
        return float("nan")
    return float(len(TOKEN_RE.findall(best_expr)))


def collect_points(
    jsonl_dir: Path,
    csv_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """
    Returns (x best_expr_size, y pct_vs_nn, depth labels per point, filename letters, run ids like 'H_3').
    """
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    letters: list[str] = []
    runs: list[str] = []

    for jpath in sorted(jsonl_dir.glob("exp_*_*.jsonl")):
        parsed = parse_jsonl_name(jpath)
        if parsed is None:
            continue
        letter, run = parsed
        cpath = csv_dir / f"{jpath.stem}.csv"
        if not cpath.is_file():
            continue

        with open(jpath, encoding="utf-8") as fj:
            jlines = [json.loads(l) for l in fj if l.strip()]
        with open(cpath, newline="", encoding="utf-8") as fc:
            crows = list(csv.DictReader(fc))

        n = min(len(jlines), len(crows))
        if n != len(jlines) or n != len(crows):
            print(f"Warning: row count mismatch {jpath.name}: jsonl={len(jlines)} csv={len(crows)}, using first {n}")

        lab = node_depth_label(letter)
        run_id = f"{letter}_{run}"
        for i in range(n):
            rec = jlines[i]
            size_best = best_expr_size(rec.get("best_expr"))
            pct = float(crows[i]["pct_vs_nn"])
            xs.append(size_best)
            ys.append(pct)
            labels.append(lab)
            letters.append(letter)
            runs.append(run_id)

    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        labels,
        letters,
        runs,
    )


def run(jsonl_dir: Path, csv_dir: Path, out_path: Path, dpi: int) -> None:
    x, y, point_labels, letters, _runs = collect_points(jsonl_dir, csv_dir)
    if x.size == 0:
        raise SystemExit(
            f"No paired exp_*_*.jsonl + matching .csv found under {jsonl_dir} / {csv_dir}"
        )

    keep = np.isfinite(x) & np.isfinite(y) & (y >= MIN_PCT_VS_NN)
    dropped = int(np.sum(~keep))
    if dropped:
        x = x[keep]
        y = y[keep]
        point_labels = np.asarray(point_labels, dtype=object)[keep].tolist()
        letters = np.asarray(letters, dtype=object)[keep].tolist()
        print(f"Excluded {dropped} point(s) with pct_vs_nn < {MIN_PCT_VS_NN}")

    if x.size == 0:
        raise SystemExit("No points left after filtering pct_vs_nn")

    letters_order = sorted(set(letters), key=node_depth_sort_tuple)
    legend_order = [node_depth_label(L) for L in letters_order]

    label_to_color = {
        lab: plt.cm.tab10(i % 10) for i, lab in enumerate(legend_order)
    }
    colors = [label_to_color[lab] for lab in point_labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, c=colors, s=28, alpha=0.65, edgecolors="k", linewidths=0.25)

    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) >= 2:
        coef = np.polyfit(x[mask], y[mask], 1)
        xx = np.linspace(float(np.nanmin(x[mask])), float(np.nanmax(x[mask])), 50)
        ax.plot(xx, np.poly1d(coef)(xx), color="gray", linestyle="--", linewidth=1.2, label="Linear fit (all points)")

    ax.set_xlabel("Best-individual tree size (nodes)")
    ax.set_ylabel("% vs nearest neighbor")
    ax.set_title("Best-individual tree size vs GP performance (per problem variant × run)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=label_to_color[lab], markersize=8, label=lab.replace("\n", " "))
        for lab in legend_order
        if lab in label_to_color
    ]
    if np.sum(mask) >= 2:
        handles.append(
            plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2, label="Linear fit (all points)")
        )
    ax.legend(handles=handles, fontsize=7, loc="best", framealpha=0.92)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_path} ({x.size} points)")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Scatter best-individual size vs pct_vs_nn from node_size JSONL+CSV.")
    parser.add_argument(
        "--jsonl_dir",
        type=Path,
        default=root / "experiments" / "node_size",
        help="Directory with exp_*_*.jsonl",
    )
    parser.add_argument(
        "--csv_dir",
        type=Path,
        default=root / "exp_results" / "node_size",
        help="Directory with matching exp_*_*.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PDF (default: <jsonl_dir>/figures/size_vs_performance.pdf)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    out = args.out or (args.jsonl_dir / "figures" / "size_vs_performance.pdf")
    run(args.jsonl_dir, args.csv_dir, out, args.dpi)


if __name__ == "__main__":
    main()
