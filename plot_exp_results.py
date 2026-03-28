#!/usr/bin/env python3
"""
Build comparison plots from exp_p{pop}_g{gens}_{run}.csv files in exp_results/.

Each CSV is one run of test_experiment_res: one row per (problem_type, capacity).
This script groups runs by (population, generations), aggregates metrics across
the 5 runs, and writes figures to compare GP vs NN / Savings settings.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# exp_p25_g400_3.csv -> pop=25, gens=400, run=3
FNAME_RE = re.compile(r"^exp_p(\d+)_g(\d+)_(\d+)\.csv$", re.IGNORECASE)


def parse_filename(path: Path) -> tuple[int, int, int] | None:
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _float(x: str | None) -> float:
    if x is None or x == "":
        return np.nan
    return float(x)


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def weighted_mean_pct(rows: list[dict[str, Any]], col: str) -> float:
    """Instance-count weighted mean of a percentage column (skips NaN cells)."""
    num = 0.0
    den = 0.0
    for row in rows:
        v = _float(row.get(col))
        n = int(row["n_instances"])
        if not np.isnan(v) and n > 0:
            num += v * n
            den += n
    return float(num / den) if den > 0 else np.nan


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (row["problem_type"], row["bool_capacity"])


def discover_groups(exp_dir: Path) -> dict[tuple[int, int], list[Path]]:
    """Map (population, generations) -> list of CSV paths (all runs)."""
    groups: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path in sorted(exp_dir.glob("*.csv")):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        pop, gens, _run = parsed
        groups[(pop, gens)].append(path)
    for key in groups:
        groups[key].sort(key=lambda p: parse_filename(p)[2])  # type: ignore
    return dict(groups)


def build_heatmap_matrix(
    groups: dict[tuple[int, int], list[Path]],
) -> tuple[list[tuple[str, str]], list[tuple[int, int]], np.ndarray]:
    """Mean pct_vs_nn per (problem_type, capacity) × (pop, gens), averaged over runs."""
    col_keys = sorted(groups.keys(), key=lambda t: (t[0], t[1]))
    any_path = groups[col_keys[0]][0]
    base_rows = load_csv_rows(any_path)
    row_keys = [row_key(r) for r in base_rows]
    mat = np.full((len(row_keys), len(col_keys)), np.nan, dtype=float)

    for j, cfg in enumerate(col_keys):
        paths = groups[cfg]
        for i, rk in enumerate(row_keys):
            vals: list[float] = []
            for path in paths:
                for r in load_csv_rows(path):
                    if row_key(r) == rk:
                        v = _float(r.get("pct_vs_nn"))
                        if not np.isnan(v):
                            vals.append(v)
                        break
            if vals:
                mat[i, j] = float(np.mean(vals))

    return row_keys, col_keys, mat


def cfg_label(cfg: tuple[int, int]) -> str:
    p, g = cfg
    return f"p{p}\ng{g}"


def run(exp_dir: Path, out_dir: Path, dpi: int) -> None:
    groups = discover_groups(exp_dir)
    if not groups:
        raise SystemExit(f"No exp_p*_g*_*.csv files found in {exp_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-run file scores: one weighted score per CSV ---
    cfg_scores_nn: dict[tuple[int, int], list[float]] = defaultdict(list)
    cfg_scores_sav: dict[tuple[int, int], list[float]] = defaultdict(list)

    for cfg, paths in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        for path in paths:
            rows = load_csv_rows(path)
            cfg_scores_nn[cfg].append(weighted_mean_pct(rows, "pct_vs_nn"))
            cfg_scores_sav[cfg].append(weighted_mean_pct(rows, "pct_vs_savings"))

    col_keys = sorted(groups.keys(), key=lambda t: (t[0], t[1]))
    means_nn = [float(np.nanmean(cfg_scores_nn[c])) for c in col_keys]
    stds_nn = [float(np.nanstd(cfg_scores_nn[c], ddof=1)) if len(cfg_scores_nn[c]) > 1 else 0.0 for c in col_keys]
    means_s = [float(np.nanmean(cfg_scores_sav[c])) for c in col_keys]
    stds_s = [float(np.nanstd(cfg_scores_sav[c], ddof=1)) if len(cfg_scores_sav[c]) > 1 else 0.0 for c in col_keys]

    x = np.arange(len(col_keys))
    labels = [cfg_label(c) for c in col_keys]

    # 1) Summary: weighted mean pct_vs_nn and pct_vs_savings across problem types
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax0, ax1 = axes
    ax0.bar(x, means_nn, yerr=stds_nn, capsize=4, color="steelblue", edgecolor="navy", alpha=0.85)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=9)
    ax0.set_ylabel("Instance-weighted mean % vs NN (↑ better)")
    ax0.set_title("GP vs Nearest Neighbor (5 runs per config)")
    ax0.axhline(0, color="gray", linewidth=0.8)
    ax0.grid(axis="y", alpha=0.35)

    ax1.bar(x, means_s, yerr=stds_s, capsize=4, color="seagreen", edgecolor="darkgreen", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Instance-weighted mean % vs Savings (↑ better)")
    ax1.set_title("GP vs Savings heuristic (5 runs per config)")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.grid(axis="y", alpha=0.35)

    fig.suptitle("Experiment comparison (population × generations)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p_summary = out_dir / "summary_pct_vs_baselines.png"
    fig.savefig(p_summary, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {p_summary}")

    # 2) Boxplot: distribution of per-run scores
    fig, ax = plt.subplots(figsize=(9, 5))
    data_nn = [cfg_scores_nn[c] for c in col_keys]
    bp = ax.boxplot(data_nn, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Weighted mean % vs NN (per run)")
    ax.set_title("Spread across 5 runs — GP vs NN")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    p_box = out_dir / "boxplot_pct_vs_nn.png"
    fig.savefig(p_box, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {p_box}")

    # 3) Heatmap: mean pct_vs_nn per problem variant × config
    row_keys, h_col_keys, mat = build_heatmap_matrix(groups)
    fig, ax = plt.subplots(figsize=(max(8, len(col_keys) * 1.2), 10))
    valid = mat[np.isfinite(mat)]
    if valid.size:
        lo, hi = float(np.nanpercentile(valid, 5)), float(np.nanpercentile(valid, 95))
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
    else:
        lo, hi = 0.0, 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=lo, vmax=hi)
    ax.set_xticks(np.arange(len(h_col_keys)))
    ax.set_xticklabels([f"p{p}\ng{g}" for p, g in h_col_keys], fontsize=8)
    ax.set_yticks(np.arange(len(row_keys)))
    ax.set_yticklabels([f"{pt} ({cap})" for pt, cap in row_keys], fontsize=7)
    ax.set_xlabel("Experiment (population / generations)")
    ax.set_ylabel("Problem variant (type, capacity constraint)")
    ax.set_title("Mean % vs NN across 5 runs (per cell)")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="% vs NN")
    fig.tight_layout()
    p_hm = out_dir / "heatmap_pct_vs_nn.png"
    fig.savefig(p_hm, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {p_hm}")

    # 4) Simple ranking table printed
    rank_nn = sorted(
        zip(col_keys, means_nn, stds_nn),
        key=lambda t: t[1],
        reverse=True,
    )
    print("\nRanking by mean weighted % vs NN (higher is better):")
    for i, (cfg, m, s) in enumerate(rank_nn, 1):
        print(f"  {i}. p{cfg[0]} g{cfg[1]}: {m:+.3f}% ± {s:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment CSV summaries from exp_results/")
    parser.add_argument(
        "--exp_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "exp_results",
        help="Directory containing exp_p*_g*_*.csv files",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for PNG figures (default: <exp_dir>/figures)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.exp_dir / "figures")
    run(args.exp_dir, out_dir, args.dpi)


if __name__ == "__main__":
    main()
