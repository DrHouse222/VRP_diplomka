#!/usr/bin/env python3
"""
Build comparison plots from experiment CSV files in exp_results/.

Supported filename patterns:
- exp_p{pop}_g{gens}_{run}.csv
- exp_cx{cx}_mu{mu}_{run}.csv
- exp_t{tournament_size}_{run}.csv
- exp_train{n_train}_{run}.csv
- exp_{set_label}_{run}.csv (set_size): e.g. exp_Full_1.csv, exp_Refined_2.csv, exp_Atomic_3.csv
- exp_{label}_{run}.csv (node_size): exp_A_1.csv, exp__A_1.csv (label is one letter A–Z after all underscores)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from node_size_labels import node_depth_label, node_depth_sort_mutation_first, node_depth_sort_tuple

FNAME_PG_RE = re.compile(r"^exp_p(\d+)_g(\d+)_(\d+)\.csv$", re.IGNORECASE)
FNAME_CXMU_RE = re.compile(r"^exp_cx(\d+)_mu(\d+)_(\d+)\.csv$", re.IGNORECASE)
FNAME_T_RE = re.compile(r"^exp_t(\d+)_(\d+)\.csv$", re.IGNORECASE)
FNAME_TRAIN_RE = re.compile(r"^exp_train(\d+)_(\d+)\.csv$", re.IGNORECASE)
FNAME_SET_RE = re.compile(r"^exp_([A-Za-z][A-Za-z0-9]*)_(\d+)\.csv$", re.IGNORECASE)
FNAME_NODES_RE = re.compile(r"^exp_+([A-Za-z])_(\d+)\.csv$", re.IGNORECASE)

ConfigKey = tuple[str, int | str, int]

def _infer_tw_green_md(problem_type: str) -> tuple[bool, bool, bool]:
    pt = problem_type.strip()
    if pt in ("VRP", "CVRP"):
        return (False, False, False)
    if pt == "VRPTW":
        return (True, False, False)
    if pt == "GVRP":
        return (False, True, False)
    if pt == "G-VRPTW":
        return (True, True, False)
    if pt in ("MDVRP", "MDCVRP"):
        return (False, False, True)
    if pt == "MDVRPTW":
        return (True, False, True)
    if pt == "GVRP-MD":
        return (False, True, True)
    if pt == "G-VRPTW-MD":
        return (True, True, True)
    return (False, False, False)


BOOL_FLAG_ROWS: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
    ("bool_capacity=True", lambda r: str(r.get("bool_capacity", "")).lower() == "true"),
    ("bool_TW=True", lambda r: _infer_tw_green_md(str(r.get("problem_type", "")))[0]),
    ("bool_green=True", lambda r: _infer_tw_green_md(str(r.get("problem_type", "")))[1]),
    ("bool_MD=True", lambda r: _infer_tw_green_md(str(r.get("problem_type", "")))[2]),
]


def parse_filename(path: Path) -> tuple[str, int | str, int, int] | None:
    """Return (mode, a, b, run). Mode is 'pg', 'cxmu', 't', 'train', 'set', or 'nodes'."""
    m = FNAME_PG_RE.match(path.name)
    if m:
        return ("pg", int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = FNAME_CXMU_RE.match(path.name)
    if m:
        return ("cxmu", int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = FNAME_T_RE.match(path.name)
    if m:
        return ("t", int(m.group(1)), 0, int(m.group(2)))
    m = FNAME_TRAIN_RE.match(path.name)
    if m:
        return ("train", int(m.group(1)), 0, int(m.group(2)))
    m = FNAME_NODES_RE.match(path.name)
    if m:
        return ("nodes", m.group(1), 0, int(m.group(2)))
    m = FNAME_SET_RE.match(path.name)
    if m:
        return ("set", m.group(1).lower(), 0, int(m.group(2)))
    return None


def cfg_sort_key(cfg: ConfigKey) -> tuple:
    """Stable sort for bar order / heatmap columns."""
    mode, a, b = cfg
    if mode == "nodes":
        return (mode, *node_depth_sort_tuple(str(a)))
    if mode == "set":
        order = {
            "full": 0,
            "reduced": 1,
            "minimal": 2,
            "refined": 3,
            "core": 4,
            "atomic": 5,
        }
        aa = str(a).lower()
        return (mode, order.get(aa, 999), aa)
    return (mode, int(a), int(b))


def cfg_sort_key_boxplot(cfg: ConfigKey) -> tuple:
    """Column order for boxplots; node_size configs sorted by mutation depth range first."""
    mode, a, b = cfg
    if mode == "nodes":
        return (mode, *node_depth_sort_mutation_first(str(a)))
    return cfg_sort_key(cfg)


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


def simple_mean_pct(rows: list[dict[str, Any]], col: str) -> float:
    """Unweighted arithmetic mean of a percentage column across rows (each row counts once)."""
    vals = [_float(row.get(col)) for row in rows]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (row["problem_type"], row["bool_capacity"])


def discover_groups(exp_dir: Path) -> dict[ConfigKey, list[Path]]:
    """Map config key -> list of CSV paths (all runs)."""
    groups: dict[ConfigKey, list[Path]] = defaultdict(list)
    for path in sorted(exp_dir.glob("*.csv")):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        mode, a, b, _run = parsed
        groups[(mode, a, b)].append(path)
    for key in groups:
        groups[key].sort(key=lambda p: parse_filename(p)[3])  # type: ignore[index]
    return dict(groups)


def build_heatmap_matrix(
    groups: dict[ConfigKey, list[Path]],
) -> tuple[list[tuple[str, str]], list[ConfigKey], np.ndarray]:
    """Mean pct_vs_nn per (problem_type, capacity) × config: each cell is the mean
    over runs of that row's pct_vs_nn (no n_instances weighting; one CSV row per cell per run)."""
    col_keys = sorted(groups.keys(), key=cfg_sort_key)
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


def build_bool_flags_heatmap_matrix(
    groups: dict[ConfigKey, list[Path]],
) -> tuple[list[str], list[ConfigKey], np.ndarray]:
    """
    Same columns as the full heatmap, but 4 rows: one per experiment bool
    (capacity from CSV; TW / green / MD inferred from problem_type).
    Each cell: mean over runs of the unweighted arithmetic mean of pct_vs_nn
    over CSV rows matching that flag (each row counts equally).
    """
    col_keys = sorted(groups.keys(), key=cfg_sort_key)
    row_labels = [
        "Capacity",
        "Time windows",
        "Green",
        "Multi-depot",
    ]
    mat = np.full((len(BOOL_FLAG_ROWS), len(col_keys)), np.nan, dtype=float)

    for j, cfg in enumerate(col_keys):
        paths = groups[cfg]
        for i, (_, pred) in enumerate(BOOL_FLAG_ROWS):
            run_scores: list[float] = []
            for path in paths:
                rows = load_csv_rows(path)
                subset = [r for r in rows if pred(r)]
                if not subset:
                    continue
                u = simple_mean_pct(subset, "pct_vs_nn")
                if not np.isnan(u):
                    run_scores.append(u)
            if run_scores:
                mat[i, j] = float(np.mean(run_scores))

    return row_labels, col_keys, mat


def cfg_label(cfg: ConfigKey) -> str:
    mode, a, b = cfg
    if mode == "pg":
        return f"p{a}\ng{b}"
    if mode == "t":
        return f"t{a}"
    if mode == "train":
        return f"train {a}"
    if mode == "set":
        return str(a)
    if mode == "nodes":
        return node_depth_label(str(a))
    return f"cx{a}\nmu{b}"


def cfg_axis_title(mode: str) -> str:
    if mode == "pg":
        return "population / generations"
    if mode == "t":
        return "tournament size"
    if mode == "train":
        return "training-set size"
    if mode == "set":
        return "feature set size"
    if mode == "nodes":
        return "initial / mutation tree depth"
    return "crossover % / mutation %"


def save_pct_nn_boxplot(
    data_nn: list[list[float]],
    labels: list[str],
    title: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data_nn, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    flat = np.concatenate(
        [np.asarray(s, dtype=float) for s in data_nn if s],
        dtype=float,
    )
    if flat.size:
        lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        pad = max(span * 0.06, 0.25)
        ax.set_ylim(lo - pad, hi + pad)
        if lo - pad <= 0 <= hi + pad:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    else:
        ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def run(exp_dir: Path, out_dir: Path, dpi: int) -> None:
    groups = discover_groups(exp_dir)
    if not groups:
        raise SystemExit(
            f"No supported CSV names found in {exp_dir}. "
            "Use exp_p*_g*_*.csv, exp_cx*_mu*_*.csv, exp_t*_*.csv, exp_train*_*.csv, "
            "exp_{setLabel}_*.csv, or exp_*_<letter>_<run>.csv."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_scores_nn: dict[ConfigKey, list[float]] = defaultdict(list)
    cfg_scores_nn_unweighted: dict[ConfigKey, list[float]] = defaultdict(list)
    cfg_scores_sav: dict[ConfigKey, list[float]] = defaultdict(list)

    for cfg, paths in sorted(groups.items(), key=lambda kv: cfg_sort_key(kv[0])):
        for path in paths:
            rows = load_csv_rows(path)
            cfg_scores_nn[cfg].append(weighted_mean_pct(rows, "pct_vs_nn"))
            cfg_scores_nn_unweighted[cfg].append(simple_mean_pct(rows, "pct_vs_nn"))
            cfg_scores_sav[cfg].append(weighted_mean_pct(rows, "pct_vs_savings"))

    col_keys = sorted(groups.keys(), key=cfg_sort_key)
    mode = col_keys[0][0]
    col_keys_box = sorted(groups.keys(), key=cfg_sort_key_boxplot) if mode == "nodes" else col_keys
    means_nn = [float(np.nanmean(cfg_scores_nn[c])) for c in col_keys]
    stds_nn = [float(np.nanstd(cfg_scores_nn[c], ddof=1)) if len(cfg_scores_nn[c]) > 1 else 0.0 for c in col_keys]
    means_nn_uw = [float(np.nanmean(cfg_scores_nn_unweighted[c])) for c in col_keys]
    stds_nn_uw = [
        float(np.nanstd(cfg_scores_nn_unweighted[c], ddof=1))
        if len(cfg_scores_nn_unweighted[c]) > 1
        else 0.0
        for c in col_keys
    ]
    means_s = [float(np.nanmean(cfg_scores_sav[c])) for c in col_keys]
    stds_s = [float(np.nanstd(cfg_scores_sav[c], ddof=1)) if len(cfg_scores_sav[c]) > 1 else 0.0 for c in col_keys]

    x = np.arange(len(col_keys))
    labels = [cfg_label(c) for c in col_keys]
    labels_box = [cfg_label(c) for c in col_keys_box]

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

    fig.suptitle("Experiment comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p_summary = out_dir / "summary_pct_vs_baselines.pdf"
    fig.savefig(p_summary, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {p_summary}")

    data_nn = [cfg_scores_nn[c] for c in col_keys_box]
    p_box = out_dir / "boxplot_pct_vs_nn.pdf"
    save_pct_nn_boxplot(
        data_nn,
        labels_box,
        "Spread across runs — GP vs NN",
        "Weighted mean % vs NN (per run)",
        p_box,
        dpi,
    )
    print(f"Wrote {p_box}")

    data_nn_uw = [cfg_scores_nn_unweighted[c] for c in col_keys_box]
    p_box_uw = out_dir / "boxplot_pct_vs_nn_unweighted.pdf"
    save_pct_nn_boxplot(
        data_nn_uw,
        labels_box,
        "Spread across runs — GP vs NN",
        "Mean % vs NN (per run)",
        p_box_uw,
        dpi,
    )
    print(f"Wrote {p_box_uw}")

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
    ax.set_xticklabels([cfg_label(c) for c in h_col_keys], fontsize=8)
    ax.set_yticks(np.arange(len(row_keys)))
    ax.set_yticklabels([f"{pt} ({cap})" for pt, cap in row_keys], fontsize=7)
    ax.set_ylabel("Problem variant (type, capacity constraint)")
    ax.set_title("Mean % vs NN")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="% vs NN")
    fig.tight_layout()
    p_hm = out_dir / "heatmap_pct_vs_nn.pdf"
    fig.savefig(p_hm, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {p_hm}")

    row_bool_labels, h_col_keys_bool, mat_bool = build_bool_flags_heatmap_matrix(groups)
    fig, ax = plt.subplots(figsize=(max(8, len(col_keys) * 1.2), 4.2))
    valid_b = mat_bool[np.isfinite(mat_bool)]
    if valid_b.size:
        lo_b, hi_b = float(np.nanpercentile(valid_b, 5)), float(np.nanpercentile(valid_b, 95))
        if lo_b == hi_b:
            lo_b, hi_b = lo_b - 1e-6, hi_b + 1e-6
    else:
        lo_b, hi_b = 0.0, 1.0
    im_b = ax.imshow(mat_bool, aspect="auto", cmap="RdYlGn", vmin=lo_b, vmax=hi_b)
    ax.set_xticks(np.arange(len(h_col_keys_bool)))
    ax.set_xticklabels([cfg_label(c) for c in h_col_keys_bool], fontsize=8)
    ax.set_yticks(np.arange(len(row_bool_labels)))
    ax.set_yticklabels(row_bool_labels, fontsize=9)
    plt.colorbar(im_b, ax=ax, fraction=0.02, pad=0.02, label="% vs NN")
    fig.tight_layout()
    p_hm_bool = out_dir / "heatmap_pct_vs_nn_by_bool_flags.pdf"
    fig.savefig(p_hm_bool, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {p_hm_bool}")

    rank_nn_uw = sorted(
        zip(col_keys, means_nn_uw, stds_nn_uw),
        key=lambda t: t[1],
        reverse=True,
    )
    print("\nRanking by unweighted mean % vs NN (higher is better):")
    for i, (cfg, m_uw, s_uw) in enumerate(rank_nn_uw, 1):
        lab = cfg_label(cfg).replace("\n", " ")
        print(
            f"  {i}. {lab}:  unweighted {m_uw:+.3f}% ± {s_uw:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Plot experiment CSV summaries from exp_results/")
    parser.add_argument(
        "--exp_dir",
        type=Path,
        default=_REPO_ROOT / "exp_results",
        help=(
            "Directory containing exp_p*_g*_*.csv, exp_cx*_mu*_*.csv, exp_t*_*.csv, "
            "exp_train*_*.csv, exp_{setLabel}_*.csv, or node_size exp_*_<letter>_* files"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for PDF figures (default: <exp_dir>/figures)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.exp_dir / "figures")
    run(args.exp_dir, out_dir, args.dpi)


if __name__ == "__main__":
    main()
