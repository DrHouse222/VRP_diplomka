#!/usr/bin/env python3
"""
Plot GP fitness convergence from experiment JSONL files, grouped by setup.

Uses the same filename conventions as plot_exp_results.py (exp_p*_g*_*.jsonl,
exp_cx*_mu*_*.jsonl, exp_t*_*.jsonl, exp_train*_*.jsonl, exp_{set}_*.jsonl,
exp_*_<letter>_<run>.jsonl for node_size). For each problem variant
(problem_type, bool_capacity) and each setup (config), aggregates log_evolution
series across runs (mean ± std over runs at each generation).

Figure where every variant line from every run is pooled, the x-axis is fitness evaluations approximated as
population_size * n_train * generation, and a 95% confidence band (mean ± 1.96 * SEM).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from node_size_labels import node_depth_label, node_depth_sort_mutation_first, node_depth_sort_tuple

# Same patterns as plot_exp_results.parse_filename but for .jsonl
FNAME_PG_RE = re.compile(r"^exp_p(\d+)_g(\d+)_(\d+)\.jsonl$", re.IGNORECASE)
FNAME_CXMU_RE = re.compile(r"^exp_cx(\d+)_mu(\d+)_(\d+)\.jsonl$", re.IGNORECASE)
FNAME_T_RE = re.compile(r"^exp_t(\d+)_(\d+)\.jsonl$", re.IGNORECASE)
FNAME_TRAIN_RE = re.compile(r"^exp_train(\d+)_(\d+)\.jsonl$", re.IGNORECASE)
FNAME_SET_RE = re.compile(r"^exp_([A-Za-z][A-Za-z0-9]*)_(\d+)\.jsonl$", re.IGNORECASE)
FNAME_NODES_RE = re.compile(r"^exp_+([A-Za-z])_(\d+)\.jsonl$", re.IGNORECASE)

ConfigKey = tuple[str, int | str, int]


def parse_jsonl_filename(path: Path) -> tuple[str, int | str, int, int] | None:
    """Return (mode, a, b, run). Same semantics as plot_exp_results.parse_filename."""
    name = path.name
    m = FNAME_NODES_RE.match(name)
    if m:
        return ("nodes", m.group(1), 0, int(m.group(2)))
    m = FNAME_PG_RE.match(name)
    if m:
        return ("pg", int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = FNAME_CXMU_RE.match(name)
    if m:
        return ("cxmu", int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = FNAME_T_RE.match(name)
    if m:
        return ("t", int(m.group(1)), 0, int(m.group(2)))
    m = FNAME_TRAIN_RE.match(name)
    if m:
        return ("train", int(m.group(1)), 0, int(m.group(2)))
    m = FNAME_SET_RE.match(name)
    if m:
        return ("set", m.group(1).lower(), 0, int(m.group(2)))
    return None


def cfg_sort_key(cfg: ConfigKey) -> tuple:
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


def cfg_sort_key_line_order(cfg: ConfigKey) -> tuple:
    """Node_size lines ordered like boxplots (mutation depth first)."""
    mode, a, b = cfg
    if mode == "nodes":
        return (mode, *node_depth_sort_mutation_first(str(a)))
    return cfg_sort_key(cfg)


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


def discover_jsonl_groups(exp_dir: Path) -> dict[ConfigKey, list[Path]]:
    groups: dict[ConfigKey, list[Path]] = defaultdict(list)
    for path in sorted(exp_dir.glob("*.jsonl")):
        parsed = parse_jsonl_filename(path)
        if parsed is None:
            continue
        mode, a, b, _run = parsed
        groups[(mode, a, b)].append(path)
    for key in groups:
        groups[key].sort(key=lambda p: parse_jsonl_filename(p)[3])  # type: ignore[index]
    return dict(groups)


def iter_jsonl_records(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def record_row_key(rec: dict[str, Any]) -> tuple[str, bool]:
    return (str(rec.get("problem_type", "")), bool(rec.get("bool_capacity", False)))


def collect_variant_keys(groups: dict[ConfigKey, list[Path]]) -> list[tuple[str, bool]]:
    keys: set[tuple[str, bool]] = set()
    for paths in groups.values():
        for path in paths:
            for rec in iter_jsonl_records(path):
                keys.add(record_row_key(rec))
    return sorted(keys, key=lambda t: (t[0], t[1]))


def curve_from_record(rec: dict[str, Any], metric: str) -> tuple[list[int], list[float]] | None:
    """metric: fitness_min, fitness_avg, fitness_max."""
    log = rec.get("log_evolution")
    if not log:
        return None
    if not all(metric in e for e in log):
        return None
    gens = [int(e["gen"]) for e in log]
    vals = [float(e[metric]) for e in log]
    return gens, vals


def curve_gen_eval_y(
    rec: dict[str, Any], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Same fitness series as curve_from_record, plus evaluation-count x-axis:
    evaluations ≈ population_size × n_train × generation (per training snapshot).
    """
    log = rec.get("log_evolution")
    if not log or not isinstance(log, list):
        return None
    if not all(metric in e for e in log):
        return None
    pop = int(rec.get("population_size") or 0)
    if pop <= 0:
        return None
    n_train = int(rec.get("n_train", 5) or 5)
    if n_train <= 0:
        n_train = 5
    gens = np.array([int(e["gen"]) for e in log], dtype=int)
    y = np.array([float(e[metric]) for e in log], dtype=float)
    eval_x = pop * n_train * gens.astype(float)
    return gens, eval_x, y


def aggregate_pooled_by_generation(
    curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pool curves aligned by generation index.

    Returns
    -------
    g_arr, x_mean, y_mean, ci_lo, ci_hi
        x_mean: mean of (population × n_train × gen) at each generation across curves
        ci_lo, ci_hi: approximate 95% CI on the mean of y (mean ± 1.96 × SEM)
    """
    gen_to_ys: dict[int, list[float]] = defaultdict(list)
    gen_to_xs: dict[int, list[float]] = defaultdict(list)
    for gens, eval_x, y in curves:
        for g, xe, ye in zip(gens.tolist(), eval_x.tolist(), y.tolist()):
            gen_to_ys[g].append(float(ye))
            gen_to_xs[g].append(float(xe))
    all_gens = sorted(gen_to_ys.keys())
    if not all_gens:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    g_arr = np.array(all_gens, dtype=int)
    y_mean = np.zeros(len(all_gens), dtype=float)
    x_mean = np.zeros(len(all_gens), dtype=float)
    ci_lo = np.zeros(len(all_gens), dtype=float)
    ci_hi = np.zeros(len(all_gens), dtype=float)
    for i, g in enumerate(all_gens):
        vals = gen_to_ys[g]
        xs = gen_to_xs[g]
        x_mean[i] = float(np.mean(xs))
        y_mean[i] = float(np.mean(vals))
        n = len(vals)
        if n > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(n))
        else:
            sem = 0.0
        delta = 1.96 * sem
        ci_lo[i] = y_mean[i] - delta
        ci_hi[i] = y_mean[i] + delta
    return g_arr, x_mean, y_mean, ci_lo, ci_hi


def aggregate_by_generation(curves: list[tuple[list[int], list[float]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen_to_vals: dict[int, list[float]] = defaultdict(list)
    for gens, ys in curves:
        for g, y in zip(gens, ys):
            gen_to_vals[g].append(y)
    all_gens = sorted(gen_to_vals.keys())
    if not all_gens:
        return np.array([]), np.array([]), np.array([])
    g_arr = np.array(all_gens, dtype=int)
    mean_y = np.array([float(np.mean(gen_to_vals[g])) for g in all_gens])
    std_y = np.array(
        [float(np.std(gen_to_vals[g], ddof=1)) if len(gen_to_vals[g]) > 1 else 0.0 for g in all_gens]
    )
    return g_arr, mean_y, std_y


def curves_for_variant_config(
    paths: list[Path],
    variant: tuple[str, bool],
    metric: str,
) -> list[tuple[list[int], list[float]]]:
    out: list[tuple[list[int], list[float]]] = []
    for path in paths:
        for rec in iter_jsonl_records(path):
            if record_row_key(rec) != variant:
                continue
            c = curve_from_record(rec, metric)
            if c is not None:
                out.append(c)
            break
    return out


def curves_all_variants_for_config(paths: list[Path], metric: str) -> list[tuple[list[int], list[float]]]:
    """Every JSONL line in every run file (all problem_type × bool_capacity)."""
    out: list[tuple[list[int], list[float]]] = []
    for path in paths:
        for rec in iter_jsonl_records(path):
            c = curve_from_record(rec, metric)
            if c is not None:
                out.append(c)
    return out


def curves_all_variants_eval_curves(
    paths: list[Path], metric: str
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(gens, eval_x, y) for every JSONL record (all variants × runs)."""
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for path in paths:
        for rec in iter_jsonl_records(path):
            t = curve_gen_eval_y(rec, metric)
            if t is not None:
                out.append(t)
    return out


def run(
    exp_dir: Path,
    out_dir: Path,
    metric: str,
    dpi: int,
    *,
    write_combined: bool,
) -> None:
    groups = discover_jsonl_groups(exp_dir)
    if not groups:
        raise SystemExit(
            f"No supported exp_*.jsonl names in {exp_dir}. "
            "Same patterns as plot_exp_results (.csv swapped for .jsonl)."
        )

    variants = collect_variant_keys(groups)
    if not variants:
        raise SystemExit(f"No readable records with problem_type / bool_capacity in {exp_dir}")

    col_keys = sorted(groups.keys(), key=cfg_sort_key_line_order)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = exp_dir.resolve().name
    pdf_path = out_dir / f"{base}_convergence_{metric}.pdf"

    ncol = len(col_keys)

    with PdfPages(pdf_path) as pdf:
        for variant in variants:
            fig, ax = plt.subplots(figsize=(9, 5))

            for j, cfg in enumerate(col_keys):
                curves = curves_for_variant_config(groups[cfg], variant, metric)
                if not curves:
                    continue
                g_arr, mean_y, std_y = aggregate_by_generation(curves)
                if g_arr.size == 0:
                    continue
                color = f"C{j % 10}"
                label = cfg_label(cfg).replace("\n", " ")
                ax.plot(g_arr, mean_y, label=label, color=color, linewidth=1.4)
                lo = mean_y - std_y
                hi = mean_y + std_y
                ax.fill_between(g_arr, lo, hi, color=color, alpha=0.12)

            ax.set_xlabel("Generation")
            ax.set_ylabel(metric)
            ax.grid(True, axis="both", alpha=0.35)
            if ax.lines:
                ncol_leg = min(4, len(ax.lines)) if ncol else 1
                ax.legend(fontsize=7, ncol=ncol_leg, loc="upper right", framealpha=0.9)
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    print(f"Wrote {pdf_path} ({len(variants)} variant pages × {ncol} setups)")

    if write_combined:
        pdf_combined = out_dir / f"{base}_convergence_{metric}_all_variants.pdf"
        fig, ax = plt.subplots(figsize=(9, 5))
        for j, cfg in enumerate(col_keys):
            eval_curves = curves_all_variants_eval_curves(groups[cfg], metric)
            if not eval_curves:
                continue
            _g_arr, x_mean, y_mean, ci_lo, ci_hi = aggregate_pooled_by_generation(eval_curves)
            if x_mean.size == 0:
                continue
            color = f"C{j % 10}"
            label = cfg_label(cfg).replace("\n", " ")
            ax.fill_between(
                x_mean,
                ci_lo,
                ci_hi,
                color=color,
                alpha=0.05,
                linewidth=0,
                zorder=1,
            )
            ax.plot(x_mean, y_mean, label=label, color=color, linewidth=1.4, zorder=2)

        ax.set_xlabel("Fitness evaluations (population × n_train × generation)")
        ax.set_ylabel(metric)
        ax.grid(True, axis="both", alpha=0.35)
        if ax.lines:
            ncol_leg = min(4, len(ax.lines)) if ncol else 1
            ax.legend(fontsize=7, ncol=ncol_leg, loc="upper right", framealpha=0.9)
        ax.set_ylim(top=12000)
        fig.tight_layout()
        fig.savefig(pdf_combined, dpi=dpi)
        plt.close(fig)
        n_runs = len(next(iter(groups.values()))) if groups else 0
        print(
            f"Wrote {pdf_combined} (one page, each setup pooled over "
            f"{len(variants)} variants × {n_runs} run files)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot GP fitness convergence curves from experiment JSONL files (grouped like plot_exp_results)."
    )
    parser.add_argument(
        "--exp_dir",
        type=Path,
        required=True,
        help="Directory with exp_*_*.jsonl (same naming as CSV experiment dirs)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: <exp_dir>/figures)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="fitness_min",
        choices=["fitness_min", "fitness_avg", "fitness_max"],
        help="log_evolution field to plot (default: fitness_min)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--no-combined-figure",
        action="store_true",
        help="Skip the single-page PDF that pools every variant (all problem_type × capacity lines).",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or (args.exp_dir / "figures")
    run(
        args.exp_dir.resolve(),
        out_dir.resolve(),
        args.metric,
        args.dpi,
        write_combined=not args.no_combined_figure,
    )


if __name__ == "__main__":
    main()
