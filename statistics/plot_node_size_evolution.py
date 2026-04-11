#!/usr/bin/env python3
"""
Plot GP tree size evolution from node_size experiment JSONL files.

Reads experiments/node_size/exp_{VARIANT}_{run}.jsonl (e.g. exp_A_1.jsonl).
Uses log_evolution[*].size_avg (and optional min/max band) per generation.

One combined figure for all depth configurations, plus one figure per mutation-depth
group from notes.txt. Legends show only initial / mutation depth ranges (no A–I letters).
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from node_size_labels import node_depth_label, node_depth_sort_tuple

# Smaller graphs: group by mutation tree depth range
MUTATION_GROUPS: dict[str, list[str]] = {
    "mutation depth 0–1": ["A", "B", "D"],
    "mutation depth 1–5": ["C", "E", "F"],
    "mutation depth 0–3": ["G", "H", "I"],
}

FNAME_RE = re.compile(r"^exp_([A-Za-z])_(\d+)\.jsonl$", re.IGNORECASE)


def parse_jsonl_name(path: Path) -> tuple[str, int] | None:
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2))


def load_all_curves(
    exp_dir: Path,
) -> dict[str, list[tuple[list[int], list[float]]]]:
    """
    For each variant letter, list of (gens, size_avg) per JSONL line
    (one problem-type experiment).
    """
    by_variant: dict[str, list[tuple[list[int], list[float]]]] = defaultdict(list)

    for path in sorted(exp_dir.glob("exp_*_*.jsonl")):
        parsed = parse_jsonl_name(path)
        if parsed is None:
            continue
        letter, _run = parsed
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                log = rec.get("log_evolution") or []
                if not log:
                    continue
                gens = [int(e["gen"]) for e in log]
                avg = [float(e["size_avg"]) for e in log]
                by_variant[letter].append((gens, avg))

    return dict(by_variant)


def aggregate_by_generation(
    curves: list[tuple[list[int], list[float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For all curves (possibly different lengths), at each generation g
    mean and sample std of size_avg across curves that have that gen.
    """
    gen_to_avgs: dict[int, list[float]] = defaultdict(list)

    for gens, avg in curves:
        for g, a in zip(gens, avg):
            gen_to_avgs[g].append(a)

    all_gens = sorted(gen_to_avgs.keys())
    if not all_gens:
        return np.array([]), np.array([]), np.array([])

    g_arr = np.array(all_gens, dtype=int)
    mean_avg = np.array([float(np.mean(gen_to_avgs[g])) for g in all_gens])
    std_avg = np.array(
        [float(np.std(gen_to_avgs[g], ddof=1)) if len(gen_to_avgs[g]) > 1 else 0.0 for g in all_gens]
    )
    return g_arr, mean_avg, std_avg


def plot_variant_curves(
    ax: Any,
    letters: list[str],
    by_variant: dict[str, list[tuple[list[int], list[float]]]],
    *,
    title: str,
    cmap_name: str = "tab10",
) -> None:
    cmap = plt.get_cmap(cmap_name)
    for i, letter in enumerate(letters):
        curves = by_variant.get(letter, [])
        if not curves:
            continue
        g, m, s = aggregate_by_generation(curves)
        if g.size == 0:
            continue
        ncolors = getattr(cmap, "N", 10)
        color = cmap(i % ncolors)
        lab = node_depth_label(letter)
        ax.plot(g, m, label=lab, color=color, linewidth=1.8)
        ax.fill_between(g, m - s, m + s, color=color, alpha=0.15)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Tree size (nodes)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def run(exp_dir: Path, out_dir: Path, dpi: int) -> None:
    by_variant = load_all_curves(exp_dir)
    if not by_variant:
        raise SystemExit(f"No exp_*_*.jsonl files found in {exp_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    letters_all = sorted(by_variant.keys(), key=node_depth_sort_tuple)

    # --- One big figure: all variants ---
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_variant_curves(
        ax,
        letters_all,
        by_variant,
        title="Mean population tree size (size_avg) vs generation — all depth configs",
    )
    fig.tight_layout()
    p_all = out_dir / "node_size_evolution_all_variants.png"
    fig.savefig(p_all, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {p_all}")

    # --- One figure per mutation group ---
    for group_name, letters in MUTATION_GROUPS.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_variant_curves(
            ax,
            sorted(letters, key=node_depth_sort_tuple),
            by_variant,
            title=None,
        )
        fig.tight_layout()
        safe = group_name.replace(" ", "_").replace("–", "-")
        p_g = out_dir / f"node_size_evolution_{safe}.png"
        fig.savefig(p_g, dpi=dpi)
        plt.close(fig)
        print(f"Wrote {p_g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tree size evolution from node_size JSONL experiments.")
    parser.add_argument(
        "--exp_dir",
        type=Path,
        default=_REPO_ROOT / "experiments" / "node_size",
        help="Directory with exp_A_1.jsonl, …",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: <exp_dir>/figures)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.exp_dir / "figures")
    run(args.exp_dir, out_dir, args.dpi)


if __name__ == "__main__":
    main()
