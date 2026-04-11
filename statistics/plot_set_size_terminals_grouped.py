#!/usr/bin/env python3
"""
Grouped terminal-usage histogram for set_size experiments.

Reads best_expr from any `exp_<variant>_<run>.jsonl` files in set_size.
Outputs one grouped bar chart with one bar per variant for each terminal.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FILE_RE = re.compile(r"^exp_([A-Za-z][A-Za-z0-9]*)_(\d+)\.jsonl$", re.IGNORECASE)
FUNC_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")

def variant_sort_key(v: str) -> tuple[int, str]:
    order = {
        "full": 0,
        "reduced": 1,
        "minimal": 2,
        "refined": 3,
        "core": 4,
        "atomic": 5,
    }
    vv = v.lower()
    return (order.get(vv, 999), vv)


def collect_terminals_by_variant(exp_dir: Path) -> dict[str, Counter[str]]:
    counts_by_variant: dict[str, Counter[str]] = {}

    for path in sorted(exp_dir.glob("exp_*_*.jsonl")):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        variant = m.group(1)
        if variant not in counts_by_variant:
            counts_by_variant[variant] = Counter()

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                expr = rec.get("best_expr")
                if not isinstance(expr, str) or not expr.strip():
                    continue

                funcs = set(FUNC_RE.findall(expr))
                terms = [name for name in IDENT_RE.findall(expr) if name not in funcs]
                counts_by_variant[variant].update(terms)

    return counts_by_variant


def run(exp_dir: Path, out_path: Path, top_n: int) -> None:
    counts_by_variant = collect_terminals_by_variant(exp_dir)
    variants = sorted(counts_by_variant.keys(), key=variant_sort_key)
    if not variants or all(not counts_by_variant[v] for v in variants):
        raise SystemExit(f"No set_size terminal data found in {exp_dir}")

    total = Counter()
    for v in variants:
        total.update(counts_by_variant[v])
    terminals = [t for t, _ in total.most_common(top_n)]

    x = np.arange(len(terminals))
    n_variants = max(1, len(variants))
    width = 0.8 / n_variants
    offsets = {
        v: ((i - (n_variants - 1) / 2.0) * width)
        for i, v in enumerate(variants)
    }
    cmap = plt.get_cmap("tab10")
    colors = {v: cmap(i % getattr(cmap, "N", 10)) for i, v in enumerate(variants)}

    fig, ax = plt.subplots(figsize=(max(12, len(terminals) * 0.75), 5.6))
    for v in variants:
        y = [counts_by_variant[v].get(t, 0) for t in terminals]
        ax.bar(x + offsets[v], y, width=width, label=v, color=colors[v], alpha=0.9, edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(terminals, rotation=30, ha="right")
    ax.set_ylabel("Count")
    #ax.set_title("Terminal usage by set_size variant (best_expr)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Variant", loc="best")
    ax.margins(x=0.004)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_separate(exp_dir: Path, out_dir: Path, top_n: int) -> None:
    counts_by_variant = collect_terminals_by_variant(exp_dir)
    variants = sorted(counts_by_variant.keys(), key=variant_sort_key)
    if not variants or all(not counts_by_variant[v] for v in variants):
        raise SystemExit(f"No set_size terminal data found in {exp_dir}")

    cmap = plt.get_cmap("tab10")
    colors = {v: cmap(i % getattr(cmap, "N", 10)) for i, v in enumerate(variants)}
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in variants:
        counts = counts_by_variant[v]
        terms = [t for t, _ in counts.most_common(top_n)]
        vals = [counts[t] for t in terms]

        fig, ax = plt.subplots(figsize=(max(10, len(terms) * 0.7), 5.2))
        ax.bar(range(len(terms)), vals, color=colors[v], alpha=0.9, edgecolor="black", linewidth=0.3)
        ax.set_xticks(range(len(terms)))
        ax.set_xticklabels(terms, rotation=30, ha="right")
        ax.set_ylabel("Count")
        #ax.set_title(f"Terminal usage")
        ax.grid(axis="y", alpha=0.3)
        ax.margins(x=0.004)
        fig.tight_layout()

        out_path = out_dir / f"set_size_terminals_{v.lower()}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Grouped terminal histogram for experiments/set_size JSONL files.")
    parser.add_argument(
        "--exp_dir",
        type=Path,
        default=root / "experiments" / "set_size",
        help="Directory containing exp_<variant>_*.jsonl files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <exp_dir>/figures/set_size_terminals_grouped.png)",
    )
    parser.add_argument(
        "--out_dir_separate",
        type=Path,
        default=None,
        help="Output directory for separate variant plots (default: <exp_dir>/figures)",
    )
    parser.add_argument("--top_n", type=int, default=20, help="Top N terminals by total count")
    args = parser.parse_args()

    out = args.out or (args.exp_dir / "figures" / "set_size_terminals_grouped.png")
    run(args.exp_dir, out, args.top_n)
    out_dir_sep = args.out_dir_separate or (args.exp_dir / "figures")
    run_separate(args.exp_dir, out_dir_sep, args.top_n)


if __name__ == "__main__":
    main()

