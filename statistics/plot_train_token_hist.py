#!/usr/bin/env python3
"""
Build token-usage histograms from train_size JSONL experiments.

Default behavior uses exp_train10_1..5.jsonl and counts token occurrences in
the `best_expr` field across all records in those files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

FILE_RE = re.compile(r"^exp_train10_(\d+)\.jsonl$", re.IGNORECASE)
FILE_ALL_RE = re.compile(r"^exp_train\d+_(\d+)\.jsonl$", re.IGNORECASE)
FUNC_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")


def collect_expr_tokens(paths: list[Path]) -> tuple[Counter[str], Counter[str], Counter[str]]:
    fn_counts: Counter[str] = Counter()
    term_counts: Counter[str] = Counter()
    const_counts: Counter[str] = Counter()

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                expr = rec.get("best_expr")
                if not isinstance(expr, str) or not expr.strip():
                    continue

                funcs = FUNC_RE.findall(expr)
                fn_counts.update(funcs)

                consts = NUM_RE.findall(expr)
                const_counts.update(consts)

                func_set = set(funcs)
                terms = [name for name in IDENT_RE.findall(expr) if name not in func_set]
                term_counts.update(terms)

    return fn_counts, term_counts, const_counts


def _plot_counter(ax: plt.Axes, counts: Counter[str], title: str, top_n: int) -> None:
    items = counts.most_common(top_n)
    if not items:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    ax.bar(range(len(labels)), vals, color="steelblue", edgecolor="navy", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def run(exp_dir: Path, out_path: Path, top_n: int) -> None:
    paths = sorted([p for p in exp_dir.glob("exp_train10_*.jsonl") if FILE_RE.match(p.name)])
    if len(paths) != 5:
        print(f"Warning: expected 5 exp_train10 files, found {len(paths)}")
    if not paths:
        raise SystemExit(f"No exp_train10_*.jsonl files found in {exp_dir}")

    fn_counts, term_counts, const_counts = collect_expr_tokens(paths)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    _plot_counter(axes[0], fn_counts, "Functions used in best_expr", top_n)
    _plot_counter(axes[1], term_counts, "Terminals used in best_expr", top_n)
    _plot_counter(axes[2], const_counts, "Constants used in best_expr", top_n)
    fig.suptitle("Token usage across exp_train10_1..5 best individuals", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_terminals_only(exp_dir: Path, out_path: Path, top_n: int) -> None:
    paths = sorted([p for p in exp_dir.glob("exp_train10_*.jsonl") if FILE_RE.match(p.name)])
    if len(paths) != 5:
        print(f"Warning: expected 5 exp_train10 files, found {len(paths)}")
    if not paths:
        raise SystemExit(f"No exp_train10_*.jsonl files found in {exp_dir}")

    _fn_counts, term_counts, _const_counts = collect_expr_tokens(paths)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    _plot_counter(ax, term_counts, "Terminals used in best_expr", top_n)
    fig.suptitle("Terminal usage across exp_train10_1..5 best individuals", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_functions_all_files(exp_dir: Path, out_path: Path, top_n: int) -> None:
    paths = sorted([p for p in exp_dir.glob("exp_train*_*.jsonl") if FILE_ALL_RE.match(p.name)])
    if not paths:
        raise SystemExit(f"No exp_train*_*.jsonl files found in {exp_dir}")

    fn_counts, _term_counts, _const_counts = collect_expr_tokens(paths)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    _plot_counter(ax, fn_counts, "Functions used in best_expr", top_n)
    fig.suptitle("Function usage across all experiments/train_size files", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_terminals_all_files(exp_dir: Path, out_path: Path, top_n: int) -> None:
    paths = sorted([p for p in exp_dir.glob("exp_train*_*.jsonl") if FILE_ALL_RE.match(p.name)])
    if not paths:
        raise SystemExit(f"No exp_train*_*.jsonl files found in {exp_dir}")

    _fn_counts, term_counts, _const_counts = collect_expr_tokens(paths)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    _plot_counter(ax, term_counts, "Terminals used in best_expr", top_n)
    fig.suptitle("Terminal usage across all experiments/train_size files", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_all_files_combined(exp_dir: Path, out_path: Path, top_n: int) -> None:
    paths = sorted([p for p in exp_dir.glob("exp_train*_*.jsonl") if FILE_ALL_RE.match(p.name)])
    if not paths:
        raise SystemExit(f"No exp_train*_*.jsonl files found in {exp_dir}")

    fn_counts, term_counts, const_counts = collect_expr_tokens(paths)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    _plot_counter(axes[0], fn_counts, "Functions used in best_expr", top_n)
    _plot_counter(axes[1], term_counts, "Terminals used in best_expr", top_n)
    _plot_counter(axes[2], const_counts, "Constants used in best_expr", top_n)
    fig.suptitle("Token usage across all experiments/train_size files", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Plot histogram of function/terminal/constant usage from exp_train10 JSONL.")
    parser.add_argument(
        "--exp_dir",
        type=Path,
        default=root / "experiments" / "train_size",
        help="Directory containing exp_train10_*.jsonl files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <exp_dir>/figures/train10_token_hist.png)",
    )
    parser.add_argument(
        "--out_terminals",
        type=Path,
        default=None,
        help="Optional output path for terminals-only histogram (default: <exp_dir>/figures/train10_terminals_hist.png)",
    )
    parser.add_argument(
        "--out_functions_all",
        type=Path,
        default=None,
        help="Optional output path for all-files function histogram (default: <exp_dir>/figures/train_all_functions_hist.png)",
    )
    parser.add_argument(
        "--out_terminals_all",
        type=Path,
        default=None,
        help="Optional output path for all-files terminal histogram (default: <exp_dir>/figures/train_all_terminals_hist.png)",
    )
    parser.add_argument(
        "--out_all_combined",
        type=Path,
        default=None,
        help="Optional output path for all-files combined histogram (default: <exp_dir>/figures/train_all_token_hist.png)",
    )
    parser.add_argument("--top_n", type=int, default=20, help="Top N tokens shown in each subplot")
    args = parser.parse_args()

    out = args.out or (args.exp_dir / "figures" / "train10_token_hist.png")
    run(args.exp_dir, out, args.top_n)
    out_t = args.out_terminals or (args.exp_dir / "figures" / "train10_terminals_hist.png")
    run_terminals_only(args.exp_dir, out_t, args.top_n)
    out_fa = args.out_functions_all or (args.exp_dir / "figures" / "train_all_functions_hist.png")
    run_functions_all_files(args.exp_dir, out_fa, args.top_n)
    out_ta = args.out_terminals_all or (args.exp_dir / "figures" / "train_all_terminals_hist.png")
    run_terminals_all_files(args.exp_dir, out_ta, args.top_n)
    out_ca = args.out_all_combined or (args.exp_dir / "figures" / "train_all_token_hist.png")
    run_all_files_combined(args.exp_dir, out_ca, args.top_n)


if __name__ == "__main__":
    main()

