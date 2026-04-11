#!/usr/bin/env python3
"""
Analyze heuristic_code usage in generated heuristics JSON and plot frequencies.

Counts:
- function calls (e.g., add, min, protected_div)
- terminals/variables (name tokens that are not called as functions)
- numeric constants (e.g., 1.0, 2)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze generated heuristic code usage.")
    parser.add_argument(
        "--json_path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "old_experiments" / "all_generated_heuristics.json",
        help="Path to JSON file containing items with heuristic_code.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "old_experiments" / "heuristic_analysis",
        help="Directory for output figures and CSV summary.",
    )
    parser.add_argument("--top_n", type=int, default=20, help="Top N items to show in each chart.")
    parser.add_argument("--dpi", type=int, default=150, help="Output figure DPI.")
    return parser.parse_args()


class HeuristicVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: Counter[str] = Counter()
        self.terminals: Counter[str] = Counter()
        self.constants: Counter[str] = Counter()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.functions[node.func.id] += 1
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Function names are loaded as Name too, skip them here.
        parent = getattr(node, "_parent", None)
        if isinstance(parent, ast.Call) and parent.func is node:
            return
        self.terminals[node.id] += 1

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, (int, float)):
            # Skip constants that are part of a unary signed literal;
            # they are handled in visit_UnaryOp to preserve the sign.
            parent = getattr(node, "_parent", None)
            if isinstance(parent, ast.UnaryOp) and parent.operand is node:
                return
            self.constants[str(node.value)] += 1

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        # Python AST encodes "-1.0" as UnaryOp(USub, Constant(1.0)).
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            if isinstance(node.op, ast.USub):
                self.constants[str(-node.operand.value)] += 1
                return
            if isinstance(node.op, ast.UAdd):
                self.constants[str(+node.operand.value)] += 1
                return
        self.generic_visit(node)


def attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "_parent", parent)


def load_rows(json_path: Path) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_codes(
    rows: list[dict],
) -> tuple[Counter[str], Counter[str], Counter[str], int, list[dict[str, int | str]]]:
    functions: Counter[str] = Counter()
    terminals: Counter[str] = Counter()
    constants: Counter[str] = Counter()
    parsed = 0
    per_expr_stats: list[dict[str, int | str]] = []

    for idx, row in enumerate(rows):
        code = row.get("heuristic_code")
        if not isinstance(code, str) or not code.strip():
            continue
        try:
            tree = ast.parse(code, mode="eval")
        except SyntaxError:
            continue

        parsed += 1
        attach_parents(tree)
        visitor = HeuristicVisitor()
        visitor.visit(tree)
        functions.update(visitor.functions)
        terminals.update(visitor.terminals)
        constants.update(visitor.constants)
        f_count = sum(visitor.functions.values())
        t_count = sum(visitor.terminals.values())
        c_count = sum(visitor.constants.values())
        per_expr_stats.append(
            {
                "row_index": idx,
                "variant_name": str(row.get("variant_name", "")),
                "functions_count": f_count,
                "terminals_count": t_count,
                "constants_count": c_count,
                "size_total": f_count + t_count + c_count,
            }
        )

    return functions, terminals, constants, parsed, per_expr_stats


def save_counter_bar(
    counter: Counter[str],
    title: str,
    out_path: Path,
    top_n: int,
    dpi: int,
    *,
    orientation: str = "h",
) -> None:
    items = counter.most_common(top_n)
    if not items:
        return

    if orientation == "v":
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        fig_w = max(8.0, min(18.0, 0.55 * len(labels) + 2.0))
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        x = list(range(len(labels)))
        ax.bar(x, values, color="steelblue", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        # Trim extra whitespace at the ends of the x-axis
        if x:
            ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
    else:
        labels = [k for k, _ in items][::-1]
        values = [v for _, v in items][::-1]

        fig_h = max(4.0, min(12.0, 0.4 * len(labels) + 1.2))
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.barh(labels, values, color="steelblue", alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(
    out_path: Path,
    functions: Counter[str],
    terminals: Counter[str],
    constants: Counter[str],
) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "token", "count"])
        for token, count in functions.most_common():
            writer.writerow(["function", token, count])
        for token, count in terminals.most_common():
            writer.writerow(["terminal", token, count])
        for token, count in constants.most_common():
            writer.writerow(["constant", token, count])


def save_size_stats_csv(out_path: Path, stats_rows: list[dict[str, int | str]]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_index",
                "variant_name",
                "functions_count",
                "terminals_count",
                "constants_count",
                "size_total",
            ]
        )
        for row in stats_rows:
            writer.writerow(
                [
                    row["row_index"],
                    row["variant_name"],
                    row["functions_count"],
                    row["terminals_count"],
                    row["constants_count"],
                    row["size_total"],
                ]
            )


def save_size_plots(out_dir: Path, stats_rows: list[dict[str, int | str]], dpi: int) -> tuple[Path, Path]:
    sizes = [int(r["size_total"]) for r in stats_rows]
    if not sizes:
        return out_dir / "size_histogram.png", out_dir / "size_components_boxplot.png"

    p_hist = out_dir / "size_histogram.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = min(20, max(5, len(set(sizes))))
    ax.hist(sizes, bins=bins, color="slateblue", alpha=0.85, edgecolor="black")
    ax.set_title("Heuristic size distribution")
    ax.set_xlabel("Total size")
    ax.set_ylabel("Number of heuristics")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p_hist, dpi=dpi)
    plt.close(fig)

    p_box = out_dir / "size_components_boxplot.png"
    f_vals = [int(r["functions_count"]) for r in stats_rows]
    t_vals = [int(r["terminals_count"]) for r in stats_rows]
    c_vals = [int(r["constants_count"]) for r in stats_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot([f_vals, t_vals, c_vals, sizes], tick_labels=["functions", "terminals", "constants", "total"], patch_artist=True)
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#B279A2"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title("Heuristic size components per expression")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p_box, dpi=dpi)
    plt.close(fig)
    return p_hist, p_box


def main() -> None:
    args = parse_args()
    rows = load_rows(args.json_path)
    functions, terminals, constants, parsed_count, per_expr_stats = analyze_codes(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    save_counter_bar(
        functions,
        "Function usage",
        args.out_dir / "functions_usage.png",
        args.top_n,
        args.dpi,
    )
    save_counter_bar(
        terminals,
        "Terminal usage",
        args.out_dir / "terminals_usage.png",
        args.top_n,
        args.dpi,
        orientation="v",
    )
    save_counter_bar(
        constants,
        "Constant usage",
        args.out_dir / "constants_usage.png",
        args.top_n,
        args.dpi,
    )
    save_summary_csv(args.out_dir / "usage_summary.csv", functions, terminals, constants)
    save_size_stats_csv(args.out_dir / "size_stats_per_heuristic.csv", per_expr_stats)
    p_hist, p_box = save_size_plots(args.out_dir, per_expr_stats, args.dpi)

    sizes = [int(r["size_total"]) for r in per_expr_stats]
    if sizes:
        print(
            "Heuristic size stats (total): "
            f"min={min(sizes)}, median={statistics.median(sizes):.1f}, "
            f"mean={statistics.mean(sizes):.2f}, max={max(sizes)}"
        )

    print(f"Read rows: {len(rows)}")
    print(f"Parsed heuristic_code rows: {parsed_count}")
    print(f"Wrote: {args.out_dir / 'functions_usage.png'}")
    print(f"Wrote: {args.out_dir / 'terminals_usage.png'}")
    print(f"Wrote: {args.out_dir / 'constants_usage.png'}")
    print(f"Wrote: {args.out_dir / 'usage_summary.csv'}")
    print(f"Wrote: {args.out_dir / 'size_stats_per_heuristic.csv'}")
    print(f"Wrote: {p_hist}")
    print(f"Wrote: {p_box}")


if __name__ == "__main__":
    main()
