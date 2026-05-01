#!/usr/bin/env python3
"""
Analyze GP expressions behind exp_results/best_per_variant.csv rows.

For each row:
- locate the source JSONL in experiments/<same_subdir>/<same_stem>.jsonl
- take the JSONL line at `index` (same alignment as evaluation CSVs)
- extract best_expr
- compute expression size and token usage:
  - functions (identifiers followed by "(")
  - terminals (identifier tokens not used as function names in that expression)
  - constants (numeric literals)

Outputs:
1) exp_results/best_per_variant_expr_analysis.csv
2) exp_results/best_per_variant_expr_token_totals.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

FUNC_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_tokens(expr: str) -> tuple[list[str], list[str], list[str]]:
    funcs = FUNC_RE.findall(expr)
    func_set = set(funcs)
    idents = IDENT_RE.findall(expr)
    terms = [t for t in idents if t not in func_set]
    consts = NUM_RE.findall(expr)
    return funcs, terms, consts


def counter_to_str(c: Counter[str]) -> str:
    if not c:
        return ""
    return "; ".join(f"{k}:{v}" for k, v in c.most_common())


def resolve_jsonl_path(root: Path, source_file: str) -> Path | None:
    src_path = Path(source_file)
    direct = root / "experiments" / src_path.parent / f"{src_path.stem}.jsonl"
    if direct.is_file():
        return direct
    candidates = sorted((root / "experiments").glob(f"**/{src_path.stem}.jsonl"))
    if candidates:
        return candidates[0]
    return None


def load_jsonl_record(path: Path | None, idx: int) -> dict | None:
    if path is None:
        return None
    if idx < 0 or not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == idx:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    return None
    return None


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Analyze best_expr tokens for best_per_variant rows.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "exp_results" / "best_per_variant.csv",
        help="Input best_per_variant CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "exp_results" / "best_per_variant_expr_analysis.csv",
        help="Detailed output CSV",
    )
    parser.add_argument(
        "--out_totals",
        type=Path,
        default=root / "exp_results" / "best_per_variant_expr_token_totals.csv",
        help="Token totals output CSV",
    )
    parser.add_argument(
        "--out_size_stats",
        type=Path,
        default=root / "exp_results" / "best_per_variant_expr_size_stats.csv",
        help="Expression size summary stats CSV",
    )
    parser.add_argument(
        "--fig_dir",
        type=Path,
        default=root / "exp_results" / "figures" / "best_per_variant_exprs",
        help="Directory for generated PDF charts",
    )
    parser.add_argument("--top_n", type=int, default=20, help="Top N tokens in usage charts")
    args = parser.parse_args()

    func_totals: Counter[str] = Counter()
    term_totals: Counter[str] = Counter()
    const_totals: Counter[str] = Counter()

    rows_out: list[dict[str, str]] = []
    with open(args.input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src = row.get("source_file", "")
            idx_s = row.get("index", "")
            try:
                idx = int(idx_s)
            except ValueError:
                idx = -1

            jsonl_path = resolve_jsonl_path(root, src)
            rec = load_jsonl_record(jsonl_path, idx)
            expr = rec.get("best_expr", "") if rec else ""
            expr = expr if isinstance(expr, str) else ""

            funcs, terms, consts = parse_tokens(expr) if expr else ([], [], [])
            fc = Counter(funcs)
            tc = Counter(terms)
            cc = Counter(consts)
            func_totals.update(fc)
            term_totals.update(tc)
            const_totals.update(cc)

            expr_size = len(funcs) + len(terms) + len(consts)
            status = "ok"
            if jsonl_path is None:
                status = "jsonl_not_found"
            elif rec is None:
                status = "row_not_found_or_invalid"

            rows_out.append(
                {
                    "index": row.get("index", ""),
                    "problem_type": row.get("problem_type", ""),
                    "bool_capacity": row.get("bool_capacity", ""),
                    "n_instances": row.get("n_instances", ""),
                    "gp_avg_cost": row.get("gp_avg_cost", ""),
                    "pct_vs_nn": row.get("pct_vs_nn", ""),
                    "source_file": src,
                    "jsonl_file": str(jsonl_path.relative_to(root)) if jsonl_path is not None else "",
                    "analysis_status": status,
                    "expr_size_nodes": str(expr_size) if expr else "",
                    "n_functions": str(len(funcs)) if expr else "",
                    "n_terminals": str(len(terms)) if expr else "",
                    "n_constants": str(len(consts)) if expr else "",
                    "functions_used": counter_to_str(fc),
                    "terminals_used": counter_to_str(tc),
                    "constants_used": counter_to_str(cc),
                    "best_expr": expr,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "problem_type",
            "bool_capacity",
            "n_instances",
            "gp_avg_cost",
            "pct_vs_nn",
            "source_file",
            "jsonl_file",
            "analysis_status",
            "expr_size_nodes",
            "n_functions",
            "n_terminals",
            "n_constants",
            "functions_used",
            "terminals_used",
            "constants_used",
            "best_expr",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    with open(args.out_totals, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token_type", "token", "count"])
        for token, count in func_totals.most_common():
            w.writerow(["function", token, count])
        for token, count in term_totals.most_common():
            w.writerow(["terminal", token, count])
        for token, count in const_totals.most_common():
            w.writerow(["constant", token, count])

    def plot_counter(counter: Counter[str], title: str, out_name: str, top_n: int) -> None:
        items = counter.most_common(top_n)
        if not items:
            return
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.55), 4.8))
        ax.bar(range(len(labels)), vals, color="steelblue", edgecolor="navy", alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / out_name)
        plt.close(fig)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_counter(func_totals, "Functions in best_per_variant expressions", "functions_usage.pdf", args.top_n)
    plot_counter(term_totals, "Terminals in best_per_variant expressions", "terminals_usage.pdf", args.top_n)
    plot_counter(const_totals, "Constants in best_per_variant expressions", "constants_usage.pdf", args.top_n)

    size_rows = [
        r for r in rows_out
        if r.get("analysis_status") == "ok" and r.get("expr_size_nodes")
    ]
    if size_rows:
        labels = [f"{r['problem_type']} cap={r['bool_capacity']}" for r in size_rows]
        vals = [int(r["expr_size_nodes"]) for r in size_rows]
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 5.0))
        ax.bar(range(len(labels)), vals, color="darkcyan", edgecolor="black", alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Expression size (token count)")
        ax.set_title("Best expression size per problem variant")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "expr_size_by_variant.pdf")
        plt.close(fig)

        stats_rows = [
            ("min", float(min(vals))),
            ("max", float(max(vals))),
            ("median", float(statistics.median(vals))),
            ("mean", float(statistics.mean(vals))),
        ]
        with open(args.out_size_stats, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in stats_rows:
                w.writerow([k, f"{v:.6f}"])
            w.writerow(["n_expressions", str(len(vals))])

    print(f"Wrote {args.out} ({len(rows_out)} rows)")
    print(f"Wrote {args.out_totals}")
    print(f"Wrote {args.out_size_stats}")
    print(f"Wrote charts in {args.fig_dir}")


if __name__ == "__main__":
    main()

