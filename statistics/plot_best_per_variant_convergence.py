#!/usr/bin/env python3
"""
Plot convergence curves for best_per_variant rows.

For each row in exp_results/best_per_variant.csv:
- resolve source_file -> experiments/**/<stem>.jsonl
- load the JSONL record at `index`
- plot fitness_min from log_evolution with ± fitness_std band (population spread)

Outputs one multi-page PDF, one page per best evolved function.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def resolve_jsonl_path(repo_root: Path, source_file: str) -> Path | None:
    src = Path(source_file)
    direct = repo_root / "experiments" / src.parent / f"{src.stem}.jsonl"
    if direct.is_file():
        return direct
    candidates = sorted((repo_root / "experiments").glob(f"**/{src.stem}.jsonl"))
    if candidates:
        return candidates[0]
    return None


def load_jsonl_record_by_index(path: Path | None, target_index: int) -> dict | None:
    """Index is based on successfully parsed non-empty JSON lines."""
    if path is None or target_index < 0 or not path.is_file():
        return None
    current = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if current == target_index:
                return rec
            current += 1
    return None


def make_page(
    pdf: PdfPages,
    row: dict[str, str],
    rec: dict | None,
    source_jsonl: Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    title_left = f"{row.get('problem_type', '')} cap={row.get('bool_capacity', '')} idx={row.get('index', '')}"
    title_right = row.get("source_file", "")

    if rec is None:
        ax.text(
            0.5,
            0.5,
            "Record not found or invalid JSONL.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        #fig.suptitle(f"{title_left}\n{title_right}", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)
        return

    log = rec.get("log_evolution") or []
    if not isinstance(log, list) or not log:
        ax.text(
            0.5,
            0.5,
            "No log_evolution available in record.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        #fig.suptitle(f"{title_left}\n{title_right}", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)
        return

    gens: list[int] = []
    fit_min: list[float] = []
    fit_std: list[float] = []
    for i, e in enumerate(log):
        if "fitness_min" not in e:
            continue
        if "fitness_std" not in e:
            continue
        gens.append(int(e.get("gen", i)))
        fit_min.append(float(e["fitness_min"]))
        fit_std.append(float(e["fitness_std"]))

    if gens:
        gmin = np.array(fit_min, dtype=float)
        gstd = np.array(fit_std, dtype=float)
        ax.plot(gens, gmin.tolist(), color="C0", linewidth=1.5, label="fitness_min")
        lo = gmin - gstd
        hi = gmin + gstd
        #ax.fill_between(gens, lo, hi, color="C0", alpha=0.2, label="± fitness_std")
    else:
        ax.text(
            0.5,
            0.5,
            "Missing fitness_min / fitness_std in log_evolution entries.",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("fitness_min")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend(fontsize=8)

    src_note = str(source_jsonl.relative_to(source_jsonl.parents[1])) if source_jsonl else "jsonl_not_found"
    #fig.suptitle(f"{title_left}\n{title_right} -> {src_note}", fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def run(input_csv: Path, output_pdf: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with open(input_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"No rows in {input_csv}")

    with PdfPages(output_pdf) as pdf:
        for row in rows:
            idx_s = row.get("index", "")
            try:
                idx = int(idx_s)
            except ValueError:
                idx = -1

            src = row.get("source_file", "")
            jsonl_path = resolve_jsonl_path(repo_root, src)
            rec = load_jsonl_record_by_index(jsonl_path, idx)
            make_page(pdf, row, rec, jsonl_path)

    print(f"Wrote {output_pdf} ({len(rows)} pages)")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Plot convergence for each best_per_variant function.")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=repo_root / "exp_results" / "best_per_variant.csv",
        help="Input best_per_variant CSV path",
    )
    parser.add_argument(
        "--out_pdf",
        type=Path,
        default=repo_root / "exp_results" / "figures" / "best_per_variant_convergence.pdf",
        help="Output multi-page PDF",
    )
    args = parser.parse_args()
    run(args.input_csv, args.out_pdf)


if __name__ == "__main__":
    main()
