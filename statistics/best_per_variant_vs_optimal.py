#!/usr/bin/env python3
"""
Build a table like best_per_variant.csv but compare GP cost to known optima
(from Sets/Solutions) instead of NN / savings.

Reads: exp_results/best_per_variant.csv (or override with --input).
Writes: exp_results/best_per_variant_vs_optimal.csv

For problem types with published optima (VRP/CVRP Set A, VRPTW HG, MDVRP / MDVRPTW Cordeau),
computes mean optimal cost over the same instance list as in DEAP_gen.load_instances_by_type,
then pct_gap_vs_optimal = (gp_avg_cost - optimal_avg) / optimal_avg * 100
(positive => GP worse than optimum).

Green / GVRP variants have no matching folder in Sets/Solutions — optimal columns left empty.

**bool_capacity=False:** published optima assume feasible (capacity/TW) solutions; runs without
capacity often produce infeasible-but-cheaper tours, so GP/NN costs are **not comparable** to
those optima — optimal columns are left blank for bool_capacity=False rows.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from DEAP_gen import load_instances_by_type


def parse_sol_cost(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.strip().splitlines()):
        m = re.search(r"Cost\s+([\d.]+)\s*$", line.strip(), re.I)
        if m:
            return float(m.group(1))
    return None


def parse_res_cost(path: Path) -> float | None:
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if not lines:
        return None
    parts = lines[0].strip().split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def problem_type_to_key(problem_type: str) -> tuple[bool, bool, bool] | None:
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
    return None


def family_with_optima(key: tuple[bool, bool, bool]) -> str | None:
    """Subset of Sets/Solutions with optimal files."""
    tw, green, md = key
    if green:
        return None
    if not tw and not md:
        return "cvrp"
    if tw and not md:
        return "cvrptw"
    if not tw and md:
        return "mdcvrp"
    if tw and md:
        return "mdcvrptw"
    return None


def mean_optimal_cost(
    instances: list[Any],
    family: str,
    solutions_root: Path,
) -> tuple[float | None, int, int]:
    """Returns (mean optimal, n_found, n_total)."""
    costs: list[float] = []
    for inst in instances:
        name = getattr(inst, "name", "")
        if family == "cvrp":
            p = solutions_root / "Set_A-sol" / f"{name}.sol"
            v = parse_sol_cost(p) if p.is_file() else None
        elif family == "cvrptw":
            p = solutions_root / "Vrp-Set-HG-sol" / f"{name}.sol"
            v = parse_sol_cost(p) if p.is_file() else None
        elif family == "mdcvrp":
            p = solutions_root / "C-mdvrp-sol" / f"{name}.res"
            v = parse_res_cost(p) if p.is_file() else None
        elif family == "mdcvrptw":
            p = solutions_root / "C-mdvrptw-sol" / f"{name}.res"
            v = parse_res_cost(p) if p.is_file() else None
        else:
            v = None
        if v is not None and math.isfinite(v):
            costs.append(v)
    n_total = len(instances)
    n_found = len(costs)
    if not costs:
        return None, n_found, n_total
    return sum(costs) / len(costs), n_found, n_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment best_per_variant with optimal-cost gaps.")
    parser.add_argument(
        "--input",
        type=Path,
        default=_REPO_ROOT / "exp_results" / "best_per_variant.csv",
        help="Input CSV (default: exp_results/best_per_variant.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "exp_results" / "best_per_variant_vs_optimal.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        default=_REPO_ROOT / "Sets" / "Solutions",
        help="Root folder containing Set_A-sol, Vrp-Set-HG-sol, C-mdvrp-sol, C-mdvrptw-sol",
    )
    args = parser.parse_args()

    cvrp, vrptw, _gvrp, mdvrp, mdvrptw = load_instances_by_type()
    inst_by_key: dict[tuple[bool, bool, bool], list[Any]] = {
        (False, False, False): cvrp,
        (True, False, False): vrptw,
        (False, False, True): mdvrp,
        (True, False, True): mdvrptw,
    }

    rows_in = []
    with open(args.input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_in.append(row)

    fieldnames = [
        "index",
        "problem_type",
        "bool_capacity",
        "n_instances",
        "gp_avg_cost",
        "optimal_avg_cost",
        "pct_gap_vs_optimal",
        "n_optima_found",
        "n_instances_expected",
        "source_file",
    ]

    out_rows: list[dict[str, Any]] = []
    for row in rows_in:
        pt = str(row.get("problem_type", ""))
        cap_raw = str(row.get("bool_capacity", "")).lower()
        capacity_on = cap_raw in ("true", "1", "yes")
        key = problem_type_to_key(pt)
        gp = row.get("gp_avg_cost", "")
        try:
            gp_f = float(gp) if gp not in (None, "") else float("nan")
        except ValueError:
            gp_f = float("nan")

        optimal_avg: str = ""
        pct_gap: str = ""
        n_found_s = ""
        n_exp_s = ""

        if key is not None and key in inst_by_key and capacity_on:
            fam = family_with_optima(key)
            if fam is not None:
                instances = inst_by_key[key]
                o_opt, n_found, n_tot = mean_optimal_cost(
                    instances, fam, args.solutions.resolve()
                )
                n_found_s = str(n_found)
                n_exp_s = str(n_tot)
                if o_opt is not None and math.isfinite(gp_f) and o_opt > 0:
                    optimal_avg = str(o_opt)
                    pct_gap = str((gp_f - o_opt) / o_opt * 100.0)

        out_rows.append(
            {
                "index": row.get("index", ""),
                "problem_type": row.get("problem_type", ""),
                "bool_capacity": row.get("bool_capacity", ""),
                "n_instances": row.get("n_instances", ""),
                "gp_avg_cost": row.get("gp_avg_cost", ""),
                "optimal_avg_cost": optimal_avg,
                "pct_gap_vs_optimal": pct_gap,
                "n_optima_found": n_found_s,
                "n_instances_expected": n_exp_s,
                "source_file": row.get("source_file", ""),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"Wrote {args.out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
