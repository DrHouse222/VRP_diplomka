#!/usr/bin/env python3
"""
Test GP-evolved VRP heuristics from experiment_results.jsonl and compare
their best_expr to Nearest Neighbor and Savings heuristics.
"""

import json
import copy
import math
import operator
import os

from deap import gp

from parser import VRPFeatureExtractor
from problem_types import VRP_PROBLEM_TYPE
from basic_heuristics import nearest_neighbor_heuristic, saving_heuristic2
from data_generation import remove_tw_from_gvrp, evrptw_to_multi_depot
from DEAP_gen import load_instances_by_type, create_toolbox


def load_problem_map():
    cvrp, vrptw, gvrp, mdvrp, mdvrptw = load_instances_by_type()
    return {
        (False, False, False): ("VRP", cvrp),
        (True, False, False): ("VRPTW", vrptw),
        (False, True, False): ("GVRP", remove_tw_from_gvrp(copy.deepcopy(gvrp))),
        (True, True, False): ("G-VRPTW", gvrp),
        (False, False, True): ("MDVRP", mdvrp),
        (True, False, True): ("MDVRPTW", mdvrptw),
        (False, True, True): ("GVRP-MD", evrptw_to_multi_depot(remove_tw_from_gvrp(copy.deepcopy(gvrp)))),
        (True, True, True): ("G-VRPTW-MD", evrptw_to_multi_depot(gvrp)),
    }


def build_scoring_from_expr(expr_str: str):
    """
    Build a callable scoring function from a GP expression string 'best_expr'.
    Uses the same primitive set as in DEAP_gen.
    """
    toolbox, pset = create_toolbox()
    # Parse the string into a PrimitiveTree, then compile
    tree = gp.PrimitiveTree.from_string(expr_str, pset)
    func = gp.compile(expr=tree, pset=pset)
    return func


def main(max_instances_per_variant=None, results_path: str = "experiment_results.jsonl"):
    """
    Evaluate best_expr from experiment_results.jsonl against NN and Savings.

    max_instances_per_variant: if set (e.g. 3), only use that many instances per
    variant for a quicker run. None = use all instances.
    """
    problem_map = load_problem_map()

    # Load all experiment records
    records = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue

    if not records:
        print(f"No records found in {results_path}")
        return

    results = []

    for idx, rec in enumerate(records):
        problem_type = rec.get("problem_type")
        bool_capacity = rec.get("bool_capacity", False)
        best_expr = rec.get("best_expr")
        # Infer flags TW/green/MD from problem_type name
        if problem_type == "VRP":
            bool_TW = False
            bool_green = False
            bool_MD = False
        elif problem_type == "VRPTW":
            bool_TW = True
            bool_green = False
            bool_MD = False
        elif problem_type == "GVRP":
            bool_TW = False
            bool_green = True
            bool_MD = False
        elif problem_type == "G-VRPTW":
            bool_TW = True
            bool_green = True
            bool_MD = False
        elif problem_type == "MDVRP":
            bool_TW = False
            bool_green = False
            bool_MD = True
        elif problem_type == "MDVRPTW":
            bool_TW = True
            bool_green = False
            bool_MD = True
        elif problem_type == "GVRP-MD":
            bool_TW = False
            bool_green = True
            bool_MD = True
        elif problem_type == "G-VRPTW-MD":
            bool_TW = True
            bool_green = True
            bool_MD = True
        else:
            print(f"[{idx}] Unknown problem_type {problem_type}, skipping")
            continue

        key = (bool_TW, bool_green, bool_MD)
        if key not in problem_map:
            print(f"[{idx}] No instance set for key {key}, skipping")
            continue

        _, instances = problem_map[key]
        if not instances:
            print(f"[{idx}] No instances for {problem_type}, skipping")
            continue

        if max_instances_per_variant is not None:
            instances = instances[: max_instances_per_variant]

        if not best_expr:
            print(f"[{idx}] Empty best_expr for {problem_type}, skipping")
            continue

        try:
            scoring_func = build_scoring_from_expr(best_expr)
        except Exception as e:
            print(f"[{idx}] Failed to compile best_expr for {problem_type}: {e}")
            results.append(
                {
                    "index": idx,
                    "problem_type": problem_type,
                    "bool_capacity": bool_capacity,
                    "n_instances": len(instances),
                    "gp_avg_cost": None,
                    "nn_avg_cost": None,
                    "savings_avg_cost": None,
                    "error": str(e),
                }
            )
            continue

        gp_costs = []
        nn_costs = []
        s_costs = []

        for inst in instances:
            try:
                fe = VRPFeatureExtractor(inst)
                # Choose solver depending on green flag
                if bool_green:
                    gp_routes = VRP_PROBLEM_TYPE.solve_with_scoring(
                        inst, fe, scoring_func, bool_capacity
                    )
                else:
                    gp_routes = VRP_PROBLEM_TYPE.solve_with_scoring_without_green(
                        inst, fe, scoring_func, bool_capacity
                    )
                gp_costs.append(VRP_PROBLEM_TYPE.compute_cost(inst, gp_routes))
            except Exception:
                gp_costs.append(float("nan"))

            try:
                nn_routes = nearest_neighbor_heuristic(inst, bool_capacity=bool_capacity)
                nn_costs.append(VRP_PROBLEM_TYPE.compute_cost(inst, nn_routes))
            except Exception:
                nn_costs.append(float("nan"))

            try:
                s_routes = saving_heuristic2(inst, bool_capacity=bool_capacity)
                s_costs.append(VRP_PROBLEM_TYPE.compute_cost(inst, s_routes))
            except Exception:
                s_costs.append(float("nan"))

        n = len(instances)
        valid_gp = [c for c in gp_costs if not math.isnan(c)]
        gp_avg = sum(valid_gp) / len(valid_gp) if valid_gp else float("nan")
        nn_avg = sum(nn_costs) / n if n else float("nan")
        s_avg = sum(s_costs) / n if n else float("nan")

        pct_vs_nn = None
        pct_vs_s = None
        if not math.isnan(gp_avg):
            if nn_avg > 0 and not math.isnan(nn_avg):
                pct_vs_nn = ((nn_avg - gp_avg) / nn_avg) * 100.0
            if s_avg > 0 and not math.isnan(s_avg):
                pct_vs_s = ((s_avg - gp_avg) / s_avg) * 100.0

        results.append(
            {
                "index": idx,
                "problem_type": problem_type,
                "bool_capacity": bool_capacity,
                "n_instances": n,
                "gp_avg_cost": gp_avg if not math.isnan(gp_avg) else None,
                "nn_avg_cost": nn_avg if not math.isnan(nn_avg) else None,
                "savings_avg_cost": s_avg if not math.isnan(s_avg) else None,
                "pct_vs_nn": pct_vs_nn,
                "pct_vs_savings": pct_vs_s,
            }
        )

    # Report table
    print("\n" + "=" * 110)
    print("GP best_expr vs NN vs Savings (average cost per problem type)")
    print("=" * 110)
    print(
        f"{'Idx':<5} {'Type':<14} {'Cap':<5} {'N':<4} "
        f"{'GP avg':<12} {'NN avg':<12} {'Savings avg':<12} "
        f"{'% vs NN':<10} {'% vs Sav':<10}"
    )
    print("-" * 110)
    for r in results:
        idx = r["index"]
        t = r["problem_type"]
        cap = "Y" if r["bool_capacity"] else "N"
        n = r["n_instances"]
        gp = r["gp_avg_cost"]
        nn = r["nn_avg_cost"]
        s = r["savings_avg_cost"]
        gp_s = f"{gp:.2f}" if gp is not None else "N/A"
        nn_s = f"{nn:.2f}" if nn is not None else "N/A"
        s_s = f"{s:.2f}" if s is not None else "N/A"
        pct_nn = r.get("pct_vs_nn")
        pct_s = r.get("pct_vs_savings")
        vs_nn = f"{pct_nn:+.1f}%" if pct_nn is not None else "N/A"
        vs_s = f"{pct_s:+.1f}%" if pct_s is not None else "N/A"
        print(
            f"{idx:<5} {t:<14} {cap:<5} {n:<4} "
            f"{gp_s:<12} {nn_s:<12} {s_s:<12} {vs_nn:<10} {vs_s:<10}"
        )
    print("=" * 110)

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), "test_experiment_res_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "index,problem_type,bool_capacity,n_instances,"
            "gp_avg_cost,nn_avg_cost,savings_avg_cost,pct_vs_nn,pct_vs_savings\n"
        )
        for r in results:
            f.write(
                f"{r['index']},{r['problem_type']},{r['bool_capacity']},"
                f"{r['n_instances']},"
                f"{r['gp_avg_cost'] if r['gp_avg_cost'] is not None else ''},"
                f"{r['nn_avg_cost'] if r['nn_avg_cost'] is not None else ''},"
                f"{r['savings_avg_cost'] if r['savings_avg_cost'] is not None else ''},"
                f"{r['pct_vs_nn'] if r['pct_vs_nn'] is not None else ''},"
                f"{r['pct_vs_savings'] if r['pct_vs_savings'] is not None else ''}\n"
            )
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    import sys

    n_inst = None
    if len(sys.argv) > 1:
        try:
            n_inst = int(sys.argv[1])
        except ValueError:
            n_inst = None
    main(max_instances_per_variant=n_inst)

