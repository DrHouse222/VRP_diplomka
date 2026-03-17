#!/usr/bin/env python3
"""
Test LLM-generated VRP heuristics from generated_heuristics.json and compare
them to Nearest Neighbor and Savings heuristics.
"""

import inspect
import json
import copy
import math
import operator
import os

from parser import VRPFeatureExtractor
from problem_types import VRP_PROBLEM_TYPE
from basic_heuristics import nearest_neighbor_heuristic, saving_heuristic2
from data_generation import remove_tw_from_gvrp, evrptw_to_multi_depot
from DEAP_gen import load_instances_by_type


def _protected_div(left, right):
    try:
        return left / right if abs(right) > 1e-6 else 1.0
    except Exception:
        return 1.0


def _make_operator_namespace():
    return {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "protected_div": _protected_div,
        "max": max,
        "min": min,
    }


def _strip_code(raw: str) -> str:
    s = raw.strip()
    if "```python" in s:
        s = s.split("```python")[1].split("```")[0].strip()
    elif "```" in s:
        s = s.split("```")[1].split("```")[0].strip()
    return s


def build_scoring_callable(heuristic_code: str):
    """
    Build a scoring function (*args) -> float from heuristic_code.
    *args must be in the order of VRP_PROBLEM_TYPE.feature_names.
    """
    code = _strip_code(heuristic_code)
    feature_names = VRP_PROBLEM_TYPE.feature_names
    op_ns = _make_operator_namespace()

    if not code.strip():
        raise ValueError("Empty heuristic code")

    if code.strip().startswith("def "):
        # Define function in namespace with operators, then wrap to accept *args
        exec_ns = dict(op_ns)
        exec(code, exec_ns)
        fn = (
            exec_ns.get("score")
            or exec_ns.get("scoring_function")
            or exec_ns.get("score_function")
        )
        if fn is None:
            # Use any user-defined function from the exec (e.g. def my_score(...))
            fn = next(
                (v for k, v in exec_ns.items() if callable(v) and k not in op_ns),
                None,
            )
        if fn is None:
            raise ValueError("Code defines no 'score' or 'scoring_function'")
        def scoring_func(*args):
            kwargs = dict(zip(feature_names, args))
            # Pass only the keys the function accepts
            sig = inspect.signature(fn)
            params = {k: kwargs.get(k, 0.0) for k in sig.parameters}
            return float(fn(**params))
        return scoring_func

    # Expression only: eval with feature names bound to args
    try:
        compiled = compile(code, "<llm_heuristic>", "eval")
    except SyntaxError:
        raise ValueError(f"Invalid expression: {code[:80]}...")

    def scoring_func(*args):
        ns = dict(op_ns)
        ns.update(zip(feature_names, args))
        return float(eval(compiled, ns))

    return scoring_func


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


def main(max_instances_per_variant=None):
    """
    max_instances_per_variant: if set (e.g. 3), only use that many instances per
    variant for a quicker run. None = use all instances.
    """
    json_path = os.path.join(os.path.dirname(__file__), "generated_heuristics.json")
    with open(json_path, "r", encoding="utf-8") as f:
        heuristics = json.load(f)

    problem_map = load_problem_map()
    results = []

    for idx, entry in enumerate(heuristics):
        variant_name = entry["variant_name"]
        bool_capacity = entry.get("bool_capacity", False)
        bool_TW = entry.get("bool_TW", False)
        bool_green = entry.get("bool_green", False)
        bool_MD = entry.get("bool_MD", False)
        code = entry.get("heuristic_code", "")

        key = (bool_TW, bool_green, bool_MD)
        if key not in problem_map:
            print(f"[{idx}] Skip {variant_name}: no instance set for key {key}")
            continue
        problem_type, instances = problem_map[key]
        if not instances:
            print(f"[{idx}] Skip {variant_name}: no instances")
            continue
        if max_instances_per_variant is not None:
            instances = instances[: max_instances_per_variant]

        try:
            scoring_func = build_scoring_callable(code)
        except Exception as e:
            print(f"[{idx}] {variant_name} compile error: {e}")
            results.append({
                "index": idx,
                "variant": variant_name,
                "bool_capacity": bool_capacity,
                "llm_avg_cost": None,
                "nn_avg_cost": None,
                "savings_avg_cost": None,
                "error": str(e),
            })
            continue

        llm_costs = []
        nn_costs = []
        savings_costs = []
        for inst in instances:
            try:
                fe = VRPFeatureExtractor(inst)
                routes_llm = VRP_PROBLEM_TYPE.solve_with_scoring(
                    inst, fe, scoring_func, bool_capacity
                )
                cost_llm = VRP_PROBLEM_TYPE.compute_cost(inst, routes_llm)
                llm_costs.append(cost_llm)
            except Exception as e:
                llm_costs.append(float("nan"))
            try:
                routes_nn = nearest_neighbor_heuristic(inst, bool_capacity=bool_capacity)
                nn_costs.append(VRP_PROBLEM_TYPE.compute_cost(inst, routes_nn))
            except Exception:
                nn_costs.append(float("nan"))
            try:
                routes_s = saving_heuristic2(inst, bool_capacity=bool_capacity)
                savings_costs.append(VRP_PROBLEM_TYPE.compute_cost(inst, routes_s))
            except Exception:
                savings_costs.append(float("nan"))

        n = len(instances)
        valid_llm = [c for c in llm_costs if not math.isnan(c)]
        llm_avg = sum(valid_llm) / len(valid_llm) if valid_llm else float("nan")
        nn_avg = sum(nn_costs) / n if n else float("nan")
        s_avg = sum(savings_costs) / n if n else float("nan")

        # Percentual difference: (baseline - llm) / baseline * 100; positive = LLM better
        pct_vs_nn = None
        pct_vs_sav = None
        if llm_avg is not None and not math.isnan(llm_avg):
            if nn_avg is not None and not math.isnan(nn_avg) and nn_avg > 0:
                pct_vs_nn = ((nn_avg - llm_avg) / nn_avg) * 100.0
            if s_avg is not None and not math.isnan(s_avg) and s_avg > 0:
                pct_vs_sav = ((s_avg - llm_avg) / s_avg) * 100.0

        results.append({
            "index": idx,
            "variant": variant_name,
            "bool_capacity": bool_capacity,
            "n_instances": n,
            "llm_avg_cost": llm_avg if not math.isnan(llm_avg) else None,
            "nn_avg_cost": nn_avg if not math.isnan(nn_avg) else None,
            "savings_avg_cost": s_avg if not math.isnan(s_avg) else None,
            "pct_vs_nn": pct_vs_nn,
            "pct_vs_savings": pct_vs_sav,
        })

    # Report
    print("\n" + "=" * 100)
    print("LLM vs NN vs Savings (average cost per instance set)")
    print("=" * 100)
    print(f"{'Idx':<5} {'Variant':<14} {'Cap':<5} {'N':<4} {'LLM avg':<12} {'NN avg':<12} {'Savings avg':<12} {'pct vs NN':<10} {'pct vs Sav':<10}")
    print("-" * 100)
    for r in results:
        idx = r["index"]
        var = r["variant"]
        cap = "Y" if r["bool_capacity"] else "N"
        n = r.get("n_instances", 0)
        llm = r.get("llm_avg_cost")
        nn = r.get("nn_avg_cost")
        sav = r.get("savings_avg_cost")
        llm_s = f"{llm:.2f}" if llm is not None else "N/A"
        nn_s = f"{nn:.2f}" if nn is not None else "N/A"
        sav_s = f"{sav:.2f}" if sav is not None else "N/A"
        pct_nn = r.get("pct_vs_nn")
        pct_sav = r.get("pct_vs_savings")
        vs_nn = f"{pct_nn:+.1f}%" if pct_nn is not None else "N/A"
        vs_sav = f"{pct_sav:+.1f}%" if pct_sav is not None else "N/A"
        print(f"{idx:<5} {var:<14} {cap:<5} {n:<4} {llm_s:<12} {nn_s:<12} {sav_s:<12} {vs_nn:<10} {vs_sav:<10}")
    print("=" * 100)

    # Optional: save CSV (with percentual differences)
    csv_path = os.path.join(os.path.dirname(__file__), "res_test_LLM.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,variant,bool_capacity,n_instances,llm_avg_cost,nn_avg_cost,savings_avg_cost,pct_vs_nn,pct_vs_savings\n")
        for r in results:
            llm = r.get("llm_avg_cost")
            nn = r.get("nn_avg_cost")
            sav = r.get("savings_avg_cost")
            pct_nn = r.get("pct_vs_nn")
            pct_sav = r.get("pct_vs_savings")
            f.write(f"{r['index']},{r['variant']},{r['bool_capacity']},{r.get('n_instances', 0)},")
            f.write(f"{llm if llm is not None else ''},{nn if nn is not None else ''},{sav if sav is not None else ''},")
            f.write(f"{pct_nn if pct_nn is not None else ''},{pct_sav if pct_sav is not None else ''}\n")
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    import sys
    n_inst = None
    if len(sys.argv) > 1:
        try:
            n_inst = int(sys.argv[1])
        except ValueError:
            pass
    main(max_instances_per_variant=n_inst)
