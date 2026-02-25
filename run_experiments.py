#!/usr/bin/env python3
"""
Batch runner for GP VRP experiments.

Iterates over all supported problem variants (CVRP, VRPTW, GVRP, G-VRPTW,
MDCVRP, MDVRPTW, GVRP-MD, G-VRPTW-MD) and runs GP for each combination
of capacity constraint (on/off) and problem type.

Results (best evolved expression, fitness, timings, flags) are saved
to a JSONL file for later analysis.
"""

import json
import os
import time
import copy
from typing import Any, Dict, List

from deap import gp

from DEAP_gen import (
    load_instances_by_type,
    train_and_test_problem_type,
    remove_tw_from_gvrp,
    evrptw_to_multi_depot,
)


def run_all_experiments(
    output_path: str = "experiment_results.jsonl",
    population_size: int = 100,
    generations: int = 10,
    time_limit_sec: float | None = None,
    n_train: int = 5,
    n_test: int = -1,
    base_seed: int | None = 0,
    cxpb: float = 0.8,
    mutpb: float = 0.15,
) -> None:
    """
    Run GP experiments across all supported problem variants and capacity settings.

    Parameters
    ----------
    output_path : str
        Path to JSONL file where results will be appended.
    population_size : int
        GP population size.
    generations : int
        Maximum number of generations (upper bound if time_limit_sec is set).
    time_limit_sec : float or None
        If provided, GP will stop once this time budget (seconds) is exceeded
        for a given experiment (per variant), even if 'generations' not reached.
    n_train : int
        Number of instances to use for training in each variant.
    n_test : int
        Number of instances to use for testing (0 = no test set).
    base_seed : int or None
        Base random seed; per-experiment seed = base_seed + experiment_index.
        If None, no explicit seeding is applied.
    cxpb : float
        Crossover probability used inside GP.
    mutpb : float
        Mutation probability used inside GP.
    """

    # Make sure directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Load all instance groups once
    (
        cvrp_instances,
        vrptw_instances,
        gvrp_instances,
        mdvrp_instances,
        mdvrptw_instances,
    ) = load_instances_by_type()

    # Problem map matches DEAP_gen.main
    def build_problem_map():
        return {
            (False, False, False): ("CVRP", cvrp_instances),
            (True, False, False): ("VRPTW", vrptw_instances),
            (False, True, False): (
                "GVRP",
                remove_tw_from_gvrp(copy.deepcopy(gvrp_instances)),
            ),
            (True, True, False): ("G-VRPTW", gvrp_instances),
            (False, False, True): ("MDCVRP", mdvrp_instances),
            (True, False, True): ("MDVRPTW", mdvrptw_instances),
            (False, True, True): (
                "GVRP-MD",
                evrptw_to_multi_depot(
                    remove_tw_from_gvrp(copy.deepcopy(gvrp_instances))
                ),
            ),
            (True, True, True): (
                "G-VRPTW-MD",
                evrptw_to_multi_depot(gvrp_instances),
            ),
        }

    problem_map = build_problem_map()

    with open(output_path, "w", encoding="utf-8") as f_out:
        exp_index = 0
        for bool_capacity in (False, True):
            for bool_TW in (False, True):
                for bool_green in (False, True):
                    for bool_MD in (False, True):
                        key = (bool_TW, bool_green, bool_MD)
                        if key not in problem_map:
                            continue

                        problem_type, instances = problem_map[key]
                        if not instances:
                            continue

                        # Per-experiment seed
                        seed = None
                        if base_seed is not None:
                            seed = int(base_seed) + exp_index
                        exp_index += 1

                        print(
                            f"\n=== Running experiment: "
                            f"cap={bool_capacity}, TW={bool_TW}, green={bool_green}, MD={bool_MD}, "
                            f"type={problem_type} ==="
                        )

                        start = time.time()
                        results = train_and_test_problem_type(
                            all_instances=instances,
                            problem_type=problem_type,
                            bool_capacity=bool_capacity,
                            n_train=n_train,
                            n_test=n_test,
                            population_size=population_size,
                            generations=generations,
                            time_limit_sec=time_limit_sec,
                            seed=seed,
                            cxpb=cxpb,
                            mutpb=mutpb,
                        )
                        elapsed = time.time() - start

                        if not results:
                            continue

                        best_individual, logbook, pset = results
                        best_expr = str(best_individual)
                        best_fitness = (
                            float(best_individual.fitness.values[0])
                            if best_individual.fitness.valid
                            else None
                        )

                        # Extract terminals used in the best individual
                        used_terminals = sorted(
                            {str(node) for node in best_individual if isinstance(node, gp.Terminal)}
                        )

                        # Serialize logbook (fitness/size evolution)
                        def _serialize_log_entry(entry: Any) -> Dict[str, Any]:
                            d: Dict[str, Any] = dict(entry)
                            for k, v in list(d.items()):
                                # Convert numpy scalars to Python scalars
                                try:
                                    if hasattr(v, "item"):
                                        d[k] = v.item()
                                except Exception:
                                    pass
                            return d

                        log_evolution: List[Dict[str, Any]] = [
                            _serialize_log_entry(rec) for rec in logbook
                        ]

                        last_gen = log_evolution[-1]["gen"] if log_evolution else None

                        record = {
                            "problem_type": problem_type,
                            "n_train": n_train,
                            "n_test": n_test,
                            "population_size": population_size,
                            "generations_max": generations,
                            "generations_run": last_gen,
                            "time_limit_sec": time_limit_sec,
                            "elapsed_sec": elapsed,
                            "seed": seed,
                            "cxpb": cxpb,
                            "mutpb": mutpb,
                            "best_expr": best_expr,
                            "best_fitness": best_fitness,
                            "terminals_used": used_terminals,
                            "log_evolution": log_evolution,
                        }
                        f_out.write(json.dumps(record) + "\n")
                        f_out.flush()


if __name__ == "__main__":
    # Example: run quick experiments with a time budget per variant
    run_all_experiments(
        output_path="experiment_results.jsonl",
        population_size=50,
        generations=50,
        time_limit_sec=None,  # e.g. set to 300 for 5-minute budget per variant
        n_train=5,
        n_test=-1,
    )

