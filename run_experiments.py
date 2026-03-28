#!/usr/bin/env python3
"""
Batch runner for GP VRP experiments.

Iterates over all supported problem variants (VRP, VRPTW, GVRP, G-VRPTW,
MDVRP, MDVRPTW, GVRP-MD, G-VRPTW-MD) and runs GP for each combination
of capacity constraint (on/off) and problem type.

Results (best evolved expression, fitness, timings, flags) are saved
to a JSONL file for later analysis.
"""

import json
import os
import time
import copy
import argparse
from typing import Any, Dict, List
import datetime

from deap import gp

from DEAP_gen import (
    load_instances_by_type,
    train_and_test_problem_type,
    train_and_test_problem_type_with_test_csv,
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
    test_csv_dir: str | None = None,
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
    test_csv_dir : str or None
        If set, per-experiment test-set summaries are written as CSV files
        (same columns as test_experiment_res.py) under this directory.
        If None, uses train_and_test_problem_type (detailed per-instance test prints).
    """

    # Make sure directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if test_csv_dir:
        os.makedirs(test_csv_dir, exist_ok=True)

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
            (False, False, False): ("VRP", cvrp_instances),
            (True, False, False): ("VRPTW", vrptw_instances),
            (False, True, False): (
                "GVRP",
                remove_tw_from_gvrp(copy.deepcopy(gvrp_instances)),
            ),
            (True, True, False): ("G-VRPTW", gvrp_instances),
            (False, False, True): ("MDVRP", mdvrp_instances),
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
                        if base_seed != 0:
                            seed = int(base_seed) + exp_index
                        else:
                            # Use a random seed when base_seed is not provided
                            import random
                            seed = random.randint(0, 2**31 - 1)
                        exp_index += 1

                        print(
                            f"\n=== Running experiment: "
                            f"cap={bool_capacity}, TW={bool_TW}, green={bool_green}, MD={bool_MD}, "
                            f"type={problem_type} ==="
                        )

                        start = time.time()
                        if test_csv_dir:
                            test_csv_path = os.path.join(
                                test_csv_dir,
                                f"exp_{exp_index:04d}_{problem_type}_cap{int(bool_capacity)}_seed{seed}.csv",
                            )
                            results = train_and_test_problem_type_with_test_csv(
                                all_instances=instances,
                                problem_type=problem_type,
                                bool_capacity=bool_capacity,
                                bool_green=bool_green,
                                n_train=n_train,
                                n_test=n_test,
                                population_size=population_size,
                                generations=generations,
                                time_limit_sec=time_limit_sec,
                                seed=seed,
                                cxpb=cxpb,
                                mutpb=mutpb,
                                test_csv_path=test_csv_path,
                            )
                        else:
                            results = train_and_test_problem_type(
                                all_instances=instances,
                                problem_type=problem_type,
                                bool_capacity=bool_capacity,
                                bool_green=bool_green,
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

                        # Serialize logbook (fitness/size evolution).
                        # DEAP's MultiStatistics stores per-chapter data in
                        # logbook.chapters['fitness'] and ['size'], while logbook
                        # entries themselves only contain gen / nevals.
                        log_evolution: List[Dict[str, Any]] = []
                        fitness_chapter = getattr(logbook, "chapters", {}).get("fitness")
                        size_chapter = getattr(logbook, "chapters", {}).get("size")

                        for i, rec in enumerate(logbook):
                            base: Dict[str, Any] = {}
                            for k, v in dict(rec).items():
                                try:
                                    if hasattr(v, "item"):
                                        v = v.item()
                                except Exception:
                                    pass
                                base[k] = v

                            # Attach fitness stats for this generation, if present
                            if fitness_chapter is not None and i < len(fitness_chapter):
                                fit_stats = fitness_chapter[i]
                                for stat_name, v in dict(fit_stats).items():
                                    try:
                                        if hasattr(v, "item"):
                                            v = v.item()
                                    except Exception:
                                        pass
                                    base[f"fitness_{stat_name}"] = v

                            # Attach size stats for this generation, if present
                            if size_chapter is not None and i < len(size_chapter):
                                sz_stats = size_chapter[i]
                                for stat_name, v in dict(sz_stats).items():
                                    try:
                                        if hasattr(v, "item"):
                                            v = v.item()
                                    except Exception:
                                        pass
                                    base[f"size_{stat_name}"] = v

                            log_evolution.append(base)

                        last_gen = log_evolution[-1]["gen"] if log_evolution else None

                        record = {
                            "problem_type": problem_type,
                            "bool_capacity": bool_capacity,
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
                            "log_evolution": log_evolution,
                        }
                        f_out.write(json.dumps(record) + "\n")
                        f_out.flush()


if __name__ == "__main__":
    print(f"Starting GP VRP experiments at {datetime.datetime.now().isoformat()}")
    parser = argparse.ArgumentParser(description="Run GP VRP batch experiments.")
    parser.add_argument("--n_train", type=int, default=5, help="Number of training instances per variant.")
    parser.add_argument("--n_test", type=int, default=-1, help="Number of test instances per variant (-1 uses remaining).")
    parser.add_argument("--cxpb", type=float, default=1.0, help="Crossover probability.")
    parser.add_argument("--mutpb", type=float, default=0.2, help="Mutation probability.")

    # Keep existing defaults for the rest
    parser.add_argument("--output_path", type=str, default="experiment_results_test.jsonl")
    parser.add_argument("--population_size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--time_limit_sec", type=float, default=10000)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument(
        "--test_csv_dir",
        type=str,
        default="exp_results",
        help="Directory for per-experiment test-set CSV summaries (test_experiment_res schema). "
        "Set empty string to disable and use detailed test prints only.",
    )

    args = parser.parse_args()

    args.output_path = "experiments/" + args.output_path

    run_all_experiments(
        output_path=args.output_path,
        population_size=args.population_size,
        generations=args.generations,
        time_limit_sec=args.time_limit_sec,
        n_train=args.n_train,
        n_test=args.n_test,
        base_seed=args.base_seed,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        test_csv_dir=test_csv_dir,
    )
    print(f"Finished GP VRP experiments at {datetime.datetime.now().isoformat()}")

