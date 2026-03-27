#!/usr/bin/env python3
"""
Generate graphs from GP experiment results stored in experiment_results.jsonl.

For each experiment (one line in the JSONL file), this script creates:
  - a fitness evolution plot (avg / min / max per generation)
  - a tree-size evolution plot (avg / min / max per generation)

The plots are saved as PNG files in the specified output directory.
"""

import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def load_experiments(path: str) -> List[Dict[str, Any]]:
    """Load all experiment records from a JSONL file."""
    experiments: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                experiments.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return experiments


def plot_single_experiment(record: Dict[str, Any], out_dir: str) -> None:
    """
    Create fitness and size evolution plots for a single experiment record.
    """
    problem_type = record.get("problem_type", "UNKNOWN")
    bool_capacity = record.get("bool_capacity", False)
    log_evolution: List[Dict[str, Any]] = record.get("log_evolution", [])

    if not log_evolution:
        return

    gens = [entry.get("gen", i) for i, entry in enumerate(log_evolution)]

    def extract_series(prefix: str, key: str) -> List[float] | None:
        col = f"{prefix}_{key}"
        vals: List[float] = []
        for entry in log_evolution:
            if col not in entry:
                return None
            vals.append(entry.get(col))
        return vals

    # Fitness stats (may be None if you only logged gen/nevals)
    fit_avg = extract_series("fitness", "avg")
    fit_min = extract_series("fitness", "min")
    fit_max = extract_series("fitness", "max")

    size_avg = extract_series("size", "avg")
    size_min = extract_series("size", "min")
    size_max = extract_series("size", "max")

    os.makedirs(out_dir, exist_ok=True)

    # Fitness plot
    if fit_avg is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(gens, fit_avg, label="avg", color="C0")
        if fit_min is not None and fit_max is not None:
            plt.fill_between(gens, fit_min, fit_max, color="C0", alpha=0.2, label="min–max")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        title_suffix = " (capacity ON)" if bool_capacity else " (capacity OFF)"
        plt.title(f"Fitness evolution: {problem_type}{title_suffix}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()

        fname = f"fitness_{problem_type}_cap{int(bool_capacity)}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    # Tree size plot
    if size_avg is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(gens, size_avg, label="avg", color="C1")
        if size_min is not None and size_max is not None:
            plt.fill_between(gens, size_min, size_max, color="C1", alpha=0.2, label="min–max")
        plt.xlabel("Generation")
        plt.ylabel("Tree size (nodes)")
        title_suffix = " (capacity ON)" if bool_capacity else " (capacity OFF)"
        plt.title(f"Tree size evolution: {problem_type}{title_suffix}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()

        fname = f"size_{problem_type}_cap{int(bool_capacity)}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()


def main(
    results_path: str = "exp_test.jsonl",
    output_dir: str = "graphs2",
) -> None:
    """
    Load experiment results and generate graphs for each experiment.

    You can run this script directly:
        python graph_generation.py
    or override paths:
        python graph_generation.py experiment_results.jsonl graphs_out
    """
    experiments = load_experiments(results_path)
    if not experiments:
        print(f"No experiments found in {results_path}")
        return

    print(f"Loaded {len(experiments)} experiments from {results_path}")
    for i, rec in enumerate(experiments):
        print(f"  Plotting experiment {i+1}/{len(experiments)}: {rec.get('problem_type')} (cap={rec.get('bool_capacity')})")
        plot_single_experiment(rec, output_dir)

    print(f"Graphs saved to {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        main(results_path=sys.argv[1], output_dir=sys.argv[2])
    elif len(sys.argv) == 2:
        main(results_path=sys.argv[1])
    else:
        main()

