# AUTOMATED DESIGN OF CONSTRUCTIVE HEURISTICS FOR THE VEHICLE ROUTING PROBLEM

Evolutionary search for **scoring functions** used in route construction across several **vehicle routing problem** variants (CVRP, VRPTW, GVRP, MDVRP, and their combinations). The pipeline uses **DEAP** genetic programming, compares evolved heuristics to **nearest neighbor** and **Clarke–Wright savings** baselines, and includes scripts to aggregate results and produce **PDF** figures.

## Requirements

- Python 3.10+ (recommended)
- Install dependencies from the requirements.txt:

```bash
pip install -r requirements.txt
```

## Repository layout

| Path | Role |
|------|------|
| `DEAP_gen.py` | DEAP toolbox, GP loop, instance loading, training/evaluation helpers |
| `run_experiments.py` | Batch GP over all variants and result evaluation, writes JSONL for logging GP and CSV result comparations |
| `vrp_problem.py` | Unified problem config, primitive set, route construction function `solve_with_scoring`, fitness function `compute_cost` |
| `vrp_feature_extractor.py` | Feature vectors used during route build |
| `parser.py` | VRP instances parsers |
| `basic_heuristics.py` | Nearest-neighbor and savings baselines |
| `data_generation.py` | Instance transformations |
| `API.py` | LLM-generated heuristics via OpenRouter |
| `test_experiment_res.py` | Evaluate GP evolved JSONL vs NN / savings, writes CSV summaries |
| `test_LLM.py` | Evaluate LLM-generated heuristics |
| `statistics/` | Plotting and analysis scripts (CSV / JSONL → PDF, aggregations) |
| `experiments/` | Typical location for `*.jsonl` outputs |
| `exp_results/` | Typical location for `*.csv` outputs |
| `Sets/` | Benchmark instances and known optimal solutions |

## Running GP experiments

`run_experiments.py` iterates over supported problem types, runs GP per variant, and appends one JSON object per line to the output file. By default the path is prefixed with `experiments/`

Example:

```bash
cd /path/to/VRP_diplomka
python run_experiments.py \
  --output_path pop_gen/exp_p50_g200_1.jsonl \
  --population_size 50 \
  --generations 200 \
  --n_train 5 \
  --n_test -1 \
  --cxpb 0.8 \
  --mutpb 0.15
```

Use `--no_eval` to skip the post-run CSV evaluation.

Each JSONL record includes fields such as `problem_type`, `best_expr`, `best_fitness`, and `log_evolution` (per-generation fitness and size statistics).

## Statistics and figures

Scripts under `statistics/`. Typical workflow: place experiment CSVs under `exp_results/<experiment_name>/`, JSONL under `experiments/<experiment_name>/`, then run the matching plotter.

| Script | Purpose |
|--------|---------|
| `plot_exp_results.py` | Compare experiment CSVs (bar/box/heatmap) |
| `plot_jsonl_convergence.py` | Fitness convergence from JSONL, grouped by experiment naming |
| `plot_node_size_evolution.py` | Mean tree size over generations for node-size experiments |
| `plot_size_vs_performance.py` | Best-individual size vs `% vs NN` |
| `plot_train_token_hist.py` | Token usage histograms from train-size JSONL |
| `plot_set_size_terminals_grouped.py` | Terminal usage by feature-set variant |
| `aggregate_best_exp_results.py` | Build `exp_results/best_per_variant.csv` |
| `analyze_best_per_variant_exprs.py` | Expression statistics for best rows |
| `plot_best_per_variant_convergence.py` | Convergence PDF per `best_per_variant` row |
| `best_per_variant_vs_optimal.py` | Compare best GP costs to known optima where available |
| `graph_generation.py` | Per-record fitness/size plots from a single JSONL |

Example:

```bash
python statistics/plot_exp_results.py --exp_dir exp_results/pop_gen --out_dir exp_results/figures/pop_gen
python statistics/plot_jsonl_convergence.py --exp_dir experiments/pop_gen
```

## LLM heuristics

`API.py` builds prompts and calls OpenRouter to propose scoring expressions, `test_LLM.py` evaluates exported JSON against the baselines. Requires an API key.

## License / attribution

This repository is part of a diploma thesis project (FIT VUT). Dataset files under `Sets/` follow the licenses of their respective sources; cite those sources when publishing results derived from them.