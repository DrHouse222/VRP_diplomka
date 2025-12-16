#!/usr/bin/env python3
"""
Genetic Programming solution for VRP variants using DEAP framework.
"""

import random
import numpy as np
import operator
from typing import List, Tuple, Dict, Any, Callable, Optional
from deap import base, creator, tools, gp, algorithms
from parser import VRPInstance, VRPTWInstance, VRPFeatureExtractor
from basic_heuristics import nearest_neighbor_heuristic, savings_heuristic
from problem_types import ProblemType, CVRPProblemType, CVRPTWProblemType, ProblemTypeRegistry, PROBLEM_REGISTRY


def protected_div(left, right):
    try:
        return left / right if abs(right) > 1e-6 else 1.0
    except:
        return 1.0

def max_func(a, b):
    return max(a, b)

def min_func(a, b):
    return min(a, b)


def create_primitive_set(problem_type: str = "CVRP"):
    """Create the primitive set for genetic programming."""
    problem_config = PROBLEM_REGISTRY.get(problem_type)
    if problem_config is None:
        raise ValueError(f"Unknown problem type: {problem_type}")
    return problem_config.create_primitive_set(gp)


def create_individual(pset):
    """Create a GP individual."""
    return gp.PrimitiveTree(gp.genHalfAndHalf(pset, min_=1, max_=3))


def create_toolbox(problem_type: str = "CVRP"):
    """Create the DEAP toolbox for the given problem type."""
    pset = create_primitive_set(problem_type)
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", create_individual, pset)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    # Store problem type in toolbox
    toolbox.problem_type = problem_type
    
    return toolbox, pset


def evaluate_individual(individual, instances, problem_type: str = "CVRP"):
    """Evaluate a GP individual on VRP instances."""
    problem_config = PROBLEM_REGISTRY.get(problem_type)
    if problem_config is None:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    # Compile the individual
    func = gp.compile(expr=individual, pset=problem_config.create_primitive_set(gp))
    
    total_fitness = 0.0
    
    for instance in instances:
        feature_extractor = VRPFeatureExtractor(instance)
        
        # Solve using the GP function
        solution = problem_config.solve_with_scoring(instance, feature_extractor, func)
        # Problem type now returns scalar fitness via unified compute_cost
        fitness = problem_config.evaluate_solution(instance, solution)
        total_fitness += fitness
    
    return (total_fitness,)


def run_genetic_programming(instances: List, problem_type: str = "auto", 
                          population_size: int = 50, generations: int = 50) -> Tuple[Any, Any, Any, str]:
    """Run genetic programming to evolve VRP scoring function."""
    
    # Auto-detect problem type
    if problem_type == "auto":
        problem_type = PROBLEM_REGISTRY.auto_detect(instances[0])
    
    print(f"Using problem type: {problem_type}")
    print(f"Running GP on {len(instances)} {problem_type} instances...")
    
    # Create toolbox
    toolbox, pset = create_toolbox(problem_type)
    
    # Create evaluation function with problem type
    def evaluate_with_problem_type(individual):
        return evaluate_individual(individual, instances, problem_type)
    
    # Register evaluation function
    toolbox.register("evaluate", evaluate_with_problem_type)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # Statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    
    # Create population
    population = toolbox.population(n=population_size)
    
    # Run evolution
    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=0.5, mutpb=0.1, ngen=generations,
        stats=mstats, verbose=True
    )
    
    # Get best individual
    best_individual = tools.selBest(population, 1)[0]
    
    return best_individual, logbook, pset, problem_type


def load_instances_by_type():
    """Load instances grouped by problem type."""
    cvrp_instances = [
        VRPInstance("Sets/Set_A/A-n32-k5.vrp"),
        # TODO: Add more
    ]
    
    vrptw_instances = [
        VRPTWInstance("Sets/Vrp-Set-HG/C1_2_2.txt"),
        # TODO: Add more
    ]
    
    return cvrp_instances, vrptw_instances


def train_and_test_problem_type(instances, problem_type, population_size=30, generations=30):
    """Train and test a specific problem type."""
    if not instances:
        print(f"No {problem_type} instances available, skipping...")
        return None
    
    print(f"\n{'='*60}")
    print(f"Training {problem_type} model with {len(instances)} instances")
    print(f"{'='*60}")
    
    for i, instance in enumerate(instances):
        print(f"  {i+1}. {instance.name} (n={instance.dimension}, cap={instance.capacity})")
    
    # Run genetic programming
    best_individual, logbook, pset, detected_type = run_genetic_programming(
        instances=instances,
        problem_type=problem_type,
        population_size=population_size,
        generations=generations
    )
    
    # Get the evolved function
    func = gp.compile(expr=best_individual, pset=pset)
    print(f"\n{problem_type} Evolved scoring formula:")
    print(f"{str(best_individual)}")
    
    # Test the evolved function
    print(f"\nTesting {problem_type} evolved scoring function...")
    
    problem_config = PROBLEM_REGISTRY.get(problem_type)
    
    for i, instance in enumerate(instances):
        print(f"\nInstance {i+1}: {instance.name}")
        
        # Create feature extractor
        feature_extractor = VRPFeatureExtractor(instance)
        
        # Solve using evolved function
        gp_solution = problem_config.solve_with_scoring(instance, feature_extractor, func)
        gp_fitness = problem_config.evaluate_solution(instance, gp_solution)
        
        # Compare with heuristics (map problem types to heuristic format)
        heuristic_problem_type = "vrp" if problem_type == "CVRP" else "vrptw"
        nn_routes = nearest_neighbor_heuristic(instance, problem=heuristic_problem_type)
        nn_fitness = instance.cost(nn_routes)
        
        savings_routes = savings_heuristic(instance, problem=heuristic_problem_type)
        savings_fitness = instance.cost(savings_routes)
        
        print(f"  GP-DEAP Solution: Fitness = {gp_fitness:.2f}")
        print(f"  Nearest Neighbor: Fitness = {nn_fitness:.2f}")
        print(f"  Savings: Fitness = {savings_fitness:.2f}")
        
        if nn_fitness > 0:
            improvement_nn = ((nn_fitness - gp_fitness) / nn_fitness) * 100
            print(f"  Improvement over NN (fitness): {improvement_nn:.2f}%")
        
        if savings_fitness > 0:
            improvement_savings = ((savings_fitness - gp_fitness) / savings_fitness) * 100
            print(f"  Improvement over Savings (fitness): {improvement_savings:.2f}%")
    
    return best_individual, logbook, pset


def main():
    """Main function to run the genetic programming solver with separate training."""
    # Load instances grouped by problem type
    cvrp_instances, vrptw_instances = load_instances_by_type()
    
    print("Loaded VRP instances by type:")
    print(f"CVRP instances: {len(cvrp_instances)}")
    print(f"CVRPTW instances: {len(vrptw_instances)}")
    
    # Train and test CVRP model
    cvrp_results = None
    if cvrp_instances:
        cvrp_results = train_and_test_problem_type(
            instances=cvrp_instances,
            problem_type="CVRP",
            population_size=30,
            generations=30
        )
    
    # Train and test VRPTW model
    vrptw_results = None
    if vrptw_instances:
        vrptw_results = train_and_test_problem_type(
            instances=vrptw_instances,
            problem_type="CVRPTW",
            population_size=30,
            generations=30
        )
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    if cvrp_results:
        cvrp_best, cvrp_logbook, cvrp_pset = cvrp_results
        print(f"CVRP Model: Best fitness = {cvrp_best.fitness.values[0]:.2f}")
    
    if vrptw_results:
        vrptw_best, vrptw_logbook, vrptw_pset = vrptw_results
        print(f"VRPTW Model: Best fitness = {vrptw_best.fitness.values[0]:.2f}")
    
    return cvrp_results, vrptw_results


if __name__ == "__main__":
    main()
