#!/usr/bin/env python3
"""
Genetic Programming solution for VRP variants using DEAP framework.
"""

import random
import numpy as np
import operator
from typing import List, Tuple, Dict, Any, Callable, Optional
from deap import base, creator, tools, gp, algorithms
from parser import VRPInstance, VRPTWInstance, GVRPMultiTechInstance, VRPFeatureExtractor
from basic_heuristics import nearest_neighbor_heuristic, savings_heuristic
from problem_types import VRPProblemType, VRP_PROBLEM_TYPE
from data_generation import convert_vrptw_to_gvrptw


def create_individual(pset):
    """Create a GP individual."""
    return gp.PrimitiveTree(gp.genHalfAndHalf(pset, min_=1, max_=3))


def create_toolbox():
    """Create the DEAP toolbox"""
    pset = VRP_PROBLEM_TYPE.create_primitive_set(gp)
    
    # Create DEAP classes only if they don't already exist
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", create_individual, pset)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    return toolbox, pset


def evaluate_individual(individual, instances, bool_capacity=True):
    """Evaluate a GP individual on VRP instances."""
    # Compile the individual
    func = gp.compile(expr=individual, pset=VRP_PROBLEM_TYPE.create_primitive_set(gp))
    
    total_fitness = 0.0
    
    for instance in instances:
        feature_extractor = VRPFeatureExtractor(instance)
        
        # Solve using the GP function
        solution = VRP_PROBLEM_TYPE.solve_with_scoring(instance, feature_extractor, func, bool_capacity)
        # Problem type now returns scalar fitness via unified compute_cost
        fitness = VRP_PROBLEM_TYPE.compute_cost(instance, solution)
        total_fitness += fitness
    
    # Add tree size penalty to encourage simpler trees
    tree_size = len(individual)
    tree_size_penalty = 0.1 * tree_size  # Small penalty per node
    total_fitness += tree_size_penalty
    
    return (total_fitness,)  # DEAP expects a tuple, not a float


def run_genetic_programming(instances, bool_capacity = True, population_size = 50, generations = 50):
    """Run genetic programming to evolve VRP scoring function."""
    
    # Create toolbox
    toolbox, pset = create_toolbox()
    
    # Create evaluation function
    def evaluate_with_problem_type(individual):
        return evaluate_individual(individual, instances, bool_capacity)
    
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
    
    return best_individual, logbook, pset


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
    
    gvrp_instances = [
        GVRPMultiTechInstance("Sets/felipe-et-al-2014/data-A0-N030_red.xml"),
        # TODO: Add more
    ]
    
    return cvrp_instances, vrptw_instances, gvrp_instances


def train_and_test_problem_type(instances, problem_type, bool_capacity=True, population_size=30, generations=30):
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
    best_individual, logbook, pset = run_genetic_programming(
        instances=instances,
        bool_capacity=bool_capacity,
        population_size=population_size,
        generations=generations
    )
    
    # Get the evolved function
    func = gp.compile(expr=best_individual, pset=pset)
    print(f"\n{problem_type} Evolved scoring formula:")
    print(f"{str(best_individual)}")
    
    # Test the evolved function
    print(f"\nTesting {problem_type} evolved scoring function...")
    
    for i, instance in enumerate(instances):
        print(f"\nInstance {i+1}: {instance.name}")
        
        # Create feature extractor
        feature_extractor = VRPFeatureExtractor(instance)
        
        # Solve using evolved function
        gp_solution = VRP_PROBLEM_TYPE.solve_with_scoring(instance, feature_extractor, func, bool_capacity)
        gp_fitness = VRP_PROBLEM_TYPE.compute_cost(instance, gp_solution)

        # Compare with heuristics (map problem types to heuristic format)
        nn_routes = nearest_neighbor_heuristic(instance, bool_capacity=bool_capacity)
        nn_fitness = VRP_PROBLEM_TYPE.compute_cost(instance, nn_routes)
        
        #savings_routes = savings_heuristic(instance, problem=heuristic_problem_type)
        #savings_fitness = VRP_PROBLEM_TYPE.compute_cost(instance, savings_routes)
        
        print(f"  GP-DEAP Solution: Fitness = {gp_fitness:.2f}")
        
        print(f"  Nearest Neighbor: Fitness = {nn_fitness:.2f}")
        #print(f"  Savings: Fitness = {savings_fitness:.2f}")
        
        if nn_fitness > 0:
            improvement_nn = ((nn_fitness - gp_fitness) / nn_fitness) * 100
            print(f"  Improvement over NN (fitness): {improvement_nn:.2f}%")
        
        #if savings_fitness > 0:
        #    improvement_savings = ((savings_fitness - gp_fitness) / savings_fitness) * 100
        #    print(f"  Improvement over Savings (fitness): {improvement_savings:.2f}%")
        
    
    return best_individual, logbook, pset


def main():
    # Choose variants
    bool_capacity = True
    bool_TW = False
    bool_green = False

    # Load instances
    cvrp_instances, vrptw_instances, gvrp_instances = load_instances_by_type()
    results = None

    # Mapping: (green, TW) -> (Problem Name, Data List)
    problem_map = {
        (False, False): ("CVRP", cvrp_instances),
        (True, False):  ("VRPTW", vrptw_instances),
        (False, True):  ("GVRP", gvrp_instances),
        (True, True):   ("G-VRPTW", convert_vrptw_to_gvrptw(vrptw_instances))
    }

    problem_type, instances = problem_map.get((bool_TW, bool_green))

    if instances:
        results = train_and_test_problem_type(
            instances=instances,
            problem_type=problem_type,
            bool_capacity=bool_capacity,
            population_size=50,
            generations=50
        )
    else:
        print(f"No instances loaded for {problem_type}")
            
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    if results:
        vrp_best, vrp_logbook, vrp_pset = results
        print(f"{problem_type} Model: Best fitness = {vrp_best.fitness.values[0]:.2f}")
    

if __name__ == "__main__":
    main()
