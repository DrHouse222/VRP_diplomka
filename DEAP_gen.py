#!/usr/bin/env python3
"""
Genetic Programming solution for CVRP using DEAP framework.
Evolves a scoring function that evaluates requests based on VRP features.
"""

import random
import numpy as np
import operator
from typing import List, Tuple, Dict, Any
from deap import base, creator, tools, gp, algorithms
from parser import VRPInstance, VRPFeatureExtractor
from basic_heuristics import nearest_neighbor_heuristic, savings_heuristic


def create_primitive_set():
    """Create the primitive set for genetic programming."""
    pset = gp.PrimitiveSet("MAIN", 12)  # 12 input features
    
    # Rename arguments to meaningful names
    pset.renameArguments(ARG0='dist_to_depot')
    pset.renameArguments(ARG1='dist_from_current')
    pset.renameArguments(ARG2='dist_to_depot_from_request')
    pset.renameArguments(ARG3='demand')
    pset.renameArguments(ARG4='remaining_capacity')
    pset.renameArguments(ARG5='capacity_utilization')
    pset.renameArguments(ARG6='demand_ratio')
    pset.renameArguments(ARG7='route_length')
    pset.renameArguments(ARG8='is_empty_route')
    pset.renameArguments(ARG9='dist_last_to_depot')
    pset.renameArguments(ARG10='savings')
    pset.renameArguments(ARG11='norm_savings')
    
    # Mathematical operators
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    #pset.addPrimitive(operator.neg, 1)
    #pset.addPrimitive(operator.abs, 1)
    
    # Additional mathematical functions
    def protected_div(left, right):
        """Protected division to avoid division by zero."""
        try:
            return left / right if abs(right) > 1e-6 else 1.0
        except:
            return 1.0
    
    def protected_sqrt(x):
        """Protected square root."""
        try:
            return np.sqrt(abs(x))
        except:
            return 0.0
    
    def protected_log(x):
        """Protected logarithm."""
        try:
            return np.log(abs(x) + 1e-6)
        except:
            return 0.0
    
    def min_func(a, b):
        """Minimum function."""
        return min(a, b)
    
    def max_func(a, b):
        """Maximum function."""
        return max(a, b)
    
    pset.addPrimitive(protected_div, 2)
    #pset.addPrimitive(protected_sqrt, 1)
    #pset.addPrimitive(protected_log, 1)
    #pset.addPrimitive(min_func, 2)
    #pset.addPrimitive(max_func, 2)
    
    # Constants
    pset.addTerminal(0.0)
    pset.addTerminal(1.0)
    pset.addTerminal(2.0)
    pset.addTerminal(0.5)
    pset.addTerminal(-1.0)
    
    return pset


def create_individual(pset):
    """Create a GP individual."""
    return gp.PrimitiveTree(gp.genHalfAndHalf(pset, min_=1, max_=3))


def create_toolbox(pset):
    """Create DEAP toolbox with operators."""
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", create_individual, pset)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Genetic operators
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # Compile function for evaluation
    toolbox.register("compile", gp.compile, pset=pset)
    
    # Store pset in toolbox for later use
    toolbox.pset = pset
    
    return toolbox


def evaluate_individual(individual, instances=None, feature_extractors=None, pset=None):
    """
    Evaluate a GP individual on VRP instances.
    Lower score means better solution.
    """
    if instances is None or feature_extractors is None or pset is None:
        return (float('inf'),)
    
    try:
        # Compile the individual into a function
        func = gp.compile(expr=individual, pset=pset)
        
        total_cost = 0.0
        num_evaluations = 0
        
        for instance, feature_extractor in zip(instances, feature_extractors):
            # Generate solution using the GP-evolved scoring function
            solution = solve_with_gp_scoring(instance, feature_extractor, func)
            
            # Evaluate solution quality
            total_distance, capacity_violation = instance.cost(solution)
            
            # Cost includes distance and penalty for violations
            cost = total_distance + capacity_violation * 1000
            total_cost += cost
            num_evaluations += 1
        
        # Return average cost (lower is better)
        avg_cost = total_cost / num_evaluations if num_evaluations > 0 else float('inf')
        return (avg_cost,)
    
    except Exception as e:
        # Return very high cost for invalid individuals
        print(f"Error in evaluation: {e}")
        return (float('inf'),)


def solve_with_gp_scoring(instance: VRPInstance, feature_extractor: VRPFeatureExtractor, 
                         scoring_func) -> List[List[int]]:
    """
    Solve VRP using GP-evolved scoring function.
    
    Args:
        instance: VRP instance
        feature_extractor: Feature extractor for the instance
        scoring_func: GP-evolved scoring function
    
    Returns:
        List of routes
    """
    n = instance.dimension
    unvisited = set(range(1, n))  # All customers except depot
    routes = []
    
    while unvisited:
        route = [instance.depot]
        load = 0
        current_position = instance.depot
        
        while True:
            # Get feasible candidates
            candidates = [c for c in unvisited if load + instance.demands[c] <= instance.capacity]
            
            if not candidates:
                # Return to depot
                route.append(instance.depot)
                break
            
            # Score each candidate using GP function
            scores = []
            for candidate in candidates:
                features = feature_extractor.extract_features(
                    candidate, route, load, current_position
                )
                
                # Convert features to the order expected by the GP function
                feature_values = [
                    features['dist_to_depot'],
                    features['dist_from_current'],
                    features['dist_to_depot_from_request'],
                    features['demand'],
                    features['remaining_capacity'],
                    features['capacity_utilization'],
                    features['demand_ratio'],
                    features['route_length'],
                    features['is_empty_route'],
                    features['dist_last_to_depot'],
                    features['savings'],
                    features['norm_savings']
                ]
                
                try:
                    score = scoring_func(*feature_values)
                    scores.append(score)
                except:
                    # If GP function fails, use distance as fallback
                    scores.append(instance.dist_matrix[current_position, candidate])
            
            # Choose candidate with lowest score (most desirable)
            best_candidate = candidates[np.argmin(scores)]
            
            # Update route
            route.append(best_candidate)
            load += instance.demands[best_candidate]
            unvisited.remove(best_candidate)
            current_position = best_candidate
        
        routes.append(route)
    
    return routes


def run_genetic_programming(instances: List[VRPInstance], population_size: int = 50, 
                          generations: int = 50, cxpb: float = 0.7, mutpb: float = 0.3):
    """
    Run genetic programming to evolve VRP scoring function.
    
    Args:
        instances: List of VRP instances for training
        population_size: Size of population
        generations: Number of generations
        cxpb: Crossover probability
        mutpb: Mutation probability
    
    Returns:
        Best evolved individual and statistics
    """
    # Create feature extractors for each instance
    feature_extractors = [VRPFeatureExtractor(inst) for inst in instances]
    
    # Create primitive set and toolbox
    pset = create_primitive_set()
    toolbox = create_toolbox(pset)
    
    # Set up evaluation function with instances
    def evaluate_with_instances(individual):
        return evaluate_individual(individual, instances, feature_extractors, pset)
    
    toolbox.register("evaluate", evaluate_with_instances)
    
    # Create initial population
    population = toolbox.population(n=population_size)
    
    # Statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    
    # Hall of fame
    hof = tools.HallOfFame(1)
    
    # Run evolution
    print(f"Starting Genetic Programming evolution...")
    print(f"Population size: {population_size}, Generations: {generations}")
    print(f"Crossover probability: {cxpb}, Mutation probability: {mutpb}")
    
    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb, mutpb, generations, 
        stats=mstats, halloffame=hof, verbose=True
    )
    
    return hof[0], logbook, pset


def main():
    """Main function to run the genetic programming solver."""
    # Load VRP instances for training
    instances = [
        VRPInstance("Set_A/A-n32-k5.vrp"),
        # Add more instances for better generalization
    ]
    
    print("Loaded VRP instances:")
    for i, instance in enumerate(instances):
        print(f"  {i+1}. {instance.name} (n={instance.dimension}, cap={instance.capacity})")
    
    # Run genetic programming
    best_individual, logbook, pset = run_genetic_programming(
        instances=instances,
        population_size=30,
        generations=30,
        cxpb=0.7,
        mutpb=0.3
    )
    
    print(f"\nEvolution completed!")
    print(f"Best individual fitness: {best_individual.fitness.values[0]:.2f}")
    
    # Test the evolved function
    print(f"\nTesting evolved scoring function...")
    
    for i, instance in enumerate(instances):
        print(f"\nInstance {i+1}: {instance.name}")
        
        # Create feature extractor
        feature_extractor = VRPFeatureExtractor(instance)
        
        # Compile best individual
        best_func = gp.compile(expr=best_individual, pset=pset)
        
        # Solve using evolved function
        gp_solution = solve_with_gp_scoring(instance, feature_extractor, best_func)
        gp_distance, gp_violation = instance.cost(gp_solution)
        
        # Compare with heuristics
        nn_routes = nearest_neighbor_heuristic(instance)
        nn_distance, nn_violation = instance.cost(nn_routes)
        
        savings_routes = savings_heuristic(instance)
        savings_distance, savings_violation = instance.cost(savings_routes)
        
        print(f"  GP Solution: Distance = {gp_distance:.2f}, Violation = {gp_violation:.2f}")
        print(f"  Nearest Neighbor: Distance = {nn_distance:.2f}, Violation = {nn_violation:.2f}")
        print(f"  Savings: Distance = {savings_distance:.2f}, Violation = {savings_violation:.2f}")
        
        if nn_distance > 0:
            improvement_nn = ((nn_distance - gp_distance) / nn_distance) * 100
            print(f"  Improvement over NN: {improvement_nn:.2f}%")
        
        if savings_distance > 0:
            improvement_savings = ((savings_distance - gp_distance) / savings_distance) * 100
            print(f"  Improvement over Savings: {improvement_savings:.2f}%")
    
    # Print the evolved formula
    print(f"\nEvolved scoring formula:")
    print(f"{best_individual}")
    
    return best_individual, logbook


if __name__ == "__main__":
    main()