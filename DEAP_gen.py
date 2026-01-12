#!/usr/bin/env python3
"""
Genetic Programming solution for VRP variants using DEAP framework.
"""

import numpy as np
from deap import base, creator, tools, gp, algorithms
from parser import VRPInstance, VRPTWInstance, GVRPMultiTechInstance, VRPFeatureExtractor
from basic_heuristics import nearest_neighbor_heuristic, savings_heuristic
from problem_types import VRP_PROBLEM_TYPE
from data_generation import convert_vrptw_to_gvrptw
import time
import matplotlib.pyplot as plt
import glob
import os


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
        if solve_with_scoring == 1:
            solution = VRP_PROBLEM_TYPE.solve_with_scoring(instance, feature_extractor, func, bool_capacity)
        else:
            solution = VRP_PROBLEM_TYPE.solve_with_scoring2(instance, feature_extractor, func, bool_capacity)
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
    
    # Hall of Fame: Tracks the best individual across all generations
    hof = tools.HallOfFame(1)
    
    # Create population
    population = toolbox.population(n=population_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    hof.update(population)
    
    # Evolution loop with elitism: combine population + offspring, select best
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
    
    # Record initial statistics
    record = mstats.compile(population) if mstats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if mstats:
        print(logbook.stream)
    
    # Evolution loop with (μ+λ) elitism
    for gen in range(1, generations + 1):
        # Select and clone the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.5:  # cxpb
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if np.random.random() < 0.1:  # mutpb
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update Hall of Fame
        hof.update(offspring)
        
        # ELITISM: Select best individuals from population + offspring (μ+λ selection)
        # This guarantees the best individual is always preserved
        population[:] = tools.selBest(population + offspring, len(population))
        
        # Append the current generation statistics to the logbook
        record = mstats.compile(population) if mstats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if mstats:
            print(logbook.stream)
    
    # Get best individual from Hall of Fame (guaranteed to be the best ever found)
    best_individual = hof[0]
    
    return best_individual, logbook, pset


def train_and_test_problem_type(all_instances, problem_type, n_train=6, bool_capacity=True, population_size=30, generations=30):
    """
    Train on 'n_train' instances, Test on the rest.
    Prints detailed fitness comparison for every instance.
    """
    if not all_instances:
        print(f"No {problem_type} instances available, skipping...")
        return None
    
    # --- 1. THE SPLIT ---
    total_count = len(all_instances)
    
    if n_train >= total_count:
        train_instances = all_instances
        test_instances = []
        print(f"Warning: Using ALL {total_count} instances for training (Overfitting mode).")
    else:
        train_instances = all_instances[:n_train]
        test_instances = all_instances[n_train:]
    
    print(f"\n{'='*60}")
    print(f"Experiment Setup: {problem_type}")
    print(f"Training Set    : {len(train_instances)} instances")
    print(f"Testing Set     : {len(test_instances)} instances")
    print(f"{'='*60}")
    
    # --- 2. TRAINING ---
    print(f"Starting Genetic Programming on {len(train_instances)} training instances...")
    best_individual, logbook, pset = run_genetic_programming(
        instances=train_instances,
        bool_capacity=bool_capacity,
        population_size=population_size,
        generations=generations
    )
    
    # Get the evolved function
    func = gp.compile(expr=best_individual, pset=pset)
    print(f"\n{problem_type} Evolved scoring formula:")
    print(f"{str(best_individual)}")
    
    # --- 3. DETAILED EVALUATION FUNCTION ---
    def evaluate_set(name, dataset):
        if not dataset:
            return 0.0
            
        print(f"\n{'-'*30}")
        print(f"Evaluating on {name} Set")
        print(f"{'-'*30}")
        
        total_improvement = 0.0
        
        for i, instance in enumerate(dataset):

            time_start = time.time()
            
            # Create feature extractor
            feature_extractor = VRPFeatureExtractor(instance)
            
            # Solve using GP (Evolved Rule)
            if solve_with_scoring == 1:
                gp_routes = VRP_PROBLEM_TYPE.solve_with_scoring(instance, feature_extractor, func, bool_capacity)
            else:
                gp_routes = VRP_PROBLEM_TYPE.solve_with_scoring2(instance, feature_extractor, func, bool_capacity)
            gp_fitness = VRP_PROBLEM_TYPE.compute_cost(instance, gp_routes)

            # Solve using Baseline (Nearest Neighbor)
            nn_routes = nearest_neighbor_heuristic(instance, bool_capacity=bool_capacity)
            nn_fitness = VRP_PROBLEM_TYPE.compute_cost(instance, nn_routes)
            
            # Calculate Improvement
            if nn_fitness > 0:
                imp = ((nn_fitness - gp_fitness) / nn_fitness) * 100
            else:
                imp = 0.0

            time_end = time.time()
            elapsed = time_end - time_start
            
            #print(f"\nInstance {i+1}: {instance.name}")
            #print(f"  GP-DEAP Solution: Fitness = {gp_fitness:.2f}")
            #print(f"  Nearest Neighbor: Fitness = {nn_fitness:.2f}")
            #print(f"  Improvement over NN (fitness): {imp:.2f}%")
            #print(f"  Time taken: {elapsed:.2f} seconds")
            
            total_improvement += imp
            
        avg_imp = total_improvement / len(dataset)
        print(f"\n>> AVERAGE IMPROVEMENT on {name}: {avg_imp:.2f}%")
        return avg_imp

    # --- 4. RUN EVALUATION ---
    
    # Evaluate Training Data
    train_imp = evaluate_set("TRAINING", train_instances)
    
    # Evaluate Testing Data (if any)
    if test_instances:
        test_imp = evaluate_set("TESTING", test_instances)
    else:
        test_imp = 0.0
    
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"Training Avg Improvement: {train_imp:.2f}%")
    if test_instances:
        print(f"Testing Avg Improvement : {test_imp:.2f}%")
    else:
        print(f"Testing Avg Improvement : N/A")
    print(f"{'='*60}")
    
    return best_individual, logbook, pset

def plot_route(instance, route, title=None, ax=None, fitness=None):
    coords = instance.coords
    depot = getattr(instance, "depot", 0)
    n = instance.dimension
    
    # Check if route is a single route or list of routes
    if route and isinstance(route[0], list):
        routes = route  # Multiple routes
    else:
        routes = [route]  # Single route, wrap in list
    
    # Auto-detect problem type for title
    if title is None:
        has_tw = all(hasattr(instance, attr) for attr in ("ready_times", "due_dates", "service_times"))
        has_battery = getattr(instance, "battery_capacity", 0.0) > 0.0
        if has_battery and has_tw:
            base_title = "GVRPTW"
        elif has_battery:
            base_title = "GVRP"
        elif has_tw:
            base_title = "VRPTW"
        else:
            base_title = "CVRP"
        
        if len(routes) > 1:
            title = f"{base_title} - All Routes"
        else:
            title = f"{base_title} Route"
    
    # Add fitness to title if provided
    if fitness is not None:
        if len(routes) > 1:
            title = f"{title} (Fitness: {fitness:.2f}, {len(routes)} routes)"
        else:
            title = f"{title} (Fitness: {fitness:.2f})"
    
    # Create figure if needed
    if ax is None:
        fig_size = (12, 10) if len(routes) > 1 else (10, 8)
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    # Get node types if available (GVRP only)
    node_types = getattr(instance, "node_types", None)
    
    # Identify different node types
    if node_types is not None:
        # GVRP: use node_types array
        depots = [i for i in range(n) if node_types[i] == 0]
        customers = [i for i in range(n) if node_types[i] == 1]
        stations = [i for i in range(n) if node_types[i] == 2]
    else:
        # CVRP/VRPTW: depot is at index 0 (or instance.depot), rest are customers
        depots = [depot]
        customers = [i for i in range(n) if i != depot]
        stations = []
    
    # Plot all nodes as grey dots (background)
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgrey', s=20, alpha=0.5, zorder=1)
    
    # Plot customers (blue circles)
    if customers:
        customer_coords = coords[customers]
        ax.scatter(customer_coords[:, 0], customer_coords[:, 1], 
                  c='blue', marker='o', s=50, label='Customer', zorder=2)
    
    # Plot charging stations (red triangles) - only for GVRP
    if stations:
        station_coords = coords[stations]
        ax.scatter(station_coords[:, 0], station_coords[:, 1], 
                  c='red', marker='^', s=80, label='Charging Station', zorder=3)
    
    # Plot depot (black square)
    depot_coords = coords[depots]
    ax.scatter(depot_coords[:, 0], depot_coords[:, 1], 
              c='black', marker='s', s=150, label='Depot', zorder=4, edgecolors='white', linewidths=2)
    
    # Plot all routes
    if len(routes) > 1:
        # Multiple routes: use different colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
        for idx, route_path in enumerate(routes):
            if len(route_path) > 1:
                path_coords = coords[route_path]
                ax.plot(path_coords[:, 0], path_coords[:, 1], 
                       '-', linewidth=2, alpha=0.7, color=colors[idx], 
                       label=f'Route {idx+1}', zorder=2)
                
                # Add arrows to show direction
                for i in range(len(route_path) - 1):
                    start = coords[route_path[i]]
                    end = coords[route_path[i + 1]]
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    if np.sqrt(dx**2 + dy**2) > 0.01:
                        ax.annotate('', xy=end, xytext=start,
                                   arrowprops=dict(arrowstyle='->', 
                                                  color=colors[idx], 
                                                  lw=1.5, alpha=0.6))
    else:
        # Single route: use blue
        route_path = routes[0]
        if len(route_path) > 1:
            path_coords = coords[route_path]
            ax.plot(path_coords[:, 0], path_coords[:, 1], 
                   'b-', linewidth=2, alpha=0.7, label='Route', zorder=2)
            
            # Add arrows to show direction
            for i in range(len(route_path) - 1):
                start = coords[route_path[i]]
                end = coords[route_path[i + 1]]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                if np.sqrt(dx**2 + dy**2) > 0.01:
                    ax.annotate('', xy=end, xytext=start,
                               arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.6))
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    ax.legend(loc='best', ncol=2 if len(routes) > 1 else 1)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    return ax

def load_instances_by_type():
    """Load instances grouped by problem type. Automatically discovers all .vrp, .txt, and .xml files."""
    cvrp_instances = []
    # Load all .vrp files from Sets/Set_A
    vrp_files = sorted(glob.glob("Sets/Set_A/*.vrp"))
    for filepath in vrp_files:
        try:
            cvrp_instances.append(VRPInstance(filepath))
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    vrptw_instances = []
    
    # Load first 30 .txt files from Sets/Vrp-Set-HG
    # Use natural sort (numeric-aware) so C_1_B comes before C_10_B
    def natural_sort_key(filename):
        """Natural sort key that handles numbers in filenames."""
        import re
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'(\d+)', os.path.basename(filename))]
    
    txt_files = sorted(glob.glob("Sets/Vrp-Set-HG/*.txt"), key=natural_sort_key)[:30]
    del txt_files[25]
    for filepath in txt_files:
        try:
            vrptw_instances.append(VRPTWInstance(filepath))
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    
    gvrp_instances = []
    
    # Load all .xml files from Sets/felipe-et-al-2014
    xml_files = sorted(glob.glob("Sets/felipe-et-al-2014/*.xml"))
    for filepath in xml_files:
        try:
            gvrp_instances.append(GVRPMultiTechInstance(filepath))
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    
    print(f"Loaded {len(cvrp_instances)} CVRP instances")
    print(f"Loaded {len(vrptw_instances)} VRPTW instances")
    print(f"Loaded {len(gvrp_instances)} GVRP instances")
    
    return cvrp_instances, vrptw_instances, gvrp_instances



solve_with_scoring = 2

def main():
    # Choose variants
    bool_capacity = True
    bool_TW = True
    bool_green = True

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

    start_time = time.time()

    if instances:
        results = train_and_test_problem_type(
            all_instances=instances,
            problem_type=problem_type,
            bool_capacity=bool_capacity,
            n_train=5,
            population_size=20,
            generations=20
        )
    else:
        print(f"No instances loaded for {problem_type}")
            
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    
    if results:
        vrp_best, vrp_logbook, vrp_pset = results
        
        # Plot all routes for the first instance
        if instances:
            instance = instances[0]
            feature_extractor = VRPFeatureExtractor(instance)
            func = gp.compile(expr=vrp_best, pset=vrp_pset)
            if solve_with_scoring == 1:
                gp_solution = VRP_PROBLEM_TYPE.solve_with_scoring(instance, feature_extractor, func, bool_capacity)
            else:
                gp_solution = VRP_PROBLEM_TYPE.solve_with_scoring2(instance, feature_extractor, func, bool_capacity)
            gp_fitness = VRP_PROBLEM_TYPE.compute_cost(instance, gp_solution)
            
            if gp_solution and len(gp_solution) > 0:
                plot_route(instance, gp_solution, title=f"{instance.name}", fitness=gp_fitness)
                #plt.show()


if __name__ == "__main__":
    main()
