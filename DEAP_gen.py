#!/usr/bin/env python3
"""
Genetic Programming solution for VRP variants using DEAP framework.
"""

import numpy as np
from deap import base, creator, tools, gp, algorithms
from parser import VRPInstance, VRPTWInstance, GVRPMultiTechInstance, CordeauMDVRPInstance
from vrp_feature_extractor import VRPFeatureExtractor
from basic_heuristics import nearest_neighbor_heuristic, saving_heuristic
from vrp_problem import VRP_PROBLEM_TYPE
from data_generation import remove_tw_from_gvrp, evrptw_to_multi_depot
import time
import copy
import matplotlib.pyplot as plt
import glob
import os
import operator

def create_individual(pset, min_initial=2, max_initial=6):
    """Create a GP individual."""
    return gp.PrimitiveTree(gp.genHalfAndHalf(pset, min_=min_initial, max_=max_initial))


def create_toolbox(min_initial=2, max_initial=6):
    """Create the DEAP toolbox"""
    pset = VRP_PROBLEM_TYPE.create_primitive_set(gp)
    
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", create_individual, pset, min_initial=min_initial, max_initial=max_initial)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    return toolbox, pset


def evaluate_individual(individual, instances, bool_capacity=True):
    """Evaluate a GP individual on VRP instances."""
    func = gp.compile(expr=individual, pset=VRP_PROBLEM_TYPE.create_primitive_set(gp))
    
    total_fitness = 0.0
    
    for instance in instances:
        feature_extractor = VRPFeatureExtractor(instance)
        
        solution = VRP_PROBLEM_TYPE.solve_with_scoring(
            instance, feature_extractor, func, bool_capacity
        )
        fitness = VRP_PROBLEM_TYPE.compute_cost(instance, solution)
        total_fitness += fitness
    
    tree_size = len(individual)
    tree_size_penalty = 0.1 * tree_size 
    total_fitness += tree_size_penalty
    
    return (total_fitness,)


def run_genetic_programming(
    instances,
    bool_capacity: bool = True,
    population_size: int = 50,
    generations: int = 50,
    time_limit_sec: float | None = None,
    seed: int | None = None,
    cxpb: float = 1.0,
    mutpb: float = 0.2,
    tournsize: int = 3,
    min_initial=2,
    max_initial=6,
    min_mut=0,
    max_mut=3,
):
    """Run genetic programming to evolve VRP scoring function."""
    
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    toolbox, pset = create_toolbox(min_initial=min_initial, max_initial=max_initial)
    
    def evaluate_with_problem_type(individual):
        return evaluate_individual(individual, instances, bool_capacity)
    
    toolbox.register("evaluate", evaluate_with_problem_type)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=min_mut, max_=max_mut)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    
    hof = tools.HallOfFame(1)
    
    population = toolbox.population(n=population_size)
    
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    hof.update(population)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
    
    record = mstats.compile(population) if mstats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if mstats:
        print(logbook.stream)
    
    start_time = time.time()
    for gen in range(1, generations + 1):
        if time_limit_sec is not None and (time.time() - start_time) >= time_limit_sec:
            break
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if np.random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        hof.update(offspring)
        
        population[:] = offspring
        if hof:
            idx = np.random.randint(len(population))
            population[idx] = toolbox.clone(hof[0])
        
        record = mstats.compile(population) if mstats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if mstats:
            print(logbook.stream)
    
    best_individual = hof[0]
    
    return best_individual, logbook, pset


def train_and_test_problem_type(
    all_instances,
    problem_type,
    n_train: int = 6,
    n_test: int = -1,
    bool_capacity: bool = True,
    population_size: int = 30,
    generations: int = 30,
    time_limit_sec: float | None = None,
    seed: int | None = None,
    cxpb: float = 0.8,
    mutpb: float = 0.15,
    evaluate_test_split: bool = True,
    tournsize: int = 3,
    min_initial: int = 2,
    max_initial: int = 6,
    min_mut: int = 0,
    max_mut: int = 3,
):
    """
    Train a GP scoring function for one problem variant and evaluate it.

    Parameters
    ----------
    all_instances : list
        Full list of parsed instances for the selected problem variant.
    problem_type : str
        Variant label used in logs/output (for example `VRP`, `VRPTW`).
    n_train : int
        Number of instances used for training.
    n_test : int
        Number of held-out test instances. If set to `-1`, all remaining
        instances after the training split are used.
    bool_capacity : bool
        If True, enforce vehicle capacity constraints during routing.
    population_size : int
        GP population size.
    generations : int
        Maximum number of GP generations.
    time_limit_sec : float or None
        Optional wall-clock budget (seconds) for GP evolution. If None,
        evolution runs for the full number of generations.
    seed : int or None
        Random seed for reproducibility. If None, no explicit seed is set.
    cxpb : float
        Crossover probability used by GP.
    mutpb : float
        Mutation probability used by GP.
    evaluate_test_split : bool
        If True, evaluate and print metrics on the held-out test split.
        If False, only training metrics are evaluated.
    tournsize : int
        Tournament size for selection.
    min_initial : int
        Minimum initial tree depth when creating individuals.
    max_initial : int
        Maximum initial tree depth when creating individuals.
    min_mut : int
        Minimum mutation tree depth.
    max_mut : int
        Maximum mutation tree depth.

    Returns
    -------
    tuple or None
        `(best_individual, logbook, pset)` when training runs, otherwise
        `None` if `all_instances` is empty.
    """
    if not all_instances:
        print(f"No {problem_type} instances available, skipping...")
        return None
    
    total_count = len(all_instances)
    if n_test == -1:
        if n_train >= total_count:
            train_instances = all_instances
            test_instances = []
            print(f"Warning: Using ALL {total_count} instances for training (Overfitting mode).")
        else:
            train_instances = all_instances[:n_train]
            test_instances = all_instances[n_train:]
    else:
        if n_train + n_test > total_count:
            n_test = total_count - n_train
            print(f"Warning: Adjusted n_test to {n_test} due to insufficient instances.")
        train_instances = all_instances[:n_train]
        test_instances = all_instances[n_train:n_train + n_test]
    
    print(f"\n{'='*60}")
    print(f"Experiment Setup: {problem_type}")
    print(f"Training Set    : {len(train_instances)} instances")
    if evaluate_test_split:
        print(f"Testing Set     : {len(test_instances)} instances")
    else:
        print(
            f"Held-out instances (not evaluated here): {len(test_instances)}"
            if test_instances
            else "Testing Set     : 0 instances"
        )
    print(f"{'='*60}")
    
    print(f"Starting Genetic Programming on {len(train_instances)} training instances...")
    best_individual, logbook, pset = run_genetic_programming(
        instances=train_instances,
        bool_capacity=bool_capacity,
        population_size=population_size,
        generations=generations,
        time_limit_sec=time_limit_sec,
        seed=seed,
        cxpb=cxpb,
        mutpb=mutpb,
        tournsize=tournsize,
        min_initial=min_initial,
        max_initial=max_initial,
        min_mut=min_mut,
        max_mut=max_mut,
    )
    
    func = gp.compile(expr=best_individual, pset=pset)
    print(f"\n{problem_type} Evolved scoring formula:")
    print(f"{str(best_individual)}")
    
    def evaluate_set(name, dataset):
        if not dataset:
            return 0.0
        
        total_improvementNN = 0.0
        total_improvementS = 0.0

        for i, instance in enumerate(dataset):

            time_start = time.time()
            
            feature_extractor = VRPFeatureExtractor(instance)
            
            gp_routes = VRP_PROBLEM_TYPE.solve_with_scoring(
                instance, feature_extractor, func, bool_capacity
            )
            gp_fitness = VRP_PROBLEM_TYPE.compute_cost(instance, gp_routes)

            nn_routes = nearest_neighbor_heuristic(instance, bool_capacity=bool_capacity)
            nn_fitness = VRP_PROBLEM_TYPE.compute_cost(instance, nn_routes)

            s_routes2 = saving_heuristic(instance, bool_capacity=bool_capacity)

            s_fitness = VRP_PROBLEM_TYPE.compute_cost(instance, s_routes2)
            
            if nn_fitness > 0:
                impNN = ((nn_fitness - gp_fitness) / nn_fitness) * 100
            else:
                impNN = 0.0
            if s_fitness > 0:
                impS = ((s_fitness - gp_fitness) / s_fitness) * 100
            else:                
                impS = 0.0

            time_end = time.time()
            elapsed = time_end - time_start
            
            print(f"\nInstance {i+1}: {instance.name}")
            print(f"  GP-DEAP Solution: Fitness = {gp_fitness:.2f}")
            print(f"  Nearest Neighbor: Fitness = {nn_fitness:.2f}")
            print(f"  Saving Heuristic : Fitness = {s_fitness:.2f}")
            print(f"  Improvement over NN (fitness): {impNN:.2f}%")
            print(f"  Improvement over Saving (fitness): {impS:.2f}%")
            print(f"  Time taken: {elapsed:.2f} seconds")
            
            total_improvementNN += impNN
            total_improvementS += impS
            
        return total_improvementNN / len(dataset), total_improvementS / len(dataset)

    
    train_impNN, train_impS = evaluate_set("TRAINING", train_instances)
    
    if evaluate_test_split and test_instances:
        test_impNN, test_impS = evaluate_set("TESTING", test_instances)
    else:
        test_impNN = 0.0
        test_impS = 0.0
    
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"Training Avg Improvement (NN): {train_impNN:.2f}%")
    print(f"Training Avg Improvement (Saving): {train_impS:.2f}%")
    if evaluate_test_split:
        print(f"Testing Avg Improvement (NN): {test_impNN:.2f}%")
        print(f"Testing Avg Improvement (Saving): {test_impS:.2f}%")
    else:
        print("Held-out test split: not evaluated (evaluate_test_split=False).")
    print(f"{'='*60}")
    
    return best_individual, logbook, pset


def plot_route(instance, route, title=None, ax=None, fitness=None):
    """
    Function to help visualize constructed routes.
    """
    coords = instance.coords
    depot = getattr(instance, "depot", 0)
    n = instance.dimension
    
    if route and isinstance(route[0], list):
        routes = route
    else:
        routes = [route]
    
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
            base_title = "VRP"
        
        if len(routes) > 1:
            title = f"{base_title} - All Routes"
        else:
            title = f"{base_title} Route"
    
    if fitness is not None:
        if len(routes) > 1:
            title = f"{title} (Fitness: {fitness:.2f}, {len(routes)} routes)"
        else:
            title = f"{title} (Fitness: {fitness:.2f})"
    
    if ax is None:
        fig_size = (12, 10) if len(routes) > 1 else (10, 8)
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    node_types = getattr(instance, "node_types", None)
    
    if node_types is not None:
        depots = [i for i in range(n) if node_types[i] == 0]
        customers = [i for i in range(n) if node_types[i] == 1]
        stations = [i for i in range(n) if node_types[i] == 2]
    else:
        depots = [depot]
        customers = [i for i in range(n) if i != depot]
        stations = []
    
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgrey', s=20, alpha=0.5, zorder=1)
    
    if customers:
        customer_coords = coords[customers]
        ax.scatter(customer_coords[:, 0], customer_coords[:, 1], 
                  c='blue', marker='o', s=50, label='Customer', zorder=2)
    
    if stations:
        station_coords = coords[stations]
        ax.scatter(station_coords[:, 0], station_coords[:, 1], 
                  c='red', marker='^', s=80, label='Charging Station', zorder=3)
    
    depot_coords = coords[depots]
    ax.scatter(depot_coords[:, 0], depot_coords[:, 1], 
              c='black', marker='s', s=150, label='Depot', zorder=4, edgecolors='white', linewidths=2)
    
    if len(routes) > 1:
        colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
        for idx, route_path in enumerate(routes):
            if len(route_path) > 1:
                path_coords = coords[route_path]
                ax.plot(path_coords[:, 0], path_coords[:, 1], 
                       '-', linewidth=2, alpha=0.7, color=colors[idx], 
                       label=f'Route {idx+1}', zorder=2)
                
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
        route_path = routes[0]
        if len(route_path) > 1:
            path_coords = coords[route_path]
            ax.plot(path_coords[:, 0], path_coords[:, 1], 
                   'b-', linewidth=2, alpha=0.7, label='Route', zorder=2)
            
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
    vrp_files = sorted(glob.glob("Sets/Set_A/*.vrp"))
    for filepath in vrp_files:
        try:
            cvrp_instances.append(VRPInstance(filepath))
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    vrptw_instances = []
    def natural_sort_key(filename):
        """Natural sort key that handles numbers in filenames."""
        import re
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'(\d+)', os.path.basename(filename))]
    txt_files = sorted(glob.glob("Sets/Vrp-Set-HG/*.txt"), key=natural_sort_key)

    preferred_vrptw = ["C1_2_1.txt", "R1_2_1.txt", "RC1_2_1.txt"] # Smaller instances into beginning
    ordered_txt_files = []
    remaining_txt_files = list(txt_files)
    for name in preferred_vrptw:
        for f in list(remaining_txt_files):
            if os.path.basename(f) == name:
                ordered_txt_files.append(f)
                remaining_txt_files.remove(f)
                break
    ordered_txt_files.extend(remaining_txt_files)

    for filepath in ordered_txt_files:
        try:
            vrptw_instances.append(VRPTWInstance(filepath))
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    
    gvrp_instances = []
    evrptw_files = sorted(
        [
            f
            for f in glob.glob("Sets/evrptw_instances/*.txt")
            if not os.path.basename(f) in ["readme.txt"]
        ]
    )

    preferred_gvrp = ["c101_21.txt", "r101_21.txt", "rc101_21.txt"] # Smaller instances into beginning
    ordered_evrptw_files = []
    remaining_evrptw = list(evrptw_files)
    for name in preferred_gvrp:
        for f in list(remaining_evrptw):
            if os.path.basename(f) == name:
                ordered_evrptw_files.append(f)
                remaining_evrptw.remove(f)
                break
    ordered_evrptw_files.extend(remaining_evrptw)

    for filepath in ordered_evrptw_files:
        try:
            gvrp_instances.append(GVRPMultiTechInstance(filepath))
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    mdvrp_instances = []
    mdvrptw_instances = []
    mdvrp_files = sorted(glob.glob("Sets/C-mdvrp/*"))
    for filepath in mdvrp_files:
        try:
            mdvrp_instances.append(CordeauMDVRPInstance(filepath))
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    mdvrptw_files = sorted(glob.glob("Sets/C-mdvrptw/*"))
    for filepath in mdvrptw_files:
        try:
            mdvrptw_instances.append(CordeauMDVRPInstance(filepath))
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    print(f"Loaded {len(cvrp_instances)} VRP instances")
    print(f"Loaded {len(vrptw_instances)} VRPTW instances")
    print(f"Loaded {len(gvrp_instances)} GVRP instances")
    print(f"Loaded {len(mdvrp_instances)} MDVRP instances")
    print(f"Loaded {len(mdvrptw_instances)} MDVRPTW instances")
    return cvrp_instances, vrptw_instances, gvrp_instances, mdvrp_instances, mdvrptw_instances


def main():

    bool_capacity = 1
    bool_TW = 0
    bool_green = 0
    bool_MD = 1

    cvrp_instances, vrptw_instances, gvrp_instances, mdvrp_instances, mdvrptw_instances = load_instances_by_type()
    results = None

    problem_map = { # Problem map for all problem types
        (False, False, False): ("VRP", cvrp_instances),
        (True, False, False):  ("VRPTW", vrptw_instances),
        (False, True, False):  ("GVRP", remove_tw_from_gvrp(copy.deepcopy(gvrp_instances))),
        (True, True, False):   ("G-VRPTW", gvrp_instances),
        (False, False, True):  ("MDVRP", mdvrp_instances),
        (True, False, True):   ("MDVRPTW", mdvrptw_instances),
        (False, True, True):   ("GVRP-MD", evrptw_to_multi_depot(remove_tw_from_gvrp(copy.deepcopy(gvrp_instances)))),
        (True, True, True):    ("G-VRPTW-MD", evrptw_to_multi_depot(gvrp_instances))
    }

    problem_type, instances = problem_map.get((bool_TW, bool_green, bool_MD))

    start_time = time.time()

    if instances:
        results = train_and_test_problem_type(
            all_instances=instances,
            problem_type=problem_type,
            bool_capacity=bool_capacity,
            n_train=5,
            n_test=-1,
            population_size=1,
            generations=1,
            cxpb=1.0,
            mutpb=0.2,
        )
    else:
        print(f"No instances loaded for {problem_type}")
            
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    
    # Graph generation
    '''
    if results:
        vrp_best, vrp_logbook, vrp_pset = results
        
        if instances:
            instance = instances[3]
            feature_extractor = VRPFeatureExtractor(instance)
            func = gp.compile(expr=vrp_best, pset=vrp_pset)
            gp_solution = VRP_PROBLEM_TYPE.solve_with_scoring(instance, feature_extractor, func, bool_capacity)
            gp_fitness = VRP_PROBLEM_TYPE.compute_cost(instance, gp_solution)
            
            if gp_solution and len(gp_solution) > 0:
                plot_route(instance, gp_solution, title=f"{instance.name}", fitness=gp_fitness)
                plt.show()
    '''

if __name__ == "__main__":
    main()
