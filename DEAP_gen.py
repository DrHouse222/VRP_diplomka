#!/usr/bin/env python3
"""
Genetic Programming solution for VRP variants using DEAP framework.
Evolves a scoring function that evaluates requests based on problem-specific features.
Supports CVRP, CVRPTW, and extensible for future problem types.
"""

import random
import numpy as np
import operator
from typing import List, Tuple, Dict, Any, Callable, Optional
from abc import ABC, abstractmethod
from deap import base, creator, tools, gp, algorithms
from parser import VRPInstance, VRPTWInstance, VRPFeatureExtractor
from basic_heuristics import nearest_neighbor_heuristic, savings_heuristic


class ProblemType(ABC):
    """Abstract base class for problem type configurations."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Problem type name."""
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """List of feature names for this problem type."""
        pass
    
    @property
    @abstractmethod
    def num_features(self) -> int:
        """Number of features for this problem type."""
        pass
    
    @abstractmethod
    def create_primitive_set(self) -> gp.PrimitiveSet:
        """Create GP primitive set for this problem type."""
        pass
    
    @abstractmethod
    def extract_feature_values(self, features: Dict[str, float]) -> List[float]:
        """Extract feature values in the order expected by GP function."""
        pass
    
    @abstractmethod
    def solve_with_scoring(self, instance, feature_extractor, scoring_func) -> List[List[int]]:
        """Solve problem using GP-evolved scoring function."""
        pass
    
    @abstractmethod
    def evaluate_solution(self, instance, solution) -> Tuple[float, float]:
        """Evaluate solution quality. Returns (distance, violation)."""
        pass


class CVRPProblemType(ProblemType):
    """Configuration for Capacitated VRP."""
    
    @property
    def name(self) -> str:
        return "CVRP"
    
    @property
    def feature_names(self) -> List[str]:
        return [
            'dist_to_depot', 'dist_from_current', 'dist_to_depot_from_request',
            'demand', 'remaining_capacity', 'capacity_utilization', 'demand_ratio',
            'route_length', 'is_empty_route', 'dist_last_to_depot', 'savings', 'norm_savings'
        ]
    
    @property
    def num_features(self) -> int:
        return 12
    
    def create_primitive_set(self) -> gp.PrimitiveSet:
        """Create the primitive set for CVRP."""
        pset = gp.PrimitiveSet("MAIN", self.num_features)
        
        # Rename arguments to meaningful names
        for i, name in enumerate(self.feature_names):
            pset.renameArguments(**{f"ARG{i}": name})
        
        # Mathematical operators
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        
        # Protected division
        def protected_div(left, right):
            try:
                return left / right if abs(right) > 1e-6 else 1.0
            except:
                return 1.0
        pset.addPrimitive(protected_div, 2)
        
        # Constants
        pset.addTerminal(0.0)
        pset.addTerminal(1.0)
        pset.addTerminal(2.0)
        pset.addTerminal(0.5)
        pset.addTerminal(-1.0)
        
        return pset
    
    def extract_feature_values(self, features: Dict[str, float]) -> List[float]:
        """Extract feature values in the order expected by GP function."""
        return [features[name] for name in self.feature_names]
    
    def solve_with_scoring(self, instance, feature_extractor, scoring_func) -> List[List[int]]:
        """Solve CVRP using GP-evolved scoring function."""
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
                    
                    feature_values = self.extract_feature_values(features)
                    
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
    
    def evaluate_solution(self, instance, solution) -> Tuple[float, float]:
        """Evaluate CVRP solution quality."""
        return instance.cost(solution)


class CVRPTWProblemType(ProblemType):
    """Configuration for Capacitated VRP with Time Windows."""
    
    @property
    def name(self) -> str:
        return "CVRPTW"
    
    @property
    def feature_names(self) -> List[str]:
        return [
            'dist_to_depot', 'dist_from_current', 'dist_to_depot_from_request',
            'demand', 'remaining_capacity', 'capacity_utilization', 'demand_ratio',
            'route_length', 'is_empty_route', 'dist_last_to_depot', 'savings', 'norm_savings',
            'arrival_time', 'ready_time', 'due_time', 'service_time', 'start_service_time',
            'finish_service_time', 'wait_time', 'tw_feasible', 'slack_to_due', 'remaining_tw_from_now'
        ]
    
    @property
    def num_features(self) -> int:
        return 22  # 12 base + 10 TW features
    
    def create_primitive_set(self) -> gp.PrimitiveSet:
        """Create the primitive set for CVRPTW."""
        pset = gp.PrimitiveSet("MAIN", self.num_features)
        
        # Rename arguments to meaningful names
        for i, name in enumerate(self.feature_names):
            pset.renameArguments(**{f"ARG{i}": name})
        
        # Mathematical operators
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        
        # Protected division
        def protected_div(left, right):
            try:
                return left / right if abs(right) > 1e-6 else 1.0
            except:
                return 1.0
        pset.addPrimitive(protected_div, 2)
        
        # Additional functions for time windows
        def max_func(a, b):
            return max(a, b)
        def min_func(a, b):
            return min(a, b)
        
        pset.addPrimitive(max_func, 2)
        pset.addPrimitive(min_func, 2)
        
        # Constants
        pset.addTerminal(0.0)
        pset.addTerminal(1.0)
        pset.addTerminal(2.0)
        pset.addTerminal(0.5)
        pset.addTerminal(-1.0)
        
        return pset
    
    def extract_feature_values(self, features: Dict[str, float]) -> List[float]:
        """Extract feature values in the order expected by GP function."""
        # Ensure we have exactly 23 features for VRPTW
        feature_values = []
        for name in self.feature_names:
            feature_values.append(features.get(name, 0.0))
        
        return feature_values
    
    def solve_with_scoring(self, instance, feature_extractor, scoring_func) -> List[List[int]]:
        """Solve CVRPTW using GP-evolved scoring function."""
        n = instance.dimension
        unvisited = set(range(1, n))  # All customers except depot
        routes = []
        
        while unvisited:
            route = [instance.depot]
            load = 0
            current_position = instance.depot
            current_time = 0.0
            
            while True:
                # Get capacity-feasible candidates
                capacity_ok = [c for c in unvisited if load + instance.demands[c] <= instance.capacity]
                
                if not capacity_ok:
                    # Return to depot
                    route.append(instance.depot)
                    break
                
                # Check time window feasibility and score candidates
                feasible_candidates = []
                scores = []
                
                for candidate in capacity_ok:
                    travel = instance.dist_matrix[current_position, candidate]
                    arrival = current_time + travel
                    
                    # Check time window feasibility
                    if arrival > instance.due_dates[candidate]:
                        continue  # Skip infeasible candidates
                    
                    feasible_candidates.append(candidate)
                    
                    # Extract features with current time
                    features = feature_extractor.extract_features(
                        candidate, route, load, current_position, current_time
                    )
                    
                    feature_values = self.extract_feature_values(features)
                    
                    try:
                        score = scoring_func(*feature_values)
                        scores.append(score)
                    except Exception as e:
                        # If GP function fails, use distance as fallback
                        scores.append(travel)
                
                if not feasible_candidates:
                    # No time-feasible candidates, return to depot
                    route.append(instance.depot)
                    break
                
                # Choose candidate with lowest score
                best_candidate = feasible_candidates[np.argmin(scores)]
                
                # Update route and time
                route.append(best_candidate)
                load += instance.demands[best_candidate]
                unvisited.remove(best_candidate)
                
                # Update time
                travel = instance.dist_matrix[current_position, best_candidate]
                arrival = current_time + travel
                start_service = max(arrival, instance.ready_times[best_candidate])
                current_time = start_service + instance.service_times[best_candidate]
                current_position = best_candidate
            
            routes.append(route)
        
        return routes
    
    def evaluate_solution(self, instance, solution) -> Tuple[float, float]:
        """Evaluate CVRPTW solution quality."""
        return instance.cost(solution)


class ProblemTypeRegistry:
    """Registry for different problem types."""
    
    def __init__(self):
        self._types = {}
        self._register_default_types()
    
    def _register_default_types(self):
        """Register default problem types."""
        self.register(CVRPProblemType())
        self.register(CVRPTWProblemType())
    
    def register(self, problem_type: ProblemType):
        """Register a new problem type."""
        self._types[problem_type.name] = problem_type
    
    def get(self, name: str) -> Optional[ProblemType]:
        """Get problem type by name."""
        return self._types.get(name)
    
    def list_types(self) -> List[str]:
        """List all registered problem types."""
        return list(self._types.keys())
    
    def detect_problem_type(self, instance) -> str:
        """Auto-detect problem type from instance."""
        if hasattr(instance, 'ready_times') and hasattr(instance, 'due_dates'):
            return "CVRPTW"
        else:
            return "CVRP"


# Global registry
PROBLEM_REGISTRY = ProblemTypeRegistry()


def create_primitive_set(problem_type: str = "CVRP"):
    """Create the primitive set for genetic programming."""
    problem_config = PROBLEM_REGISTRY.get(problem_type)
    if problem_config is None:
        raise ValueError(f"Unknown problem type: {problem_type}")
    return problem_config.create_primitive_set()


def create_individual(pset):
    """Create a GP individual."""
    return gp.PrimitiveTree(gp.genHalfAndHalf(pset, min_=1, max_=3))


def create_toolbox(pset, problem_type: str = "CVRP"):
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
    
    # Store pset and problem type in toolbox for later use
    toolbox.pset = pset
    toolbox.problem_type = problem_type
    
    return toolbox


def evaluate_individual(individual, instances=None, feature_extractors=None, problem_type="CVRP"):
    """
    Evaluate a GP individual on VRP instances.
    Lower score means better solution.
    """
    if instances is None or feature_extractors is None:
        return (float('inf'),)
    
    try:
        problem_config = PROBLEM_REGISTRY.get(problem_type)
        if problem_config is None:
            return (float('inf'),)
        
        # Compile the individual into a function
        pset = problem_config.create_primitive_set()
        func = gp.compile(expr=individual, pset=pset)
        
        total_cost = 0.0
        num_evaluations = 0
        
        for instance, feature_extractor in zip(instances, feature_extractors):
            # Generate solution using the GP-evolved scoring function
            solution = problem_config.solve_with_scoring(instance, feature_extractor, func)
            
            # Evaluate solution quality
            total_distance, capacity_violation = problem_config.evaluate_solution(instance, solution)
            
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




def run_genetic_programming(instances: List, problem_type: str = "auto", population_size: int = 50, 
                          generations: int = 50, cxpb: float = 0.7, mutpb: float = 0.3):
    """
    Run genetic programming to evolve VRP scoring function.
    
    Args:
        instances: List of VRP instances for training
        problem_type: Problem type ("auto", "CVRP", "CVRPTW")
        population_size: Size of population
        generations: Number of generations
        cxpb: Crossover probability
        mutpb: Mutation probability
    
    Returns:
        Best evolved individual and statistics
    """
    
    # Auto-detect problem type if needed
    if problem_type == "auto":
        problem_type = PROBLEM_REGISTRY.detect_problem_type(instances[0])
    
    print(f"Using problem type: {problem_type}")
    
    feature_extractors = [VRPFeatureExtractor(inst) for inst in instances]
    
    pset = create_primitive_set(problem_type)
    toolbox = create_toolbox(pset, problem_type)
    
    # Set up evaluation function with instances
    def evaluate_with_instances(individual):
        return evaluate_individual(individual, instances, feature_extractors, problem_type)
    
    toolbox.register("evaluate", evaluate_with_instances)
    
    # Initial population
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
    print(f"Problem type: {problem_type}")
    print(f"Population size: {population_size}, Generations: {generations}")
    print(f"Crossover probability: {cxpb}, Mutation probability: {mutpb}")
    
    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb, mutpb, generations, 
        stats=mstats, halloffame=hof, verbose=True
    )
    
    return hof[0], logbook, pset, problem_type


def train_and_test_problem_type(instances, problem_type, population_size=30, generations=30):
    """Train and test GP for a specific problem type."""
    print(f"\n{'='*60}")
    print(f"Training {problem_type} model with {len(instances)} instances")
    print(f"{'='*60}")
    
    # Show instances
    for i, instance in enumerate(instances):
        print(f"  {i+1}. {instance.name} (n={instance.dimension}, cap={instance.capacity})")
    
    # Train GP model
    best_individual, logbook, pset, detected_type = run_genetic_programming(
        instances=instances,
        problem_type=problem_type,
        population_size=population_size,
        generations=generations,
        cxpb=0.7,
        mutpb=0.3
    )
    
    print(f"\n{problem_type} Evolution completed!")
    print(f"Best individual fitness: {best_individual.fitness.values[0]:.2f}")
    
    # Test the evolved function
    print(f"\nTesting {problem_type} evolved scoring function...")
    
    problem_config = PROBLEM_REGISTRY.get(problem_type)
    
    for i, instance in enumerate(instances):
        print(f"\nInstance {i+1}: {instance.name}")
        
        # Create feature extractor
        feature_extractor = VRPFeatureExtractor(instance)
        
        # Compile best individual
        best_func = gp.compile(expr=best_individual, pset=pset)
        
        # Solve using evolved function
        gp_solution = problem_config.solve_with_scoring(instance, feature_extractor, best_func)
        gp_distance, gp_violation = problem_config.evaluate_solution(instance, gp_solution)
        
        # Compare with heuristics (map problem types to heuristic format)
        heuristic_problem_type = "vrp" if problem_type == "CVRP" else "vrptw"
        
        nn_routes = nearest_neighbor_heuristic(instance, problem=heuristic_problem_type)
        nn_distance, nn_violation = instance.cost(nn_routes)
        
        savings_routes = savings_heuristic(instance, problem=heuristic_problem_type)
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
    print(f"\n{problem_type} Evolved scoring formula:")
    print(f"{best_individual}")
    
    return best_individual, logbook, pset


def load_instances_by_type():
    """Load instances grouped by problem type."""
    cvrp_instances = [
        VRPInstance("Set_A/A-n32-k5.vrp"),
        # Add more CVRP instances as needed
        # VRPInstance("Set_A/A-n33-k5.vrp"),
        # VRPInstance("Set_A/A-n34-k5.vrp"),
    ]
    
    vrptw_instances = [
        #VRPTWInstance("Vrp-Set-HG/C1_2_1.txt"),
        VRPTWInstance("Vrp-Set-HG/C1_2_2.txt"),
        #VRPTWInstance("Vrp-Set-HG/C1_2_3.txt"),
        #VRPTWInstance("Vrp-Set-HG/C1_2_4.txt"),
    ]
    
    return cvrp_instances, vrptw_instances


def main():
    """Main function to run the genetic programming solver with separate training."""
    # Load instances grouped by problem type
    cvrp_instances, vrptw_instances = load_instances_by_type()
    
    print("Loaded VRP instances by type:")
    print(f"CVRP instances: {len(cvrp_instances)}")
    print(f"CVRPTW instances: {len(vrptw_instances)}")
    
    # Train and test CVRP model
    
    cvrp_results = None
    '''
    if cvrp_instances:
        cvrp_results = train_and_test_problem_type(
            instances=cvrp_instances,
            problem_type="CVRP",
            population_size=30,
            generations=30
        )'''
    
    # Train and test VRPTW model
    vrptw_results = None
    if vrptw_instances:
        vrptw_results = train_and_test_problem_type(
            instances=vrptw_instances,
            problem_type="CVRPTW",
            population_size=10,
            generations=10
        )
    
    # Summary
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