#!/usr/bin/env python3
"""
Genetic Programming solution for VRP variants using pyGPGOMEA framework.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from pyGPGOMEA import GPGOMEARegressor as GPG
from sklearn.metrics import mean_squared_error
from parser import VRPInstance, VRPTWInstance, GVRPMultiTechInstance, VRPFeatureExtractor
from basic_heuristics import nearest_neighbor_heuristic, savings_heuristic
from problem_types import VRPProblemType, VRP_PROBLEM_TYPE


class VRPDataGenerator:
    """Generates training data for VRP scoring function using multiple instances."""
    
    def __init__(self, instances: List, problem_type: str = "auto"):
        self.instances = instances
        self.problem_type = problem_type
        
        # Auto-detect problem type (for display only; unified model handles all)
        if problem_type == "auto":
            inst = instances[0]
            if hasattr(inst, "node_types") and hasattr(inst, "battery_capacity"):
                self.problem_type = "GVRP"
            elif hasattr(inst, "ready_times") and hasattr(inst, "due_dates"):
                self.problem_type = "CVRPTW"
            else:
                self.problem_type = "CVRP"
        
        self.feature_extractors = [VRPFeatureExtractor(inst) for inst in instances]
    
    def generate_training_data(self, num_samples_per_instance: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for the GP regressor.
        
        Args:
            num_samples_per_instance: Number of samples to generate per VRP instance
        
        Returns:
            Tuple of (X, y) where X is feature matrix and y is target scores
        """
        X_list = []
        y_list = []
        
        for instance, feature_extractor in zip(self.instances, self.feature_extractors):
            # Generate random VRP scenarios for training
            for _ in range(num_samples_per_instance):
                # Create a random partial route
                n = instance.dimension
                unvisited = set(range(1, n))  # All customers except depot
                
                # Randomly select some customers for the current route
                route_length = np.random.randint(0, min(10, len(unvisited) + 1))
                current_route = [instance.depot]
                current_load = 0.0
                current_position = instance.depot
                current_time = 0.0  # For VRPTW
                
                # Add random customers to route
                for _ in range(route_length):
                    feasible = [c for c in unvisited if current_load + instance.demands[c] <= instance.capacity]
                    
                    # For VRPTW, also check time window feasibility
                    if self.problem_type == "CVRPTW":
                        time_feasible = []
                        for c in feasible:
                            travel = instance.dist_matrix[current_position, c]
                            arrival = current_time + travel
                            if arrival <= instance.due_dates[c]:
                                time_feasible.append(c)
                        feasible = time_feasible
                    
                    if not feasible:
                        break
                    
                    customer = np.random.choice(feasible)
                    current_route.append(customer)
                    current_load += instance.demands[customer]
                    unvisited.remove(customer)
                    current_position = customer
                    
                    # Update time for VRPTW
                    if self.problem_type == "CVRPTW":
                        travel = instance.dist_matrix[current_position, customer]
                        arrival = current_time + travel
                        start_service = max(arrival, instance.ready_times[customer])
                        current_time = start_service + instance.service_times[customer]
                
                # Generate features for remaining unvisited customers
                remaining_customers = list(unvisited)
                if not remaining_customers:
                    continue
                
                # Sample a few customers to create training examples
                num_samples = min(5, len(remaining_customers))
                sampled_customers = np.random.choice(remaining_customers, num_samples, replace=False)
                
                for customer in sampled_customers:
                    # Extract features (with current_time for TW; battery handled inside extractor)
                    if self.problem_type == "CVRPTW":
                        features = feature_extractor.extract_features(
                            customer, current_route, current_load, current_position, current_time
                        )
                    else:
                        features = feature_extractor.extract_features(
                            customer, current_route, current_load, current_position
                        )
                    
                    # Convert to array using unified feature extraction
                    feature_array = np.array(VRP_PROBLEM_TYPE.extract_feature_values(features))
                    
                    # Calculate target score
                    # Use a combination of distance and capacity considerations
                    dist_score = features['dist_from_current']
                    capacity_penalty = 0 if features['remaining_capacity'] >= features['demand'] else 1000
                    savings_bonus = -features['savings']  # Negative because we want to minimize
                    
                    # Add time window penalty for VRPTW
                    tw_penalty = 0
                    if self.problem_type == "CVRPTW":
                        if not features.get('tw_feasible', True):
                            tw_penalty = 1000
                        else:
                            # Penalty based on time window tightness
                            tw_penalty = features.get('wait_time', 0) * 0.1
                    
                    target_score = dist_score + capacity_penalty + savings_bonus + tw_penalty #TODO replace with a solver
                    
                    X_list.append(feature_array)
                    y_list.append(target_score)
        
        return np.array(X_list), np.array(y_list)


def solve_with_gp_scoring(instance, feature_extractor: VRPFeatureExtractor, 
                         gp_model, problem_type: str = "auto") -> List[List[int]]:
    """
    Solve VRP using GP-evolved scoring function.
    """
    return VRP_PROBLEM_TYPE.solve_with_scoring(instance, feature_extractor, gp_model.predict)


def run_gomea_genetic_programming(instances: List, 
                                 problem_type: str = "auto",
                                 time_limit: int = 60,
                                 popsize: int = 64,
                                 parallel: int = 4) -> GPG:
    """
    Run GOMEA genetic programming to evolve VRP scoring function.
    
    Args:
        instances: List of VRP instances for training
        problem_type: Problem type ("auto", "CVRP", "CVRPTW")
        time_limit: Time limit in seconds
        popsize: Population size
        parallel: Number of parallel cores
    
    Returns:
        Trained GP model
    """
    # Auto-detect problem (for display only)
    if problem_type == "auto":
        inst = instances[0]
        if hasattr(inst, "node_types") and hasattr(inst, "battery_capacity"):
            problem_type = "GVRP"
        elif hasattr(inst, "ready_times") and hasattr(inst, "due_dates"):
            problem_type = "CVRPTW"
        else:
            problem_type = "CVRP"
    
    print(f"Using problem type: {problem_type}")
    print(f"Generating training data from {len(instances)} {problem_type} instances...")
    
    # Generate training data
    data_generator = VRPDataGenerator(instances, problem_type=problem_type)
    X_train, y_train = data_generator.generate_training_data(num_samples_per_instance=100)
    
    print(f"Generated {len(X_train)} training samples")
    print(f"Feature matrix shape: {X_train.shape}")
    if len(y_train) > 0:
        print(f"Target range: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
    
    # Create and configure GP-GOMEA
    print(f"\nRunning GP-GOMEA...")
    ea = GPG(
        gomea=True,  # Use GOMEA as search algorithm
        functions="+_-_*_p/",  # Functions to use
        coeffmut='0.5_0.5_0.5_10',  # Constant mutation parameters
        time=time_limit,  # Time limit
        generations=-1,  # No generation limit
        evaluations=-1,  # No evaluation limit
        initmaxtreeheight=3,  # Initial tree height
        ims='4_1',  # Interleaved multistart scheme
        popsize=popsize,  # Population size
        batchsize=256,  # Batch size
        parallel=parallel,  # Parallel cores
        linearscaling=True,  # Use linear scaling
        silent=False  # Show progress
    )
    
    # Train the model
    ea.fit(X_train, y_train)
    
    print(f"\nTraining completed!")
    print(f"Evaluations taken: {ea.get_evaluations()}")
    print(f"Final training RMSE: {np.sqrt(mean_squared_error(y_train, ea.predict(X_train))):.4f}")
    
    return ea


def load_instances_by_type():
    """Load instances grouped by problem type."""
    cvrp_instances = [
        VRPInstance("Sets/Set_A/A-n32-k5.vrp"),
        # TODO: Add more
    ]
    
    vrptw_instances = [
        VRPTWInstance("Sets/Vrp-Set-HG/C1_2_2.txt"),
        #VRPTWInstance("Sets/Vrp-Set-HG/C1_2_3.txt"),
        #VRPTWInstance("Sets/Vrp-Set-HG/C1_2_4.txt"),
        # TODO: Add more
    ]
    
    gvrp_instances = [
        GVRPMultiTechInstance("Sets/felipe-et-al-2014/data-A0-N030_red.xml"),
        # TODO: Add more
    ]
    
    return cvrp_instances, vrptw_instances, gvrp_instances


def train_and_test_problem_type(instances, problem_type, time_limit=30, popsize=32, parallel=2):
    """Train and test a specific problem type."""
    if not instances:
        print(f"No {problem_type} instances available, skipping...")
        return None
    
    print(f"\n{'='*60}")
    print(f"Training {problem_type} model with {len(instances)} instances")
    print(f"{'='*60}")
    
    for i, instance in enumerate(instances):
        print(f"  {i+1}. {instance.name} (n={instance.dimension}, cap={instance.capacity})")
    
    # Run GOMEA genetic programming
    gp_model = run_gomea_genetic_programming(
        instances=instances,
        problem_type=problem_type,
        time_limit=time_limit,
        popsize=popsize,
        parallel=parallel
    )
    
    # Get the evolved model
    model_str = gp_model.get_model().replace("p/", "/").replace("plog", "log")
    print(f"\n{problem_type} Evolved scoring formula:")
    print(f"{model_str}")
    
    # Test the evolved function
    print(f"\nTesting {problem_type} evolved scoring function...")
    
    for i, instance in enumerate(instances):
        print(f"\nInstance {i+1}: {instance.name}")
        
        # Create feature extractor
        feature_extractor = VRPFeatureExtractor(instance)
        
        # Solve using evolved function
        gp_solution = solve_with_gp_scoring(instance, feature_extractor, gp_model, problem_type)
        gp_fitness = VRP_PROBLEM_TYPE.compute_cost(instance, gp_solution)
        print(f"  GP-GOMEA Solution: Fitness = {gp_fitness:.2f}")
        
        # Compare with heuristics when applicable (heuristics do not handle GVRP battery constraints)
        '''
        if problem_type != "GVRP":
            heuristic_problem_type = "vrp" if problem_type == "CVRP" else "vrptw"
            nn_routes = nearest_neighbor_heuristic(instance, problem=heuristic_problem_type)
            nn_fitness = instance.cost(nn_routes)
            
            savings_routes = savings_heuristic(instance, problem=heuristic_problem_type)
            savings_fitness = instance.cost(savings_routes)
            
            print(f"  Nearest Neighbor: Fitness = {nn_fitness:.2f}")
            print(f"  Savings: Fitness = {savings_fitness:.2f}")
            
            if nn_fitness > 0:
                improvement_nn = ((nn_fitness - gp_fitness) / nn_fitness) * 100
                print(f"  Improvement over NN: {improvement_nn:.2f}%")
            
            if savings_fitness > 0:
                improvement_savings = ((savings_fitness - gp_fitness) / savings_fitness) * 100
                print(f"  Improvement over Savings: {improvement_savings:.2f}%")
        else:
            print("  Heuristic comparison skipped for GVRP (battery not supported by heuristics).")
        '''
    
    return gp_model


def main():
    """Main function to run the GOMEA genetic programming solver with separate training."""
    # Load instances grouped by problem type
    cvrp_instances, vrptw_instances, gvrp_instances = load_instances_by_type()
    
    print("Loaded VRP instances by type:")
    print(f"CVRP instances: {len(cvrp_instances)}")
    print(f"CVRPTW instances: {len(vrptw_instances)}")
    print(f"GVRP instances: {len(gvrp_instances)}")
    
    # Train and test CVRP model
    cvrp_results = None
    if cvrp_instances:
        cvrp_results = train_and_test_problem_type(
            instances=cvrp_instances,
            problem_type="CVRP",
            time_limit=10,
            popsize=16,
            parallel=2
        )
    
    # Train and test VRPTW model
    vrptw_results = None
    if vrptw_instances:
        vrptw_results = train_and_test_problem_type(
            instances=vrptw_instances,
            problem_type="CVRPTW",
            time_limit=10,
            popsize=16,
            parallel=2
        )
    
    # Train and test GVRP model
    '''
    gvrp_results = None
    if gvrp_instances:
        gvrp_results = train_and_test_problem_type(
            instances=gvrp_instances,
            problem_type="GVRP",
            time_limit=30,
            popsize=32,
            parallel=2
        )'''
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    if cvrp_results:
        print(f"CVRP Model: Training completed successfully")
    
    if vrptw_results:
        print(f"VRPTW Model: Training completed successfully")
    '''
    if gvrp_results:
        print(f"GVRP Model: Training completed successfully")
    '''
    return cvrp_results, vrptw_results, #gvrp_results


if __name__ == "__main__":
    main()
