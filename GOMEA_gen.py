#!/usr/bin/env python3
"""
Genetic Programming solution for CVRP using pyGPGOMEA framework.
Evolves a scoring function that evaluates requests based on VRP features.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from pyGPGOMEA import GPGOMEARegressor as GPG
from sklearn.metrics import mean_squared_error
from parser import VRPInstance, VRPFeatureExtractor
from basic_heuristics import nearest_neighbor_heuristic, savings_heuristic


class VRPDataGenerator: # TODO
    """Generates training data for VRP scoring function using multiple instances."""
    
    def __init__(self, instances: List[VRPInstance]):
        self.instances = instances
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
                
                # Add random customers to route
                for _ in range(route_length):
                    feasible = [c for c in unvisited if current_load + instance.demands[c] <= instance.capacity]
                    if not feasible:
                        break
                    
                    customer = np.random.choice(feasible)
                    current_route.append(customer)
                    current_load += instance.demands[customer]
                    unvisited.remove(customer)
                    current_position = customer
                
                # Generate features for remaining unvisited customers
                remaining_customers = list(unvisited)
                if not remaining_customers:
                    continue
                
                # Sample a few customers to create training examples
                num_samples = min(5, len(remaining_customers))
                sampled_customers = np.random.choice(remaining_customers, num_samples, replace=False)
                
                for customer in sampled_customers:
                    # Extract features
                    features = feature_extractor.extract_features(
                        customer, current_route, current_load, current_position
                    )
                    
                    # Convert to array in the expected order
                    feature_array = np.array([
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
                    ])
                    
                    # Calculate target score (lower is better for VRP)
                    # Use a combination of distance and capacity considerations
                    dist_score = features['dist_from_current']
                    capacity_penalty = 0 if features['remaining_capacity'] >= features['demand'] else 1000
                    savings_bonus = -features['savings']  # Negative because we want to minimize
                    
                    target_score = dist_score + capacity_penalty + savings_bonus
                    
                    X_list.append(feature_array)
                    y_list.append(target_score)
        
        return np.array(X_list), np.array(y_list)


def solve_with_gp_scoring(instance: VRPInstance, feature_extractor: VRPFeatureExtractor, 
                         gp_model) -> List[List[int]]:
    """
    Solve VRP using GP-evolved scoring function.
    
    Args:
        instance: VRP instance
        feature_extractor: Feature extractor for the instance
        gp_model: Trained GP model
    
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
            
            # Score each candidate using GP model
            scores = []
            for candidate in candidates:
                features = feature_extractor.extract_features(
                    candidate, route, load, current_position
                )
                
                # Convert features to the order expected by the GP model
                feature_values = np.array([[
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
                ]])
                
                try:
                    score = gp_model.predict(feature_values)[0]
                    scores.append(score)
                except:
                    # If GP model fails, use distance as fallback
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


def run_gomea_genetic_programming(instances: List[VRPInstance], 
                                 time_limit: int = 60,
                                 popsize: int = 64,
                                 parallel: int = 4) -> GPG:
    """
    Run GOMEA genetic programming to evolve VRP scoring function.
    
    Args:
        instances: List of VRP instances for training
        time_limit: Time limit in seconds
        popsize: Population size
        parallel: Number of parallel cores
    
    Returns:
        Trained GP model
    """
    print(f"Generating training data from {len(instances)} VRP instances...")
    
    # Generate training data
    data_generator = VRPDataGenerator(instances)
    X_train, y_train = data_generator.generate_training_data(num_samples_per_instance=500)
    
    print(f"Generated {len(X_train)} training samples")
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Target range: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
    
    # Create and configure GP-GOMEA
    print(f"\nRunning GP-GOMEA...")
    ea = GPG(
        gomea=True,  # Use GOMEA as search algorithm
        functions="+_-_*_p/_sqrt_plog",  # Functions to use
        coeffmut='0.5_0.5_0.5_10',  # Constant mutation parameters
        time=time_limit,  # Time limit
        generations=-1,  # No generation limit
        evaluations=-1,  # No evaluation limit
        initmaxtreeheight=3,  # Initial tree height
        ims='4_1',  # Interleaved multistart scheme
        popsize=popsize,  # Population size
        batchsize=256,  # Batch size
        parallel=parallel,  # Parallel cores
        linearscaling=True,  # Use linear scaling (recommended for real-world data)
        silent=False  # Show progress
    )
    
    # Train the model
    ea.fit(X_train, y_train)
    
    print(f"\nTraining completed!")
    print(f"Evaluations taken: {ea.get_evaluations()}")
    print(f"Final training RMSE: {np.sqrt(mean_squared_error(y_train, ea.predict(X_train))):.4f}")
    
    return ea


def main():
    """Main function to run the GOMEA genetic programming solver."""
    # Load VRP instances for training
    instances = [
        VRPInstance("Set_A/A-n32-k5.vrp"),
        VRPInstance("Set_A/A-n33-k5.vrp"),
        VRPInstance("Set_A/A-n33-k6.vrp"),
        # Add more instances for better generalization
    ]
    
    print("Loaded VRP instances:")
    for i, instance in enumerate(instances):
        print(f"  {i+1}. {instance.name} (n={instance.dimension}, cap={instance.capacity})")
    
    # Run GOMEA genetic programming
    gp_model = run_gomea_genetic_programming(
        instances=instances,
        time_limit=30,  # 30 seconds
        popsize=32,
        parallel=2
    )
    
    # Get the evolved model
    model_str = gp_model.get_model().replace("p/", "/").replace("plog", "log")
    print(f"\nEvolved scoring formula:")
    print(f"{model_str}")
    
    # Test the evolved function
    print(f"\nTesting evolved scoring function...")
    
    for i, instance in enumerate(instances):
        print(f"\nInstance {i+1}: {instance.name}")
        
        # Create feature extractor
        feature_extractor = VRPFeatureExtractor(instance)
        
        # Solve using evolved function
        gp_solution = solve_with_gp_scoring(instance, feature_extractor, gp_model)
        gp_distance, gp_violation = instance.cost(gp_solution)
        
        # Compare with heuristics
        nn_routes = nearest_neighbor_heuristic(instance)
        nn_distance, nn_violation = instance.cost(nn_routes)
        
        savings_routes = savings_heuristic(instance)
        savings_distance, savings_violation = instance.cost(savings_routes)
        
        print(f"  GP-GOMEA Solution: Distance = {gp_distance:.2f}, Violation = {gp_violation:.2f}")
        print(f"  Nearest Neighbor: Distance = {nn_distance:.2f}, Violation = {nn_violation:.2f}")
        print(f"  Savings: Distance = {savings_distance:.2f}, Violation = {savings_violation:.2f}")
        
        if nn_distance > 0:
            improvement_nn = ((nn_distance - gp_distance) / nn_distance) * 100
            print(f"  Improvement over NN: {improvement_nn:.2f}%")
        
        if savings_distance > 0:
            improvement_savings = ((savings_distance - gp_distance) / savings_distance) * 100
            print(f"  Improvement over Savings: {improvement_savings:.2f}%")
    
    return gp_model


if __name__ == "__main__":
    main()
