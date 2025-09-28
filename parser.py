import vrplib
import numpy as np
import re
from typing import List, Dict

# Read VRPLIB formatted instances (default)
instance = vrplib.read_instance("Set_A/A-n32-k5.vrp")

#dict_keys(['name', 'comment', 'type', 'dimension', 'edge_weight_type', 'capacity', 'node_coord', 'demand', 'depot', 'edge_weight'])

class VRPInstance:
    def __init__(self, instance):
        instance_dict = vrplib.read_instance(instance)

        self.name = instance_dict["name"]
        self.comment = instance_dict["comment"]
        # Extract number of trucks from comment
        match = re.search(r"No of trucks:\s*(\d+)", self.comment)
        self.num_trucks = int(match.group(1)) if match else 1

        self.edge_weight_type = instance_dict["edge_weight_type"]
        self.type = instance_dict["type"]
        self.dimension = instance_dict["dimension"]
        self.capacity = instance_dict["capacity"]

        # numpy arrays
        self.coords = instance_dict["node_coord"]
        self.demands = instance_dict["demand"]

        # depot TODO only one
        self.depot = int(instance_dict["depot"][0])

        # distance matrix
        self.dist_matrix = instance_dict["edge_weight"]

    def cost(self, routes):
        """
        Compute total distance and penalties for a set of routes.
        routes = list of lists [[0, 5, 7, 0], [0, 2, 3, 0]]
        """
        total_distance = 0.0
        capacity_violation = 0

        for route in routes:
            load = sum(self.demands[node] for node in route if node != self.depot)
            if load > self.capacity:
                capacity_violation += (load - self.capacity)

            for i in range(len(route) - 1):
                total_distance += self.dist_matrix[route[i], route[i+1]]

        return total_distance, capacity_violation

    def __repr__(self):
        return f"VRPInstance({self.name}, n={self.dimension}, cap={self.capacity})"


class VRPFeatureExtractor:
    """Extracts VRP features for request evaluation."""
    
    def __init__(self, instance: VRPInstance):
        self.instance = instance
        self.depot = instance.depot
        self.dist_matrix = instance.dist_matrix
        self.demands = instance.demands
        self.capacity = instance.capacity
    
    def extract_features(self, request: int, current_route: List[int], 
                        current_load: float, current_position: int) -> Dict[str, float]:
        """
        Extract VRP features for a given request.
        
        Args:
            request: Customer node to evaluate
            current_route: Current route being built
            current_load: Current load of the route
            current_position: Current position in the route (last customer)
        
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        # Basic distance features
        features['dist_to_depot'] = self.dist_matrix[self.depot, request]
        features['dist_from_current'] = self.dist_matrix[current_position, request]
        features['dist_to_depot_from_request'] = self.dist_matrix[request, self.depot]
        
        # Demand and capacity features
        features['demand'] = self.demands[request]
        features['remaining_capacity'] = self.capacity - current_load
        features['capacity_utilization'] = current_load / self.capacity if self.capacity > 0 else 0
        features['demand_ratio'] = self.demands[request] / self.capacity if self.capacity > 0 else 0
        
        # Route-specific features
        features['route_length'] = len([c for c in current_route if c != self.depot])
        features['is_empty_route'] = 1.0 if len([c for c in current_route if c != self.depot]) == 0 else 0.0
        
        # Distance-based features
        if len(current_route) > 2:  # More than just depot
            # Distance from last customer to depot
            last_customer = current_route[-2] if current_route[-1] == self.depot else current_route[-1]
            features['dist_last_to_depot'] = self.dist_matrix[last_customer, self.depot]
        else:
            features['dist_last_to_depot'] = 0.0
        
        # Savings-like features
        features['savings'] = (self.dist_matrix[self.depot, current_position] + 
                              self.dist_matrix[request, self.depot] - 
                              self.dist_matrix[current_position, request])
        
        # Normalized features for better GP performance
        max_dist = np.max(self.dist_matrix)
        features['norm_dist_to_depot'] = features['dist_to_depot'] / max_dist if max_dist > 0 else 0
        features['norm_dist_from_current'] = features['dist_from_current'] / max_dist if max_dist > 0 else 0
        features['norm_savings'] = features['savings'] / max_dist if max_dist > 0 else 0
        
        return features