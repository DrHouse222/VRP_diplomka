#!/usr/bin/env python3
"""
Unified VRP problem type configuration.
Handles both CVRP and CVRPTW automatically.
"""

from typing import List, Dict, Any, Optional
import operator
import numpy as np


class VRPProblemType:
    """Unified configuration for Capacitated VRP and CVRP with Time Windows.
    
    Automatically detects if instance has time windows and adapts accordingly.
    """
    
    def __init__(self):
        # Base features (always available)
        self.base_feature_names = [
            'dist_to_depot', 'dist_from_current', 'dist_to_depot_from_request',
            'demand', 'remaining_capacity', 'capacity_utilization', 'demand_ratio',
            'route_length', 'is_empty_route', 'dist_last_to_depot', 'savings', 'norm_savings'
        ]
        
        # Time window features (only for CVRPTW)
        self.tw_feature_names = [
            'arrival_time', 'ready_time', 'due_time', 'service_time', 'start_service_time',
            'finish_service_time', 'wait_time', 'tw_feasible', 'slack_to_due', 'remaining_tw_from_now'
        ]
    
    @property
    def name(self) -> str:
        return "VRP"  # Unified name
    
    def has_time_windows(self, instance) -> bool:
        """Check if instance has time windows."""
        return all(hasattr(instance, attr) for attr in ("ready_times", "due_dates", "service_times"))
    
    def get_feature_names(self, instance=None) -> List[str]:
        """Get feature names based on instance type."""
        if instance is None or not self.has_time_windows(instance):
            return self.base_feature_names
        else:
            return self.base_feature_names + self.tw_feature_names
    
    @property
    def feature_names(self) -> List[str]:
        """Return all possible features (for GP primitive set creation)."""
        return self.base_feature_names + self.tw_feature_names
    
    @property
    def num_features(self) -> int:
        """Return maximum number of features (22 for CVRPTW)."""
        return len(self.base_feature_names) + len(self.tw_feature_names)
    
    def extract_feature_values(self, features: Dict[str, float]) -> List[float]:
        """Extract feature values in the order expected by GP function."""
        return [features.get(name, 0.0) for name in self.feature_names]
    
    def create_primitive_set(self, gp_module) -> Optional[Any]:
        """Create GP primitive set for unified VRP (DEAP-specific).
        
        Uses maximum number of features (22) to support both CVRP and CVRPTW.
        Missing features for CVRP instances will be filled with 0.0.
        """
        if gp_module is None:
            return None
            
        pset = gp_module.PrimitiveSet("MAIN", self.num_features)
        
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
    
    def compute_cost(self, instance, routes) -> float:
        """Compute cost for VRP solution (handles both CVRP and CVRPTW)."""
        total_distance = 0.0
        demands = getattr(instance, "demands", None)
        dist_matrix = getattr(instance, "dist_matrix", None)

        if demands is None or dist_matrix is None:
            raise ValueError("Instance must have 'demands' and 'dist_matrix' attributes.")
        
        for route in routes:
            if not route:
                continue
            
            # Distance calculation
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                travel = float(dist_matrix[u, v])
                total_distance += travel
                
        return total_distance
    
    
    def solve_with_scoring(self, instance, feature_extractor, scoring_func) -> List[List[int]]:
        """
        Construct VRP routes greedily using the GP-evolved scoring function.

        - Works LINEARLY: builds one route at a time.
        - Uses an UNLIMITED number of trucks: starts a new route whenever the
          current one cannot be extended (capacity / time-window constraints).
        """
        n = instance.dimension
        depot = getattr(instance, "depot", 0)
        max_capacity = getattr(instance, "capacity", 0.0)
        has_tw = self.has_time_windows(instance)

        unvisited = set(range(1, n))  # All customers except depot
        routes: List[List[int]] = []

        while unvisited:
            # Start a new route (new truck)
            route = [depot]
            load = 0.0
            current_node = depot
            current_time = 0.0

            while True:
                candidates = list(unvisited)

                # Check feasibility and score candidates
                feasible_candidates: List[int] = []
                scores: List[float] = []

                for customer in candidates:
                    demand = instance.demands[customer]

                    # Capacity feasibility
                    if load + demand > max_capacity:
                        continue

                    # Time-window feasibility (if applicable)
                    if has_tw:
                        # Arrival at customer
                        travel = instance.dist_matrix[current_node, customer]
                        arrival = current_time + travel
                        due_cust = instance.due_dates[customer]
                        if arrival > due_cust:
                            continue  # infeasible at customer

                        # Time at departure from customer (respect ready time + service)
                        ready_cust = instance.ready_times[customer]
                        service_cust = instance.service_times[customer]
                        start_service = max(arrival, ready_cust)
                        depart_time = start_service + service_cust

                        # Check that we can still return to depot before its due date
                        return_travel = instance.dist_matrix[customer, depot]
                        arrival_depot = depart_time + return_travel
                        due_depot = instance.due_dates[depot]
                        if arrival_depot > due_depot:
                            continue  # route would violate depot due date

                    # Extract features for this (route, customer) pair
                    features = feature_extractor.extract_features(
                        request=customer,
                        current_route=route,
                        current_load=load,
                        current_position=current_node,
                        current_time=current_time if has_tw else 0.0,
                    )

                    # Remaining capacity AFTER adding this customer
                    remaining_capacity_after = max_capacity - (load + demand)
                    features["remaining_capacity"] = remaining_capacity_after

                    feature_values = self.extract_feature_values(features)

                    try:
                        score = scoring_func(*feature_values)
                    except Exception:
                        score = 1e6  # Large penalty on errors

                    feasible_candidates.append(customer)
                    scores.append(score)

                # No feasible extension -> close route (truck returns to depot)
                if not feasible_candidates:
                    route.append(depot)
                    break

                # Pick best candidate (lowest score)
                best_idx = int(np.argmin(scores))
                best_customer = feasible_candidates[best_idx]

                # Update route state
                travel = instance.dist_matrix[current_node, best_customer]
                if has_tw:
                    arrival = current_time + travel
                    ready = instance.ready_times[best_customer]
                    service = instance.service_times[best_customer]
                    start_service = max(arrival, ready)
                    current_time = start_service + service
                else:
                    current_time += travel

                route.append(best_customer)
                load += instance.demands[best_customer]
                current_node = best_customer
                unvisited.remove(best_customer)

            routes.append(route)

        return routes
    
    def evaluate_solution(self, instance, solution) -> float:
        """Evaluate VRP solution quality (scalar fitness)."""
        return self.compute_cost(instance, solution)


# Global instance - only one problem type needed
VRP_PROBLEM_TYPE = VRPProblemType()
