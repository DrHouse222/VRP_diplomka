#!/usr/bin/env python3
"""
Unified VRP problem type configuration.
Handles both CVRP and CVRPTW automatically.
"""

from typing import List, Dict, Any, Optional
import operator
import numpy as np


class VRPProblemType:
    """Unified configuration for VRP variants: CVRP, CVRPTW, and GVRP.
    
    Automatically detects instance type and adapts accordingly:
    - CVRP: Capacity constraints only
    - CVRPTW: Capacity + Time Windows
    - GVRP: Capacity + Battery constraints + Charging stations
    """
    
    def __init__(self):
        # Base features (always available)
        self.base_feature_names = [
            'dist_to_depot', 'dist_from_current', 'demand', 'remaining_capacity', 'savings'
        ]
    
        # Time window features (only for CVRPTW)
        self.tw_feature_names = [
            'arrival_time', 'due_time', 'wait_time', 'tw_feasible', 'slack_to_due'
        ]
        
        # GVRP battery features (only for GVRP)
        self.gvrp_feature_names = [
            'current_battery', 'energy_to_customer', 'is_directly_reachable',
            'dist_to_nearest_charger', 'battery_safety_margin'
        ]
    
    @property
    def name(self) -> str:
        return "VRP"  # Unified name
    
    def has_time_windows(self, instance) -> bool:
        """
        Check if instance has time windows.
        
        GVRP instances typically don't have time windows, but this method
        checks for the presence of ready_times and due_dates attributes.
        """
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
        return self.base_feature_names + self.tw_feature_names + self.gvrp_feature_names
    
    @property
    def num_features(self) -> int:
        """Return maximum number of features (15 total: 5 base + 5 TW + 5 GVRP)."""
        return len(self.base_feature_names) + len(self.tw_feature_names) + len(self.gvrp_feature_names)
    
    def extract_feature_values(self, features: Dict[str, float]) -> List[float]:
        """Extract feature values in the order expected by GP function."""
        return [features.get(name, 0.0) for name in self.feature_names]
    
    def create_primitive_set(self, gp_module) -> Optional[Any]:
        """Create GP primitive set for unified VRP (DEAP-specific).
        
        Uses maximum number of features (32) to support CVRP, CVRPTW, and GVRP.
        Missing features for specific instance types will be filled with 0.0.
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
        """
        Compute cost for VRP solution.
        
        Handles CVRP, CVRPTW, and GVRP instances.
        For GVRP: routes may include charging stations (node_type == 2).
        Returns total distance traveled (including detours to charging stations).
        """
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

    def solve_with_scoring(self, instance, feature_extractor, scoring_func, bool_capacity=True) -> List[List[int]]:
        """
        Construct GVRP routes greedily (Sequential Construction) using GP-evolved scoring.
        Handles:
          - Capacity
          - Time Windows (Hard)
          - Battery Constraints (Reachability + Safety Buffer)
          - Automatic Charging Station Insertion
        """
        n = instance.dimension
        depot = getattr(instance, "depot", 0)

        if bool_capacity:
            max_capacity = getattr(instance, "capacity", 0.0)
        else:
            max_capacity = float('inf')
        
        # --- GREEN VRP SETUP ---
        has_battery = getattr(instance, "battery_capacity", 0.0) > 0.0
        battery_cap = getattr(instance, "battery_capacity", float('inf'))
        energy_per_dist = getattr(instance, "energy_consumption", 1.0)  # Energy per unit distance
        #charge_time_fixed = getattr(instance, "charge_time", 0.0)  # Fixed charging time at stations
        charge_time_fixed = 10.0  # Fixed charging time at stations (for simplicity)
        
        # Identify Charging Stations GVRP instances use node_types: 0=depot, 1=customer, 2=charging_station
        node_types = getattr(instance, "node_types", None)
        if node_types is not None:
            stations = [i for i in range(n) if node_types[i] == 2]  # Type 2 are charging stations
        else:
            stations = []
        
        # PRE-CALCULATION: Distance to nearest charger for every node (Safe Haven)
        # This prevents the truck from getting stranded at a customer.
        dist_to_nearest_charger = {}
        for i in range(n):
            if not stations:
                dist_to_nearest_charger[i] = 0
            else:
                # Find closest station to node 'i'
                min_dist = min(instance.dist_matrix[i, cs] for cs in stations)
                dist_to_nearest_charger[i] = min_dist
    
        has_tw = self.has_time_windows(instance)
        unvisited = set(range(0, n))
        # Remove depot from unvisited
        unvisited.discard(depot)
        # Remove charging stations (type 2) from 'unvisited' - they are not customers
        if node_types is not None:
            for i in range(n):
                if node_types[i] == 2:  # Charging station
                    unvisited.discard(i)
    
        routes: List[List[int]] = []
        
        while unvisited:
            # Start a new route (new truck)
            route = [depot]
            load = 0.0
            current_node = depot
            current_time = 0.0
            current_battery = battery_cap
            
            while True:
                candidates = list(unvisited)
                
                feasible_candidates: List[int] = []
                candidate_infos = [] # Stores: {is_direct, station_idx, cost_dist, arrival_time}
                scores: List[float] = []
                
                for customer in candidates:
                    demand = instance.demands[customer]
                    
                    # 1. Capacity Check
                    if load + demand > max_capacity:
                        continue
                    
                    # 2. GREEN MOVE ANALYSIS
                    # We try two ways to reach the customer: Direct vs. Via Station
                    valid_move_found = False
                    best_move_info = None
                    
                    # --- OPTION A: DIRECT MOVE ---
                    dist_direct = instance.dist_matrix[current_node, customer]
                    energy_direct = dist_direct * energy_per_dist
                    
                    # A1. Can we physically reach it?
                    if current_battery >= energy_direct:
                        # A2. Safety Check: Can we leave the customer to a safe haven?
                        batt_after = current_battery - energy_direct
                        energy_safety = dist_to_nearest_charger[customer] * energy_per_dist
                        
                        if batt_after >= energy_safety:
                            # A3. Time Window Check
                            arrival_direct = current_time + dist_direct
                            
                            # Check Customer TW
                            if not has_tw or arrival_direct <= instance.due_dates[customer]:
                                # Check Return to Depot (Time + Battery)
                                ready = instance.ready_times[customer] if has_tw else 0.0
                                service = instance.service_times[customer] if has_tw else 0.0
                                dept_time = max(arrival_direct, ready) + service
                                dist_home = instance.dist_matrix[customer, depot]
                                
                                if not has_tw or (dept_time + dist_home <= instance.due_dates[depot]):
                                    # DIRECT IS VALID
                                    valid_move_found = True
                                    best_move_info = {
                                        'is_direct': True,
                                        'dist': dist_direct,
                                        'arrival': arrival_direct,
                                        'station': None
                                    }
    
                    # --- OPTION B: VIA CHARGING STATION ---
                    # Find the station that minimizes total distance (d1 + d2)
                    if not valid_move_found and has_battery:
                        
                        min_total_dist = float('inf') # Track the best distance
                        
                        for station in stations:
                            # Leg 1: Current -> Station
                            d1 = instance.dist_matrix[current_node, station]
                            e1 = d1 * energy_per_dist
                            
                            if current_battery < e1: continue 
                            
                            # Leg 2: Station -> Customer
                            d2 = instance.dist_matrix[station, customer]
                            e2 = d2 * energy_per_dist
                            
                            total_dist = d1 + d2
                            if total_dist >= min_total_dist: continue

                            if battery_cap < e2: continue 
                            
                            # Safety at Destination
                            if (battery_cap - e2) < (dist_to_nearest_charger[customer] * energy_per_dist):
                                continue 
                                
                            # Time Check
                            arrival_at_station = current_time + d1
                            dept_from_station = arrival_at_station + charge_time_fixed
                            arrival_at_cust = dept_from_station + d2
                            
                            if has_tw:
                                # Check Return to Depot Time
                                ready = instance.ready_times[customer]
                                service = instance.service_times[customer]
                                dept_cust = max(arrival_at_cust, ready) + service
                                dist_home = instance.dist_matrix[customer, depot]
                            
                                if arrival_at_cust > instance.due_dates[customer] or (dept_cust + dist_home > instance.due_dates[depot]):
                                    continue
                            
                            valid_move_found = True
                            min_total_dist = total_dist
                            best_move_info = {
                                'is_direct': False,
                                'dist': total_dist,
                                'arrival': arrival_at_cust,
                                'station': station,
                                'station_dist': d1
                            }
                        # Removed 'break' to ensure we find the minimum distance
                        
                    if not valid_move_found:
                        continue
                    
                    # 3. SCORING
                    # Extract features - override distance with actual travel distance
                    # (If going via station, use total distance d1+d2, not direct distance)
                    features = feature_extractor.extract_features(
                        request=customer,
                        current_route=route,
                        current_load=load,
                        current_position=current_node,
                        current_time=current_time if has_tw else 0.0,
                        current_battery=current_battery if has_battery else None,
                        dist_to_nearest_charger=dist_to_nearest_charger if has_battery else None
                    )
                    
                    # Override distance features with actual travel distance
                    # This is important when going via charging station (dist = d1 + d2)
                    actual_dist = best_move_info['dist']
                    features['dist_from_current'] = actual_dist
                    
                    # Update savings feature to reflect actual distance
                    features['savings'] = (
                        features.get('dist_to_depot', 0.0) +
                        features.get('dist_to_depot_from_request', 0.0) -
                        actual_dist
                    )
                    
                    features["remaining_capacity"] = max_capacity - (load + demand)
                    feature_values = self.extract_feature_values(features)
                    
                    try:
                        score = scoring_func(*feature_values)
                    except Exception:
                        score = 1e6 
    
                    feasible_candidates.append(customer)
                    candidate_infos.append(best_move_info)
                    scores.append(score)
                
                # --- SELECTION & EXECUTION ---
                if not feasible_candidates:
                    # Route finished, return to depot

                    # Check if we need to charge before returning to depot
                    if has_battery and current_node != depot:
                        dist_to_depot = instance.dist_matrix[current_node, depot]
                        energy_needed = dist_to_depot * energy_per_dist

                        # If we can't reach depot directly...
                        if current_battery < energy_needed and stations:
                            best_station = None
                            min_total_dist = float('inf')

                            for station in stations:
                                d1 = instance.dist_matrix[current_node, station]
                                d2 = instance.dist_matrix[station, depot]
                                e1 = d1 * energy_per_dist
                                e2 = d2 * energy_per_dist # Energy needed from Station -> Depot

                                # Check 1: Can we reach station?
                                if current_battery < e1: continue
                                # Check 2: Can full battery reach depot?
                                if battery_cap < e2: continue

                                # Check 3: Time Window, can we 
                                if has_tw:
                                    arrival_stat = current_time + d1
                                    dept_stat = arrival_stat + charge_time_fixed
                                    arrival_depot = dept_stat + d2
                                    if arrival_depot > instance.due_dates[depot]:
                                        continue
                                    
                                total_dist = d1 + d2
                                if total_dist < min_total_dist:
                                    min_total_dist = total_dist
                                    best_station = station

                            if best_station is not None:
                                route.append(best_station)
                                # Update battery and time after visiting station
                                energy_to_station = instance.dist_matrix[current_node, best_station] * energy_per_dist
                                current_battery -= energy_to_station
                                current_time += instance.dist_matrix[current_node, best_station]
                                # Charge at station
                                current_battery = battery_cap
                                current_time += charge_time_fixed
                                current_node = best_station

                    route.append(depot)
                    break
                
                # Pick best
                best_idx = int(np.argmin(scores))
                best_customer = feasible_candidates[best_idx]
                info = candidate_infos[best_idx]
    
                # Execute Move
                if not info['is_direct']:
                    # 1. Insert Station First
                    station = info['station']
                    route.append(station)
                    
                    # Update battery after leg 1 (current -> station)
                    energy_to_station = info['station_dist'] * energy_per_dist
                    current_battery -= energy_to_station
                    
                    # Update time after leg 1
                    current_time += info['station_dist']
                    
                    # Charge at station (refill battery, add charging time)
                    current_battery = battery_cap
                    current_time += charge_time_fixed
                    
                    # Update Distance for next leg
                    travel_to_cust = instance.dist_matrix[station, best_customer]
                else:
                    travel_to_cust = info['dist']
    
                # 2. Insert Customer
                route.append(best_customer)
                load += instance.demands[best_customer]
                unvisited.remove(best_customer)
                
                # Update State
                current_node = best_customer
                
                # Update Time (Standard TW logic)
                if has_tw:
                    arrival = current_time + travel_to_cust
                    ready = instance.ready_times[best_customer]
                    service = instance.service_times[best_customer]
                    start_service = max(arrival, ready)
                    current_time = start_service + service
                else:
                    current_time += travel_to_cust
                
                # Update Battery after leg 2 (station->customer or direct->customer)
                energy_used = travel_to_cust * energy_per_dist
                current_battery -= energy_used
    
            routes.append(route)
        
        return routes
    
    def evaluate_solution(self, instance, solution) -> float:
        """Evaluate VRP solution quality (scalar fitness)."""
        return self.compute_cost(instance, solution)

# Global instance - only one problem type needed
VRP_PROBLEM_TYPE = VRPProblemType()
