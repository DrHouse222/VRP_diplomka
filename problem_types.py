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
        # Linear charging: time = charge_rate * energy_needed
        # If full charge (0 to 100%) takes 20 time units, then charge_rate = 20 / battery_cap
        base_charge_time = 100.0  # Time to charge from 0% to 100%
        charge_rate = base_charge_time / battery_cap if has_battery and battery_cap > 0 else 0.0
        
        def compute_charge_time(current_batt):
            """Compute linear charging time based on current battery level."""
            if not has_battery:
                return 0.0
            energy_needed = battery_cap - current_batt
            return charge_rate * energy_needed
        
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
        has_max_travel_time = hasattr(instance, "max_travel_time")
        max_travel_time = getattr(instance, "max_travel_time", float('inf')) if has_max_travel_time else float('inf')
        
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
                                arrival_at_depot = dept_time + dist_home
                                
                                # Check time window and max travel time (if applicable)
                                tw_ok = not has_tw or (arrival_at_depot <= instance.due_dates[depot])
                                travel_time_ok = not has_max_travel_time or (arrival_at_depot <= max_travel_time)
                                
                                if tw_ok and travel_time_ok:
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
                            # Calculate battery level when arriving at station
                            batt_at_station = current_battery - (d1 * energy_per_dist)
                            charge_time = compute_charge_time(batt_at_station)
                            dept_from_station = arrival_at_station + charge_time
                            arrival_at_cust = dept_from_station + d2
                            
                            if has_tw:
                                # Check Return to Depot Time
                                ready = instance.ready_times[customer]
                                service = instance.service_times[customer]
                                dept_cust = max(arrival_at_cust, ready) + service
                                dist_home = instance.dist_matrix[customer, depot]
                                arrival_at_depot = dept_cust + dist_home
                            
                                if arrival_at_cust > instance.due_dates[customer] or (arrival_at_depot > instance.due_dates[depot]) or (has_max_travel_time and arrival_at_depot > max_travel_time):
                                    continue
                            else:
                                # For non-TW, check max travel time (if applicable)
                                if has_max_travel_time:
                                    service = getattr(instance, "service_times", np.zeros(n))[customer]
                                    dist_home = instance.dist_matrix[customer, depot]
                                    arrival_at_depot = arrival_at_cust + service + dist_home
                                    if arrival_at_depot > max_travel_time:
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

                                # Check 3: Time Window and Max Travel Time
                                if has_tw:
                                    arrival_stat = current_time + d1
                                    # Calculate battery level when arriving at station
                                    batt_at_station = current_battery - (d1 * energy_per_dist)
                                    charge_time = compute_charge_time(batt_at_station)
                                    dept_stat = arrival_stat + charge_time
                                    arrival_depot = dept_stat + d2
                                    if arrival_depot > instance.due_dates[depot] or (has_max_travel_time and arrival_depot > max_travel_time):
                                        continue
                                else:
                                    # For non-TW, still check max travel time (if applicable)
                                    if has_max_travel_time:
                                        arrival_stat = current_time + d1
                                        batt_at_station = current_battery - (d1 * energy_per_dist)
                                        charge_time = compute_charge_time(batt_at_station)
                                        arrival_depot = current_time + d1 + charge_time + d2
                                        if arrival_depot > max_travel_time:
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
                                # Charge at station (linear charging time)
                                charge_time = compute_charge_time(current_battery)
                                current_battery = battery_cap
                                current_time += charge_time
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
                    
                    # Charge at station (refill battery, add linear charging time)
                    charge_time = compute_charge_time(current_battery)
                    current_battery = battery_cap
                    current_time += charge_time
                    
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
    
    def solve_with_scoring2(self, instance, feature_extractor, scoring_func, bool_capacity=True) -> List[List[int]]:
        """
        Construct GVRP routes using Strict Algorithm B Logic (Select First, Check Later).
        Warning: This can result in under-filled trucks if the best scoring customer 
        doesn't fit, because the algorithm immediately closes the route.
        """
        n = instance.dimension
        depot = getattr(instance, "depot", 0)
        max_capacity = getattr(instance, "capacity", 0.0) if bool_capacity else float('inf')
        
        # --- GVRP Constants ---
        has_battery = getattr(instance, "battery_capacity", 0.0) > 0.0
        battery_cap = getattr(instance, "battery_capacity", float('inf')) if has_battery else float('inf')
        energy_per_dist = getattr(instance, "energy_consumption", 1.0)
        has_max_travel_time = hasattr(instance, "max_travel_time")
        max_travel_time = getattr(instance, "max_travel_time", float('inf')) if has_max_travel_time else float('inf')
        # Linear charging: time = charge_rate * energy_needed
        base_charge_time = 100.0  # Time to charge from 0% to 100%
        charge_rate = base_charge_time / battery_cap if has_battery and battery_cap > 0 else 0.0
        
        def compute_charge_time(current_batt):
            """Compute linear charging time based on current battery level."""
            if not has_battery:
                return 0.0
            energy_needed = battery_cap - current_batt
            return charge_rate * energy_needed
        
        has_tw = self.has_time_windows(instance)
        
        # Identify Stations
        node_types = getattr(instance, "node_types", None)
        stations = [i for i in range(n) if node_types[i] == 2] if node_types is not None else []
        
        # Safe Haven Pre-calculations
        dist_to_safe_haven = {}
        for i in range(n):
            targets = stations + [depot] 
            dist_to_safe_haven[i] = min(instance.dist_matrix[i, t] for t in targets)

        unvisited = set(range(n))
        unvisited.discard(depot)
        if node_types is not None: 
            for s in stations: unvisited.discard(s)

        routes = []

        while unvisited:
            route = [depot]
            load = 0.0
            curr_node = depot
            curr_time = 0.0
            curr_batt = battery_cap
            
            while True:
                # 1. SCORE FEASIBLE CANDIDATES ONLY
                candidates = list(unvisited)
                if not candidates:
                    break
                
                scored_candidates = []
                for cust in candidates:
                    # --- OPTIMIZATION: CHEAP CHECK FIRST ---
                    # If it doesn't fit capacity, don't waste CPU time extracting features!
                    if load + instance.demands[cust] > max_capacity:
                        continue 

                    # Extract features (Only for those who fit capacity)
                    feat = feature_extractor.extract_features(
                        request=cust,
                        current_route=route,
                        current_load=load,
                        current_position=curr_node,
                        current_time=curr_time if has_tw else 0.0,
                        current_battery=curr_batt if has_battery else None,
                        dist_to_nearest_charger=dist_to_safe_haven if has_battery else None
                    )
                    try:
                        score = scoring_func(*self.extract_feature_values(feat))
                    except:
                        score = 1e9
                    
                    scored_candidates.append((score, cust))
                
                # If no one fits capacity, close the route immediately
                if not scored_candidates:
                    break
                
                # Sort by score (best first)
                scored_candidates.sort(key=lambda x: x[0])
                
                # 2. SELECT BEST FEASIBLE CANDIDATE (Try-Next Logic)
                best_cust = None
                
                # ... Rest of your code (Station logic, TW checks) is perfect ...
                for score, candidate in scored_candidates:
                    demand = instance.demands[candidate]
                    
                    # --- CHECK 1: CAPACITY---
                    if load + demand > max_capacity:
                        continue  # Try next candidate

                    # --- CHECK 2: BATTERY & REACHABILITY---
                    # We determine HOW to get there (Direct or Via Station)
                    
                    move_type_candidate = "DIRECT"
                    chosen_station_candidate = None
                    travel_dist_candidate = 0.0
                    arrival_at_cust_candidate = 0.0
                    
                    dist_direct = instance.dist_matrix[curr_node, candidate]
                    energy_direct = dist_direct * energy_per_dist
                    
                    # Energy Condition 1 (Reach Cust) & 2 (Leave Cust) for DIRECT
                    can_go_direct = False
                    if has_battery:
                        if curr_batt >= energy_direct:
                            batt_after = curr_batt - energy_direct
                            if batt_after >= dist_to_safe_haven[candidate] * energy_per_dist:
                                can_go_direct = True
                    else:
                        can_go_direct = True # Infinite battery
                    
                    if can_go_direct:
                        move_type_candidate = "DIRECT"
                        travel_dist_candidate = dist_direct
                        arrival_at_cust_candidate = curr_time + dist_direct
                    else:
                        # Select best charging station that allows TW
                        move_type_candidate = "VIA_STATION"
                        
                        best_s_move = None
                        min_cost = float('inf')
                        
                        for s in stations:
                            d1 = instance.dist_matrix[curr_node, s]
                            d2 = instance.dist_matrix[s, candidate]
                            e1 = d1 * energy_per_dist
                            e2 = d2 * energy_per_dist
                            
                            # Check reachability
                            if curr_batt < e1: continue # Can't reach station
                            if battery_cap < e2: continue # Station can't reach cust
                            if (battery_cap - e2) < dist_to_safe_haven[candidate] * energy_per_dist: continue
                            

                            # 1. Calculate Timings
                            arr_s = curr_time + d1
                            # Calculate battery level when arriving at station
                            batt_at_station = curr_batt - (d1 * energy_per_dist)
                            charge_time = compute_charge_time(batt_at_station)
                            dept_s = arr_s + charge_time
                            arr_c = dept_s + d2

                            if has_tw:
                                # 2. Check Customer Time Window (Late Arrival)
                                if arr_c > instance.due_dates[candidate]: 
                                    continue

                                # 3. Check Return to Depot Feasibility (Global Constraint)
                                ready = instance.ready_times[candidate]
                                service = instance.service_times[candidate]

                                # Time we leave the customer
                                dept_cust = max(arr_c, ready) + service

                                # Distance home (Depot)
                                dist_home = instance.dist_matrix[candidate, depot]

                                # Can we get back to the depot before IT closes and within max travel time (if applicable)?
                                arrival_at_depot = dept_cust + dist_home
                                if arrival_at_depot > instance.due_dates[depot] or (has_max_travel_time and arrival_at_depot > max_travel_time):
                                    continue
                            else:
                                # For non-TW, check max travel time (if applicable)
                                if has_max_travel_time:
                                    service = getattr(instance, "service_times", np.zeros(n))[candidate]
                                    dist_home = instance.dist_matrix[candidate, depot]
                                    arrival_at_depot = arr_c + service + dist_home
                                    if arrival_at_depot > max_travel_time:
                                        continue
                            
                            # "Select best in terms of energy consumption" (roughly distance)
                            if (d1 + d2) < min_cost:
                                min_cost = d1 + d2
                                best_s_move = {
                                    'station': s,
                                    'd1': d1,
                                    'd2': d2,
                                    'arrival': arr_c
                                }
                        
                        if best_s_move:
                            chosen_station_candidate = best_s_move['station']
                            travel_dist_candidate = best_s_move['d1'] # Only leg 1 for now
                            arrival_at_cust_candidate = best_s_move['arrival']
                        else:
                            # If we can't go direct AND can't find a station -> Try next candidate
                            continue

                    # --- CHECK 3: TIME WINDOWS---
                    # Check "Hard TWs and vk would arrive after time window"
                    if has_tw:
                        # Check Arrival at Customer
                        if arrival_at_cust_candidate > instance.due_dates[candidate]:
                            continue
                        
                        # Check Return to Depot Feasibility (Global Check)
                        svc = instance.service_times[candidate]
                        dept_from_cust = max(arrival_at_cust_candidate, instance.ready_times[candidate]) + svc
                        dist_home = instance.dist_matrix[candidate, depot]
                        arrival_at_depot = dept_from_cust + dist_home
                        if arrival_at_depot > instance.due_dates[depot] or (has_max_travel_time and arrival_at_depot > max_travel_time):
                            continue
                    else:
                        # For non-TW, check max travel time (if applicable)
                        if has_max_travel_time:
                            svc = getattr(instance, "service_times", np.zeros(n))[candidate]
                            dist_home = instance.dist_matrix[candidate, depot]
                            arrival_at_depot = arrival_at_cust_candidate + svc + dist_home
                            if arrival_at_depot > max_travel_time:
                                continue
                    
                    # All checks passed! Use this candidate
                    best_cust = candidate
                    move_type = move_type_candidate
                    chosen_station = chosen_station_candidate
                    travel_dist = travel_dist_candidate
                    arrival_at_cust = arrival_at_cust_candidate
                    break  # Found a feasible candidate
                
                # If no candidate passed all checks, close the route
                if best_cust is None:
                    break

                # --- EXECUTE MOVE ---
                
                if move_type == "VIA_STATION":
                    # Move to charging station
                    route.append(chosen_station)
                    # Update battery after traveling to station
                    energy_to_station = instance.dist_matrix[curr_node, chosen_station] * energy_per_dist
                    curr_batt -= energy_to_station
                    curr_time += instance.dist_matrix[curr_node, chosen_station]
                    # Charge at station (linear charging time)
                    charge_time = compute_charge_time(curr_batt)
                    curr_batt = battery_cap
                    curr_time += charge_time
                    curr_node = chosen_station
                    # Distance for next leg (Station -> Cust)
                    travel_dist = instance.dist_matrix[chosen_station, best_cust] 
                
                # Move to destination
                route.append(best_cust)
                unvisited.remove(best_cust)
                load += instance.demands[best_cust]
                
                curr_batt -= travel_dist * energy_per_dist
                
                if has_tw:
                    curr_time = max(curr_time + travel_dist, instance.ready_times[best_cust])
                    curr_time += instance.service_times[best_cust]
                else:
                    curr_time += travel_dist
                
                curr_node = best_cust

            # End of Route (While loop broke)
            
            # Final check: Do we need to charge to get home?
            if curr_node != depot:
                 d_home = instance.dist_matrix[curr_node, depot]
                 if has_battery and curr_batt < d_home * energy_per_dist:
                     # We cannot reach home directly. Find BEST station.
                     best_s_home = None
                     min_cost_home = float('inf')
                     
                     for s in stations:
                         d1 = instance.dist_matrix[curr_node, s]
                         d2 = instance.dist_matrix[s, depot]
                         e1 = d1 * energy_per_dist
                         e2 = d2 * energy_per_dist
                         
                         # Check Reachability (Leg 1 & Leg 2)
                         if curr_batt < e1: continue
                         if battery_cap < e2: continue
                         
                         # Check Time (including charging)
                         arr_s = curr_time + d1
                         batt_at_s = curr_batt - e1
                         req_charge_time = compute_charge_time(batt_at_s) # Fill to full
                         dept_s = arr_s + req_charge_time
                         arr_depot = dept_s + d2
                         
                         if has_tw and arr_depot > instance.due_dates[depot]:
                             continue
                         if has_max_travel_time and arr_depot > max_travel_time:
                             continue
                             
                         # Greedy selection: minimize distance
                         if (d1 + d2) < min_cost_home:
                             min_cost_home = d1 + d2
                             best_s_home = s
                     
                     if best_s_home is not None:
                         route.append(best_s_home)
                         # Update logic for consistency (though loop ends immediately after)
                         d1 = instance.dist_matrix[curr_node, best_s_home]
                         curr_batt -= d1 * energy_per_dist
                         curr_time += d1
                         curr_time += compute_charge_time(curr_batt)
                         curr_batt = battery_cap
                         curr_node = best_s_home
            
            route.append(depot)
            routes.append(route)

        return routes

# Global instance - only one problem type needed
VRP_PROBLEM_TYPE = VRPProblemType()