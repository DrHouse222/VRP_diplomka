#!/usr/bin/env python3
"""
Unified VRP problem type configuration.
Handles both CVRP and CVRPTW automatically.
"""

from typing import List, Dict, Any, Optional
import operator
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)


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
        Optimized GVRP route construction with safeguards against infinite loops.
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
        energy_per_dist = getattr(instance, "energy_consumption", 1.0)
        base_charge_time = 100.0
        charge_rate = base_charge_time / battery_cap if has_battery and battery_cap > 0 else 0.0

        def compute_charge_time(current_batt):
            """Compute linear charging time based on current battery level."""
            if not has_battery:
                return 0.0
            energy_needed = battery_cap - current_batt
            return charge_rate * energy_needed

        # Identify Charging Stations
        node_types = getattr(instance, "node_types", None)
        stations = []
        if node_types is not None:
            stations = [i for i in range(n) if node_types[i] == 2]

        # ==================== Precompute Station Distances ====================
        station_distances = {}
        nearest_station_idx = {}
        dist_to_nearest_charger = {}

        if has_battery and stations:
            station_arr = np.array(stations)

            for i in range(n):
                dists_to_stations = instance.dist_matrix[i, station_arr]
                min_idx = np.argmin(dists_to_stations)

                dist_to_nearest_charger[i] = dists_to_stations[min_idx]
                nearest_station_idx[i] = stations[min_idx]
                station_distances[i] = dists_to_stations
        else:
            for i in range(n):
                dist_to_nearest_charger[i] = 0

        # ==================== Precompute K-Nearest Stations ====================
        K = 3
        K_NEAREST = min(K, len(stations)) if stations else 0
        k_nearest_stations = {}

        if has_battery and stations:
            station_arr = np.array(stations)
            for i in range(n):
                sorted_indices = np.argsort(station_distances[i])[:K_NEAREST]
                k_nearest_stations[i] = station_arr[sorted_indices].tolist()

        # Time window setup
        has_tw = self.has_time_windows(instance)
        has_max_travel_time = hasattr(instance, "max_travel_time")
        max_travel_time = getattr(instance, "max_travel_time", float('inf')) if has_max_travel_time else float('inf')

        # Initialize unvisited nodes
        unvisited = set(range(0, n))
        unvisited.discard(depot)
        if node_types is not None:
            for i in range(n):
                if node_types[i] == 2:
                    unvisited.discard(i)

        routes: List[List[int]] = []

        # ==================== MAIN ROUTING LOOP ====================
        while unvisited:
            # Start a new route
            route = [depot]
            load = 0.0
            current_node = depot
            current_time = 0.0
            current_battery = battery_cap

            customers_added_this_route = 0
            inner_iterations = 0
            MAX_INNER_ITERATIONS = len(unvisited) * 2  # Safety for inner loop

            while True:
                inner_iterations += 1

                # SAFEGUARD: Inner loop iteration check
                if inner_iterations > MAX_INNER_ITERATIONS:
                    logging.warning(f"Inner loop exceeded {MAX_INNER_ITERATIONS} iterations. Breaking route.")
                    break
                
                candidates = list(unvisited)

                # If no candidates left, must finish route
                if not candidates:
                    break
                
                feasible_candidates: List[int] = []
                candidate_infos = []
                scores: List[float] = []

                # --- EVALUATE EACH CANDIDATE CUSTOMER ---
                for customer in candidates:
                    demand = instance.demands[customer]

                    # 1. CAPACITY CHECK
                    if load + demand > max_capacity:
                        continue
                    
                    # 2. GREEN MOVE ANALYSIS
                    valid_move_found = False
                    best_move_info = None

                    # --- OPTION A: DIRECT MOVE ---
                    dist_direct = instance.dist_matrix[current_node, customer]
                    energy_direct = dist_direct * energy_per_dist

                    if current_battery >= energy_direct:
                        batt_after = current_battery - energy_direct
                        energy_safety = dist_to_nearest_charger[customer] * energy_per_dist

                        if batt_after >= energy_safety:
                            arrival_direct = current_time + dist_direct

                            if not has_tw or arrival_direct <= instance.due_dates[customer]:
                                ready = instance.ready_times[customer] if has_tw else 0.0
                                service = instance.service_times[customer] if has_tw else 0.0
                                dept_time = max(arrival_direct, ready) + service
                                dist_home = instance.dist_matrix[customer, depot]
                                arrival_at_depot = dept_time + dist_home

                                tw_ok = not has_tw or (arrival_at_depot <= instance.due_dates[depot])
                                travel_time_ok = not has_max_travel_time or (arrival_at_depot <= max_travel_time)

                                if tw_ok and travel_time_ok:
                                    valid_move_found = True
                                    best_move_info = {
                                        'is_direct': True,
                                        'dist': dist_direct,
                                        'arrival': arrival_direct,
                                        'station': None
                                    }

                    # --- OPTION B: VIA CHARGING STATION ---
                    if not valid_move_found and has_battery:
                        min_total_dist = float('inf')

                        stations_from_current = set(k_nearest_stations.get(current_node, []))
                        stations_near_dest = set(k_nearest_stations.get(customer, []))
                        candidate_stations = list(stations_from_current | stations_near_dest)

                        if not candidate_stations and stations:
                            candidate_stations = stations[:K_NEAREST]

                        for station in candidate_stations:
                            d1 = instance.dist_matrix[current_node, station]
                            e1 = d1 * energy_per_dist

                            if current_battery < e1:
                                continue
                            
                            d2 = instance.dist_matrix[station, customer]
                            e2 = d2 * energy_per_dist

                            total_dist = d1 + d2
                            if total_dist >= min_total_dist:
                                continue
                            
                            if battery_cap < e2:
                                continue
                            
                            if (battery_cap - e2) < (dist_to_nearest_charger[customer] * energy_per_dist):
                                continue
                            
                            arrival_at_station = current_time + d1
                            batt_at_station = current_battery - e1
                            charge_time = compute_charge_time(batt_at_station)
                            dept_from_station = arrival_at_station + charge_time
                            arrival_at_cust = dept_from_station + d2

                            if has_tw:
                                if arrival_at_cust > instance.due_dates[customer]:
                                    continue
                                
                                ready = instance.ready_times[customer]
                                service = instance.service_times[customer]
                                dept_cust = max(arrival_at_cust, ready) + service
                                dist_home = instance.dist_matrix[customer, depot]
                                arrival_at_depot = dept_cust + dist_home

                                if arrival_at_depot > instance.due_dates[depot]:
                                    continue
                                
                                if has_max_travel_time and arrival_at_depot > max_travel_time:
                                    continue
                            else:
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

                    if not valid_move_found:
                        continue
                    
                    # 3. SCORING
                    features = feature_extractor.extract_features(
                        request=customer,
                        current_route=route,
                        current_load=load,
                        current_position=current_node,
                        current_time=current_time if has_tw else 0.0,
                        current_battery=current_battery if has_battery else None,
                        dist_to_nearest_charger=dist_to_nearest_charger if has_battery else None
                    )

                    actual_dist = best_move_info['dist']
                    features['dist_from_current'] = actual_dist
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
                    # SAFEGUARD: Check if we added any customers this route
                    if customers_added_this_route == 0:
                        logging.warning(f"Empty route. No customers feasible from depot. {len(unvisited)} remain.")
                        logging.error(f"Unserved customers: {sorted(list(unvisited))}")
                        # Return what we have so far rather than infinite loop
                        return routes if routes else [[depot, depot]]

                    # Try to return to depot (with charging if needed)
                    if has_battery and current_node != depot:
                        dist_to_depot = instance.dist_matrix[current_node, depot]
                        energy_needed = dist_to_depot * energy_per_dist

                        if current_battery < energy_needed and stations:
                            best_station = None
                            min_total_dist = float('inf')

                            stations_from_current = set(k_nearest_stations.get(current_node, []))
                            stations_near_depot = set(k_nearest_stations.get(depot, []))
                            candidate_stations = list(stations_from_current | stations_near_depot)

                            if not candidate_stations:
                                candidate_stations = stations[:K_NEAREST]

                            for station in candidate_stations:
                                d1 = instance.dist_matrix[current_node, station]
                                d2 = instance.dist_matrix[station, depot]
                                e1 = d1 * energy_per_dist
                                e2 = d2 * energy_per_dist

                                if current_battery < e1:
                                    continue
                                if battery_cap < e2:
                                    continue
                                
                                if has_tw:
                                    arrival_stat = current_time + d1
                                    batt_at_station = current_battery - e1
                                    charge_time = compute_charge_time(batt_at_station)
                                    dept_stat = arrival_stat + charge_time
                                    arrival_depot = dept_stat + d2
                                    if arrival_depot > instance.due_dates[depot]:
                                        continue
                                    if has_max_travel_time and arrival_depot > max_travel_time:
                                        continue
                                else:
                                    if has_max_travel_time:
                                        arrival_stat = current_time + d1
                                        batt_at_station = current_battery - e1
                                        charge_time = compute_charge_time(batt_at_station)
                                        arrival_depot = arrival_stat + charge_time + d2
                                        if arrival_depot > max_travel_time:
                                            continue
                                        
                                total_dist = d1 + d2
                                if total_dist < min_total_dist:
                                    min_total_dist = total_dist
                                    best_station = station

                            if best_station is not None:
                                route.append(best_station)
                                energy_to_station = instance.dist_matrix[current_node, best_station] * energy_per_dist
                                current_battery -= energy_to_station
                                current_time += instance.dist_matrix[current_node, best_station]
                                charge_time = compute_charge_time(current_battery)
                                current_battery = battery_cap
                                current_time += charge_time
                                current_node = best_station
                            else:
                                # FALLBACK: No k-nearest stations work, try ALL stations
                                best_station = None
                                min_total_dist = float('inf')

                                for station in stations:  # Check ALL stations
                                    d1 = instance.dist_matrix[current_node, station]
                                    d2 = instance.dist_matrix[station, depot]
                                    e1 = d1 * energy_per_dist
                                    e2 = d2 * energy_per_dist

                                    if current_battery < e1:
                                        continue
                                    if battery_cap < e2:
                                        continue
                                    
                                    # Relaxed time window check - allow slight violations
                                    if has_tw:
                                        arrival_stat = current_time + d1
                                        batt_at_station = current_battery - e1
                                        charge_time = compute_charge_time(batt_at_station)
                                        dept_stat = arrival_stat + charge_time
                                        arrival_depot = dept_stat + d2

                                        # Allow small time window violations in emergency
                                        time_violation = max(0, arrival_depot - instance.due_dates[depot])
                                        if time_violation > 100:  # Only reject if violation is large
                                            continue
                                        
                                    total_dist = d1 + d2
                                    if total_dist < min_total_dist:
                                        min_total_dist = total_dist
                                        best_station = station

                                if best_station is not None:
                                    route.append(best_station)
                                    energy_to_station = instance.dist_matrix[current_node, best_station] * energy_per_dist
                                    current_battery -= energy_to_station
                                    current_time += instance.dist_matrix[current_node, best_station]
                                    charge_time = compute_charge_time(current_battery)
                                    current_battery = battery_cap
                                    current_time += charge_time
                                    current_node = best_station
                                    logging.info(f"Emergency station {best_station} used successfully.")
                                else:
                                    # Force return to depot (mark as constraint violation)
                                    logging.error(f"CONSTRAINT VIOLATION: Forcing return to depot from node {current_node}")

                    route.append(depot)
                    break
                
                # Pick best candidate
                best_idx = int(np.argmin(scores))
                best_customer = feasible_candidates[best_idx]
                info = candidate_infos[best_idx]

                # Execute Move
                if not info['is_direct']:
                    station = info['station']
                    route.append(station)

                    energy_to_station = info['station_dist'] * energy_per_dist
                    current_battery -= energy_to_station
                    current_time += info['station_dist']

                    charge_time = compute_charge_time(current_battery)
                    current_battery = battery_cap
                    current_time += charge_time

                    travel_to_cust = instance.dist_matrix[station, best_customer]
                else:
                    travel_to_cust = info['dist']

                route.append(best_customer)
                load += instance.demands[best_customer]
                unvisited.remove(best_customer)
                customers_added_this_route += 1
                consecutive_empty_routes = 0  # Reset counter on successful addition

                current_node = best_customer

                if has_tw:
                    arrival = current_time + travel_to_cust
                    ready = instance.ready_times[best_customer]
                    service = instance.service_times[best_customer]
                    start_service = max(arrival, ready)
                    current_time = start_service + service
                else:
                    current_time += travel_to_cust

                energy_used = travel_to_cust * energy_per_dist
                current_battery -= energy_used

            routes.append(route)

        # Final check
        if unvisited:
            logging.warning(f"Algorithm completed with {len(unvisited)} unvisited customers: {sorted(list(unvisited))}")

        return routes


# Global instance - only one problem type needed
VRP_PROBLEM_TYPE = VRPProblemType()