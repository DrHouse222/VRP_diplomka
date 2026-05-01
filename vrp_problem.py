#!/usr/bin/env python3
"""
Unified VRP problem type configuration.
Handles both VRP and VRPTW automatically.
"""

from typing import List, Dict, Any, Optional
import operator
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)


class VRPProblemType:
    """Unified configuration for VRP variants"""
    
    def __init__(self):
        self.base_feature_names = [
            'dist_to_depot', 'dist_from_current', 'demand', 'remaining_capacity',
            'savings', 'route_urgency',
            'depot_distance_advantage', 'depot_rank'
        ]
    
        self.tw_feature_names = [
            'current_time', 'arrival_time', 'ready_time', 'due_time',
            'wait_time'
        ]
        
        self.gvrp_feature_names = [
            'current_battery', 'energy_to_customer'
        ]
    
    def has_time_windows(self, instance) -> bool:
        """Check if instance has time windows."""
        return all(hasattr(instance, attr) for attr in ("ready_times", "due_dates", "service_times"))
    
    @property
    def feature_names(self) -> List[str]:
        """Return all possible features (for GP primitive set creation)."""
        return self.base_feature_names + self.tw_feature_names + self.gvrp_feature_names
    
    @property
    def num_features(self) -> int:
        """Return total number of features available to the GP primitive set."""
        return len(self.base_feature_names) + len(self.tw_feature_names) + len(self.gvrp_feature_names)
    
    def extract_feature_values(self, features: Dict[str, float]) -> List[float]:
        """Extract feature values in the order expected by GP function."""
        return [features.get(name, 0.0) for name in self.feature_names]
    
    def create_primitive_set(self, gp_module) -> Optional[Any]:
        """Create GP primitive set for unified VRP (DEAP-specific)."""
        if gp_module is None:
            return None
            
        pset = gp_module.PrimitiveSet("MAIN", self.num_features)
        
        for i, name in enumerate(self.feature_names):
            pset.renameArguments(**{f"ARG{i}": name})
        
        # Adding operators
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        
        def protected_div(left, right):
            try:
                return left / right if abs(right) > 1e-6 else 1.0
            except:
                return 1.0
        pset.addPrimitive(protected_div, 2)

        def safe_min(a, b):
            return a if a <= b else b

        def safe_max(a, b):
            return a if a >= b else b

        pset.addPrimitive(safe_min, 2, name="min")
        pset.addPrimitive(safe_max, 2, name="max")
        
        # Adding contsants
        pset.addTerminal(0.0)
        pset.addTerminal(1.0)
        pset.addTerminal(-1.0)
        
        return pset
    
    def compute_cost(self, instance, routes) -> float:
        """
        Compute cost for VRP solution.
        Returns total distance traveled,
        plus a penalty for any unserved customers.
        """
        total_distance = 0.0
        demands = getattr(instance, "demands", None)
        dist_matrix = getattr(instance, "dist_matrix", None)

        if demands is None or dist_matrix is None:
            raise ValueError("Instance must have 'demands' and 'dist_matrix' attributes.")
        
        for route in routes:
            if not route:
                continue
            
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                travel = float(dist_matrix[u, v])
                total_distance += travel

        node_types = getattr(instance, "node_types", None)
        depot = getattr(instance, "depot", 0)
        n = len(demands)

        if node_types is not None:
            customers = {i for i in range(n) if node_types[i] == 1}
        else:
            customers = {i for i in range(n) if i != depot}

        served_customers = set()
        for route in routes:
            for node in route:
                if node in customers:
                    served_customers.add(node)

        unserved_customers = customers - served_customers
        if unserved_customers:
            depots = getattr(instance, "depots", None)
            if depots is None:
                depots = [depot]
            for cust in unserved_customers:
                dists = [float(dist_matrix[d, cust]) for d in depots]
                min_dist = min(dists) if dists else 0.0
                total_distance += 4.0 * min_dist

        return total_distance

    def solve_with_scoring(self, instance, feature_extractor, scoring_func, bool_capacity=True) -> List[List[int]]:
        """
        Route construction with multi-depot support.
        
        For multi-depot instances (MDVRP/MDVRPTW), at the start of each route,
        all depots evaluate all potential customers, and the depot with the best
        customer is chosen to start that route.
        """
        # 1) Read instance-level configuration and normalize optional attributes.
        n = instance.dimension
        depot = getattr(instance, "depot", 0)
        
        depots = getattr(instance, "depots", None)
        is_multi_depot = depots is not None and len(depots) > 1
        if is_multi_depot:
            depots_list = depots
        else:
            depots_list = [depot]

        depot_route_counts = {d: 0 for d in depots_list}
        total_routes_started = 0

        if bool_capacity:
            max_capacity = getattr(instance, "capacity", 0.0)
        else:
            max_capacity = float('inf')

        # 2) Build constraint context (capacity, battery/charging, time windows, max travel).
        has_battery = getattr(instance, "battery_capacity", 0.0) > 0.0
        battery_cap = getattr(instance, "battery_capacity", float('inf'))
        energy_per_dist = getattr(instance, "energy_consumption", 1.0)
        
        g_inverse_refueling_rate = getattr(instance, "g_inverse_refueling_rate", None)
        if g_inverse_refueling_rate is None or g_inverse_refueling_rate == 0.0:
            base_charge_time = 100.0
            charge_rate = base_charge_time / battery_cap if has_battery and battery_cap > 0 else 0.0
        else:
            charge_rate = g_inverse_refueling_rate

        def compute_charge_time(current_batt):
            """Compute linear charging time based on current battery level.
            
            Uses g_inverse_refueling_rate if available (time per unit of energy),
            otherwise falls back to a default rate.
            """
            if not has_battery:
                return 0.0
            energy_needed = battery_cap - current_batt
            return charge_rate * energy_needed

        node_types = getattr(instance, "node_types", None)
        stations = []
        if node_types is not None:
            stations = [i for i in range(n) if node_types[i] == 2]

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

        K = 3
        K_NEAREST = min(K, len(stations)) if stations else 0
        k_nearest_stations = {}

        if has_battery and stations:
            station_arr = np.array(stations)
            for i in range(n):
                sorted_indices = np.argsort(station_distances[i])[:K_NEAREST]
                k_nearest_stations[i] = station_arr[sorted_indices].tolist()

        has_tw = self.has_time_windows(instance)
        max_travel_raw = getattr(instance, "max_travel_time", None)
        if max_travel_raw is not None and max_travel_raw > 0 and np.isfinite(max_travel_raw):
            has_max_travel_dist = True
            max_travel_dist = float(max_travel_raw)
        else:
            has_max_travel_dist = False
            max_travel_dist = float('inf')

        # 3) Track only customer nodes as unvisited (exclude depots and charging stations).
        unvisited = set(range(0, n))
        for d in depots_list:
            unvisited.discard(d)
        if node_types is not None:
            for i in range(n):
                if node_types[i] == 2:
                    unvisited.discard(i)

        routes: List[List[int]] = []

        # 4) Greedy route construction repeatedly starts a route and extends it by selecting the feasible customer with the lowest score.
        while unvisited:
            route_start_depot = depot
            if is_multi_depot:
                # Multi-depot bootstrap: choose (depot, first_customer) jointly.
                best_depot_customer_pair = None
                best_score = float('inf')
                
                for candidate_depot in depots_list:
                    for customer in unvisited:
                        demand = instance.demands[customer]
                        
                        if demand > max_capacity:
                            continue
                        
                        dist_direct = instance.dist_matrix[candidate_depot, customer]
                        energy_direct = dist_direct * energy_per_dist
                        
                        if has_battery and energy_direct > battery_cap:
                            continue
                        
                        if has_battery:
                            batt_after = battery_cap - energy_direct
                            energy_safety = dist_to_nearest_charger[customer] * energy_per_dist
                            if batt_after < energy_safety:
                                continue

                        arrival_direct = dist_direct  # Starting at time 0
                        if has_tw:
                            if arrival_direct > instance.due_dates[customer]:
                                continue
                            ready = instance.ready_times[customer]
                            service = instance.service_times[customer]
                            dept_time = max(arrival_direct, ready) + service
                            dist_home = instance.dist_matrix[customer, candidate_depot]
                            arrival_at_depot = dept_time + dist_home
                            if arrival_at_depot > instance.due_dates[candidate_depot]:
                                continue
                        else:
                            service = getattr(instance, "service_times", np.zeros(n))[customer]
                            dist_home = instance.dist_matrix[customer, candidate_depot]
                            arrival_at_depot = dist_direct + service + dist_home
                        
                        features = feature_extractor.extract_features(
                            request=customer,
                            current_route=[],
                            current_load=0.0,
                            current_position=candidate_depot,
                            current_time=0.0 if has_tw else 0.0,
                            current_battery=battery_cap if has_battery else None,
                            dist_to_nearest_charger=dist_to_nearest_charger if has_battery else None,
                            route_depot=candidate_depot,
                            bool_capacity=bool_capacity,
                        )
                        features['dist_from_current'] = dist_direct
                        features['savings'] = (
                            features.get('dist_to_depot', 0.0) +
                            features.get('dist_to_depot_from_request', 0.0) -
                            dist_direct
                        )
                        if bool_capacity:
                            features["remaining_capacity"] = max_capacity - demand
                        else:
                            features["remaining_capacity"] = 0.0
                        feature_values = self.extract_feature_values(features)

                        try:
                            score = scoring_func(*feature_values)
                        except Exception:
                            score = 1e6
                        
                        route_dist = dist_direct + instance.dist_matrix[customer, candidate_depot]
                        if has_max_travel_dist and route_dist > max_travel_dist:
                            score += (route_dist - max_travel_dist)
                        
                        if score < best_score:
                            best_score = score
                            best_depot_customer_pair = (candidate_depot, customer)

                if best_depot_customer_pair is None and has_battery and stations:
                    # Fallback allows the first move via a charging station when direct departure from depot is infeasible under battery constraints.
                    for candidate_depot in depots_list:
                        for customer in unvisited:
                            demand = instance.demands[customer]
                            if demand > max_capacity:
                                continue

                            min_total_dist = float("inf")
                            candidate_stations = stations

                            best_via_station_dist = None
                            for station in candidate_stations:
                                d1 = instance.dist_matrix[candidate_depot, station]
                                e1 = d1 * energy_per_dist
                                if battery_cap < e1:
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

                                arrival_at_station = d1  # start at time 0
                                batt_at_station = battery_cap - e1
                                charge_time = compute_charge_time(batt_at_station)
                                dept_from_station = arrival_at_station + charge_time
                                arrival_at_cust = dept_from_station + d2

                                if has_tw:
                                    if arrival_at_cust > instance.due_dates[customer]:
                                        continue
                                    ready = instance.ready_times[customer]
                                    service = instance.service_times[customer]
                                    dept_cust = max(arrival_at_cust, ready) + service
                                    dist_home = instance.dist_matrix[customer, candidate_depot]
                                    arrival_at_depot = dept_cust + dist_home
                                    if arrival_at_depot > instance.due_dates[candidate_depot]:
                                        continue
                                    route_dist_if_added = total_dist + dist_home
                                    if has_max_travel_dist and route_dist_if_added > max_travel_dist:
                                        continue
                                else:
                                    if has_max_travel_dist:
                                        dist_home = instance.dist_matrix[customer, candidate_depot]
                                        route_dist_if_added = total_dist + dist_home
                                        if route_dist_if_added > max_travel_dist:
                                            continue

                                min_total_dist = total_dist
                                best_via_station_dist = total_dist

                            if best_via_station_dist is None:
                                continue

                            features = feature_extractor.extract_features(
                                request=customer,
                                current_route=[],
                                current_load=0.0,
                                current_position=candidate_depot,
                                current_time=0.0 if has_tw else 0.0,
                                current_battery=battery_cap if has_battery else None,
                                dist_to_nearest_charger=dist_to_nearest_charger if has_battery else None,
                                route_depot=candidate_depot,
                                bool_capacity=bool_capacity,
                            )
                            features["dist_from_current"] = best_via_station_dist
                            features["savings"] = (
                                features.get("dist_to_depot", 0.0)
                                + features.get("dist_to_depot_from_request", 0.0)
                                - best_via_station_dist
                            )
                            if bool_capacity:
                                features["remaining_capacity"] = max_capacity - demand
                            else:
                                features["remaining_capacity"] = 0.0

                            feature_values = self.extract_feature_values(features)
                            try:
                                score = scoring_func(*feature_values)
                            except Exception:
                                score = 1e6

                            route_dist = best_via_station_dist + instance.dist_matrix[customer, candidate_depot]
                            if has_max_travel_dist and route_dist > max_travel_dist:
                                score += (route_dist - max_travel_dist)

                            if score < best_score:
                                best_score = score
                                best_depot_customer_pair = (candidate_depot, customer)
                
                if best_depot_customer_pair is None:
                    break
                
                route_start_depot, first_customer = best_depot_customer_pair

                depot_route_counts[route_start_depot] += 1
                total_routes_started += 1
                
            else:
                first_customer = None
            
            route = [route_start_depot]
            load = 0.0
            current_node = route_start_depot
            current_time = 0.0
            current_battery = battery_cap
            current_route_distance = 0.0  # total distance travelled so far on this route (for max_travel_dist)
            
            if is_multi_depot and first_customer is not None:
                dist_to_cust = instance.dist_matrix[route_start_depot, first_customer]
                route.append(first_customer)
                load += instance.demands[first_customer]
                unvisited.remove(first_customer)
                current_node = first_customer
                current_route_distance = dist_to_cust
                
                if has_tw:
                    arrival = current_time + dist_to_cust
                    ready = instance.ready_times[first_customer]
                    service = instance.service_times[first_customer]
                    start_service = max(arrival, ready)
                    current_time = start_service + service
                else:
                    current_time += dist_to_cust
                
                if has_battery:
                    energy_used = dist_to_cust * energy_per_dist
                    current_battery -= energy_used
                
                customers_added_this_route = 1
            else:
                customers_added_this_route = 0  # No first customer added
            inner_iterations = 0
            MAX_INNER_ITERATIONS = len(unvisited) * 2 + 1  # Safety for inner loop

            while True:
                inner_iterations += 1

                if inner_iterations > MAX_INNER_ITERATIONS:
                    logging.warning(f"Inner loop exceeded {MAX_INNER_ITERATIONS} iterations. Breaking route.")
                    route.append(route_start_depot)
                    break
                
                candidates = list(unvisited)

                if not candidates:
                    route.append(route_start_depot)
                    break
                
                feasible_candidates: List[int] = []
                candidate_infos = []
                scores: List[float] = []

                for customer in candidates:
                    # Evaluate each candidate with full feasibility checks first and score only candidates that can be served safely.
                    demand = instance.demands[customer]

                    if load + demand > max_capacity:
                        continue
                    
                    valid_move_found = False
                    best_move_info = None

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
                                dist_home = instance.dist_matrix[customer, route_start_depot]
                                arrival_at_depot = dept_time + dist_home
                                route_dist_if_added = current_route_distance + dist_direct + dist_home

                                tw_ok = not has_tw or (arrival_at_depot <= instance.due_dates[route_start_depot])
                                travel_dist_ok = not has_max_travel_dist or (route_dist_if_added <= max_travel_dist)

                                if tw_ok and travel_dist_ok:
                                    conservative_ok = True
                                    if has_battery and stations:
                                        stat = nearest_station_idx.get(customer, None)
                                        if stat is not None:
                                            d_cs = instance.dist_matrix[customer, stat]
                                            e_cs = d_cs * energy_per_dist
                                            batt_cust = batt_after
                                            if batt_cust < e_cs:
                                                conservative_ok = False
                                            else:
                                                d_sd = instance.dist_matrix[stat, route_start_depot]
                                                e_sd = d_sd * energy_per_dist
                                                if battery_cap < e_sd:
                                                    conservative_ok = False
                                                else:
                                                    full_cycle_dist = current_route_distance + dist_direct + d_cs + d_sd
                                                    if has_max_travel_dist and full_cycle_dist > max_travel_dist:
                                                        conservative_ok = False
                                                    elif has_tw:
                                                        arrival_stat2 = dept_time + d_cs
                                                        batt_at_station2 = batt_cust - e_cs
                                                        charge_time2 = compute_charge_time(batt_at_station2)
                                                        arrival_depot2 = arrival_stat2 + charge_time2 + d_sd
                                                        if arrival_depot2 > instance.due_dates[route_start_depot]:
                                                            conservative_ok = False
                                        else:
                                            conservative_ok = False

                                    if conservative_ok:
                                        valid_move_found = True
                                        best_move_info = {
                                            'is_direct': True,
                                            'dist': dist_direct,
                                            'arrival': arrival_direct,
                                            'station': None
                                        }

                    if not valid_move_found and has_battery:
                        # If direct move fails, try one charging-station detour.
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
                                dist_home = instance.dist_matrix[customer, route_start_depot]
                                arrival_at_depot = dept_cust + dist_home

                                if arrival_at_depot > instance.due_dates[route_start_depot]:
                                    continue
                                
                                route_dist_if_added = current_route_distance + total_dist + instance.dist_matrix[customer, route_start_depot]
                                if has_max_travel_dist and route_dist_if_added > max_travel_dist:
                                    continue
                            else:
                                if has_max_travel_dist:
                                    dist_home = instance.dist_matrix[customer, route_start_depot]
                                    route_dist_if_added = current_route_distance + total_dist + dist_home
                                    if route_dist_if_added > max_travel_dist:
                                        continue
                            
                            conservative_ok = True
                            if has_battery and stations:
                                stat = nearest_station_idx.get(customer, None)
                                if stat is not None:
                                    d_cs = instance.dist_matrix[customer, stat]
                                    e_cs = d_cs * energy_per_dist
                                    batt_cust = battery_cap - e2
                                    if batt_cust < e_cs:
                                        conservative_ok = False
                                    else:
                                        d_sd = instance.dist_matrix[stat, route_start_depot]
                                        e_sd = d_sd * energy_per_dist
                                        if battery_cap < e_sd:
                                            conservative_ok = False
                                        else:
                                            full_cycle_dist = current_route_distance + total_dist + d_cs + d_sd
                                            if has_max_travel_dist and full_cycle_dist > max_travel_dist:
                                                conservative_ok = False
                                            elif has_tw:
                                                arrival_stat2 = dept_cust + d_cs
                                                batt_at_station2 = batt_cust - e_cs
                                                charge_time2 = compute_charge_time(batt_at_station2)
                                                arrival_depot2 = arrival_stat2 + charge_time2 + d_sd
                                                if arrival_depot2 > instance.due_dates[route_start_depot]:
                                                    conservative_ok = False
                                else:
                                    conservative_ok = False

                            if not conservative_ok:
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
                    
                    features = feature_extractor.extract_features(
                        request=customer,
                        current_route=route,
                        current_load=load,
                        current_position=current_node,
                        current_time=current_time if has_tw else 0.0,
                        current_battery=current_battery if has_battery else None,
                        dist_to_nearest_charger=dist_to_nearest_charger if has_battery else None,
                        route_depot=route_start_depot,
                        bool_capacity=bool_capacity,
                    )

                    actual_dist = best_move_info['dist']
                    features['dist_from_current'] = actual_dist
                    features['savings'] = (
                        features.get('dist_to_depot', 0.0) +
                        features.get('dist_to_depot_from_request', 0.0) -
                        actual_dist
                    )
                    if bool_capacity:
                        features["remaining_capacity"] = max_capacity - (load + demand)
                    else:
                        features["remaining_capacity"] = 0.0
                    feature_values = self.extract_feature_values(features)

                    try:
                        score = scoring_func(*feature_values)
                    except Exception:
                        score = 1e6

                    feasible_candidates.append(customer)
                    candidate_infos.append(best_move_info)
                    scores.append(score)

                if not feasible_candidates:
                    # No feasible extension: return to depot (possibly via station).
                    if customers_added_this_route == 0:
                        return routes if routes else [[route_start_depot, route_start_depot]]

                    if has_battery and current_node != route_start_depot:
                        dist_to_depot = instance.dist_matrix[current_node, route_start_depot]
                        energy_needed = dist_to_depot * energy_per_dist

                        if current_battery < energy_needed and stations:
                            best_station = None
                            min_total_dist = float("inf")

                            for station in stations:
                                d1 = instance.dist_matrix[current_node, station]
                                d2 = instance.dist_matrix[station, route_start_depot]
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
                                    if arrival_depot > instance.due_dates[route_start_depot]:
                                        continue

                                route_dist_return = current_route_distance + d1 + d2
                                if has_max_travel_dist and route_dist_return > max_travel_dist:
                                    continue

                                total_dist = d1 + d2
                                if total_dist < min_total_dist:
                                    min_total_dist = total_dist
                                    best_station = station

                            if best_station is not None:
                                route.append(best_station)
                                d1 = instance.dist_matrix[current_node, best_station]
                                current_route_distance += d1
                                energy_to_station = d1 * energy_per_dist
                                current_battery -= energy_to_station
                                current_time += d1
                                charge_time = compute_charge_time(current_battery)
                                current_battery = battery_cap
                                current_time += charge_time
                                current_node = best_station

                    route.append(route_start_depot)
                    break
                
                # Choose the best feasible customer by score and commit the move.
                best_idx = int(np.argmin(scores))
                best_customer = feasible_candidates[best_idx]
                info = candidate_infos[best_idx]

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
                current_route_distance += info['dist']  # leg to customer (direct or via station)
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

        # Unvisited customers are handled in compute_cost() via penalties.
        if unvisited:
            logging.warning(f"Algorithm completed with {len(unvisited)} unvisited customers: {sorted(list(unvisited))}")
            logging.warning(f"Name of instance: {getattr(instance, 'name', 'unknown')}")

        return routes


VRP_PROBLEM_TYPE = VRPProblemType()

