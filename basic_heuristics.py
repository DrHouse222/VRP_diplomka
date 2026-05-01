"""
Baseline routing heuristics for VRP variants.

Implements nearest-neighbor and Clarke-Wright savings style constructors with
support for capacity constraints, time windows, battery constraints, and
multi-depot variants used in this project.
"""

import numpy as np
import os
import glob
import csv
import json
from parser import VRPInstance, VRPTWInstance, GVRPMultiTechInstance
from vrp_problem import VRP_PROBLEM_TYPE
from typing import List, Tuple, Optional


from vrp_feature_extractor import VRPFeatureExtractor

def nearest_neighbor_heuristic(instance, bool_capacity=True):
    """
    Nearest-neighbor baseline implemented via the generic solve_with_scoring
    machinery: the scoring function is simply the current distance to the
    candidate customer (dist_from_current), so the solver always prefers
    the nearest feasible customer. No force-adding of customers; any
    unserved customers are penalized via compute_cost.
    """
    try:
        idx = VRP_PROBLEM_TYPE.feature_names.index("dist_from_current")
    except ValueError:
        idx = 1  # very conservative fallback if the name is missing

    def nn_scoring_func(*feature_values):
        return float(feature_values[idx])

    feature_extractor = VRPFeatureExtractor(instance)

    return VRP_PROBLEM_TYPE.solve_with_scoring(
        instance, feature_extractor, nn_scoring_func, bool_capacity
    )


def saving_heuristic(instance, bool_capacity=True) -> List[List[int]]:
    """
    Construct GVRP routes using Clarke-Wright Savings Heuristic with Battery-Aware Savings.
    Handles:
      - Capacity
      - Time Windows (Hard)
      - Battery Constraints (Reachability + Safety Buffer)
      - Automatic Charging Station Insertion
      - Multi-depot: each customer assigned to nearest feasible depot; only merge routes sharing same depot.
    
    Uses route costs with charging stations
    """
    n = instance.dimension
    depot = getattr(instance, "depot", 0)
    depots = getattr(instance, "depots", None)
    is_multi_depot = depots is not None and len(depots) > 1
    depots_list = list(depots) if is_multi_depot else [depot]
    
    if bool_capacity:
        max_capacity = getattr(instance, "capacity", 0.0)
    else:
        max_capacity = float('inf')
    
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
    
    K_NEAREST = min(5, len(stations)) if stations else 0
    k_nearest_stations = {}
    
    if has_battery and stations:
        station_arr = np.array(stations)
        for i in range(n):
            sorted_indices = np.argsort(station_distances[i])[:K_NEAREST]
            k_nearest_stations[i] = station_arr[sorted_indices].tolist()
    
    has_tw = all(hasattr(instance, attr) for attr in ("ready_times", "due_dates", "service_times"))
    max_travel_time_raw = getattr(instance, "max_travel_time", None)
    if max_travel_time_raw is not None and max_travel_time_raw > 0 and np.isfinite(max_travel_time_raw):
        has_max_travel_time = True
        max_travel_time = float(max_travel_time_raw)
    else:
        has_max_travel_time = False
        max_travel_time = float('inf')
    
    customers = []
    if node_types is not None:
        customers = [i for i in range(n) if node_types[i] == 1]
    else:
        customers = [i for i in range(n) if i not in depots_list]
    
    customer_to_depot = {}
    for c in customers:
        if is_multi_depot:
            best_d = None
            best_dist = float('inf')
            for d in depots_list:
                if instance.demands[c] > max_capacity:
                    continue
                dist_d = instance.dist_matrix[d, c]
                if has_battery and dist_d * energy_per_dist > battery_cap:
                    continue
                if has_battery and (battery_cap - dist_d * energy_per_dist) < dist_to_nearest_charger[c] * energy_per_dist:
                    continue
                if has_tw and dist_d > instance.due_dates[c]:
                    continue
                if dist_d < best_dist:
                    best_dist = dist_d
                    best_d = d
            customer_to_depot[c] = best_d if best_d is not None else depots_list[0]
        else:
            customer_to_depot[c] = depot
    
    
    def calculate_route_distance(route: List[int]) -> float:
        """Calculate total distance of a route."""
        total_dist = 0.0
        for i in range(len(route) - 1):
            total_dist += instance.dist_matrix[route[i], route[i+1]]
        return total_dist
    
    def can_connect_customers(cust_from, cust_to, current_time, current_battery):
        """
        Check if we can travel from cust_from to cust_to (with charging if needed).
        Returns: (is_feasible, route_segment, arrival_time, battery_after)
        """
        dist_direct = instance.dist_matrix[cust_from, cust_to]
        energy_direct = dist_direct * energy_per_dist
        
        if current_battery >= energy_direct:
            batt_after = current_battery - energy_direct
            energy_safety = dist_to_nearest_charger[cust_to] * energy_per_dist
            
            if batt_after >= energy_safety:
                arrival = current_time + dist_direct
                
                if not has_tw or arrival <= instance.due_dates[cust_to]:
                    return True, [cust_to], arrival, batt_after
        
        if has_battery:
            stations_from_current = set(k_nearest_stations.get(cust_from, []))
            stations_near_dest = set(k_nearest_stations.get(cust_to, []))
            candidate_stations = list(stations_from_current | stations_near_dest)
            
            if not candidate_stations and stations:
                candidate_stations = stations[:K_NEAREST]
            
            best_station = None
            best_arrival = float('inf')
            best_battery = 0
            min_dist = float('inf')
            
            for station in candidate_stations:
                d1 = instance.dist_matrix[cust_from, station]
                e1 = d1 * energy_per_dist
                
                if current_battery < e1:
                    continue
                
                d2 = instance.dist_matrix[station, cust_to]
                e2 = d2 * energy_per_dist
                
                if battery_cap < e2:
                    continue
                
                if (battery_cap - e2) < (dist_to_nearest_charger[cust_to] * energy_per_dist):
                    continue
                
                arrival_at_station = current_time + d1
                batt_at_station = current_battery - e1
                charge_time = compute_charge_time(batt_at_station)
                dept_from_station = arrival_at_station + charge_time
                arrival_at_cust = dept_from_station + d2
                
                if has_tw and arrival_at_cust > instance.due_dates[cust_to]:
                    continue
                
                total_dist = d1 + d2
                if total_dist < min_dist:
                    min_dist = total_dist
                    best_station = station
                    best_arrival = arrival_at_cust
                    best_battery = battery_cap - e2
            
            if best_station is not None:
                return True, [best_station, cust_to], best_arrival, best_battery
        
        return False, None, None, None
    
    def validate_route(route_nodes, route_depot):
        """
        Validate entire route with all constraints.
        route_depot: depot this route starts and ends at (for multi-depot).
        Returns: (is_valid, final_route_with_stations)
        """
        if len(route_nodes) < 3:  # depot + at least 1 customer + depot
            return False, None
        
        load = 0.0
        current_time = 0.0
        current_battery = battery_cap
        current_node = route_depot
        final_route = [route_depot]
        
        for i in range(1, len(route_nodes) - 1):
            customer = route_nodes[i]
            demand = instance.demands[customer]
            
            if load + demand > max_capacity:
                return False, None
            
            feasible, segment, arrival, batt_after = can_connect_customers(
                current_node, customer, current_time, current_battery
            )
            
            if not feasible:
                return False, None
            
            for node in segment:
                final_route.append(node)
                if node in stations:
                    dist_to_station = instance.dist_matrix[current_node, node]
                    current_battery -= dist_to_station * energy_per_dist
                    current_time += dist_to_station
                    charge_time = compute_charge_time(current_battery)
                    current_battery = battery_cap
                    current_time += charge_time
                    current_node = node
            
            load += demand
            
            if has_tw:
                ready = instance.ready_times[customer]
                service = instance.service_times[customer]
                start_service = max(arrival, ready)
                current_time = start_service + service
            else:
                current_time = arrival
            
            current_battery = batt_after
            current_node = customer
        
        dist_to_depot = instance.dist_matrix[current_node, route_depot]
        energy_needed = dist_to_depot * energy_per_dist
        
        if current_battery >= energy_needed:
            arrival_depot = current_time + dist_to_depot
            
            tw_ok = not has_tw or arrival_depot <= instance.due_dates[route_depot]
            travel_ok = not has_max_travel_time or arrival_depot <= max_travel_time
            
            if tw_ok and travel_ok:
                final_route.append(route_depot)
                return True, final_route
        
        if has_battery:
            stations_from_current = set(k_nearest_stations.get(current_node, []))
            stations_near_depot = set(k_nearest_stations.get(route_depot, []))
            candidate_stations = list(stations_from_current | stations_near_depot)
            
            if not candidate_stations:
                candidate_stations = stations[:K_NEAREST]
            
            for station in candidate_stations:
                d1 = instance.dist_matrix[current_node, station]
                d2 = instance.dist_matrix[station, route_depot]
                e1 = d1 * energy_per_dist
                e2 = d2 * energy_per_dist
                
                if current_battery < e1 or battery_cap < e2:
                    continue
                
                arrival_stat = current_time + d1
                batt_at_station = current_battery - e1
                charge_time = compute_charge_time(batt_at_station)
                arrival_depot = arrival_stat + charge_time + d2
                
                tw_ok = not has_tw or arrival_depot <= instance.due_dates[route_depot]
                travel_ok = not has_max_travel_time or arrival_depot <= max_travel_time
                
                if tw_ok and travel_ok:
                    final_route.append(station)
                    final_route.append(route_depot)
                    return True, final_route
        
        return False, None
    
    def calculate_actual_route_cost(route_nodes: List[int], route_depot: int) -> float:
        """
        Calculate the actual cost of a route with charging stations inserted.
        Returns float('inf') if route is infeasible.
        """
        is_valid, final_route = validate_route(route_nodes, route_depot)
        if not is_valid:
            return float('inf')
        return calculate_route_distance(final_route)
    
    def calculate_actual_savings(cust_i: int, cust_j: int, route_depot: int) -> float:
        """
        Calculate savings using ACTUAL route costs (with charging stations).
        
        Returns:
            Positive float if merging saves distance, negative if it increases distance
        """
        route_i = [route_depot, cust_i, route_depot]
        route_j = [route_depot, cust_j, route_depot]
        
        cost_i = calculate_actual_route_cost(route_i, route_depot)
        cost_j = calculate_actual_route_cost(route_j, route_depot)
        
        if cost_i == float('inf') or cost_j == float('inf'):
            return -float('inf')  # Infeasible separate routes
        
        cost_separate = cost_i + cost_j
        
        merge_orders = [
            [route_depot, cust_i, cust_j, route_depot],
            [route_depot, cust_j, cust_i, route_depot]
        ]
        
        best_merged_cost = float('inf')
        
        for test_route in merge_orders:
            total_demand = instance.demands[cust_i] + instance.demands[cust_j]
            if total_demand > max_capacity:
                continue
            
            cost_merged = calculate_actual_route_cost(test_route, route_depot)
            best_merged_cost = min(best_merged_cost, cost_merged)
        
        if best_merged_cost == float('inf'):
            return -float('inf')  # Infeasible merge
        
        actual_savings = cost_separate - best_merged_cost
        
        return actual_savings
    
    routes = {}  # route_id -> {route: [nodes], load: float, customers: set, depot: int}
    route_id_counter = 0
    customer_to_route = {}  # customer -> route_id
    
    for customer in customers:
        route_depot = customer_to_depot[customer]
        routes[route_id_counter] = {
            'route': [route_depot, customer, route_depot],
            'load': instance.demands[customer],
            'customers': {customer},
            'depot': route_depot
        }
        customer_to_route[customer] = route_id_counter
        route_id_counter += 1
    
    savings = []
    
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            cust_i = customers[i]
            cust_j = customers[j]
            
            if customer_to_depot[cust_i] != customer_to_depot[cust_j]:
                continue
            
            route_depot = customer_to_depot[cust_i]
            
            actual_saving = calculate_actual_savings(cust_i, cust_j, route_depot)
            
            if actual_saving > 0:
                savings.append({
                    'customers': (cust_i, cust_j),
                    'saving': actual_saving
                })
    
    savings.sort(key=lambda x: x['saving'], reverse=True)
    
    
    
    merges_attempted = 0
    merges_successful = 0
    
    for saving_entry in savings:
        cust_i, cust_j = saving_entry['customers']
        
        if customer_to_route[cust_i] == customer_to_route[cust_j]:
            continue
        
        route_i_id = customer_to_route[cust_i]
        route_j_id = customer_to_route[cust_j]
        
        if route_i_id not in routes or route_j_id not in routes:
            continue
        
        route_i = routes[route_i_id]
        route_j = routes[route_j_id]
        
        if route_i['depot'] != route_j['depot']:
            continue
        
        depot_ij = route_i['depot']
        
        route_i_customers = [c for c in route_i['route'] if c != depot_ij and c not in stations]
        route_j_customers = [c for c in route_j['route'] if c != depot_ij and c not in stations]
        
        i_is_first = (route_i_customers[0] == cust_i)
        i_is_last = (route_i_customers[-1] == cust_i)
        j_is_first = (route_j_customers[0] == cust_j)
        j_is_last = (route_j_customers[-1] == cust_j)
        
        if not ((i_is_first or i_is_last) and (j_is_first or j_is_last)):
            continue
        
        new_load = route_i['load'] + route_j['load']
        
        if new_load > max_capacity:
            continue
        
        merges_attempted += 1
        
        merge_orders = []
        
        if i_is_last and j_is_first:
            merge_orders.append(route_i_customers + route_j_customers)
        
        if i_is_last and j_is_last:
            merge_orders.append(route_i_customers + route_j_customers[::-1])
        
        if i_is_first and j_is_first:
            merge_orders.append(route_i_customers[::-1] + route_j_customers)
        
        if i_is_first and j_is_last:
            merge_orders.append(route_j_customers + route_i_customers)
        
        merged = False
        for merged_customers in merge_orders:
            test_route = [depot_ij] + merged_customers + [depot_ij]
            
            is_valid, final_route = validate_route(test_route, depot_ij)
            
            if is_valid:
                del routes[route_j_id]
                
                routes[route_i_id] = {
                    'route': final_route,
                    'load': new_load,
                    'customers': route_i['customers'] | route_j['customers'],
                    'depot': depot_ij
                }
                
                for c in route_j['customers']:
                    customer_to_route[c] = route_i_id
                
                merged = True
                merges_successful += 1
                break
        
        if merged:
            continue
    
    
    final_routes = [route_data['route'] for route_data in routes.values()]
    
    
    return final_routes