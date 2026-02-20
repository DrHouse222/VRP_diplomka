# nn_heuristic.py
import numpy as np
import os
import glob
import csv
import json
from parser import VRPInstance, VRPTWInstance, GVRPMultiTechInstance
from problem_types import VRP_PROBLEM_TYPE
from data_generation import convert_vrptw_to_gvrptw
from typing import List


def nearest_neighbor_heuristic(instance, bool_capacity=True):
    """
    Build routes using a nearest neighbor heuristic.
    Supports CVRP, VRPTW, and GVRP variants.

    Args:
        instance: VRP, VRPTW, or GVRP instance
        problem: one of {"auto", "vrp", "vrptw", "gvrp"} (auto-detect if "auto")
        bool_capacity: If True, enforce capacity constraints; if False, ignore capacity

    Returns:
        list of routes (list of lists of node indices)
    """
    n = instance.dimension
    depot = getattr(instance, "depot", 0)
    
    # Auto-detect problem type
    has_tw = all(hasattr(instance, attr) for attr in ("ready_times", "due_dates", "service_times"))
    has_battery = getattr(instance, "battery_capacity", 0.0) > 0.0
    
    # GVRP setup
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
    
    # Identify charging stations (type 2)
    node_types = getattr(instance, "node_types", None)
    if node_types is not None:
        stations = [i for i in range(n) if node_types[i] == 2]
    else:
        stations = []
    
    # Pre-calculate distance to nearest charger for every node (safety check)
    dist_to_nearest_charger = {}
    for i in range(n):
        if not stations:
            dist_to_nearest_charger[i] = 0
        else:
            min_dist = min(instance.dist_matrix[i, cs] for cs in stations)
            dist_to_nearest_charger[i] = min_dist
    
    # Initialize unvisited set (exclude depot and charging stations)
    unvisited = set(range(0, n))
    unvisited.discard(depot)
    if node_types is not None:
        for i in range(n):
            if node_types[i] == 2:  # Charging station
                unvisited.discard(i)
    
    if bool_capacity:
        max_capacity = getattr(instance, "capacity", 0.0)
    else:
        max_capacity = float('inf')
    routes = []

    while unvisited:
        route = [depot]
        load = 0.0
        current_node = depot
        current_time = 0.0
        current_battery = battery_cap

        while True:
            feasible_candidates = []
            candidate_distances = []
            candidate_infos = []  # Store move info: {is_direct, dist, station, arrival}
            
            for customer in unvisited:
                demand = instance.demands[customer]
                
                # 1. Capacity check
                if load + demand > max_capacity:
                    continue
                
                # 2. Find feasible move (direct or via station)
                valid_move_found = False
                best_move_info = None
                best_dist = float('inf')
                
                # OPTION A: Direct move
                dist_direct = instance.dist_matrix[current_node, customer]
                energy_direct = dist_direct * energy_per_dist
                
                if not has_battery or current_battery >= energy_direct:
                    # Safety check: can we leave customer to reach a charger?
                    if not has_battery:
                        # No battery constraints
                        valid_direct = True
                    else:
                        batt_after = current_battery - energy_direct
                        energy_safety = dist_to_nearest_charger[customer] * energy_per_dist
                        valid_direct = (batt_after >= energy_safety)
                    
                    if valid_direct:
                        arrival_direct = current_time + dist_direct
                        
                        # Time window check
                        if not has_tw or arrival_direct <= instance.due_dates[customer]:
                            # Check return to depot
                            ready = instance.ready_times[customer] if has_tw else 0.0
                            service = instance.service_times[customer] if has_tw else 0.0
                            dept_time = max(arrival_direct, ready) + service
                            dist_home = instance.dist_matrix[customer, depot]
                            
                            if not has_tw or (dept_time + dist_home <= instance.due_dates[depot]):
                                valid_move_found = True
                                best_move_info = {
                                    'is_direct': True,
                                    'dist': dist_direct,
                                    'arrival': arrival_direct,
                                    'station': None
                                }
                                best_dist = dist_direct
                
                # OPTION B: Via charging station
                if not valid_move_found and has_battery and stations:
                    min_total_dist = float('inf')
                    
                    # Pre-compute distance from current_node to all stations (cache for this iteration)
                    station_dists = [(s, instance.dist_matrix[current_node, s]) for s in stations]
                    # Sort by distance to current node (closer stations first - likely to be better)
                    station_dists.sort(key=lambda x: x[1])
                    
                    for station, d1 in station_dists:
                        # Early skip if this station is already worse than best found
                        if d1 >= min_total_dist:
                            break  # Since stations are sorted, all remaining are worse
                        
                        e1 = d1 * energy_per_dist
                        if current_battery < e1:
                            continue
                        
                        d2 = instance.dist_matrix[station, customer]
                        e2 = d2 * energy_per_dist
                        total_dist = d1 + d2
                        
                        # Early skip if total distance is worse than best
                        if total_dist >= min_total_dist:
                            continue
                        
                        if battery_cap < e2:
                            continue
                        
                        # Safety check at destination
                        if (battery_cap - e2) < (dist_to_nearest_charger[customer] * energy_per_dist):
                            continue
                        
                        # Time window check
                        arrival_at_station = current_time + d1
                        # Calculate battery level when arriving at station
                        batt_at_station = current_battery - (d1 * energy_per_dist)
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
                            if arrival_at_depot > instance.due_dates[depot] or (has_max_travel_time and arrival_at_depot > max_travel_time):
                                continue
                        else:
                            # For non-TW, check max travel time (if applicable)
                            if has_max_travel_time:
                                service = getattr(instance, "service_times", np.zeros(n))[customer] if hasattr(instance, "service_times") else 0.0
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
                        best_dist = total_dist
                
                if not valid_move_found:
                    continue
                
                feasible_candidates.append(customer)
                candidate_distances.append(best_dist)
                candidate_infos.append(best_move_info)
            
            # No feasible candidates - finish route
            if not feasible_candidates:
                # Check if we need to charge before returning to depot
                if has_battery and current_node != depot:
                    dist_to_depot = instance.dist_matrix[current_node, depot]
                    energy_needed = dist_to_depot * energy_per_dist
                    
                    if current_battery < energy_needed and stations:
                        best_station = None
                        min_total_dist = float('inf')
                        
                        # Pre-compute and sort stations by distance to current node
                        station_dists = [(s, instance.dist_matrix[current_node, s]) for s in stations]
                        station_dists.sort(key=lambda x: x[1])
                        
                        for station, d1 in station_dists:
                            # Early termination: if d1 alone is worse than best total, skip
                            if d1 >= min_total_dist:
                                break
                            
                            d2 = instance.dist_matrix[station, depot]
                            e1 = d1 * energy_per_dist
                            e2 = d2 * energy_per_dist
                            
                            if current_battery < e1:
                                continue
                            if battery_cap < e2:
                                continue
                            
                            total_dist = d1 + d2
                            if total_dist >= min_total_dist:
                                continue
                            
                            if has_tw:
                                arrival_stat = current_time + d1
                                # Calculate battery level when arriving at station
                                batt_at_station = current_battery - (d1 * energy_per_dist)
                                charge_time = compute_charge_time(batt_at_station)
                                dept_stat = arrival_stat + charge_time
                                arrival_depot = dept_stat + d2
                                if arrival_depot > instance.due_dates[depot]:
                                    continue
                            
                            min_total_dist = total_dist
                            best_station = station
                        
                        if best_station is not None:
                            route.append(best_station)
                            energy_to_station = instance.dist_matrix[current_node, best_station] * energy_per_dist
                            current_battery -= energy_to_station
                            current_time += instance.dist_matrix[current_node, best_station]
                            # Charge at station (linear charging time)
                            charge_time = compute_charge_time(current_battery)
                            current_battery = battery_cap
                            current_time += charge_time
                            current_node = best_station
                
                route.append(depot)
                
                # Check if we added any customers to this route
                # If route is just [depot, depot], no customers were served
                if len(route) == 2 and route[0] == depot and route[1] == depot:
                    # No customers could be served - force-add nearest customer (relaxing time windows)
                    if unvisited:
                        # Find nearest customer ignoring time windows
                        nearest_customer = None
                        nearest_dist = float('inf')
                        for customer in unvisited:
                            dist = instance.dist_matrix[current_node, customer]
                            if dist < nearest_dist:
                                nearest_dist = dist
                                nearest_customer = customer
                        
                        if nearest_customer is not None:
                            # Force-add this customer (violating time windows if necessary)
                            route.insert(-1, nearest_customer)  # Insert before final depot
                            unvisited.remove(nearest_customer)
                            # Update state
                            current_node = nearest_customer
                            load += instance.demands[nearest_customer]
                            if has_tw:
                                current_time += nearest_dist
                                current_time = max(current_time, instance.ready_times[nearest_customer])
                                current_time += instance.service_times[nearest_customer]
                            else:
                                current_time += nearest_dist
                            if has_battery:
                                current_battery -= nearest_dist * energy_per_dist
                            # Continue the route to try to add more customers
                            continue
                
                break
            
            # Pick nearest neighbor (by distance)
            best_idx = int(np.argmin(candidate_distances))
            best_customer = feasible_candidates[best_idx]
            info = candidate_infos[best_idx]
            
            # Execute move
            if not info['is_direct']:
                # Insert station first
                station = info['station']
                route.append(station)
                energy_to_station = info['station_dist'] * energy_per_dist
                current_battery -= energy_to_station
                current_time += info['station_dist']
                # Charge at station (linear charging time)
                charge_time = compute_charge_time(current_battery)
                current_battery = battery_cap
                current_time += charge_time
                travel_to_cust = instance.dist_matrix[station, best_customer]
            else:
                travel_to_cust = info['dist']
            
            # Insert customer
            route.append(best_customer)
            load += instance.demands[best_customer]
            unvisited.remove(best_customer)
            current_node = best_customer
            
            # Update time
            if has_tw:
                arrival = current_time + travel_to_cust
                ready = instance.ready_times[best_customer]
                service = instance.service_times[best_customer]
                start_service = max(arrival, ready)
                current_time = start_service + service
            else:
                current_time += travel_to_cust
            
            # Update battery
            if has_battery:
                energy_used = travel_to_cust * energy_per_dist
                current_battery -= energy_used

        routes.append(route)

    return routes

def saving_heuristic(instance, bool_capacity=True) -> List[List[int]]:
    """
    Construct GVRP routes using Clarke-Wright Savings Heuristic.
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
    
    # ==================== PREPROCESSING ====================
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
    has_max_travel_time = hasattr(instance, "max_travel_time")
    max_travel_time = getattr(instance, "max_travel_time", float('inf')) if has_max_travel_time else float('inf')
    
    # Get customers only
    customers = []
    if node_types is not None:
        customers = [i for i in range(n) if node_types[i] == 1]
    else:
        customers = [i for i in range(n) if i != depot]
    
    # ==================== STEP 1: CREATE INITIAL ROUTES ====================
    # Each customer starts in its own route: depot -> customer -> depot
    routes = {}  # route_id -> {route: [nodes], load: float, customers: set}
    route_id_counter = 0
    customer_to_route = {}  # customer -> route_id
    
    for customer in customers:
        routes[route_id_counter] = {
            'route': [depot, customer, depot],
            'load': instance.demands[customer],
            'customers': {customer}
        }
        customer_to_route[customer] = route_id_counter
        route_id_counter += 1
    
    # ==================== STEP 2: CALCULATE SAVINGS ====================
    savings = []
    
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            cust_i = customers[i]
            cust_j = customers[j]
            
            # Classic Clarke-Wright savings formula
            # s(i,j) = d(depot,i) + d(depot,j) - d(i,j)
            d_0_i = instance.dist_matrix[depot, cust_i]
            d_0_j = instance.dist_matrix[depot, cust_j]
            d_i_j = instance.dist_matrix[cust_i, cust_j]
            
            saving = d_0_i + d_0_j - d_i_j
            
            savings.append({
                'customers': (cust_i, cust_j),
                'saving': saving
            })
    
    # Sort savings in descending order
    savings.sort(key=lambda x: x['saving'], reverse=True)
    
    # ==================== STEP 3: MERGE ROUTES ====================
    
    def can_connect_customers(cust_from, cust_to, current_time, current_battery):
        """
        Check if we can travel from cust_from to cust_to (with charging if needed).
        Returns: (is_feasible, route_segment, arrival_time, battery_after)
        """
        # Try direct connection
        dist_direct = instance.dist_matrix[cust_from, cust_to]
        energy_direct = dist_direct * energy_per_dist
        
        if current_battery >= energy_direct:
            batt_after = current_battery - energy_direct
            energy_safety = dist_to_nearest_charger[cust_to] * energy_per_dist
            
            if batt_after >= energy_safety:
                arrival = current_time + dist_direct
                
                if not has_tw or arrival <= instance.due_dates[cust_to]:
                    # Direct connection works
                    return True, [cust_to], arrival, batt_after
        
        # Try via charging station
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
    
    def validate_route(route_nodes):
        """
        Validate entire route with all constraints.
        Returns: (is_valid, final_route_with_stations)
        """
        if len(route_nodes) < 3:  # depot + at least 1 customer + depot
            return False, None
        
        load = 0.0
        current_time = 0.0
        current_battery = battery_cap
        current_node = depot
        final_route = [depot]
        
        # Process customers (skip first depot, last depot)
        for i in range(1, len(route_nodes) - 1):
            customer = route_nodes[i]
            demand = instance.demands[customer]
            
            # Capacity check
            if load + demand > max_capacity:
                return False, None
            
            # Try to reach customer
            feasible, segment, arrival, batt_after = can_connect_customers(
                current_node, customer, current_time, current_battery
            )
            
            if not feasible:
                return False, None
            
            # Add segment (might include charging station)
            for node in segment:
                final_route.append(node)
                if node in stations:
                    # Update for charging
                    dist_to_station = instance.dist_matrix[current_node, node]
                    current_battery -= dist_to_station * energy_per_dist
                    current_time += dist_to_station
                    charge_time = compute_charge_time(current_battery)
                    current_battery = battery_cap
                    current_time += charge_time
                    current_node = node
            
            # Now at customer
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
        
        # Return to depot
        dist_to_depot = instance.dist_matrix[current_node, depot]
        energy_needed = dist_to_depot * energy_per_dist
        
        if current_battery >= energy_needed:
            # Direct return
            arrival_depot = current_time + dist_to_depot
            
            tw_ok = not has_tw or arrival_depot <= instance.due_dates[depot]
            travel_ok = not has_max_travel_time or arrival_depot <= max_travel_time
            
            if tw_ok and travel_ok:
                final_route.append(depot)
                return True, final_route
        
        # Try via charging station
        if has_battery:
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
                
                if current_battery < e1 or battery_cap < e2:
                    continue
                
                arrival_stat = current_time + d1
                batt_at_station = current_battery - e1
                charge_time = compute_charge_time(batt_at_station)
                arrival_depot = arrival_stat + charge_time + d2
                
                tw_ok = not has_tw or arrival_depot <= instance.due_dates[depot]
                travel_ok = not has_max_travel_time or arrival_depot <= max_travel_time
                
                if tw_ok and travel_ok:
                    final_route.append(station)
                    final_route.append(depot)
                    return True, final_route
        
        return False, None
    
    # Process savings
    for saving_entry in savings:
        cust_i, cust_j = saving_entry['customers']
        
        # Skip if already in same route
        if customer_to_route[cust_i] == customer_to_route[cust_j]:
            continue
        
        route_i_id = customer_to_route[cust_i]
        route_j_id = customer_to_route[cust_j]
        
        route_i = routes[route_i_id]
        route_j = routes[route_j_id]
        
        # Get route without depot markers
        route_i_customers = [c for c in route_i['route'] if c != depot]
        route_j_customers = [c for c in route_j['route'] if c != depot]
        
        # Check if customers are at route ends (required for Clarke-Wright)
        i_is_first = (route_i_customers[0] == cust_i)
        i_is_last = (route_i_customers[-1] == cust_i)
        j_is_first = (route_j_customers[0] == cust_j)
        j_is_last = (route_j_customers[-1] == cust_j)
        
        # Can only merge if they're at ends
        if not ((i_is_first or i_is_last) and (j_is_first or j_is_last)):
            continue
        
        # Try to merge
        new_load = route_i['load'] + route_j['load']
        
        if new_load > max_capacity:
            continue
        
        # Determine merge order
        # Four possibilities: i_end-j_start, i_end-j_end, i_start-j_start, i_start-j_end
        merge_orders = []
        
        if i_is_last and j_is_first:
            # route_i -> route_j
            merge_orders.append(route_i_customers + route_j_customers)
        
        if i_is_last and j_is_last:
            # route_i -> reversed route_j
            merge_orders.append(route_i_customers + route_j_customers[::-1])
        
        if i_is_first and j_is_first:
            # reversed route_i -> route_j
            merge_orders.append(route_i_customers[::-1] + route_j_customers)
        
        if i_is_first and j_is_last:
            # route_j -> route_i
            merge_orders.append(route_j_customers + route_i_customers)
        
        # Try each merge order
        merged = False
        for merged_customers in merge_orders:
            # Build route with depots
            test_route = [depot] + merged_customers + [depot]
            
            # Validate with all constraints
            is_valid, final_route = validate_route(test_route)
            
            if is_valid:
                # Merge successful!
                # Delete old routes
                del routes[route_j_id]
                
                # Update route_i with merged route
                routes[route_i_id] = {
                    'route': final_route,
                    'load': new_load,
                    'customers': route_i['customers'] | route_j['customers']
                }
                
                # Update customer mappings
                for c in route_j['customers']:
                    customer_to_route[c] = route_i_id
                
                merged = True
                break
        
        if merged:
            continue
    
    # ==================== CONVERT TO OUTPUT FORMAT ====================
    final_routes = [route_data['route'] for route_data in routes.values()]
    
    return final_routes







if __name__ == "__main__":
    # Find all instance files
    cvrp_files = sorted(glob.glob("Sets/Set_A/*.vrp"))
    vrptw_files = sorted(
        [
            f
            for f in glob.glob("Sets/Vrp-Set-HG/*.txt")
            if not os.path.basename(f) in ["readme.txt"]
            and not os.path.basename(f).startswith("RC")
        ]
    )
    # Use EVRPTW Green VRP instances instead of the old Felipe XML or A/B datasets
    gvrp_files = sorted(
        [
            f
            for f in glob.glob("Sets/evrptw_instances/*.txt")
            if not os.path.basename(f) in ["readme.txt"]
        ]
    )
    
    print(f"Found {len(cvrp_files)} CVRP, {len(vrptw_files)} VRPTW, {len(gvrp_files)} GVRP instances")
    print("=" * 80)
    
    results = []
    
    # Process CVRP instances (with bool_capacity=True and False)
    for bool_cap in [True, False]:
        for filepath in cvrp_files:
            try:
                instance = VRPInstance(filepath)
                instance_name = os.path.basename(filepath)
                
                routes = nearest_neighbor_heuristic(instance, bool_capacity=bool_cap)
                fitness = VRP_PROBLEM_TYPE.compute_cost(instance, routes)
                
                num_customers = sum(len([n for n in route if n != instance.depot]) 
                                   for route in routes)
                
                result = {
                    "instance_name": instance_name,
                    "problem_type": "CVRP",
                    "bool_capacity": bool_cap,
                    "filepath": filepath,
                    "num_customers": num_customers,
                    "num_routes": len(routes),
                    "fitness": fitness,
                    "routes": routes
                }
                results.append(result)
                
                cap_str = "cap=True " if bool_cap else "cap=False"
                print(f"{instance_name:40s} | CVRP | {cap_str:9s} | Routes: {len(routes):3d} | Fitness: {fitness:12.2f}")
                
            except Exception as e:
                print(f"Error processing CVRP {filepath}: {e}")
                continue
    
    # Process VRPTW instances (with bool_capacity=True and False)
    vrptw_instances = []
    vrptw_filepaths = []
    for filepath in vrptw_files:
        try:
            instance = VRPTWInstance(filepath)
            vrptw_instances.append(instance)
            vrptw_filepaths.append(filepath)
        except Exception as e:
            print(f"Error loading VRPTW {filepath}: {e}")
            continue
    
    # Process loaded VRPTW instances with both capacity settings
    for bool_cap in [True, False]:
        for instance, filepath in zip(vrptw_instances, vrptw_filepaths):
            try:
                instance_name = os.path.basename(filepath)
                
                routes = nearest_neighbor_heuristic(instance, bool_capacity=bool_cap)
                fitness = VRP_PROBLEM_TYPE.compute_cost(instance, routes)
                
                num_customers = sum(len([n for n in route if n != instance.depot]) 
                                   for route in routes)
                
                result = {
                    "instance_name": instance_name,
                    "problem_type": "VRPTW",
                    "bool_capacity": bool_cap,
                    "filepath": filepath,
                    "num_customers": num_customers,
                    "num_routes": len(routes),
                    "fitness": fitness,
                    "routes": routes
                }
                results.append(result)
                
                cap_str = "cap=True " if bool_cap else "cap=False"
                print(f"{instance_name:40s} | VRPTW | {cap_str:9s} | Routes: {len(routes):3d} | Fitness: {fitness:12.2f}")
                
            except Exception as e:
                print(f"Error processing VRPTW {filepath}: {e}")
                continue
    
    # Convert VRPTW to GVRPTW and process (with bool_capacity=True and False)
    if vrptw_instances:
        try:
            gvrptw_instances = convert_vrptw_to_gvrptw(vrptw_instances)
            for gvrptw_instance, original_filepath in zip(gvrptw_instances, vrptw_filepaths):
                instance_name = os.path.basename(original_filepath)
                
                for bool_cap in [True, False]:
                    routes = nearest_neighbor_heuristic(gvrptw_instance, bool_capacity=bool_cap)
                    fitness = VRP_PROBLEM_TYPE.compute_cost(gvrptw_instance, routes)
                    
                    num_customers = sum(len([n for n in route if n != gvrptw_instance.depot and 
                                            getattr(gvrptw_instance, 'node_types', [0] * gvrptw_instance.dimension)[n] != 2])
                                       for route in routes)
                    
                    result = {
                        "instance_name": instance_name,
                        "problem_type": "GVRPTW",
                        "bool_capacity": bool_cap,
                        "filepath": original_filepath,
                        "num_customers": num_customers,
                        "num_routes": len(routes),
                        "fitness": fitness,
                        "routes": routes
                    }
                    results.append(result)
                    
                    cap_str = "cap=True " if bool_cap else "cap=False"
                    print(f"{instance_name:40s} | GVRPTW | {cap_str:9s} | Routes: {len(routes):3d} | Fitness: {fitness:12.2f}")
                    
        except Exception as e:
            print(f"Error processing GVRPTW conversions: {e}")
    
    # Process GVRP instances (with bool_capacity=True and False)
    for bool_cap in [True, False]:
        for filepath in gvrp_files:
            try:
                instance = GVRPMultiTechInstance(filepath)
                instance_name = os.path.basename(filepath)
                
                routes = nearest_neighbor_heuristic(instance, bool_capacity=bool_cap)
                fitness = VRP_PROBLEM_TYPE.compute_cost(instance, routes)
                
                num_customers = sum(len([n for n in route if n != instance.depot and 
                                        getattr(instance, 'node_types', [0] * instance.dimension)[n] != 2])
                                   for route in routes)
                
                result = {
                    "instance_name": instance_name,
                    "problem_type": "GVRP",
                    "bool_capacity": bool_cap,
                    "filepath": filepath,
                    "num_customers": num_customers,
                    "num_routes": len(routes),
                    "fitness": fitness,
                    "routes": routes
                }
                results.append(result)
                
                cap_str = "cap=True " if bool_cap else "cap=False"
                print(f"{instance_name:40s} | GVRP | {cap_str:9s} | Routes: {len(routes):3d} | Fitness: {fitness:12.2f}")
                
            except Exception as e:
                print(f"Error processing GVRP {filepath}: {e}")
                continue
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results to CSV
    csv_filename = os.path.join(results_dir, "nearest_neighbor_results.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ["instance_name", "problem_type", "bool_capacity", "filepath", "num_customers", "num_routes", "fitness"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({k: v for k, v in result.items() if k != "routes"})
    
    # Save detailed results with routes to JSON
    json_filename = os.path.join(results_dir, "nearest_neighbor_results.json")
    with open(json_filename, 'w') as jsonfile:
        json.dump(results, jsonfile, indent=2)
    
    print("=" * 80)
    print(f"Results saved to {csv_filename} and {json_filename}")
    print(f"Processed {len(results)} instances successfully (8 variants: 4 problem types × 2 capacity settings)")
