# nn_heuristic.py
import numpy as np
import os
import glob
import csv
import json
from parser import VRPInstance, VRPTWInstance, GVRPMultiTechInstance
from problem_types import VRP_PROBLEM_TYPE
from data_generation import convert_vrptw_to_gvrptw


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
    charge_time_fixed = 20.0  # Fixed charging time at stations
    
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
                    
                    for station in stations:
                        d1 = instance.dist_matrix[current_node, station]
                        d2 = instance.dist_matrix[station, customer]
                        e1 = d1 * energy_per_dist
                        e2 = d2 * energy_per_dist
                        total_dist = d1 + d2
                        
                        if current_battery < e1:
                            continue
                        if battery_cap < e2:
                            continue
                        if total_dist >= min_total_dist:
                            continue
                        
                        # Safety check at destination
                        if (battery_cap - e2) < (dist_to_nearest_charger[customer] * energy_per_dist):
                            continue
                        
                        # Time window check
                        arrival_at_station = current_time + d1
                        dept_from_station = arrival_at_station + charge_time_fixed
                        arrival_at_cust = dept_from_station + d2
                        
                        if has_tw:
                            if arrival_at_cust > instance.due_dates[customer]:
                                continue
                            ready = instance.ready_times[customer]
                            service = instance.service_times[customer]
                            dept_cust = max(arrival_at_cust, ready) + service
                            dist_home = instance.dist_matrix[customer, depot]
                            if dept_cust + dist_home > instance.due_dates[depot]:
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
                        
                        for station in stations:
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
                            energy_to_station = instance.dist_matrix[current_node, best_station] * energy_per_dist
                            current_battery -= energy_to_station
                            current_time += instance.dist_matrix[current_node, best_station]
                            current_battery = battery_cap
                            current_time += charge_time_fixed
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
                current_battery = battery_cap
                current_time += charge_time_fixed
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

def savings_heuristic(instance, problem: str = "auto"):
    """
    Clarke-Wright Savings heuristic

    Args:
        instance: VRP or VRPTW
        problem: one of {"auto", "vrp", "vrptw"}
    """
    if problem not in {"auto", "vrp", "vrptw"}:
        raise ValueError("problem must be one of {'auto','vrp','vrptw'}")
    is_tw = (problem == "vrptw") or (problem == "auto" and hasattr(instance, "ready_times") and hasattr(instance, "service_times"))

    def is_time_feasible(route):
        if not is_tw:
            return True
        current_time = 0.0
        for idx in range(1, len(route) - 1):
            c = route[idx]
            prev = route[idx - 1]
            travel = instance.dist_matrix[prev, c]
            arrival = current_time + travel
            if arrival > instance.due_dates[c]:
                return False
            start_service = max(arrival, instance.ready_times[c])
            current_time = start_service + instance.service_times[c]
        return True

    n = instance.dimension
    depot = instance.depot

    # Step 1: trivial solution
    routes = []
    route_loads = []
    for i in range(1, n):
        r = [depot, i, depot]
        if is_time_feasible(r):
            routes.append(r)
            route_loads.append(instance.demands[i])
        else:
            # If single-customer route is infeasible by TW, skip it (cannot serve)
            # This keeps behavior safe; caller can detect missing customers if needed
            continue

    # Step 2: compute savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = (instance.dist_matrix[depot, i] +
                 instance.dist_matrix[depot, j] -
                 instance.dist_matrix[i, j])
            savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    # Step 3: merge routes
    while savings:
        _, i, j = savings.pop(0)

        # find routes containing i and j
        ri = next((r for r in routes if i in r), None)
        rj = next((r for r in routes if j in r), None)
        if ri is None or rj is None or ri == rj:
            continue

        # check if i is at end of ri and j at start of rj
        if ri[-2] == i and rj[1] == j:
            new_load = (sum(instance.demands[c] for c in ri if c != depot) +
                        sum(instance.demands[c] for c in rj if c != depot))
            if new_load <= instance.capacity:
                new_route = ri[:-1] + rj[1:]
                if is_time_feasible(new_route):
                    routes.remove(ri)
                    routes.remove(rj)
                    routes.append(new_route)

        # also check the reverse (j at end, i at start)
        elif rj[-2] == j and ri[1] == i:
            new_load = (sum(instance.demands[c] for c in ri if c != depot) +
                        sum(instance.demands[c] for c in rj if c != depot))
            if new_load <= instance.capacity:
                new_route = rj[:-1] + ri[1:]
                if is_time_feasible(new_route):
                    routes.remove(ri)
                    routes.remove(rj)
                    routes.append(new_route)

    return routes







if __name__ == "__main__":
    # Find all instance files
    cvrp_files = sorted(glob.glob("Sets/Set_A/*.vrp"))
    vrptw_files = sorted([f for f in glob.glob("Sets/Vrp-Set-HG/*.txt") 
                          if not os.path.basename(f) in ["readme.txt"] 
                          and not os.path.basename(f).startswith("RC")])
    gvrp_files = sorted(glob.glob("Sets/felipe-et-al-2014/*.xml"))
    
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
