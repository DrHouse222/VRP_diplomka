def solve_with_scoring(self, instance, feature_extractor, scoring_func) -> List[List[int]]:
        """Solve CVRPTW using GP-evolved scoring function."""
        n = instance.dimension
        unvisited = set(range(1, n))  # All customers except depot
        routes = []
        
        while unvisited:
            route = [instance.depot]
            load = 0
            current_position = instance.depot
            current_time = 0.0
            
            while True:
                # Consider all unvisited candidates (allow capacity violations to be penalized)
                candidates = list(unvisited)
                
                if not candidates:
                    # Return to depot
                    route.append(instance.depot)
                    break
                
                # Check time window and capacity feasibility
                feasible_candidates = []
                scores = []
                
                for candidate in candidates:
                    travel = instance.dist_matrix[current_position, candidate]
                    arrival = current_time + travel
                    
                    # Check time window feasibility
                    if arrival > instance.due_dates[candidate]:
                        continue  # Skip time-infeasible candidates
                    
                    # Check capacity feasibility
                    if load + instance.demands[candidate] > instance.capacity:
                        continue  # Skip capacity-infeasible candidates
                    
                    feasible_candidates.append(candidate)
                    
                    # Extract features with current time
                    features = feature_extractor.extract_features(
                        candidate, route, load, current_position, current_time
                    )
                    
                    feature_values = self.extract_feature_values(features)
                    
                    try:
                        score = scoring_func(*feature_values)
                        scores.append(score)
                    except:
                        # If GP model fails, use distance as fallback
                        scores.append(travel)
                
                if not feasible_candidates:
                    # No time or capacity-feasible candidates, return to depot and start new route
                    route.append(instance.depot)
                    break
                
                # Choose candidate with lowest score
                best_candidate = feasible_candidates[scores.index(min(scores))]
                
                # Update route and time
                route.append(best_candidate)
                load += instance.demands[best_candidate]
                unvisited.remove(best_candidate)
                
                # Update time
                travel = instance.dist_matrix[current_position, best_candidate]
                arrival = current_time + travel
                start_service = max(arrival, instance.ready_times[best_candidate])
                current_time = start_service + instance.service_times[best_candidate]
                current_position = best_candidate
            
            routes.append(route)
        
        return routes


def compute_cost(instance, routes, energy_per_distance: float = 1.0) -> float:
    """
    Unified fitness function for all supported VRP variants.

    Parameters
    ----------
    instance
        Parsed instance object (VRPInstance, VRPTWInstance, GVRPMultiTechInstance, ...).
    routes : List[List[int]]
        List of routes, e.g. [[0, 5, 7, 0], ...].
    energy_per_distance : float, optional
        Energy consumption per unit distance

    Returns
    -------
    fitness : float
        Scalar fitness value (lower is better).
    """
    total_distance = 0.0
    total_time = 0.0
    capacity_violation = 0.0
    tw_violation = 0.0
    battery_violation = 0.0
    time_violation = 0.0

    depot = getattr(instance, "depot", 0)
    demands = getattr(instance, "demands", None)
    dist_matrix = getattr(instance, "dist_matrix", None)
    capacity = getattr(instance, "capacity", 0.0)

    if demands is None or dist_matrix is None:
        raise ValueError("Instance must have 'demands' and 'dist_matrix' attributes.")

    # Optional features
    has_tw = all(
        hasattr(instance, attr) for attr in ("ready_times", "due_dates", "service_times")
    )
    has_battery = getattr(instance, "battery_capacity", 0.0) > 0.0
    #max_travel_time = float(getattr(instance, "max_travel_time", 0.0))
    node_types = getattr(instance, "node_types", None)

    for route in routes:
        if not route:
            continue

        # Capacity violation (common to all)
        if capacity > 0:
            load = sum(demands[node] for node in route if node != depot)
            if load > capacity:
                capacity_violation += float(load - capacity)

        print("Num routes:", len(routes))
        print("Route:", route)
        print("Load:", load)
        print("Capacity:", capacity)
        print("Capacity violation:", capacity_violation)

        # Per-route time and battery tracking
        current_time = 0.0
        battery = getattr(instance, "battery_capacity", 0.0) if has_battery else None

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            travel = float(dist_matrix[u, v])
            total_distance += travel

            # Time accumulation (used for VRPTW and GVRP)
            current_time += travel

            # Time-window handling (VRPTW-style)
            if has_tw and v != depot:
                ready = float(instance.ready_times[v])
                due = float(instance.due_dates[v])
                service = float(instance.service_times[v])

                # Lateness beyond due date
                if current_time > due:
                    tw_violation += current_time - due

                # Start service respecting ready time
                start_service = max(current_time, ready)
                current_time = start_service + service
            else:
                # If we have generic service times (e.g. GVRP) but no TWs
                service_times = getattr(instance, "service_times", None)
                if service_times is not None and v != depot and not has_tw:
                    current_time += float(service_times[v])

            # Battery handling (GVRP-style)
            if has_battery:
                energy_consumed = travel * energy_per_distance
                battery -= energy_consumed
                if battery < 0:
                    battery_violation += -battery

                # Simple model: full recharge at charging stations (type 2)
                if node_types is not None and node_types[v] == 2:
                    battery = instance.battery_capacity
                    # Optional fixed charging time; can be tuned/disabled if needed
                    current_time += 30.0

        # Route-duration limit (used for GVRP instances that specify it)
        #if max_travel_time > 0.0 and current_time > max_travel_time:
        #    time_violation += current_time - max_travel_time
        
        total_time += current_time

    total_violation = capacity_violation + tw_violation + battery_violation + time_violation

    print("Total distance:", total_distance)
    print("Total time:", total_time)
    print("Total capacity violation:", capacity_violation)
    
    fitness = total_distance + 1000.0 * total_violation + 0.1 * total_time

    print("Computed fitness:", fitness)
    
    return fitness