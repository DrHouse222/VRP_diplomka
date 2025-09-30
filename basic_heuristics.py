# nn_heuristic.py
import numpy as np
from parser import VRPInstance, VRPTWInstance

# earliest first - only for time windows
# savings


def nearest_neighbor_heuristic(instance, problem: str = "auto"):
    """
    Build routes using a nearest neighbor heuristic.

    Args:
        instance: VRP or VRPTW instance
        problem: one of {"auto", "vrp", "vrptw"}

    Returns:
        list of routes (list of lists of node indices)
    """
    # Determine problem type
    if problem not in {"auto", "vrp", "vrptw"}:
        raise ValueError("problem must be one of {'auto','vrp','vrptw'}")
    is_tw = (problem == "vrptw") or (problem == "auto" and hasattr(instance, "ready_times") and hasattr(instance, "service_times"))

    n = instance.dimension
    unvisited = set(range(1, n))  # customers only (exclude depot=0)
    routes = []

    while unvisited:
        route = [instance.depot]
        load = 0
        current = instance.depot
        current_time = 0.0

        while True:
            # capacity-feasible candidates
            capacity_ok = [c for c in unvisited if load + instance.demands[c] <= instance.capacity]

            if is_tw:
                feasible = []
                for c in capacity_ok:
                    travel = instance.dist_matrix[current, c]
                    arrival = current_time + travel
                    # must arrive before due date
                    if arrival > instance.due_dates[c]:
                        continue
                    start_service = max(arrival, instance.ready_times[c])
                    finish_service = start_service + instance.service_times[c]
                    feasible.append((c, travel, start_service, finish_service))

                if not feasible:
                    # return to depot
                    route.append(instance.depot)
                    # advance time to depot arrival (optional bookkeeping)
                    current_time = current_time + instance.dist_matrix[current, instance.depot]
                    break

                # choose nearest by travel distance (can be changed to earliest start)
                feasible.sort(key=lambda t: t[1])
                next_customer, travel, start_service, finish_service = feasible[0]

                # update state
                route.append(next_customer)
                load += instance.demands[next_customer]
                unvisited.remove(next_customer)
                current = next_customer
                current_time = finish_service
            else:
                if not capacity_ok:
                    route.append(instance.depot)
                    break

                # pick nearest neighbor (distance only)
                dists = [instance.dist_matrix[current, c] for c in capacity_ok]
                next_customer = capacity_ok[int(np.argmin(dists))]

                # update state
                route.append(next_customer)
                load += instance.demands[next_customer]
                unvisited.remove(next_customer)
                current = next_customer

        routes.append(route)

    return routes

# savings_heuristic.py
import numpy as np

def savings_heuristic(instance, problem: str = "auto"):
    """
    Clarke-Wright Savings heuristic (parallel version).

    Args:
        instance: VRP or VRPTW instance
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


'''
instance = VRPInstance("Set_A/A-n32-k5.vrp")
routes = nearest_neighbor_heuristic(instance)
total_distance, capacity_violation = instance.cost(routes)
print(f"Routes: {routes}")
print(f"Total distance: {total_distance}, Capacity violation: {capacity_violation}")
'''


instance = VRPTWInstance("Vrp-Set-HG/C1_2_1.txt")
routes = savings_heuristic(instance)
total_distance, capacity_violation = instance.cost(routes)
print(f"Routes: {routes}")
print(f"Total distance: {total_distance}, Capacity violation: {capacity_violation}")