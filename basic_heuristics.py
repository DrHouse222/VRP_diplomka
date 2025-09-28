# nn_heuristic.py
import numpy as np
from parser import VRPInstance

# earliest first - only for time windows
# savings


def nearest_neighbor_heuristic(instance):
    """
    Build routes using a nearest neighbor heuristic.
    returns: list of routes (list of lists of node indices)
    """
    n = instance.dimension
    unvisited = set(range(1, n))  # customers only (exclude depot=0)
    routes = []

    while unvisited:
        route = [instance.depot]
        load = 0
        current = instance.depot

        while True:
            # feasible candidates: unvisited customers that fit into remaining capacity
            candidates = [c for c in unvisited if load + instance.demands[c] <= instance.capacity]

            if not candidates:
                # return to depot, end route
                route.append(instance.depot)
                break

            # pick nearest neighbor
            dists = [instance.dist_matrix[current, c] for c in candidates]
            next_customer = candidates[int(np.argmin(dists))]

            # update state
            route.append(next_customer)
            load += instance.demands[next_customer]
            unvisited.remove(next_customer)
            current = next_customer

        routes.append(route)

    return routes

# savings_heuristic.py
import numpy as np

def savings_heuristic(instance):
    """
    Clarke-Wright Savings heuristic (parallel version).
    """
    n = instance.dimension
    depot = instance.depot

    # Step 1: trivial solution
    routes = [[depot, i, depot] for i in range(1, n)]
    route_loads = [instance.demands[i] for i in range(1, n)]

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
                # merge
                new_route = ri[:-1] + rj[1:]
                routes.remove(ri)
                routes.remove(rj)
                routes.append(new_route)

        # also check the reverse (j at end, i at start)
        elif rj[-2] == j and ri[1] == i:
            new_load = (sum(instance.demands[c] for c in ri if c != depot) +
                        sum(instance.demands[c] for c in rj if c != depot))
            if new_load <= instance.capacity:
                new_route = rj[:-1] + ri[1:]
                routes.remove(ri)
                routes.remove(rj)
                routes.append(new_route)

    return routes



instance = VRPInstance("Set_A/A-n32-k5.vrp")
routes = savings_heuristic(instance)
total_distance, capacity_violation = instance.cost(routes)
print(f"Routes: {routes}")
print(f"Total distance: {total_distance}, Capacity violation: {capacity_violation}")

routes = nearest_neighbor_heuristic(instance)
total_distance, capacity_violation = instance.cost(routes)
print(f"Routes: {routes}")
print(f"Total distance: {total_distance}, Capacity violation: {capacity_violation}")