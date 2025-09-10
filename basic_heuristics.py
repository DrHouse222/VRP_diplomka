# nn_heuristic.py
import numpy as np
from parser import VRPInstance

# earliest first, savings

def nearest_neighbor_heuristic(instance):
    """
    Build routes using a simple nearest neighbor heuristic.
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


instance = VRPInstance("Set_A/A-n32-k5.vrp")
routes = nearest_neighbor_heuristic(instance)
total_distance, capacity_violation = instance.cost(routes)
print(f"Routes: {routes}")
print(f"Total distance: {total_distance}, Capacity violation: {capacity_violation}")