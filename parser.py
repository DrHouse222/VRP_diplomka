import vrplib
import numpy as np
import re

# Read VRPLIB formatted instances (default)
instance = vrplib.read_instance("Set_A/A-n32-k5.vrp")

#dict_keys(['name', 'comment', 'type', 'dimension', 'edge_weight_type', 'capacity', 'node_coord', 'demand', 'depot', 'edge_weight'])

class VRPInstance:
    def __init__(self, instance):
        instance_dict = vrplib.read_instance(instance)

        self.name = instance_dict["name"]
        self.comment = instance_dict["comment"]
        # Extract number of trucks from comment
        match = re.search(r"No of trucks:\s*(\d+)", self.comment)
        self.num_trucks = int(match.group(1)) if match else 1

        self.edge_weight_type = instance_dict["edge_weight_type"]
        self.type = instance_dict["type"]
        self.dimension = instance_dict["dimension"]
        self.capacity = instance_dict["capacity"]

        # numpy arrays
        self.coords = instance_dict["node_coord"]
        self.demands = instance_dict["demand"]

        # depot TODO only one
        self.depot = int(instance_dict["depot"][0])

        # distance matrix
        self.dist_matrix = instance_dict["edge_weight"]

    def cost(self, routes):
        """
        Compute total distance and penalties for a set of routes.
        routes = list of lists [[0, 5, 7, 0], [0, 2, 3, 0]]
        """
        total_distance = 0.0
        capacity_violation = 0

        for route in routes:
            load = sum(self.demands[node] for node in route if node != self.depot)
            if load > self.capacity:
                capacity_violation += (load - self.capacity)

            for i in range(len(route) - 1):
                total_distance += self.dist_matrix[route[i], route[i+1]]

        return total_distance, capacity_violation

    def __repr__(self):
        return f"VRPInstance({self.name}, n={self.dimension}, cap={self.capacity})"