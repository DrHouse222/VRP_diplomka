"""
Parsers for multiple VRP dataset formats used in this project.

Includes loaders for:
- CVRP/VRP instances via vrplib
- Solomon VRPTW instances
- Cordeau MDVRP / MDVRPTW instances
- EVRPTW-based GVRP instances
"""

import os
import vrplib
import numpy as np
import re
import xml.etree.ElementTree as ET

class VRPInstance:
    """Parses CVRP/VRP instances using vrplib.

    Attributes
    -----------
    name: str
    num_trucks: int
    capacity: int
    dimension: int
        Number of nodes (including depot)
    depot: int
        Depot node index (always 0)
    edge_weight_type: str
    type: str
    coords: np.ndarray shape (n, 2)
    demands: np.ndarray shape (n,)
    dist_matrix: np.ndarray shape (n, n)
        Distance matrix from vrplib (instance_dict["edge_weight"])
    """

    def __init__(self, instance):
        instance_dict = vrplib.read_instance(instance)

        self.name = instance_dict["name"]
        match = re.search(r"No of trucks:\s*(\d+)", instance_dict["comment"]) # Extract number of trucks from comment
        self.num_trucks = int(match.group(1)) if match else 1

        self.capacity = instance_dict["capacity"]
        self.dimension = instance_dict["dimension"]
        self.depot = 0

        self.edge_weight_type = instance_dict["edge_weight_type"]
        self.type = instance_dict["type"]
        
        self.coords = instance_dict["node_coord"]
        self.demands = instance_dict["demand"]

        self.dist_matrix = instance_dict["edge_weight"]

    def __repr__(self):
        return f"VRPInstance({self.name}, n={self.dimension}, cap={self.capacity})"

class VRPTWInstance:
    """Parses Solomon-style VRPTW instances using vrplib.
    
    Attributes
    -----------
    name: str
    num_vehicles: int
    capacity: int
    dimension: int
        Number of nodes (including depot)
    coords: np.ndarray shape (n, 2)
    demands: np.ndarray shape (n,)
    ready_times: np.ndarray shape (n,)
    due_dates: np.ndarray shape (n,)
    service_times: np.ndarray shape (n,)
    depot: int (always 0)
    dist_matrix: np.ndarray shape (n, n) – from vrplib
    """

    def __init__(self, file_path: str):
        instance_dict = vrplib.read_instance(file_path, instance_format="solomon")
        
        self.name = instance_dict["name"]
        self.num_vehicles = instance_dict["vehicles"]

        self.capacity = instance_dict["capacity"]
        self.dimension = instance_dict["node_coord"].shape[0]
        self.depot = 0
        
        self.coords = instance_dict["node_coord"]
        self.demands = instance_dict["demand"]
        self.ready_times = instance_dict["time_window"][:, 0]  # ready times
        self.due_dates = instance_dict["time_window"][:, 1]    # due dates
        self.service_times = instance_dict["service_time"]
        
        self.dist_matrix = instance_dict["edge_weight"]

    def __repr__(self):
        return (
            f"VRPTWInstance({self.name}, n={self.dimension}, cap={self.capacity}, "
            f"vehicles={self.num_vehicles})"
        )


class CordeauMDVRPInstance:
    """
    Parser for Cordeau MDVRP / MDVRPTW instances (`Sets/C-mdvrp`, `Sets/C-mdvrptw`).
    
    Supports:
    - type 2: MDVRP  (no time windows)
    - type 6: MDVRPTW (with time windows)
    
    Representation
    --------------
    - Nodes 0 .. n+t-1 correspond to instance lines 1 .. n+t
    - The last t nodes are depots; customers are 0 .. n-1
    - `depots` holds the indices of all depots
    - `depot` is set to the first depot index for compatibility with single-depot code
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.name = os.path.basename(file_path)

        with open(file_path, "r") as f:
            header = f.readline().split()
            if len(header) < 4:
                raise ValueError(f"Invalid Cordeau header in {file_path}: {header}")

            type_code = int(header[0])
            self.type_code = type_code

            if type_code not in (2, 6):
                raise ValueError(
                    f"CordeauMDVRPInstance only supports type 2 (MDVRP) and 6 (MDVRPTW), "
                    f"got {type_code} in {file_path}"
                )

            m = int(header[1])  # number of vehicles
            n = int(header[2])  # number of customers
            t = int(header[3])  # number of depots

            self.num_vehicles = m
            self.num_customers = n
            self.num_depots = t

            durations = []
            capacities = []
            for _ in range(t):
                line = f.readline()
                if not line:
                    raise ValueError(f"Unexpected EOF while reading depot lines in {file_path}")
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Invalid depot line in {file_path}: {line}")
                D = float(parts[0])
                Q = float(parts[1])
                durations.append(D)
                capacities.append(Q)

            self.capacity = max(capacities) if capacities else 0.0
            positive_D = [d for d in durations if d > 0]
            self.max_travel_time = max(positive_D) if positive_D else float("inf")

            node_rows = []
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 7:
                    continue

                idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                service = float(parts[3])
                demand = float(parts[4])
                freq = int(parts[5])
                a = int(parts[6])

                combo_start = 7
                combo_end = 7 + a
                rest = parts[combo_end:]

                if type_code == 6 and len(rest) >= 2:
                    e = float(rest[-2])
                    l = float(rest[-1])
                else:
                    e = None
                    l = None

                node_rows.append(
                    {
                        "idx": idx,
                        "x": x,
                        "y": y,
                        "service": service,
                        "demand": demand,
                        "freq": freq,
                        "a": a,
                        "ready": e,
                        "due": l,
                    }
                )

        expected_nodes = n + t
        if len(node_rows) != expected_nodes:
            raise ValueError(
                f"Expected {expected_nodes} nodes (customers + depots) in {file_path}, "
                f"found {len(node_rows)}"
            )

        self.dimension = expected_nodes

        self.coords = np.zeros((self.dimension, 2), dtype=float)
        self.demands = np.zeros(self.dimension, dtype=float)
        self.service_times = np.zeros(self.dimension, dtype=float)

        has_tw = self.type_code == 6
        if has_tw:
            self.ready_times = np.zeros(self.dimension, dtype=float)
            self.due_dates = np.zeros(self.dimension, dtype=float)

        self.node_types = np.ones(self.dimension, dtype=int)

        for row in node_rows:
            idx_0 = row["idx"] - 1  # convert to 0-based
            if not (0 <= idx_0 < self.dimension):
                continue

            self.coords[idx_0, 0] = row["x"]
            self.coords[idx_0, 1] = row["y"]
            self.demands[idx_0] = row["demand"]
            self.service_times[idx_0] = row["service"]

            if has_tw and row["ready"] is not None and row["due"] is not None:
                self.ready_times[idx_0] = row["ready"]
                self.due_dates[idx_0] = row["due"]

        self.depots = list(range(n, n + t))
        for d_idx in self.depots:
            self.node_types[d_idx] = 0

        self.depot = self.depots[0] if self.depots else 0

        if self.dimension > 0:
            diff = self.coords[:, None, :] - self.coords[None, :, :]
            self.dist_matrix = np.hypot(diff[..., 0], diff[..., 1])
        else:
            self.dist_matrix = np.zeros((0, 0), dtype=float)

    def __repr__(self):
        return (
            f"CordeauMDVRPInstance({self.name}, n={self.dimension}, "
            f"customers={self.num_customers}, depots={self.num_depots}, "
            f"cap={self.capacity}, vehicles={self.num_vehicles})"
        )

class GVRPMultiTechInstance:
    """
    Parser for Green VRP instances (electric VRPs).
    
    - EVRPTW instances in text format from `Sets/evrptw_instances/`
    
    Attributes
    ----------
    name : str
        Instance name.
    dimension : int
        Number of nodes in the network.
    coords : np.ndarray, shape (n, 2)
        Node coordinates ordered by internal node id.
    node_types : np.ndarray, shape (n,)
        Node type (0 = depot, 1 = customer, 2 = charging station).
    demands : np.ndarray, shape (n,)
        Demand quantity per node (0 for non-customer nodes).
    service_times : np.ndarray, shape (n,)
        Service time per node.
    depot : int
        Depot node id.
    capacity : float
        Vehicle load capacity.
    num_vehicles : int
        Number of vehicles (if known, otherwise 1).
    max_travel_time : float
        Maximum route travel time (if provided by the instance).
    battery_capacity : float
        Battery/energy capacity of the vehicle.
    energy_consumption : float
        Energy consumption per unit distance.
    dist_matrix : np.ndarray, shape (n, n)
        Symmetric distance matrix computed from coordinates.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.name = os.path.basename(file_path)
        nodes = []
        Q = None
        C = None
        r = None
        g = None
        v = None

        with open(file_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.lower().startswith("stringid"):
                    continue

                parts = stripped.split()

                if len(parts) >= 8 and parts[1] in {"d", "f", "c"}:
                    string_id = parts[0]
                    type_char = parts[1].lower()
                    try:
                        x = float(parts[2])
                        y = float(parts[3])
                        demand = float(parts[4])
                        ready = float(parts[5])
                        due = float(parts[6])
                        service = float(parts[7])
                    except ValueError:
                        continue

                    nodes.append(
                        (string_id, type_char, x, y, demand, ready, due, service)
                    )
                    continue

                if stripped.startswith("Q Vehicle fuel tank capacity"):
                    try:
                        Q = float(stripped.split("/")[1])
                    except Exception:
                        pass
                elif stripped.startswith("C Vehicle load capacity"):
                    try:
                        C = float(stripped.split("/")[1])
                    except Exception:
                        pass
                elif stripped.startswith("r fuel consumption rate"):
                    try:
                        r = float(stripped.split("/")[1])
                    except Exception:
                        pass
                elif stripped.startswith("g inverse refueling rate"):
                    try:
                        g = float(stripped.split("/")[1])
                    except Exception:
                        pass
                elif stripped.startswith("v average Velocity"):
                    try:
                        v = float(stripped.split("/")[1])
                    except Exception:
                        pass

        if not nodes:
            raise ValueError(f"No nodes parsed from EVRPTW instance: {file_path}")

        depot_nodes = [n for n in nodes if n[1] == "d"]
        other_nodes = [n for n in nodes if n[1] != "d"]

        if len(depot_nodes) != 1:
            raise ValueError(
                f"Expected exactly one depot in EVRPTW instance {file_path}, found {len(depot_nodes)}"
            )

        ordered_nodes = depot_nodes + other_nodes

        self.dimension = len(ordered_nodes)
        self.coords = np.zeros((self.dimension, 2), dtype=float)
        self.node_types = np.zeros(self.dimension, dtype=int)
        self.demands = np.zeros(self.dimension, dtype=float)
        self.service_times = np.zeros(self.dimension, dtype=float)
        self.ready_times = np.zeros(self.dimension, dtype=float)
        self.due_dates = np.zeros(self.dimension, dtype=float)

        for idx, (_, type_char, x, y, demand, ready, due, service) in enumerate(
            ordered_nodes
        ):
            self.coords[idx, 0] = x
            self.coords[idx, 1] = y

            if type_char == "d":
                self.node_types[idx] = 0
            elif type_char == "c":
                self.node_types[idx] = 1
            elif type_char == "f":
                self.node_types[idx] = 2
            else:
                self.node_types[idx] = 0

            self.demands[idx] = demand
            self.ready_times[idx] = ready
            self.due_dates[idx] = due
            self.service_times[idx] = service

        self.depot = 0

        self.battery_capacity = float(Q) if Q is not None else 0.0
        self.capacity = float(C) if C is not None else 0.0

        self.energy_consumption = float(r) if r is not None else 1.0
        self.g_inverse_refueling_rate = float(g) if g is not None else 0.0
        self.velocity = float(v) if v is not None else 1.0

        self.num_vehicles = 1

        self.max_travel_time = (
            float(np.max(self.due_dates)) if self.due_dates.size > 0 else 0.0
        )

        if self.dimension > 0:
            diff = self.coords[:, None, :] - self.coords[None, :, :]
            self.dist_matrix = np.hypot(diff[..., 0], diff[..., 1])
        else:
            self.dist_matrix = np.zeros((0, 0), dtype=float)

    def __repr__(self):
        return (
            f"GVRPMultiTechInstance({self.name}, n={self.dimension}, cap={self.capacity}, "
            f"battery={self.battery_capacity})"
        )
