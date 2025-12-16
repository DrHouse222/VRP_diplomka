import vrplib
import numpy as np
import re
from typing import List, Dict
import xml.etree.ElementTree as ET


def compute_cost(instance, routes, energy_per_distance: float = 1.0) -> float:
    """
    Unified **fitness** function for all supported VRP variants.

    This function always returns a single scalar fitness value:

        fitness = distance + 1000 * total_violation

    where `total_violation` combines capacity, time-window, battery and
    route-duration violations when applicable for the given instance type.

    Parameters
    ----------
    instance
        Parsed instance object (VRPInstance, VRPTWInstance, GVRPMultiTechInstance, ...).
    routes : List[List[int]]
        List of routes, e.g. [[0, 5, 7, 0], ...].
    energy_per_distance : float, optional
        Energy consumption per unit distance.

    Returns
    -------
    fitness : float
        Scalar fitness value to be minimised.
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
                if service_times is not None and v != depot:
                    current_time += float(service_times[v])

            # Battery handling (GVRP-style)
            if has_battery:
                energy_consumed = travel * energy_per_distance
                battery -= energy_consumed
                if battery < 0:
                    battery_violation += -battery

                # Simple model: full recharge at charging stations (type 2)
                if node_types is not None and node_types[v] == 2:
                    current_time += (instance.battery_capacity - battery) / 0.1  # Assume 0.1 unit time per energy unit
                    battery = instance.battery_capacity   
                    
        total_time += current_time

        # Route-duration limit (used for GVRP instances that specify it)
        #if max_travel_time > 0.0 and current_time > max_travel_time:
        #    time_violation += current_time - max_travel_time

    total_violation = capacity_violation + tw_violation + battery_violation + time_violation

    fitness = total_distance + 1000.0 * total_violation + 0.1 * total_time

    return fitness

class VRPInstance:
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
        
        # numpy arrays
        self.coords = instance_dict["node_coord"]
        self.demands = instance_dict["demand"]

        # distance matrix
        self.dist_matrix = instance_dict["edge_weight"]

    def cost(self, routes):
        """
        Compute total distance and capacity violation for a set of routes.

        Uses the unified :func:`compute_cost` function without time-window
        or battery terms.
        """
        return compute_cost(self, routes)

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
        # Use vrplib to parse Solomon format
        instance_dict = vrplib.read_instance(file_path, instance_format="solomon")
        
        self.name = instance_dict["name"]
        self.num_vehicles = instance_dict["vehicles"]

        self.capacity = instance_dict["capacity"]
        self.dimension = instance_dict["node_coord"].shape[0]
        self.depot = 0
        
        # Extract arrays
        self.coords = instance_dict["node_coord"]
        self.demands = instance_dict["demand"]
        self.ready_times = instance_dict["time_window"][:, 0]  # ready times
        self.due_dates = instance_dict["time_window"][:, 1]    # due dates
        self.service_times = instance_dict["service_time"]
        
        # Use precomputed distance matrix from vrplib
        self.dist_matrix = instance_dict["edge_weight"]

    def cost(self, routes):
        """
        Compute total travel distance and violations for a set of routes.

        Uses the unified :func:`compute_cost` function with time-window
        terms (since this instance has ready/due/service times).
        """
        return compute_cost(self, routes)

    def __repr__(self):
        return (
            f"VRPTWInstance({self.name}, n={self.dimension}, cap={self.capacity}, "
            f"vehicles={self.num_vehicles})"
        )

class VRPFeatureExtractor:
    """Extracts VRP/VRPTW features for request evaluation.
    """
    
    def __init__(self, instance: VRPInstance):
        self.instance = instance
        self.depot = instance.depot
        self.dist_matrix = instance.dist_matrix
        self.demands = instance.demands
        self.capacity = instance.capacity
        # Optional TW fields (duck-typed)
        self.has_tw = all(
            hasattr(instance, attr) for attr in ("ready_times", "due_dates", "service_times")
        )
        if self.has_tw:
            self.ready_times = instance.ready_times
            self.due_dates = instance.due_dates
            self.service_times = instance.service_times
            # For normalization of time-based features
            self.max_due = float(np.max(self.due_dates)) if len(self.due_dates) > 0 else 1.0
        else:
            self.max_due = 1.0
    
    def extract_features(self, request: int, current_route: List[int], 
                        current_load: float, current_position: int, current_time: float = 0.0) -> Dict[str, float]:
        """
        Extract features for a given candidate request.
        
        Args:
            request: Customer node to evaluate
            current_route: Current route being built
            current_load: Current load of the route
            current_position: Current position in the route (last customer)
            current_time: Current time at the position (used if TW present)
        
        Returns:
            Dictionary of feature values
        """
        features: Dict[str, float] = {}
        
        # Basic distance features
        features['dist_to_depot'] = float(self.dist_matrix[self.depot, request])
        features['dist_from_current'] = float(self.dist_matrix[current_position, request])
        features['dist_to_depot_from_request'] = float(self.dist_matrix[request, self.depot])
        
        # Demand and capacity features
        features['demand'] = float(self.demands[request])
        features['remaining_capacity'] = float(self.capacity - current_load)
        features['capacity_utilization'] = (current_load / self.capacity) if self.capacity > 0 else 0.0
        features['demand_ratio'] = (self.demands[request] / self.capacity) if self.capacity > 0 else 0.0
        
        # Route-specific features
        route_customers = [c for c in current_route if c != self.depot]
        features['route_length'] = float(len(route_customers))
        features['is_empty_route'] = 1.0 if len(route_customers) == 0 else 0.0
        
        # Distance-based features
        if len(current_route) > 2:  # More than just depot
            last_customer = current_route[-2] if current_route[-1] == self.depot else current_route[-1]
            features['dist_last_to_depot'] = float(self.dist_matrix[last_customer, self.depot])
        else:
            features['dist_last_to_depot'] = 0.0
        
        # Savings-like features
        features['savings'] = (
            float(self.dist_matrix[self.depot, current_position]) +
            float(self.dist_matrix[request, self.depot]) -
            float(self.dist_matrix[current_position, request])
        )
        
        # Time-window-aware features (if available)
        if self.has_tw:
            travel = float(self.dist_matrix[current_position, request])
            arrival = current_time + travel
            ready = float(self.ready_times[request])
            due = float(self.due_dates[request])
            service = float(self.service_times[request])
            start_service = max(arrival, ready)
            wait_time = max(0.0, ready - arrival)
            finish_service = start_service + service
            tw_feasible = 1.0 if arrival <= due else 0.0
            slack_to_due = max(0.0, due - arrival)
            remaining_tw_from_now = max(0.0, due - current_time - travel)

            features['arrival_time'] = arrival
            features['ready_time'] = ready
            features['due_time'] = due
            features['service_time'] = service
            features['start_service_time'] = start_service
            features['finish_service_time'] = finish_service
            features['wait_time'] = wait_time
            features['tw_feasible'] = tw_feasible
            features['slack_to_due'] = slack_to_due
            features['remaining_tw_from_now'] = remaining_tw_from_now

            # Normalized time features
            denom = self.max_due if self.max_due > 0 else 1.0
            features['norm_arrival_time'] = arrival / denom
            features['norm_ready_time'] = ready / denom
            features['norm_due_time'] = due / denom
            features['norm_wait_time'] = wait_time / denom
            features['norm_slack_to_due'] = slack_to_due / denom
            features['norm_remaining_tw_from_now'] = remaining_tw_from_now / denom
        
        # Normalized distance features
        max_dist = float(np.max(self.dist_matrix)) if np.size(self.dist_matrix) > 0 else 1.0
        if max_dist <= 0:
            max_dist = 1.0
        features['norm_dist_to_depot'] = features['dist_to_depot'] / max_dist
        features['norm_dist_from_current'] = features['dist_from_current'] / max_dist
        features['norm_savings'] = features['savings'] / max_dist
        
        return features


class GVRPMultiTechInstance:
    """
    Parser for Felipe et al. (2014) GVRP-multitech XML instances.
    
    This class is tailored to instances in `Sets/felipe-et-al-2014/`.
    
    Attributes
    ----------
    name : str
        Instance name from the XML.
    dimension : int
        Number of nodes in the network.
    coords : np.ndarray, shape (n, 2)
        Node coordinates (cx, cy) ordered by node id.
    node_types : np.ndarray, shape (n,)
        Node type (0 = depot, 1 = customer, 2 = charging station).
    demands : np.ndarray, shape (n,)
        Demand quantity per node (0 for non-customer nodes without demand).
    service_times : np.ndarray, shape (n,)
        Service time per node.
    depot : int
        Depot node id (from <departure_node>, should match node type 0).
    capacity : float
        Vehicle capacity.
    num_vehicles : int
        Number of vehicles of this profile.
    max_travel_time : float
        Maximum route travel time.
    battery_capacity : float
        Battery capacity (energy) of the vehicle.
    dist_matrix : np.ndarray, shape (n, n)
        Symmetric distance matrix computed from Euclidean coordinates.
    """

    def __init__(self, file_path: str):
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Basic info
        info = root.find("info")
        self.name = info.findtext("name") if info is not None else file_path

        # --- Network: nodes ---
        nodes_elem = root.find("./network/nodes")
        nodes = []
        max_node_id = -1
        if nodes_elem is not None:
            for node_elem in nodes_elem.findall("node"):
                node_id = int(node_elem.get("id"))
                node_type = int(node_elem.get("type"))
                cx = float(node_elem.findtext("cx"))
                cy = float(node_elem.findtext("cy"))
                nodes.append((node_id, node_type, cx, cy))
                if node_id > max_node_id:
                    max_node_id = node_id

        self.dimension = max_node_id + 1 if max_node_id >= 0 else 0

        # Initialize arrays
        self.coords = np.zeros((self.dimension, 2), dtype=float)
        self.node_types = np.zeros(self.dimension, dtype=int)

        for node_id, node_type, cx, cy in nodes:
            self.node_types[node_id] = node_type
            self.coords[node_id, 0] = cx
            self.coords[node_id, 1] = cy

        # --- Fleet info ---
        vehicle_profile = root.find("./fleet/vehicle_profile")
        self.num_vehicles = int(vehicle_profile.get("number")) if vehicle_profile is not None else 1
        self.capacity = float(vehicle_profile.findtext("capacity")) if vehicle_profile is not None else 0.0
        self.max_travel_time = (
            float(vehicle_profile.findtext("max_travel_time")) if vehicle_profile is not None else 0.0
        )

        # Depot from departure_node (should correspond to node with type 0)
        if vehicle_profile is not None:
            self.depot = int(vehicle_profile.findtext("departure_node"))
        else:
            # Fallback: first node of type 0, or 0
            depot_candidates = [nid for nid, ntype, _, _ in nodes if ntype == 0]
            self.depot = depot_candidates[0] if depot_candidates else 0

        # Battery capacity (optional)
        battery_capacity_text = None
        if vehicle_profile is not None:
            custom_elem = vehicle_profile.find("custom")
            if custom_elem is not None:
                battery_capacity_text = custom_elem.findtext("battery_capacity")
        self.battery_capacity = float(battery_capacity_text) if battery_capacity_text is not None else 0.0

        # --- Requests: demands and service times ---
        self.demands = np.zeros(self.dimension, dtype=float)
        self.service_times = np.zeros(self.dimension, dtype=float)

        requests_elem = root.find("requests")
        if requests_elem is not None:
            for req in requests_elem.findall("request"):
                node_id = int(req.get("node"))
                quantity = float(req.findtext("quantity"))
                service_time = float(req.findtext("service_time"))
                if 0 <= node_id < self.dimension:
                    self.demands[node_id] = quantity
                    self.service_times[node_id] = service_time

        # --- Distance matrix: compute from coordinates (Euclidean) ---
        # Ignore the <length> and <travel_time> fields and instead build a
        # symmetric distance matrix directly from (cx, cy) coordinates so
        # it is consistent with other coordinate-based instances.
        if self.dimension > 0:
            # coords: (n, 2)
            diff = self.coords[:, None, :] - self.coords[None, :, :]
            # Euclidean distance
            self.dist_matrix = np.hypot(diff[..., 0], diff[..., 1])
        else:
            self.dist_matrix = np.zeros((0, 0), dtype=float)

    def cost(self, routes, energy_per_distance: float = 1.0):
        """
        Compute total distance and violations for GVRP-multitech routes.

        Uses the unified :func:`compute_cost` function and additionally
        accounts for battery usage and route-duration limits.
        """
        return compute_cost(self, routes, energy_per_distance=energy_per_distance)

    def __repr__(self):
        return (
            f"GVRPMultiTechInstance({self.name}, n={self.dimension}, cap={self.capacity}, "
            f"vehicles={self.num_vehicles}, depot={self.depot})"
        )


if __name__ == "__main__":
    # Quick manual test for the Felipe et al. parser
    path = "Sets/felipe-et-al-2014/data-A0-N030_red.xml"
    inst = GVRPMultiTechInstance(path)
    print(inst)
    print("Dimension:", inst.dimension)
    print("Depot:", inst.depot)
    print("Capacity:", inst.capacity)
    print("Num vehicles:", inst.num_vehicles)
    print("Battery capacity:", inst.battery_capacity)
