import vrplib
import numpy as np
import re
from typing import List, Dict


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


class VRPFeatureExtractor:
    """Extracts VRP/VRPTW features for request evaluation.

    - Works for classic capacitated VRP.
    - If the instance also has VRPTW attributes (ready_times, due_dates, service_times),
      it augments the feature set with time-window-related metrics.
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


class VRPTWInstance:
    """Parses Solomon-style VRPTW instances (e.g., `Vrp-Set-HG/C1_2_1.txt`).

    Expected format (abridged):
        VEHICLE
        NUMBER     CAPACITY
          50          200

        CUSTOMER
        CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME
            0        70       70        0        0         1351          0
            1        33       78       20      750         809          90

    Attributes
    -----------
    name: str
        Derived from file stem
    num_vehicles: int
    capacity: int
    dimension: int
        Number of nodes (including depot)
    coords: np.ndarray shape (n, 2)
    demands: np.ndarray shape (n,)
    ready_times: np.ndarray shape (n,)
    due_dates: np.ndarray shape (n,)
    service_times: np.ndarray shape (n,)
    depot: int (always 0 for Solomon files)
    dist_matrix: np.ndarray shape (n, n) – Euclidean distances
    """

    def __init__(self, file_path: str):
        self.name = self._derive_name(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]

        vehicle_idx = self._find_line_index(lines, "VEHICLE")
        customer_idx = self._find_line_index(lines, "CUSTOMER")
        if vehicle_idx is None or customer_idx is None:
            raise ValueError("Input file missing VEHICLE or CUSTOMER section")

        # Parse vehicle section
        # Layout:
        #   VEHICLE
        #   NUMBER     CAPACITY
        #     50          200
        num_cap_line = self._next_nonempty_line(lines, start=vehicle_idx + 2)
        if num_cap_line is None:
            raise ValueError("Failed to locate vehicle number/capacity line")
        num_vehicles, capacity = self._parse_two_ints(num_cap_line)
        self.num_vehicles = num_vehicles
        self.capacity = capacity

        # Parse customer section – data begins two lines after the word CUSTOMER
        # Skip header row line with column names; then read until EOF or blank-only tail
        data_start = self._next_nonempty_index(lines, start=customer_idx + 2)
        if data_start is None:
            raise ValueError("Failed to locate customer header line")
        # After header, the next non-empty is first data row
        first_row_idx = self._next_nonempty_index(lines, start=data_start + 1)
        if first_row_idx is None:
            raise ValueError("Failed to locate first customer record")

        node_ids: List[int] = []
        xs: List[float] = []
        ys: List[float] = []
        demands: List[int] = []
        ready_times: List[float] = []
        due_dates: List[float] = []
        service_times: List[float] = []

        idx = first_row_idx
        while idx < len(lines):
            raw = lines[idx].strip()
            idx += 1
            if raw == "":
                continue
            # Some files may end with a single blank; we simply skip
            parts = raw.split()
            if len(parts) < 7:
                # Likely trailer or malformed line; stop parsing data region
                # but be tolerant and continue to next line instead of hard failing
                continue
            try:
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                dem = int(parts[3])
                rdy = float(parts[4])
                due = float(parts[5])
                svc = float(parts[6])
            except ValueError:
                # Skip non-data lines if any slipped through
                continue

            node_ids.append(nid)
            xs.append(x)
            ys.append(y)
            demands.append(dem)
            ready_times.append(rdy)
            due_dates.append(due)
            service_times.append(svc)

        if not node_ids:
            raise ValueError("No customer records parsed from file")

        # Ensure a stable internal 0..n-1 indexing by sorting by original ids
        # (If original ids do not start at 0, we still map the smallest id to 0 internally.)
        n = len(node_ids)
        order = np.argsort(np.array(node_ids))
        xs = np.array(xs, dtype=float)[order]
        ys = np.array(ys, dtype=float)[order]
        demands = np.array(demands, dtype=int)[order]
        ready_times = np.array(ready_times, dtype=float)[order]
        due_dates = np.array(due_dates, dtype=float)[order]
        service_times = np.array(service_times, dtype=float)[order]

        self.dimension = n
        self.depot = 0
        self.coords = np.stack([xs, ys], axis=1)
        self.demands = demands
        self.ready_times = ready_times
        self.due_dates = due_dates
        self.service_times = service_times

        self.dist_matrix = self._compute_euclidean_distances(self.coords)

    def cost(self, routes):
        """
        Compute total travel distance and capacity violation (ignores time-window penalties).
        routes = list of lists [[0, 5, 7, 0], ...]
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
        return (
            f"VRPTWInstance({self.name}, n={self.dimension}, cap={self.capacity}, "
            f"vehicles={self.num_vehicles})"
        )

    @staticmethod
    def _derive_name(path: str) -> str:
        import os

        base = os.path.basename(path)
        return os.path.splitext(base)[0]

    @staticmethod
    def _find_line_index(lines: List[str], starts_with: str):
        key = starts_with.strip().upper()
        for i, line in enumerate(lines):
            if line.strip().upper().startswith(key):
                return i
        return None

    @staticmethod
    def _next_nonempty_line(lines: List[str], start: int) -> str | None:
        for i in range(start, len(lines)):
            s = lines[i].strip()
            if s != "":
                return s
        return None

    @staticmethod
    def _next_nonempty_index(lines: List[str], start: int) -> int | None:
        for i in range(start, len(lines)):
            if lines[i].strip() != "":
                return i
        return None

    @staticmethod
    def _parse_two_ints(line: str) -> tuple[int, int]:
        parts = line.split()
        if len(parts) < 2:
            raise ValueError("Expected two integers on the vehicle line")
        return int(parts[0]), int(parts[1])

    @staticmethod
    def _compute_euclidean_distances(coords: np.ndarray) -> np.ndarray:
        # Pairwise Euclidean using broadcasting
        diff = coords[:, None, :] - coords[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        return dists