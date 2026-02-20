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

    def __repr__(self):
        return (
            f"GVRPMultiTechInstance({self.name}, n={self.dimension}, cap={self.capacity}, "
            f"vehicles={self.num_vehicles}, depot={self.depot})"
        )