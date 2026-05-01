"""
vrp_feature_extractor: feature extraction for VRP scoring.

Contains the VRPFeatureExtractor used by GP and heuristic scoring logic.
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np


class VRPFeatureExtractor:
    """Extracts features for request evaluation."""

    def __init__(self, instance: Any):
        self.instance = instance
        self.depot = instance.depot
        self.dist_matrix = instance.dist_matrix
        self.demands = instance.demands
        self.capacity = instance.capacity
        self.has_tw = all(
            hasattr(instance, attr) for attr in ("ready_times", "due_dates", "service_times")
        )
        if self.has_tw:
            self.ready_times = instance.ready_times
            self.due_dates = instance.due_dates
            self.service_times = instance.service_times
            self.max_due = float(np.max(self.due_dates)) if len(self.due_dates) > 0 else 1.0
        else:
            self.max_due = 1.0

        self.has_battery = getattr(instance, "battery_capacity", 0.0) > 0.0
        if self.has_battery:
            self.battery_capacity = getattr(instance, "battery_capacity", 0.0)
            self.energy_consumption = getattr(instance, "energy_consumption", 1.0)
            self.node_types = getattr(instance, "node_types", None)
            if self.node_types is not None:
                n = len(self.node_types)
                self.stations = [i for i in range(n) if self.node_types[i] == 2]
            else:
                self.stations = []
        else:
            self.battery_capacity = float("inf")
            self.energy_consumption = 1.0
            self.stations = []

        self.depots = getattr(instance, "depots", None)
        if self.depots is None:
            self.depots = [self.depot]
        self.depots = list(self.depots)

        n_all = len(self.demands)
        if hasattr(self, "node_types") and self.node_types is not None:
            self.customers = [i for i in range(n_all) if self.node_types[i] == 1]
        else:
            self.customers = [i for i in range(n_all) if i not in self.depots]
        self.total_customer_demand = float(
            sum(float(self.demands[i]) for i in self.customers)
        ) if self.customers else 1.0

    def extract_features(
        self,
        request: int,
        current_route: List[int],
        current_load: float,
        current_position: int,
        current_time: float = 0.0,
        current_battery: float = None,
        dist_to_nearest_charger: Dict[int, float] = None,
        route_depot: int = None,
        bool_capacity: bool = True,
    ) -> Dict[str, float]:
        """
        Extract features for a given candidate request.

        Parameters
        ----------
        request : int
            Customer node to evaluate.
        current_route : list[int]
            Current route being built.
        current_load : float
            Current load of the route.
        current_position : int
            Current position in the route (last customer).
        current_time : float
            Current time at the position (used if TW present).
        current_battery : float or None
            Current battery level (for GVRP).
        dist_to_nearest_charger : dict[int, float] or None
            Mapping node -> distance to nearest charger (for GVRP).
        route_depot : int or None
            Depot of the current route (for multi-depot). If None, uses instance.depot.
        bool_capacity : bool
            If False, capacity-related features are zeroed out.

        Returns
        -------
        dict[str, float]
            Dictionary of extracted feature values.
        """
        features: Dict[str, float] = {}
        depot = route_depot if route_depot is not None else self.depot

        dist_from_current = float(self.dist_matrix[current_position, request])
        features["dist_to_depot"] = float(self.dist_matrix[depot, request])
        features["dist_from_current"] = dist_from_current

        features["savings"] = (
            float(self.dist_matrix[depot, current_position])
            + float(self.dist_matrix[request, depot])
            - float(self.dist_matrix[current_position, request])
        )

        if bool_capacity: # Capacitated
            features["demand"] = float(self.demands[request])
            features["remaining_capacity"] = float(self.capacity - current_load)
            if self.capacity > 0:
                features["load_percentage"] = current_load / self.capacity
            else:
                features["load_percentage"] = 0.0

        if len(self.depots) > 1: # Multi-depot
            d_this = float(self.dist_matrix[depot, request])
            depot_dists = [float(self.dist_matrix[d, request]) for d in self.depots]
            sorted_dists = sorted(depot_dists)
            rank = sorted_dists.index(d_this) + 1 if d_this in sorted_dists else len(self.depots)
            features["depot_rank"] = float(rank)
            if len(sorted_dists) > 1:
                second_best = sorted_dists[1]
                features["depot_distance_advantage"] = max(0.0, second_best - d_this)
            else:
                features["depot_distance_advantage"] = 0.0

        if self.has_tw: # Time Window
            travel = float(self.dist_matrix[current_position, request])
            arrival = current_time + travel
            ready = float(self.ready_times[request])
            due = float(self.due_dates[request])
            wait_time = max(0.0, ready - arrival)
            slack_to_due = max(0.0, due - arrival)

            features["current_time"] = float(current_time)
            features["arrival_time"] = arrival
            features["ready_time"] = ready
            features["due_time"] = due
            features["wait_time"] = wait_time
            features["slack_to_due"] = slack_to_due
            d_safe = max(dist_from_current, 1e-6)
            features["route_urgency"] = (due - float(current_time)) / d_safe

        if self.has_battery and current_battery is not None: # Green VRP
            features["current_battery"] = float(current_battery)
            features["battery_percentage"] = current_battery / self.battery_capacity

            dist_to_customer = float(self.dist_matrix[current_position, request])
            energy_to_customer = dist_to_customer * self.energy_consumption
            features["energy_to_customer"] = energy_to_customer

            if dist_to_nearest_charger is not None and request in dist_to_nearest_charger:
                features["dist_to_nearest_charger"] = dist_to_nearest_charger[request]
            else:
                features["dist_to_nearest_charger"] = 0.0

        return features
