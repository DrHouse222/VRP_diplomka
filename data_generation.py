#!/usr/bin/env python3
"""
Data generation utilities for VRP instances.
"""

import copy
import math
import random
import numpy as np
from typing import Optional, List, Any
from parser import VRPTWInstance


class GVRPTWInstance:
    """
    A VRPTW instance with GVRP attributes (battery constraints) added.
    Combines time windows with battery capacity and charging stations.
    Clones 20% of customers into charging stations at the same coordinates.
    """
    
    def __init__(self, vrptw_instance, battery_capacity, energy_consumption, percent_hybrid):
        """
        Convert a VRPTW instance to a GVRPTW instance.
        Clones percent_hybrid of customers into charging stations at the same coordinates.
        """
        # Get original dimension
        self.name = vrptw_instance.name
        self.num_vehicles = getattr(vrptw_instance, 'num_vehicles', 1)
        self.capacity = vrptw_instance.capacity
        self.depot = vrptw_instance.depot
        original_dim = vrptw_instance.dimension

        # Attributes for GVRP
        self.battery_capacity = battery_capacity
        self.energy_consumption = energy_consumption
        
        # Get all customer nodes
        customer_nodes = [i for i in range(original_dim) if i != self.depot]
        
        # Select customers to clone into charging stations
        num_stations = max(1, int(len(customer_nodes) * percent_hybrid))
        customers_to_clone = random.sample(customer_nodes, min(num_stations, len(customer_nodes)))
        
        # Create mapping: original_customer_id -> new_charging_station_id
        self.customer_to_station = {}
        new_dim = original_dim + len(customers_to_clone)
        
        # extend array
        self.dimension = new_dim
        
        # Extend coordinates: add charging stations at same coordinates as selected customers
        self.coords = np.zeros((new_dim, 2), dtype=float)
        self.coords[:original_dim] = vrptw_instance.coords
        station_id = original_dim
        for customer_id in customers_to_clone:
            self.coords[station_id] = vrptw_instance.coords[customer_id]
            self.customer_to_station[customer_id] = station_id
            station_id += 1
        
        # Extend demands, service times, time windows
        self.demands = np.zeros(new_dim, dtype=float)
        self.demands[:original_dim] = vrptw_instance.demands
        self.service_times = np.zeros(new_dim, dtype=float)
        self.service_times[:original_dim] = vrptw_instance.service_times
        self.ready_times = np.zeros(new_dim, dtype=float)
        self.ready_times[:original_dim] = vrptw_instance.ready_times
        self.due_dates = np.zeros(new_dim, dtype=float)
        self.due_dates[:original_dim] = vrptw_instance.due_dates
        self.due_dates[original_dim:] = float('inf')
        
        # Create distance matrix for new dimension
        self.dist_matrix = np.zeros((new_dim, new_dim), dtype=float)
        self.dist_matrix[:original_dim, :original_dim] = vrptw_instance.dist_matrix
        
        # Fill distances for new nodes
        for customer_id, station_id in self.customer_to_station.items():
            # Distance from customer to its station = 0
            self.dist_matrix[customer_id, station_id] = 0.0
            self.dist_matrix[station_id, customer_id] = 0.0
            
            # Distance from any original node to this station = distance to the customer
            for i in range(original_dim):
                if i != customer_id:  # Skip the customer itself
                    dist = vrptw_instance.dist_matrix[i, customer_id]
                    self.dist_matrix[i, station_id] = dist
                    self.dist_matrix[station_id, i] = dist
        
        # Distance between two charging stations = distance between their customers
        station_list = list(self.customer_to_station.items())
        for i, (customer1, station1) in enumerate(station_list):
            for customer2, station2 in station_list[i+1:]:
                dist = vrptw_instance.dist_matrix[customer1, customer2]
                self.dist_matrix[station1, station2] = dist
                self.dist_matrix[station2, station1] = dist
        
        # Create node_types array: 0=depot, 1=customer, 2=charging_station
        self.node_types = np.ones(new_dim, dtype=int)
        self.node_types[self.depot] = 0 
        # Mark charging stations as type 2
        for station_id in self.customer_to_station.values():
            self.node_types[station_id] = 2
    
    def __repr__(self):
        num_stations = np.sum(self.node_types == 2)
        return (
            f"GVRPTWInstance({self.name}, n={self.dimension}, cap={self.capacity}, "
            f"vehicles={self.num_vehicles}, battery={self.battery_capacity}, "
            f"stations={num_stations})"
        )


def convert_vrptw_to_gvrptw(vrptw_instances, battery_capacity = 200, energy_consumption = 1.0, percent_hybrid = 0.2):
    #Convert a VRPTW instance to a GVRPTW instance
    gvrp_instances = []
    for vrptw_instance in vrptw_instances:
        gvrp_instances.append(GVRPTWInstance(
            vrptw_instance=vrptw_instance,
            battery_capacity=battery_capacity,
            energy_consumption=energy_consumption,
            percent_hybrid=percent_hybrid
        ))
    return gvrp_instances

def remove_tw_from_gvrp(gvrp_instances):
    #Remove time windows from a GVRPTW instance
    # Delete the time window attributes so has_time_windows() returns False
    for gvrp_instance in gvrp_instances:
        if hasattr(gvrp_instance, 'ready_times'):
            delattr(gvrp_instance, 'ready_times')
        if hasattr(gvrp_instance, 'due_dates'):
            delattr(gvrp_instance, 'due_dates')
        if hasattr(gvrp_instance, 'service_times'):
            delattr(gvrp_instance, 'service_times')
    return gvrp_instances


def evrptw_to_multi_depot(
    instance: Any,
    num_depots: int | None = None,
) -> Any:
    """
    Convert a single-depot EVRPTW instance to a multi-depot instance.
    Returns a new instance (or list of new instances); originals are not modified.
    New depot positions are created deterministically (no random seed): they are
    placed on a circle around the centroid of all nodes, evenly spaced.

    The original depot stays at index 0. Additional depots are appended at indices
    dimension, dimension+1, ... and instance.depots is set to [0, dimension, ...].
    problem_types.solve_with_scoring uses instance.depots when present.

    Parameters
    ----------
    instance : EVRPTW-like instance or list of such instances
        If a list, each instance is copied and converted; returns list of new instances.
        Otherwise the instance is copied and the copy is converted and returned.
    num_depots : int or None
        If None, the number of depots is chosen dynamically based on the number
        of customers in the instance:
            - minimum 2 depots, maximum 5 depots
            - mapping is based on customer count in [5, 100]
        If an int is provided, it is used directly (still clipped to [2, 5]).

    Returns
    -------
    New instance(s) with depots attribute set and arrays extended. Originals unchanged.
    """
    if isinstance(instance, list):
        return [evrptw_to_multi_depot(inst, num_depots=num_depots) for inst in instance]
    # Work on a copy so the original is not modified
    instance = copy.deepcopy(instance)

    # Determine number of customers
    n_all = getattr(instance, "dimension", None)
    if n_all is None and hasattr(instance, "demands"):
        n_all = len(instance.demands)
    node_types = getattr(instance, "node_types", None)
    if node_types is not None:
        # By convention, node_type == 1 are customers
        n_customers = int(sum(1 for t in node_types if t == 1))
    else:
        # Fallback: treat all non-depot nodes as customers
        depot_idx = getattr(instance, "depot", 0)
        n_customers = max(0, (n_all or 0) - 1) if n_all is not None else 0

    if num_depots is None:
        # Clamp customer count to [5, 100] for the mapping
        c = max(5, min(100, n_customers))
        # Simple piecewise mapping from customers -> depots
        if c <= 20:
            num_depots = 2
        elif c <= 40:
            num_depots = 3
        elif c <= 70:
            num_depots = 4
        else:
            num_depots = 5

    # Ensure depots are within requested [2, 5] range
    num_depots = max(2, min(5, int(num_depots)))

    if num_depots < 2:
        setattr(instance, "depots", [getattr(instance, "depot", 0)])
        return instance

    n_old = instance.dimension
    num_new = num_depots - 1  # keep original depot at index 0
    coords = instance.coords

    # Deterministic depot positions: centroid + circle
    x_min, x_max = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
    y_min, y_max = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
    range_x = x_max - x_min if x_max > x_min else 1.0
    range_y = y_max - y_min if y_max > y_min else 1.0
    radius = 0.4 * max(range_x, range_y)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    new_depot_coords = np.zeros((num_new, 2), dtype=float)
    for k in range(num_new):
        angle = 2.0 * math.pi * k / num_new
        new_depot_coords[k, 0] = cx + radius * math.cos(angle)
        new_depot_coords[k, 1] = cy + radius * math.sin(angle)

    n_new = n_old + num_new
    instance.dimension = n_new

    # Extend coords
    new_coords = np.zeros((n_new, 2), dtype=float)
    new_coords[:n_old] = coords
    new_coords[n_old:] = new_depot_coords
    instance.coords = new_coords

    # Extend node_types: new nodes are depots (0)
    new_node_types = np.ones(n_new, dtype=int)
    new_node_types[:n_old] = instance.node_types
    new_node_types[n_old:] = 0
    instance.node_types = new_node_types

    # Extend demands (0 for depots)
    new_demands = np.zeros(n_new, dtype=float)
    new_demands[:n_old] = np.asarray(instance.demands, dtype=float)
    instance.demands = new_demands

    # Extend service_times only if present (0 for depots)
    if hasattr(instance, "service_times") and instance.service_times is not None:
        new_service_times = np.zeros(n_new, dtype=float)
        new_service_times[:n_old] = np.asarray(instance.service_times, dtype=float)
        instance.service_times = new_service_times

    # Time windows for new depots: same as original depot (only if present)
    if hasattr(instance, "ready_times") and instance.ready_times is not None:
        new_ready = np.zeros(n_new, dtype=float)
        new_ready[:n_old] = instance.ready_times
        new_ready[n_old:] = instance.ready_times[0]
        instance.ready_times = new_ready
    if hasattr(instance, "due_dates") and instance.due_dates is not None:
        new_due = np.zeros(n_new, dtype=float)
        new_due[:n_old] = instance.due_dates
        new_due[n_old:] = instance.due_dates[0]
        instance.due_dates = new_due

    # New distance matrix (Euclidean)
    diff = instance.coords[:, None, :] - instance.coords[None, :, :]
    instance.dist_matrix = np.hypot(diff[..., 0], diff[..., 1])

    # Multi-depot attribute for problem_types.solve_with_scoring
    instance.depots = [0] + list(range(n_old, n_new))
    # Keep depot = 0 for single-depot compatibility
    if not hasattr(instance, "depot"):
        instance.depot = 0

    return instance