#!/usr/bin/env python3
"""
Data generation utilities for VRP instances.
"""

import copy
import math
import numpy as np
from typing import Any

def remove_tw_from_gvrp(gvrp_instances):
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
    New depot positions are created deterministically, they are
    placed on a circle around the centroid of all nodes, evenly spaced.

    The original depot stays at index 0. Additional depots are appended at indices
    dimension, dimension+1, ... and instance.depots is set to [0, dimension, ...].

    Parameters
    ----------
    instance : EVRPTW-like instance or list of instances
    num_depots : int or None
        If None, the number of depots is chosen dynamically based on the number
        of customers in the instance:
            - minimum 2 depots, maximum 5 depots
            - mapping is based on customer count in [5, 100]
        If an int is provided, it is used directly (clipped to [2, 5]).

    Returns
    -------
    New instance(s) with depots attribute set and arrays extended. Originals unchanged.
    """
    if isinstance(instance, list):
        return [evrptw_to_multi_depot(inst, num_depots=num_depots) for inst in instance]
    instance = copy.deepcopy(instance)

    n_all = getattr(instance, "dimension", None)
    if n_all is None and hasattr(instance, "demands"):
        n_all = len(instance.demands)
    node_types = getattr(instance, "node_types", None)
    if node_types is not None:
        n_customers = int(sum(1 for t in node_types if t == 1))
    else:
        depot_idx = getattr(instance, "depot", 0)
        n_customers = max(0, (n_all or 0) - 1) if n_all is not None else 0

    if num_depots is None:
        c = max(5, min(100, n_customers))
        if c <= 20:
            num_depots = 2
        elif c <= 40:
            num_depots = 3
        elif c <= 70:
            num_depots = 4
        else:
            num_depots = 5

    num_depots = max(2, min(5, int(num_depots)))

    if num_depots < 2:
        setattr(instance, "depots", [getattr(instance, "depot", 0)])
        return instance

    n_old = instance.dimension
    num_new = num_depots - 1  # keep original depot at index 0
    coords = instance.coords

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

    new_coords = np.zeros((n_new, 2), dtype=float)
    new_coords[:n_old] = coords
    new_coords[n_old:] = new_depot_coords
    instance.coords = new_coords

    new_node_types = np.ones(n_new, dtype=int)
    new_node_types[:n_old] = instance.node_types
    new_node_types[n_old:] = 0
    instance.node_types = new_node_types

    new_demands = np.zeros(n_new, dtype=float)
    new_demands[:n_old] = np.asarray(instance.demands, dtype=float)
    instance.demands = new_demands

    if hasattr(instance, "service_times") and instance.service_times is not None:
        new_service_times = np.zeros(n_new, dtype=float)
        new_service_times[:n_old] = np.asarray(instance.service_times, dtype=float)
        instance.service_times = new_service_times

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

    diff = instance.coords[:, None, :] - instance.coords[None, :, :]
    instance.dist_matrix = np.hypot(diff[..., 0], diff[..., 1])

    instance.depots = [0] + list(range(n_old, n_new))
    if not hasattr(instance, "depot"):
        instance.depot = 0

    return instance