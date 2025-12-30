#!/usr/bin/env python3
"""
Data generation utilities for VRP instances.
"""

import numpy as np
import random
from typing import Optional, List
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


def convert_vrptw_to_gvrptw(vrptw_instances, battery_capacity = 160, energy_consumption = 1.0, percent_hybrid = 0.2):
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