#!/usr/bin/env python3
"""
Problem type abstractions for VRP variants.
Shared between DEAP_gen.py and GOMEA_gen.py.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import operator


class ProblemType(ABC):
    """Abstract base class for problem type configurations."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Problem type name."""
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """List of feature names for this problem type."""
        pass
    
    @property
    @abstractmethod
    def num_features(self) -> int:
        """Number of features for this problem type."""
        pass
    
    @abstractmethod
    def extract_feature_values(self, features: Dict[str, float]) -> List[float]:
        """Extract feature values in the order expected by GP function."""
        pass
    
    @abstractmethod
    def solve_with_scoring(self, instance, feature_extractor, scoring_func) -> List[List[int]]:
        """Solve problem using GP-evolved scoring function."""
        pass
    
    @abstractmethod
    def evaluate_solution(self, instance, solution) -> Tuple[float, float]:
        """Evaluate solution quality. Returns (distance, violation)."""
        pass
    
    def create_primitive_set(self, gp_module) -> Optional[Any]:
        """Create GP primitive set for this problem type (DEAP-specific)."""
        return None


class CVRPProblemType(ProblemType):
    """Configuration for Capacitated VRP."""
    
    @property
    def name(self) -> str:
        return "CVRP"
    
    @property
    def feature_names(self) -> List[str]:
        return [
            'dist_to_depot', 'dist_from_current', 'dist_to_depot_from_request',
            'demand', 'remaining_capacity', 'capacity_utilization', 'demand_ratio',
            'route_length', 'is_empty_route', 'dist_last_to_depot', 'savings', 'norm_savings'
        ]
    
    @property
    def num_features(self) -> int:
        return 12
    
    def extract_feature_values(self, features: Dict[str, float]) -> List[float]:
        """Extract feature values in the order expected by GP function."""
        return [features.get(name, 0.0) for name in self.feature_names]
    
    def create_primitive_set(self, gp_module) -> Optional[Any]:
        """Create GP primitive set for CVRP (DEAP-specific)."""
        if gp_module is None:
            return None
            
        pset = gp_module.PrimitiveSet("MAIN", self.num_features)
        
        # Rename arguments to meaningful names
        for i, name in enumerate(self.feature_names):
            pset.renameArguments(**{f"ARG{i}": name})
        
        # Mathematical operators
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        
        # Protected division
        def protected_div(left, right):
            try:
                return left / right if abs(right) > 1e-6 else 1.0
            except:
                return 1.0
        pset.addPrimitive(protected_div, 2)
        
        # Constants
        pset.addTerminal(0.0)
        pset.addTerminal(1.0)
        pset.addTerminal(2.0)
        pset.addTerminal(0.5)
        pset.addTerminal(-1.0)
        
        return pset
    
    def solve_with_scoring(self, instance, feature_extractor, scoring_func) -> List[List[int]]:
        """Solve CVRP using GP-evolved scoring function."""
        n = instance.dimension
        unvisited = set(range(1, n))  # All customers except depot
        routes = []
        
        while unvisited:
            route = [instance.depot]
            load = 0
            current_position = instance.depot
            
            while True:
                # Get capacity-feasible candidates
                candidates = [c for c in unvisited if load + instance.demands[c] <= instance.capacity]
                
                if not candidates:
                    # Return to depot
                    route.append(instance.depot)
                    break
                
                # Score each candidate using GP model
                scores = []
                for candidate in candidates:
                    features = feature_extractor.extract_features(
                        candidate, route, load, current_position
                    )
                    
                    feature_values = self.extract_feature_values(features)
                    
                    try:
                        score = scoring_func(*feature_values)
                        scores.append(score)
                    except:
                        # If GP model fails, use distance as fallback
                        scores.append(instance.dist_matrix[current_position, candidate])
                
                # Choose candidate with lowest score (most desirable)
                best_candidate = candidates[scores.index(min(scores))]
                
                # Update route
                route.append(best_candidate)
                load += instance.demands[best_candidate]
                unvisited.remove(best_candidate)
                current_position = best_candidate
            
            routes.append(route)
        
        return routes
    
    def evaluate_solution(self, instance, solution) -> Tuple[float, float]:
        """Evaluate CVRP solution quality."""
        return instance.cost(solution)


class CVRPTWProblemType(ProblemType):
    """Configuration for Capacitated VRP with Time Windows."""
    
    @property
    def name(self) -> str:
        return "CVRPTW"
    
    @property
    def feature_names(self) -> List[str]:
        return [
            'dist_to_depot', 'dist_from_current', 'dist_to_depot_from_request',
            'demand', 'remaining_capacity', 'capacity_utilization', 'demand_ratio',
            'route_length', 'is_empty_route', 'dist_last_to_depot', 'savings', 'norm_savings',
            'arrival_time', 'ready_time', 'due_time', 'service_time', 'start_service_time',
            'finish_service_time', 'wait_time', 'tw_feasible', 'slack_to_due', 'remaining_tw_from_now'
        ]
    
    @property
    def num_features(self) -> int:
        return 22  # 12 base + 10 TW features
    
    def extract_feature_values(self, features: Dict[str, float]) -> List[float]:
        """Extract feature values in the order expected by GP function."""
        return [features.get(name, 0.0) for name in self.feature_names]
    
    def create_primitive_set(self, gp_module) -> Optional[Any]:
        """Create GP primitive set for CVRPTW (DEAP-specific)."""
        if gp_module is None:
            return None
            
        pset = gp_module.PrimitiveSet("MAIN", self.num_features)
        
        # Rename arguments to meaningful names
        for i, name in enumerate(self.feature_names):
            pset.renameArguments(**{f"ARG{i}": name})
        
        # Mathematical operators
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        
        # Protected division
        def protected_div(left, right):
            try:
                return left / right if abs(right) > 1e-6 else 1.0
            except:
                return 1.0
        pset.addPrimitive(protected_div, 2)
        
        # Additional functions for time windows
        def max_func(a, b):
            return max(a, b)
        def min_func(a, b):
            return min(a, b)
        
        pset.addPrimitive(max_func, 2)
        pset.addPrimitive(min_func, 2)
        
        # Constants
        pset.addTerminal(0.0)
        pset.addTerminal(1.0)
        pset.addTerminal(2.0)
        pset.addTerminal(0.5)
        pset.addTerminal(-1.0)
        
        return pset
    
    def solve_with_scoring(self, instance, feature_extractor, scoring_func) -> List[List[int]]:
        """Solve CVRPTW using GP-evolved scoring function."""
        n = instance.dimension
        unvisited = set(range(1, n))  # All customers except depot
        routes = []
        
        while unvisited:
            route = [instance.depot]
            load = 0
            current_position = instance.depot
            current_time = 0.0
            
            while True:
                # Get capacity-feasible candidates
                capacity_ok = [c for c in unvisited if load + instance.demands[c] <= instance.capacity]
                
                if not capacity_ok:
                    # Return to depot
                    route.append(instance.depot)
                    break
                
                # Check time window feasibility and score candidates
                feasible_candidates = []
                scores = []
                
                for candidate in capacity_ok:
                    travel = instance.dist_matrix[current_position, candidate]
                    arrival = current_time + travel
                    
                    # Check time window feasibility
                    if arrival > instance.due_dates[candidate]:
                        continue  # Skip infeasible candidates
                    
                    feasible_candidates.append(candidate)
                    
                    # Extract features with current time
                    features = feature_extractor.extract_features(
                        candidate, route, load, current_position, current_time
                    )
                    
                    feature_values = self.extract_feature_values(features)
                    
                    try:
                        score = scoring_func(*feature_values)
                        scores.append(score)
                    except:
                        # If GP model fails, use distance as fallback
                        scores.append(travel)
                
                if not feasible_candidates:
                    # No time-feasible candidates, return to depot
                    route.append(instance.depot)
                    break
                
                # Choose candidate with lowest score
                best_candidate = feasible_candidates[scores.index(min(scores))]
                
                # Update route and time
                route.append(best_candidate)
                load += instance.demands[best_candidate]
                unvisited.remove(best_candidate)
                
                # Update time
                travel = instance.dist_matrix[current_position, best_candidate]
                arrival = current_time + travel
                start_service = max(arrival, instance.ready_times[best_candidate])
                current_time = start_service + instance.service_times[best_candidate]
                current_position = best_candidate
            
            routes.append(route)
        
        return routes
    
    def evaluate_solution(self, instance, solution) -> Tuple[float, float]:
        """Evaluate CVRPTW solution quality."""
        return instance.cost(solution)


class ProblemTypeRegistry:
    """Registry for managing problem type configurations."""
    
    def __init__(self):
        self._types = {}
        self._register_default_types()
    
    def _register_default_types(self):
        """Register default problem types."""
        self.register(CVRPProblemType())
        self.register(CVRPTWProblemType())
    
    def register(self, problem_type: ProblemType):
        """Register a problem type."""
        self._types[problem_type.name] = problem_type
    
    def get(self, name: str) -> ProblemType:
        """Get a problem type by name."""
        if name not in self._types:
            raise ValueError(f"Unknown problem type: {name}")
        return self._types[name]
    
    def auto_detect(self, instance) -> str:
        """Auto-detect problem type from instance."""
        if hasattr(instance, 'ready_times') and hasattr(instance, 'due_dates'):
            return "CVRPTW"
        else:
            return "CVRP"
    
    def list_types(self) -> List[str]:
        """List all registered problem types."""
        return list(self._types.keys())


# Global registry
PROBLEM_REGISTRY = ProblemTypeRegistry()
