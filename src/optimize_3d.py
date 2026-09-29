"""
3D Pattern Optimization Module for BlastOpt Botswana.
Provides optimization algorithms for 3D blast pattern design.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from scipy.optimize import differential_evolution, minimize
from src.blast_pattern_3d import (
    BlastHole,
    BenchGeometry,
    PatternType,
    PatternGenerator3D,
    HoleDeviationCompensator,
    create_bench_geometry
)
from src.predict import predict_single_blast


class PatternOptimizer3D:
    """Optimizer for 3D blast patterns."""

    def __init__(self, prediction_function: Optional[Callable] = None):
        """
        Initialize the 3D pattern optimizer.

        Args:
            prediction_function: Function to predict blast outcomes
        """
        self.prediction_function = prediction_function or predict_single_blast
        self.pattern_generator = PatternGenerator3D()

    def optimize_pattern(
        self,
        bench_geometry: BenchGeometry,
        pattern_type: PatternType,
        constraints: Dict[str, Any],
        max_iterations: int = 50,
        population_size: int = 15
    ) -> Dict[str, Any]:
        """
        Optimize 3D blast pattern design.

        Args:
            bench_geometry: Bench geometry parameters
            pattern_type: Type of pattern to optimize
            constraints: Optimization constraints
            max_iterations: Maximum optimization iterations
            population_size: Population size for genetic algorithm

        Returns:
            Dictionary with optimization results
        """
        # Define optimization variables based on pattern type
        variable_bounds = self._get_variable_bounds(pattern_type, constraints)

        # Objective function wrapper
        def objective_function(x):
            return self._calculate_objective(x, bench_geometry, pattern_type, constraints)

        # Run optimization
        result = differential_evolution(
            objective_function,
            variable_bounds,
            maxiter=max_iterations,
            popsize=population_size,
            seed=42,
            polish=True
        )

        # Extract optimized parameters
        optimized_params = self._extract_parameters(result.x, pattern_type)

        # Generate optimized pattern
        holes = self._generate_pattern_from_params(
            optimized_params,
            bench_geometry,
            pattern_type
        )

        # Calculate final objective value
        final_objective = objective_function(result.x)

        return {
            'optimized_parameters': optimized_params,
            'holes': holes,
            'objective_value': final_objective,
            'optimization_success': result.success,
            'optimization_message': result.message,
            'num_iterations': result.nfev
        }

    def _get_variable_bounds(
        self,
        pattern_type: PatternType,
        constraints: Dict[str, Any]
    ) -> List[Tuple[float, float]]:
        """
        Get variable bounds for optimization.

        Args:
            pattern_type: Type of pattern
            constraints: Constraint dictionary

        Returns:
            List of (min, max) bounds for each variable
        """
        bounds = []

        # Common bounds
        burden_min = constraints.get('burden_min_m', 3.0)
        burden_max = constraints.get('burden_max_m', 10.0)
        spacing_min = constraints.get('spacing_min_m', 4.0)
        spacing_max = constraints.get('spacing_max_m', 12.0)
        stemming_min = constraints.get('stemming_min_m', 2.0)
        stemming_max = constraints.get('stemming_max_m', 8.0)
        pf_min = constraints.get('powder_factor_min_kg_m3', 0.3)
        pf_max = constraints.get('powder_factor_max_kg_m3', 1.2)

        # Core variables
        bounds.extend([
            (burden_min, burden_max),      # burden
            (spacing_min, spacing_max),    # spacing
            (stemming_min, stemming_max),  # stemming
            (pf_min, pf_max)               # powder factor
        ])

        # Pattern-specific variables
        if pattern_type == PatternType.ECHELON:
            echelon_min = constraints.get('echelon_min_m', 0.5)
            echelon_max = constraints.get('echelon_max_m', 5.0)
            bounds.append((echelon_min, echelon_max))  # echelon offset
        elif pattern_type == PatternType.V_PATTERN:
            angle_min = constraints.get('v_angle_min_deg', 15.0)
            angle_max = constraints.get('v_angle_max_deg', 60.0)
            bounds.append((angle_min, angle_max))  # v-pattern angle

        return bounds

    def _calculate_objective(
        self,
        x: np.ndarray,
        bench_geometry: BenchGeometry,
        pattern_type: PatternType,
        constraints: Dict[str, Any]
    ) -> float:
        """
        Calculate objective function value.

        Args:
            x: Array of optimized variables
            bench_geometry: Bench geometry
            pattern_type: Pattern type
            constraints: Constraints

        Returns:
            float: Objective value (to be minimized)
        """
        # Extract parameters
        params = self._extract_parameters(x, pattern_type)

        # Generate pattern
        holes = self._generate_pattern_from_params(params, bench_geometry, pattern_type)

        # Calculate pattern metrics
        pattern_metrics = self._calculate_pattern_metrics(holes, bench_geometry)

        # Base cost (objective to minimize)
        cost = pattern_metrics['powder_factor_kg_m3'] * 100  # Weight powder factor heavily

        # Add penalties for constraint violations
        penalty = 0.0

        # Burden-spacing ratio penalty
        burden_spacing_ratio = params['burden_m'] / params['spacing_m']
        if burden_spacing_ratio < 0.7 or burden_spacing_ratio > 1.0:
            penalty += 500 * abs(burden_spacing_ratio - 0.85)

        # Stemming penalty (should be reasonable proportion of hole length)
        stemming_ratio = params['stemming_m'] / bench_geometry.bench_height_m
        if stemming_ratio < 0.2 or stemming_ratio > 0.6:
            penalty += 300 * abs(stemming_ratio - 0.4)

        # Powder factor penalty
        if params['powder_factor_kg_m3'] < 0.4 or params['powder_factor_kg_m3'] > 1.0:
            penalty += 200 * abs(params['powder_factor_kg_m3'] - 0.65)

        # Burden penalty based on hole diameter
        optimal_burden = params['hole_diameter_mm'] / 40.0  # Rule of thumb
        burden_deviation = abs(params['burden_m'] - optimal_burden)
        penalty += 100 * burden_deviation

        # Safety constraints
        if 'max_ppv_limit_mms' in constraints:
            # Estimate PPV based on max charge
            max_charge = max(hole.charge_mass_kg for hole in holes)
            estimated_ppv = 1140 * (max_charge ** 0.5) / (constraints.get('monitoring_distance_m', 400) ** 1.5)
            if estimated_ppv > constraints['max_ppv_limit_mms']:
                penalty += 1000 * (estimated_ppv - constraints['max_ppv_limit_mms'])

        return cost + penalty

    def _extract_parameters(
        self,
        x: np.ndarray,
        pattern_type: PatternType
    ) -> Dict[str, Any]:
        """
        Extract parameters from optimization array.

        Args:
            x: Array of optimized variables
            pattern_type: Pattern type

        Returns:
            Dictionary of parameters
        """
        params = {
            'burden_m': x[0],
            'spacing_m': x[1],
            'stemming_m': x[2],
            'powder_factor_kg_m3': x[3],
            'hole_diameter_mm': 250.0,  # Default value
        }

        # Pattern-specific parameters
        if pattern_type == PatternType.ECHELON and len(x) > 4:
            params['echelon_offset_m'] = x[4]
        elif pattern_type == PatternType.V_PATTERN and len(x) > 4:
            params['v_angle_deg'] = x[4]

        return params

    def _generate_pattern_from_params(
        self,
        params: Dict[str, Any],
        bench_geometry: BenchGeometry,
        pattern_type: PatternType
    ) -> List[BlastHole]:
        """
        Generate pattern from parameters.

        Args:
            params: Design parameters
            bench_geometry: Bench geometry
            pattern_type: Pattern type

        Returns:
            List of blast holes
        """
        num_rows = 5  # Default rows
        num_cols = 6  # Default cols

        if pattern_type == PatternType.RECTANGULAR:
            holes = self.pattern_generator.generate_rectangular_pattern(
                bench_geometry=bench_geometry,
                burden_m=params['burden_m'],
                spacing_m=params['spacing_m'],
                hole_diameter_mm=params['hole_diameter_mm'],
                stemming_m=params['stemming_m'],
                powder_factor_kg_m3=params['powder_factor_kg_m3'],
                num_rows=num_rows,
                num_cols=num_cols
            )
        elif pattern_type == PatternType.STAGGERED:
            holes = self.pattern_generator.generate_staggered_pattern(
                bench_geometry=bench_geometry,
                burden_m=params['burden_m'],
                spacing_m=params['spacing_m'],
                hole_diameter_mm=params['hole_diameter_mm'],
                stemming_m=params['stemming_m'],
                powder_factor_kg_m3=params['powder_factor_kg_m3'],
                num_rows=num_rows,
                num_cols=num_cols
            )
        elif pattern_type == PatternType.ECHELON:
            holes = self.pattern_generator.generate_echelon_pattern(
                bench_geometry=bench_geometry,
                burden_m=params['burden_m'],
                spacing_m=params['spacing_m'],
                hole_diameter_mm=params['hole_diameter_mm'],
                stemming_m=params['stemming_m'],
                powder_factor_kg_m3=params['powder_factor_kg_m3'],
                num_rows=num_rows,
                num_cols=num_cols,
                echelon_offset_m=params.get('echelon_offset_m', 2.0)
            )
        elif pattern_type == PatternType.V_PATTERN:
            holes = self.pattern_generator.generate_v_pattern(
                bench_geometry=bench_geometry,
                burden_m=params['burden_m'],
                spacing_m=params['spacing_m'],
                hole_diameter_mm=params['hole_diameter_mm'],
                stemming_m=params['stemming_m'],
                powder_factor_kg_m3=params['powder_factor_kg_m3'],
                num_rows=num_rows,
                num_cols=num_cols,
                v_angle_deg=params.get('v_angle_deg', 30.0)
            )
        else:
            holes = []

        return holes

    def _calculate_pattern_metrics(
        self,
        holes: List[BlastHole],
        bench_geometry: BenchGeometry
    ) -> Dict[str, Any]:
        """
        Calculate pattern metrics.

        Args:
            holes: List of blast holes
            bench_geometry: Bench geometry

        Returns:
            Dictionary of pattern metrics
        """
        if not holes:
            return {}

        total_charge = sum(hole.charge_mass_kg for hole in holes)
        avg_hole_length = np.mean([hole.hole_length for hole in holes])

        # Calculate pattern area
        x_coords = [hole.x for hole in holes]
        y_coords = [hole.y for hole in holes]
        pattern_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))

        # Calculate powder factor
        bench_volume = pattern_area * bench_geometry.bench_height_m
        powder_factor = total_charge / bench_volume if bench_volume > 0 else 0

        return {
            'total_charge_mass_kg': total_charge,
            'avg_hole_length_m': avg_hole_length,
            'pattern_area_m2': pattern_area,
            'bench_volume_m3': bench_volume,
            'powder_factor_kg_m3': powder_factor,
            'num_holes': len(holes)
        }


class MultiObjectivePatternOptimizer:
    """Multi-objective optimizer for 3D patterns."""

    def __init__(self):
        """Initialize multi-objective optimizer."""
        self.pattern_generator = PatternGenerator3D()

    def optimize_pareto_front(
        self,
        bench_geometry: BenchGeometry,
        pattern_type: PatternType,
        constraints: Dict[str, Any],
        population_size: int = 50,
        generations: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate Pareto front for multi-objective optimization.

        Args:
            bench_geometry: Bench geometry
            pattern_type: Pattern type
            constraints: Constraints
            population_size: Population size
            generations: Number of generations

        Returns:
            List of Pareto-optimal solutions
        """
        # Simple NSGA-II-like implementation
        solutions = []

        # Generate initial population
        for _ in range(population_size):
            # Random parameters within bounds
            params = self._generate_random_params(constraints)

            # Generate pattern
            holes = self._generate_pattern_from_params(params, bench_geometry, pattern_type)

            # Calculate objectives
            objectives = self._calculate_objectives(holes, bench_geometry, constraints)

            solutions.append({
                'parameters': params,
                'holes': holes,
                'objectives': objectives
            })

        # Evolve population (simplified)
        for generation in range(generations):
            # Tournament selection and crossover would go here
            # For simplicity, we'll just keep the best solutions
            solutions.sort(key=lambda x: x['objectives']['total_cost'])
            solutions = solutions[:population_size // 2]

        # Return non-dominated solutions
        pareto_front = self._extract_pareto_front(solutions)

        return pareto_front

    def _generate_random_params(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random parameters within constraints."""
        return {
            'burden_m': np.random.uniform(
                constraints.get('burden_min_m', 3.0),
                constraints.get('burden_max_m', 10.0)
            ),
            'spacing_m': np.random.uniform(
                constraints.get('spacing_min_m', 4.0),
                constraints.get('spacing_max_m', 12.0)
            ),
            'stemming_m': np.random.uniform(
                constraints.get('stemming_min_m', 2.0),
                constraints.get('stemming_max_m', 8.0)
            ),
            'powder_factor_kg_m3': np.random.uniform(
                constraints.get('powder_factor_min_kg_m3', 0.3),
                constraints.get('powder_factor_max_kg_m3', 1.2)
            ),
            'hole_diameter_mm': 250.0
        }

    def _generate_pattern_from_params(
        self,
        params: Dict[str, Any],
        bench_geometry: BenchGeometry,
        pattern_type: PatternType
    ) -> List[BlastHole]:
        """Generate pattern from parameters."""
        # Use pattern generator
        return self.pattern_generator.generate_rectangular_pattern(
            bench_geometry=bench_geometry,
            burden_m=params['burden_m'],
            spacing_m=params['spacing_m'],
            hole_diameter_mm=params['hole_diameter_mm'],
            stemming_m=params['stemming_m'],
            powder_factor_kg_m3=params['powder_factor_kg_m3'],
            num_rows=5,
            num_cols=6
        )

    def _calculate_objectives(
        self,
        holes: List[BlastHole],
        bench_geometry: BenchGeometry,
        constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate multiple objectives."""
        total_charge = sum(hole.charge_mass_kg for hole in holes)

        # Calculate powder factor
        x_coords = [hole.x for hole in holes]
        y_coords = [hole.y for hole in holes]
        pattern_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        bench_volume = pattern_area * bench_geometry.bench_height_m
        powder_factor = total_charge / bench_volume if bench_volume > 0 else 0

        # Estimate vibration
        max_charge = max(hole.charge_mass_kg for hole in holes)
        estimated_ppv = 1140 * (max_charge ** 0.5) / (constraints.get('monitoring_distance_m', 400) ** 1.5)

        return {
            'total_cost': powder_factor * 100,  # Cost objective
            'vibration': estimated_ppv,            # Vibration objective
            'fragmentation': powder_factor * 50   # Fragmentation proxy
        }

    def _extract_pareto_front(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract non-dominated solutions."""
        pareto_front = []

        for solution in solutions:
            is_dominated = False
            objectives = solution['objectives']

            for other in solutions:
                if other is solution:
                    continue

                other_objectives = other['objectives']

                # Check if other dominates this solution
                dominates = True
                for obj_key in objectives:
                    if other_objectives[obj_key] > objectives[obj_key]:
                        dominates = False
                        break

                if dominates:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(solution)

        return pareto_front


def optimize_pattern_for_cost(
    bench_geometry: BenchGeometry,
    pattern_type: PatternType,
    target_d50_mm: float,
    max_ppv_mms: float,
    monitoring_distance_m: float
) -> Dict[str, Any]:
    """
    Optimize pattern for minimum cost with constraints.

    Args:
        bench_geometry: Bench geometry
        pattern_type: Pattern type
        target_d50_mm: Target fragmentation
        max_ppv_mms: Maximum PPV
        monitoring_distance_m: Monitoring distance

    Returns:
        Optimization results
    """
    optimizer = PatternOptimizer3D()

    constraints = {
        'burden_min_m': 3.0,
        'burden_max_m': 10.0,
        'spacing_min_m': 4.0,
        'spacing_max_m': 12.0,
        'stemming_min_m': 2.0,
        'stemming_max_m': 8.0,
        'powder_factor_min_kg_m3': 0.3,
        'powder_factor_max_kg_m3': 1.2,
        'max_ppv_limit_mms': max_ppv_mms,
        'monitoring_distance_m': monitoring_distance_m
    }

    return optimizer.optimize_pattern(
        bench_geometry=bench_geometry,
        pattern_type=pattern_type,
        constraints=constraints,
        max_iterations=30,
        population_size=12
    )
