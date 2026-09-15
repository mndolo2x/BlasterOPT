"""
Genetic Algorithm & Differential Evolution Blast Optimization Module.

Optimizes controllable blast design parameters (Burden, Spacing, Stemming, Powder Factor)
to minimize drilling and blasting cost while satisfying environmental/safety constraints:
- Maximum allowed Ground Vibration (PPV mm/s)
- Maximum allowed Flyrock Distance (m)
- Target Mean Fragmentation size (d50 mm) range
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional, Tuple, List
from scipy.optimize import differential_evolution

from src.models import BlastMLPipeline
from src.predict import predict_single_blast


class BlastOptimizer:
    """
    Genetic Algorithm / Differential Evolution Optimizer for blast design parameters.
    """

    def __init__(
        self,
        fixed_parameters: Dict[str, float],
        max_ppv_limit_mms: float = 10.0,
        max_flyrock_limit_m: float = 100.0,
        target_d50_range_mm: Tuple[float, float] = (100.0, 300.0),
        ml_pipeline: Optional[BlastMLPipeline] = None,
        penalty_weight: float = 1000.0,
    ):
        self.fixed_params = fixed_parameters
        self.max_ppv = max_ppv_limit_mms
        self.max_flyrock = max_flyrock_limit_m
        self.target_d50_min, self.target_d50_max = target_d50_range_mm
        self.ml_pipeline = ml_pipeline
        self.penalty_weight = penalty_weight

        # Parameter bounds for optimization:
        # 1. Burden (m)
        # 2. Spacing (m)
        # 3. Stemming (m)
        # 4. Powder factor (kg/m3)
        self.bounds = [
            (2.5, 10.0),  # burden_m
            (3.0, 12.0),  # spacing_m
            (2.0, 8.0),   # stemming_m
            (0.2, 1.8),   # powder_factor_kg_m3
        ]

    def _objective_function(self, vector: np.ndarray) -> float:
        burden, spacing, stemming, pf = vector

        # Construct full input payload
        inputs = self.fixed_params.copy()
        inputs["burden_m"] = burden
        inputs["spacing_m"] = spacing
        inputs["stemming_m"] = stemming
        inputs["powder_factor_kg_m3"] = pf

        # Derive required dependent features
        bench_h = inputs.get("bench_height_m", 12.0)
        hole_diam = inputs.get("hole_diameter_mm", 250.0)
        hole_vol = burden * spacing * bench_h
        rock_density = inputs.get("rock_density_t_m3", 2.65)

        # Charge per hole kg = pf * hole_vol
        charge_mass = pf * hole_vol
        inputs["charge_mass_per_hole_kg"] = charge_mass

        if "max_charge_per_delay_kg" not in inputs:
            inputs["max_charge_per_delay_kg"] = charge_mass * 2.0

        # Run prediction engine
        preds = predict_single_blast(inputs, model_pipeline=self.ml_pipeline)

        cost = preds["cost_per_tonne_usd"]
        d50 = preds["d50_mm"]
        ppv = preds["ppv_mms"]
        flyrock = preds["flyrock_m"]

        # Calculate constraint penalties
        penalty = 0.0

        # PPV Constraint
        if ppv > self.max_ppv:
            penalty += self.penalty_weight * ((ppv - self.max_ppv) ** 2)

        # Flyrock Constraint
        if flyrock > self.max_flyrock:
            penalty += self.penalty_weight * ((flyrock - self.max_flyrock) ** 2)

        # Fragmentation d50 target range penalty
        if d50 < self.target_d50_min:
            penalty += self.penalty_weight * (((self.target_d50_min - d50) / 10.0) ** 2)
        elif d50 > self.target_d50_max:
            penalty += self.penalty_weight * (((d50 - self.target_d50_max) / 10.0) ** 2)

        return float(cost + penalty)

    def optimize(
        self, popsize: int = 15, maxiter: int = 50, seed: int = 42
    ) -> Dict[str, Any]:
        """
        Executes Differential Evolution (GA) optimization.
        """
        convergence_history = []

        def callback(xk, convergence):
            score = self._objective_function(xk)
            convergence_history.append(score)

        result = differential_evolution(
            self._objective_function,
            bounds=self.bounds,
            popsize=popsize,
            maxiter=maxiter,
            seed=seed,
            callback=callback,
        )

        opt_burden, opt_spacing, opt_stemming, opt_pf = result.x

        # Re-run best vector
        opt_inputs = self.fixed_params.copy()
        opt_inputs["burden_m"] = round(opt_burden, 2)
        opt_inputs["spacing_m"] = round(opt_spacing, 2)
        opt_inputs["stemming_m"] = round(opt_stemming, 2)
        opt_inputs["powder_factor_kg_m3"] = round(opt_pf, 3)

        bench_h = opt_inputs.get("bench_height_m", 12.0)
        hole_vol = opt_burden * opt_spacing * bench_h
        opt_inputs["charge_mass_per_hole_kg"] = round(opt_pf * hole_vol, 2)
        if "max_charge_per_delay_kg" not in opt_inputs:
            opt_inputs["max_charge_per_delay_kg"] = round(opt_inputs["charge_mass_per_hole_kg"] * 2.0, 2)

        final_preds = predict_single_blast(opt_inputs, model_pipeline=self.ml_pipeline)

        return {
            "success": bool(result.success),
            "optimized_parameters": opt_inputs,
            "predicted_outputs": final_preds,
            "best_cost_usd_t": final_preds["cost_per_tonne_usd"],
            "convergence_history": convergence_history,
            "optimization_message": str(result.message),
        }
