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

from src.models import BlastMLPipeline, HAS_TORCH
from src.predict import predict_single_blast

if HAS_TORCH:
    import torch


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
        hole_vol = burden * spacing * bench_h

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
        Executes Differential Evolution (GA) optimization and extracts top 5 recommended designs.
        """
        convergence_history = []
        evaluated_candidates = []

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

        # Generate a diverse population sample around the bounds to form the top 5 recommended designs
        rng = np.random.RandomState(seed)
        grid_burdens = np.linspace(self.bounds[0][0], self.bounds[0][1], 8)
        grid_spacings = np.linspace(self.bounds[1][0], self.bounds[1][1], 8)
        grid_stemmings = np.linspace(self.bounds[2][0], self.bounds[2][1], 6)
        grid_pfs = np.linspace(self.bounds[3][0], self.bounds[3][1], 6)

        # Evaluate best vector first
        candidate_vectors = [result.x]

        for _ in range(120):
            vec = np.array([
                rng.choice(grid_burdens),
                rng.choice(grid_spacings),
                rng.choice(grid_stemmings),
                rng.choice(grid_pfs),
            ])
            candidate_vectors.append(vec)

        for vec in candidate_vectors:
            burden, spacing, stemming, pf = vec
            opt_inputs = self.fixed_params.copy()
            opt_inputs["burden_m"] = round(float(burden), 2)
            opt_inputs["spacing_m"] = round(float(spacing), 2)
            opt_inputs["stemming_m"] = round(float(stemming), 2)
            opt_inputs["powder_factor_kg_m3"] = round(float(pf), 3)

            bench_h = opt_inputs.get("bench_height_m", 12.0)
            hole_vol = opt_inputs["burden_m"] * opt_inputs["spacing_m"] * bench_h
            opt_inputs["charge_mass_per_hole_kg"] = round(opt_inputs["powder_factor_kg_m3"] * hole_vol, 2)
            if "max_charge_per_delay_kg" not in opt_inputs:
                opt_inputs["max_charge_per_delay_kg"] = round(opt_inputs["charge_mass_per_hole_kg"] * 2.0, 2)

            preds = predict_single_blast(opt_inputs, model_pipeline=self.ml_pipeline)
            obj_score = self._objective_function(vec)

            from src.domain.safety_checks import evaluate_safety
            from src.config import dump_model
            safety_rep = evaluate_safety(
                preds,
                limits={"max_ppv_mms": self.max_ppv, "max_flyrock_m": self.max_flyrock},
                blast_params=opt_inputs,
            )
            s_dict = dump_model(safety_rep)
            preds["safety_report"] = s_dict

            evaluated_candidates.append({
                "score": obj_score,
                "parameters": opt_inputs,
                "outputs": preds,
                "safety_report": s_dict,
            })

        # Sort candidates by objective score (cost + penalties) and pick top 5 distinct designs
        evaluated_candidates.sort(key=lambda x: x["score"])

        top_designs = []
        seen_keys = set()

        for item in evaluated_candidates:
            p = item["parameters"]
            key = (p["burden_m"], p["spacing_m"], p["stemming_m"], p["powder_factor_kg_m3"])
            if key not in seen_keys:
                seen_keys.add(key)
                top_designs.append(item)
            if len(top_designs) >= 5:
                break

        best_design = top_designs[0]
        best_safety = best_design["outputs"].get("safety_report", {})

        return {
            "success": bool(result.success),
            "optimized_parameters": best_design["parameters"],
            "predicted_outputs": best_design["outputs"],
            "best_cost_usd_t": best_design["outputs"]["cost_per_tonne_usd"],
            "top_5_designs": top_designs,
            "convergence_history": convergence_history,
            "optimization_message": str(result.message),
            "safety_report": best_safety,
            "overall_status": best_safety.get("overall_status", "SAFE"),
            "requires_engineer_review": best_safety.get("requires_engineer_review", False),
            "blocks_export": best_safety.get("blocks_export", False),
        }


def optimize_blast_design(model, target_fragmentation, max_vibration, max_airblast):
    """
    Use gradient descent to find blast parameters that achieve
    target fragmentation while staying within vibration and airblast limits.

    This implements the inverse design method from the GA-ANN research.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for gradient descent blast optimization.")

    # Initialize parameters (burden, spacing, stemming, etc., size 10)
    params = torch.ones(10, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=0.01)

    for step in range(1000):
        optimizer.zero_grad()
        out = model(params)
        frag, vib, air = out[0], out[1], out[2]

        # Loss: minimize fragmentation difference/maximize frag, minimize vibration/airblast violations
        frag_loss = torch.abs(frag - target_fragmentation) if isinstance(target_fragmentation, (int, float, torch.Tensor)) else -frag
        loss = frag_loss + 10 * torch.relu(vib - max_vibration) + 10 * torch.relu(air - max_airblast)
        loss.backward()
        optimizer.step()

    return params.detach()
