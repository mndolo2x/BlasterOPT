"""
Prediction helper module with machine learning and physics fallback models.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from src.models import BlastMLPipeline, FEATURE_COLS
from src.data_ingestion import engineer_features


def predict_physics_fallback(inputs: Dict[str, float]) -> Dict[str, float]:
    """
    Computes predictions directly using analytical domain physics equations
    when trained ML models are unavailable.
    """
    rock_factor_A = inputs.get("rock_factor_A", 7.0)
    bench_height_m = inputs.get("bench_height_m", 12.0)
    hole_diameter_mm = inputs.get("hole_diameter_mm", 250.0)
    burden_m = inputs.get("burden_m", 6.0)
    spacing_m = inputs.get("spacing_m", 7.0)
    stemming_m = inputs.get("stemming_m", 5.0)
    charge_mass_kg = inputs.get("charge_mass_per_hole_kg", 300.0)
    powder_factor_kg_m3 = inputs.get("powder_factor_kg_m3", 0.6)
    max_charge_per_delay_kg = inputs.get("max_charge_per_delay_kg", 600.0)
    monitoring_dist_m = inputs.get("monitoring_distance_m", 500.0)
    rws = inputs.get("explosive_rws", 100.0)

    # 1. Kuz-Ram d50 (mm)
    # d50 (cm) = A * K^-0.8 * Q^(1/6) * (115/RWS)^(19/30)
    d50_cm = (
        rock_factor_A
        * (np.maximum(powder_factor_kg_m3, 0.05) ** (-0.8))
        * (np.maximum(charge_mass_kg, 1.0) ** (1/6))
        * ((115.0 / max(rws, 10.0)) ** (19/30))
    )
    d50_mm = float(np.clip(d50_cm * 10.0, 20.0, 1500.0))

    # Kuz-Ram Uniformity Index n
    spacing_burden_ratio = spacing_m / max(burden_m, 0.1)
    hole_depth_m = bench_height_m + 0.3 * burden_m
    charge_length_m = max(0.5, hole_depth_m - stemming_m)
    n_uniformity = (2.2 - 14 * (burden_m / (hole_diameter_mm / 1000.0))) * (
        1 + (spacing_burden_ratio - 1) / 2
    ) * (charge_length_m / max(bench_height_m, 0.1))
    n_uniformity = float(np.clip(np.abs(n_uniformity) + 0.8, 0.7, 2.2))

    # 2. USBM PPV (mm/s)
    # SD = Distance / sqrt(Q_delay)
    sd = monitoring_dist_m / np.sqrt(max(max_charge_per_delay_kg, 1.0))
    ppv = float(np.clip(1140.0 * (sd ** (-1.6)), 0.1, 250.0))

    # 3. Flyrock Distance (m)
    flyrock = float(
        np.clip(20.0 * ((max(charge_mass_kg, 1.0) ** (2/3)) / max(burden_m, 0.5)), 5.0, 500.0)
    )

    # 4. Estimated Drilling & Blasting Cost ($ / tonne)
    rock_volume_per_hole = burden_m * spacing_m * bench_height_m
    rock_mass_per_hole = rock_volume_per_hole * 2.65
    drilling_cost = (bench_height_m + 1.0) * (12.0 + (hole_diameter_mm / 100.0) * 8.0)
    explosives_cost = charge_mass_kg * (1.2 + (rws / 100.0) * 0.8) + 15.0
    cost_per_tonne = float(np.clip((drilling_cost + explosives_cost) / max(rock_mass_per_hole, 1.0), 0.2, 50.0))

    return {
        "d50_mm": round(d50_mm, 2),
        "uniformity_index_n": round(n_uniformity, 2),
        "ppv_mms": round(ppv, 2),
        "flyrock_m": round(flyrock, 2),
        "cost_per_tonne_usd": round(cost_per_tonne, 2),
    }


def predict_outcomes(input_params: Any, model_pipeline: Optional[BlastMLPipeline] = None) -> Dict[str, Any]:
    """
    Validates input parameters and predicts outcomes for all five blast target variables.
    Handles missing or out-of-range inputs gracefully by returning error messages.
    """
    if not isinstance(input_params, dict):
        return {"error": "Invalid input: input_params must be a dictionary."}

    required_keys = [
        "rock_factor_A",
        "bench_height_m",
        "hole_diameter_mm",
        "burden_m",
        "spacing_m",
        "stemming_m",
        "charge_mass_per_hole_kg",
        "powder_factor_kg_m3",
        "max_charge_per_delay_kg",
        "monitoring_distance_m",
    ]

    missing_keys = [k for k in required_keys if k not in input_params or input_params[k] is None]
    if missing_keys:
        return {"error": f"Missing required input parameter(s): {', '.join(missing_keys)}"}

    # Define acceptable physical range validation rules
    valid_ranges = {
        "rock_factor_A": (1.0, 20.0),
        "bench_height_m": (1.0, 50.0),
        "hole_diameter_mm": (50.0, 500.0),
        "burden_m": (0.5, 20.0),
        "spacing_m": (0.5, 30.0),
        "stemming_m": (0.1, 15.0),
        "charge_mass_per_hole_kg": (1.0, 5000.0),
        "powder_factor_kg_m3": (0.01, 5.0),
        "max_charge_per_delay_kg": (1.0, 10000.0),
        "monitoring_distance_m": (10.0, 10000.0),
    }

    out_of_range = []
    for key, (min_val, max_val) in valid_ranges.items():
        val = input_params[key]
        if not isinstance(val, (int, float)) or not (min_val <= val <= max_val):
            out_of_range.append(f"{key} (value: {val}, expected range: [{min_val}, {max_val}])")

    if out_of_range:
        return {"error": f"Out-of-range input parameter(s): {'; '.join(out_of_range)}"}

    results = predict_single_blast(input_params, model_pipeline=model_pipeline)

    # Ensure all five target keys are present in output
    all_targets = ["d50_mm", "uniformity_index_n", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"]
    fallback = predict_physics_fallback(input_params)
    for target in all_targets:
        if target not in results or results[target] is None or np.isnan(results[target]):
            results[target] = fallback[target]

    return results


def predict_single_blast(
    inputs: Dict[str, float], model_pipeline: Optional[BlastMLPipeline] = None
) -> Dict[str, float]:
    """
    Predicts fragmentation, PPV, flyrock, and cost for a single blast design input.
    """
    if model_pipeline is not None and len(model_pipeline.models) > 0:
        df_single = pd.DataFrame([inputs])
        df_single = engineer_features(df_single)
        preds_df = model_pipeline.predict(df_single)

        results = {}
        for target in ["d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"]:
            col = f"pred_{target}"
            if col in preds_df.columns:
                results[target] = float(np.round(preds_df[col].iloc[0], 2))

        # Fallback for any missing prediction
        fallback = predict_physics_fallback(inputs)
        for target in ["d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"]:
            if target not in results or np.isnan(results[target]):
                results[target] = fallback[target]

        return results
    else:
        return predict_physics_fallback(inputs)
