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
        "ppv_mms": round(ppv, 2),
        "flyrock_m": round(flyrock, 2),
        "cost_per_tonne_usd": round(cost_per_tonne, 2),
    }


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
