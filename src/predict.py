"""
Prediction helper module with machine learning and physics fallback models.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from sklearn.neighbors import NearestNeighbors
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


def find_similar_blasts(new_blast_params: Union[Dict[str, float], pd.DataFrame], historical_blasts: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Find the top_k most similar past blasts based on input parameters.
    Return their outcomes so the blaster can learn from history.

    Parameters:
    -----------
    new_blast_params : Dict[str, float] or pd.DataFrame
        Input features for the target blast design.
    historical_blasts : pd.DataFrame
        Historical dataset of past blast logs.
    top_k : int, default=5
        Number of top similar past blasts to retrieve.

    Returns:
    --------
    pd.DataFrame
        DataFrame of top_k most similar historical blast records.
    """
    if isinstance(new_blast_params, dict):
        df_new = pd.DataFrame([new_blast_params])
    else:
        df_new = new_blast_params.copy()

    # Identify common numeric feature columns
    num_cols = [c for c in FEATURE_COLS if c in historical_blasts.columns and c in df_new.columns]
    if not num_cols:
        num_cols = [c for c in historical_blasts.select_dtypes(include=[np.number]).columns if c in df_new.columns]

    X_hist = historical_blasts[num_cols].fillna(historical_blasts[num_cols].median())
    X_new = df_new[num_cols].fillna(X_hist.median())

    # Fit NearestNeighbors
    k = min(top_k, len(historical_blasts))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(X_hist)

    distances, indices = nn.kneighbors(X_new.iloc[[0]])
    similar_df = historical_blasts.iloc[indices[0]].copy()
    similar_df["similarity_distance"] = np.round(distances[0], 3)

    return similar_df


def total_cost_per_tonne(blast_params: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate total cost per tonne from blast design through milling.

    Components:
    - Explosive cost (function of powder factor)
    - Drilling cost (function of hole depth, diameter, spacing)
    - Digging cost (function of fragmentation)
    - Hauling cost (function of fragmentation)
    - Crushing cost (function of fragmentation)
    - Milling cost (function of fragmentation)

    Returns: total_cost_per_tonne (USD or BWP) breakdown dictionary
    """
    pf = float(blast_params.get("powder_factor_kg_m3", 0.65))
    bench_h = float(blast_params.get("bench_height_m", 12.0))
    hole_d = float(blast_params.get("hole_diameter_mm", 250.0))
    burden = float(blast_params.get("burden_m", 6.0))
    spacing = float(blast_params.get("spacing_m", 7.0))
    d50 = float(blast_params.get("d50_mm", 220.0))

    rock_vol = burden * spacing * bench_h
    rock_mass_t = max(rock_vol * 2.65, 1.0)

    # 1. Drilling Cost
    drilling_rate_per_m = 12.0 + (hole_d / 100.0) * 8.0
    drilling_cost = float(((bench_h + 1.0) * drilling_rate_per_m) / rock_mass_t)

    # 2. Explosive Cost
    rws = float(blast_params.get("explosive_rws", 100.0))
    exp_price_per_kg = 1.2 + (rws / 100.0) * 0.8
    charge_mass = pf * rock_vol
    explosive_cost = float((charge_mass * exp_price_per_kg + 15.0) / rock_mass_t)

    # 3. Digging Cost (muckpile diggability dependent on fragmentation size d50)
    digging_cost = float(np.clip(0.40 + (d50 / 1000.0) * 0.60, 0.30, 2.50))

    # 4. Hauling Cost (truck fill factor dependent on boulder/fine ratio)
    hauling_cost = float(np.clip(0.80 + (d50 / 1000.0) * 0.40, 0.50, 3.00))

    # 5. Crushing Cost (primary crushing energy requirement)
    crushing_cost = float(np.clip(0.30 + (d50 / 500.0) * 0.50, 0.20, 2.00))

    # 6. Milling Cost (SAG/ball mill specific energy consumption)
    milling_cost = float(np.clip(2.50 + (d50 / 300.0) * 2.00, 1.50, 10.00))

    total = drilling_cost + explosive_cost + digging_cost + hauling_cost + crushing_cost + milling_cost

    return {
        "drilling_cost_usd_t": round(drilling_cost, 2),
        "explosive_cost_usd_t": round(explosive_cost, 2),
        "digging_cost_usd_t": round(digging_cost, 2),
        "hauling_cost_usd_t": round(hauling_cost, 2),
        "crushing_cost_usd_t": round(crushing_cost, 2),
        "milling_cost_usd_t": round(milling_cost, 2),
        "total_cost_usd_t": round(total, 2),
    }


def predict_crusher_throughput(d80_cm: float, ore_hardness: float, crusher_settings: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Predict crusher throughput (t/h) and specific energy (kWh/t) based on blast fragmentation.

    Train on historical data pairing blast fragmentation (D80) with crusher performance.
    This closes the loop between blast design and processing plant economics.

    Parameters:
    -----------
    d80_cm : float
        80% passing fragmentation diameter size in centimeters.
    ore_hardness : float
        Ore hardness index / Bond Work Index (kWh/t).
    crusher_settings : Dict[str, float], optional
        Crusher operational settings (e.g. css_mm: closed side setting mm).

    Returns:
    --------
    Dict[str, float]
        Dictionary containing throughput_tph (t/h) and specific_energy_kwh_t (kWh/t).
    """
    if crusher_settings is None:
        crusher_settings = {"css_mm": 150.0, "power_rating_kw": 400.0}

    css = float(crusher_settings.get("css_mm", 150.0))
    power_kw = float(crusher_settings.get("power_rating_kw", 400.0))

    # Bond Work Index equation for specific energy estimate: W = 10 * Wi * (1/sqrt(P80) - 1/sqrt(F80))
    # F80 in microns = d80_cm * 10,000; P80 in microns = css_mm * 1000
    f80_um = max(d80_cm * 10000.0, 1000.0)
    p80_um = max(css * 1000.0, 1000.0)

    specific_energy = 10.0 * max(ore_hardness, 5.0) * (1.0 / np.sqrt(p80_um) - 1.0 / np.sqrt(f80_um))
    specific_energy = float(np.clip(specific_energy, 0.2, 15.0))

    throughput_tph = float(np.clip(power_kw / max(specific_energy, 0.1), 50.0, 5000.0))

    return {
        "throughput_tph": round(throughput_tph, 2),
        "specific_energy_kwh_t": round(specific_energy, 2),
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
