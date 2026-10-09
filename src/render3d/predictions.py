"""
Integrate trained models with the 3D design.
"""
import pandas as pd
from typing import Dict, Any
from src.render3d.hole_pattern import HolePattern


def pattern_to_features(pattern: HolePattern) -> pd.DataFrame:
    """
    Convert a HolePattern into the 10-feature row expected by the models.
    """
    return pd.DataFrame([{
        "burden_m": pattern.burden_m,
        "spacing_m": pattern.spacing_m,
        "powder_factor_kg_m3": pattern.powder_factor_kg_m3,
        "stemming_m": pattern.stemming_m,
        "rock_factor_A": pattern.hole_angle_deg,  # placeholder, override below
        "hole_depth_m": pattern.hole_depth_m,
        "hole_diameter_mm": 165.0,  # override below
        "max_charge_per_delay_kg": pattern.hole_depth_m,  # override below
        "explosive_rws": 115.0,
        "bench_height_m": pattern.bench_height_m,
    }])


def predict_pattern(
    model: Any,
    pattern: HolePattern,
    extra_features: Dict[str, Any],
) -> Dict[str, float]:
    """
    Run the trained model on the pattern features.

    Args:
        model: trained model with .predict(X) method
        pattern: HolePattern
        extra_features: dict with rock_factor_A, hole_diameter_mm,
                        max_charge_per_delay_kg, explosive_rws

    Returns:
        dict with fragmentation_d80_cm, vibration_ppv_mms, airblast_db
    """
    features = pattern_to_features(pattern)
    for key, value in extra_features.items():
        if key in features.columns:
            features[key] = value

    preds = model.predict(features)
    if isinstance(preds, pd.DataFrame):
        d80 = float(preds["fragmentation_d80_cm"].iloc[0]) if "fragmentation_d80_cm" in preds.columns else float(preds.iloc[0, 0])
        ppv = float(preds["vibration_ppv_mms"].iloc[0]) if "vibration_ppv_mms" in preds.columns else float(preds.iloc[0, 1])
        airblast = float(preds["airblast_db"].iloc[0]) if "airblast_db" in preds.columns else float(preds.iloc[0, 2])
    else:
        d80 = float(preds[0][0])
        ppv = float(preds[0][1])
        airblast = float(preds[0][2])

    return {
        "fragmentation_d80_cm": d80,
        "vibration_ppv_mms": ppv,
        "airblast_db": airblast,
    }
