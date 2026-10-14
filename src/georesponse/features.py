"""
Feature Engineering and Specific Energy Computation for GeoResponse MWD telemetry.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from src.georesponse.config import ALL_FEATURE_COLUMNS


def compute_mwd_features(df: pd.DataFrame, bit_diameter_mm: float = 250.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely computes MWD derived features (Teale Specific Energy, ROP/Torque ratio, Feed/Percussion ratio)
    based on available sensor telemetry without unhandled KeyErrors or fake measurement injection.

    Returns:
        (df_engineered, feature_engineering_report)
    """
    df_feat = df.copy()
    report: Dict[str, Any] = {
        "computed_features": [],
        "skipped_features": [],
        "skip_reasons": {},
    }

    # 1. Base Sensor Columns Present
    has_rop = "penetration_rate_m_hr" in df_feat.columns
    has_torque = "torque_nm" in df_feat.columns
    has_wob = "weight_on_bit_kg" in df_feat.columns
    has_percussion = "percussion_pressure_bar" in df_feat.columns
    has_feed = "feed_pressure_bar" in df_feat.columns

    area_m2 = np.pi * ((bit_diameter_mm / 2000.0) ** 2)

    # 2. Teale Mechanical Specific Energy (MSE) Computation
    if has_rop and has_torque and has_wob:
        rop = np.maximum(df_feat["penetration_rate_m_hr"].values, 0.1)
        wob = np.maximum(df_feat["weight_on_bit_kg"].values, 0.0)
        torque = np.maximum(df_feat["torque_nm"].values, 0.0)

        thrust_component = (wob * 9.81) / area_m2 / 1e6
        rotational_component = (torque * 2.0 * np.pi) / (area_m2 * (rop / 3600.0) * 1e6)

        df_feat["mwd_specific_energy_mj_m3"] = np.clip(thrust_component + rotational_component, 1.0, 500.0)
        report["computed_features"].append("mwd_specific_energy_mj_m3")
    else:
        missing_inputs = []
        if not has_rop:
            missing_inputs.append("penetration_rate_m_hr")
        if not has_torque:
            missing_inputs.append("torque_nm")
        if not has_wob:
            missing_inputs.append("weight_on_bit_kg")
        report["skipped_features"].append("mwd_specific_energy_mj_m3")
        report["skip_reasons"]["mwd_specific_energy_mj_m3"] = f"Missing required telemetry columns: {missing_inputs}"

    # 3. ROP to Torque Ratio
    if has_rop and has_torque:
        rop = np.maximum(df_feat["penetration_rate_m_hr"].values, 0.1)
        torque = np.maximum(df_feat["torque_nm"].values, 0.0)
        df_feat["rop_torque_ratio"] = rop / np.maximum(torque, 1.0)
        report["computed_features"].append("rop_torque_ratio")
    else:
        missing_inputs = []
        if not has_rop:
            missing_inputs.append("penetration_rate_m_hr")
        if not has_torque:
            missing_inputs.append("torque_nm")
        report["skipped_features"].append("rop_torque_ratio")
        report["skip_reasons"]["rop_torque_ratio"] = f"Missing required telemetry columns: {missing_inputs}"

    # 4. Feed to Percussion Ratio
    if has_feed and has_percussion:
        feed = np.maximum(df_feat["feed_pressure_bar"].values, 0.0)
        percussion = np.maximum(df_feat["percussion_pressure_bar"].values, 0.1)
        df_feat["feed_percussion_ratio"] = feed / percussion
        report["computed_features"].append("feed_percussion_ratio")

    return df_feat, report


def compute_mwd_specific_energy(df: pd.DataFrame, bit_diameter_mm: float = 250.0) -> pd.DataFrame:
    """
    Backward-compatible wrapper returning engineered DataFrame only.
    """
    df_feat, _ = compute_mwd_features(df, bit_diameter_mm=bit_diameter_mm)
    return df_feat


def prepare_feature_matrix(
    df: pd.DataFrame,
    required_features: List[str] = None,
    bit_diameter_mm: float = 250.0,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Engineers available features and extracts numeric matrix X.

    Returns:
        (X_matrix, feature_columns_list, feature_engineering_report)
    """
    df_engineered, report = compute_mwd_features(df, bit_diameter_mm=bit_diameter_mm)

    if required_features is not None:
        target_features = [c for c in required_features if c in df_engineered.columns]
    else:
        target_features = [c for c in ALL_FEATURE_COLUMNS if c in df_engineered.columns]

    X = df_engineered[target_features].copy()
    if not X.empty:
        X = X.fillna(X.median())

    return X, list(X.columns), report
