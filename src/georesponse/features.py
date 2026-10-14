"""
Feature Engineering and Specific Energy Computation for GeoResponse MWD telemetry.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from src.georesponse.config import FEATURE_COLUMNS


def compute_mwd_specific_energy(df: pd.DataFrame, bit_diameter_mm: float = 250.0) -> pd.DataFrame:
    """
    Calculates Teale (1965) MWD Mechanical Specific Energy (MSE) in MJ/m3:
        MSE = (WOB / Area) + (120 * pi * RPM * Torque) / (Area * ROP)

    For standard MWD telemetry without explicit RPM:
        MSE_approx = (weight_on_bit_kg * 9.81 / Area) + (torque_nm / (Area * penetration_rate_m_hr / 3600.0))
    """
    df_feat = df.copy()
    area_m2 = np.pi * ((bit_diameter_mm / 2000.0) ** 2)

    rop = np.maximum(df_feat["penetration_rate_m_hr"].values, 0.1)
    wob = np.maximum(df_feat["weight_on_bit_kg"].values, 0.0)
    torque = np.maximum(df_feat["torque_nm"].values, 0.0)

    # MSE components (converted to MJ/m3)
    thrust_component = (wob * 9.81) / area_m2 / 1e6
    rotational_component = (torque * 2.0 * np.pi) / (area_m2 * (rop / 3600.0) * 1e6)

    df_feat["mwd_specific_energy_mj_m3"] = np.clip(thrust_component + rotational_component, 1.0, 500.0)
    df_feat["rop_torque_ratio"] = rop / np.maximum(torque, 1.0)

    return df_feat


def compute_mwd_features(df: pd.DataFrame, bit_diameter_mm: float = 250.0) -> pd.DataFrame:
    """
    Alias for compute_mwd_specific_energy.
    """
    return compute_mwd_specific_energy(df, bit_diameter_mm=bit_diameter_mm)


def prepare_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Engineers MSE features and extracts clean numeric feature matrix X.
    """
    df_engineered = compute_mwd_specific_energy(df)
    X = df_engineered[[c for c in FEATURE_COLUMNS if c in df_engineered.columns]].copy()
    X = X.fillna(X.median())
    return X, list(X.columns)
