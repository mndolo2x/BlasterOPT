"""
Data Ingestion, Validation, Cleaning, and Feature Engineering Module.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

# Expected required columns in raw dataset
REQUIRED_COLUMNS = [
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

TARGET_COLUMNS = ["d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"]

VALIDATION_RANGES = {
    "rock_factor_A": (1.0, 20.0),
    "bench_height_m": (2.0, 50.0),
    "hole_diameter_mm": (50.0, 500.0),
    "burden_m": (0.5, 15.0),
    "spacing_m": (0.5, 20.0),
    "stemming_m": (0.1, 15.0),
    "charge_mass_per_hole_kg": (1.0, 3000.0),
    "powder_factor_kg_m3": (0.05, 5.0),
    "max_charge_per_delay_kg": (1.0, 10000.0),
    "monitoring_distance_m": (10.0, 5000.0),
    "d50_mm": (1.0, 2000.0),
    "ppv_mms": (0.01, 500.0),
    "flyrock_m": (0.0, 2000.0),
    "cost_per_tonne_usd": (0.01, 100.0),
}


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw blasting dataset from a CSV file.

    Parameters:
    -----------
    filepath : str
        Relative or absolute path to the raw CSV file.

    Returns:
    --------
    pd.DataFrame
        Loaded raw dataset as a Pandas DataFrame.

    Raises:
    -------
    FileNotFoundError
        If the specified file path does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at path: {filepath}")
    df = pd.read_csv(filepath)
    return df


def validate_data_schema(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validates if input DataFrame contains all required blasting parameter columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to validate.

    Returns:
    --------
    Tuple[bool, list]
        A tuple containing (is_valid boolean, list of missing column names).
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    is_valid = len(missing_cols) == 0
    return is_valid, missing_cols


def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw DataFrame:
    - Drops duplicate rows
    - Handles missing values with domain median
    - Clips extreme non-physical outliers based on VALIDATION_RANGES
    """
    cleaned_df = df.copy()

    # Drop duplicate records
    cleaned_df = cleaned_df.drop_duplicates()

    # Clip values to physically realistic boundaries
    for col, (vmin, vmax) in VALIDATION_RANGES.items():
        if col in cleaned_df.columns:
            # Coerce numeric values
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
            # Fill NaNs with median if present
            if cleaned_df[col].isna().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            cleaned_df[col] = cleaned_df[col].clip(lower=vmin, upper=vmax)

    return cleaned_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers mining & blasting domain features:
    - Spacing to Burden Ratio (spacing_m / burden_m)
    - Stiffness Ratio (bench_height_m / burden_m)
    - Charge-to-Burden Ratio (charge_mass_per_hole_kg / burden_m)
    - Scaled Distance (monitoring_distance_m / sqrt(max_charge_per_delay_kg))
    - Powder Factor check / recalculation
    - Energy Factor estimate
    - Interaction Terms:
      1. Powder Factor * Burden (pf_burden_interaction)
      2. Spacing * Stemming (spacing_stemming_interaction)
    """
    df_feat = df.copy()

    # Spacing to Burden Ratio
    df_feat["spacing_burden_ratio"] = df_feat["spacing_m"] / np.maximum(df_feat["burden_m"], 0.1)

    # Stiffness Ratio (Bench Height / Burden)
    df_feat["stiffness_ratio"] = df_feat["bench_height_m"] / np.maximum(df_feat["burden_m"], 0.1)

    # Stemming to Burden Ratio
    df_feat["stemming_burden_ratio"] = df_feat["stemming_m"] / np.maximum(df_feat["burden_m"], 0.1)

    # Scaled Distance (USBM Ground Vibration)
    df_feat["scaled_distance"] = df_feat["monitoring_distance_m"] / np.sqrt(
        np.maximum(df_feat["max_charge_per_delay_kg"], 1.0)
    )

    # Energy Factor Approximation (MJ / m3 or MJ / tonne assuming RWS if available)
    rws = df_feat["explosive_rws"] if "explosive_rws" in df_feat.columns else 100.0
    # Base ANFO energy = 3.7 MJ/kg
    df_feat["energy_factor_mj_m3"] = df_feat["powder_factor_kg_m3"] * 3.7 * (rws / 100.0)

    # Area per hole (Burden * Spacing)
    df_feat["hole_area_m2"] = df_feat["burden_m"] * df_feat["spacing_m"]

    # Requested Interaction Features:
    # 1. Interaction term between powder factor and burden
    df_feat["pf_burden_interaction"] = df_feat["powder_factor_kg_m3"] * df_feat["burden_m"]

    # 2. Interaction term between spacing and stemming length
    df_feat["spacing_stemming_interaction"] = df_feat["spacing_m"] * df_feat["stemming_m"]

    return df_feat


def load_real_blast_data(
    filepath: str, anomaly_log_path: str = "data/processed/data_anomalies.log"
) -> pd.DataFrame:
    """
    Loads, validates, and cleans real mine blasting logs from CSV when a data-sharing agreement is active.

    Real Data vs. Synthetic Data Domain Context:
    -------------------------------------------
    Synthetic blast datasets rely on empirical Kuz-Ram equations and simplified geology assumptions.
    In contrast, real mine production logs (e.g. from Debswana Jwaneng or Orapa pits) reflect site-specific
    geological heterogeneity, joint plane orientations, bench water conditions, explosive product degradation,
    and actual measured fragmentation or seismograph PPV waveforms. Consuming real blast data is essential
    for production model calibration, reducing generalization error on site.

    Parameters:
    -----------
    filepath : str
        File path to real blast data CSV file.
    anomaly_log_path : str, default="data/processed/data_anomalies.log"
        File path to record out-of-bounds anomaly entries and missing value logs.

    Returns:
    --------
    pd.DataFrame
        Validated, cleaned, and feature-engineered real blast dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Real blast data file not found at: {filepath}")

    raw_df = pd.read_csv(filepath)

    # Prepare directory for anomaly log file
    os.makedirs(os.path.dirname(anomaly_log_path) if os.path.dirname(anomaly_log_path) else ".", exist_ok=True)

    anomalies = []

    # 1. Check schema
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing_cols:
        anomalies.append(f"CRITICAL: Missing required schema column(s): {', '.join(missing_cols)}")

    cleaned_df = raw_df.copy()

    # 2. Check missing values and impute with median
    for col in REQUIRED_COLUMNS + [c for c in TARGET_COLUMNS if c in cleaned_df.columns]:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
            num_missing = cleaned_df[col].isna().sum()
            if num_missing > 0:
                median_val = cleaned_df[col].median()
                anomalies.append(
                    f"MISSING VALUE: Column '{col}' had {num_missing} missing entry/entries. Imputed with median = {median_val:.2f}."
                )
                cleaned_df[col] = cleaned_df[col].fillna(median_val)

    # 3. Check physical range validation bounds
    for col, (vmin, vmax) in VALIDATION_RANGES.items():
        if col in cleaned_df.columns:
            out_of_bounds = cleaned_df[(cleaned_df[col] < vmin) | (cleaned_df[col] > vmax)]
            if len(out_of_bounds) > 0:
                anomalies.append(
                    f"OUT OF RANGE: Column '{col}' contained {len(out_of_bounds)} row(s) outside physical range [{vmin}, {vmax}]. Clipped."
                )
                cleaned_df[col] = cleaned_df[col].clip(lower=vmin, upper=vmax)

    # Write anomaly report to log file
    with open(anomaly_log_path, "a", encoding="utf-8") as f_log:
        f_log.write(f"\n--- Data Ingestion Anomaly Report for {filepath} ---\n")
        if anomalies:
            for log_entry in anomalies:
                f_log.write(f"[{pd.Timestamp.now()}] {log_entry}\n")
        else:
            f_log.write(f"[{pd.Timestamp.now()}] No anomalies detected in real dataset.\n")

    # Perform feature engineering
    engineered_df = engineer_features(cleaned_df)

    return engineered_df


def prepare_ingested_dataset(
    raw_df: pd.DataFrame, save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Complete ETL pipeline: Clean, validate, engineer features, and optionally save.
    """
    is_valid, missing = validate_data_schema(raw_df)
    if not is_valid:
        raise ValueError(f"Dataset missing required columns: {missing}")

    cleaned = clean_and_preprocess(raw_df)
    engineered = engineer_features(cleaned)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        engineered.to_csv(save_path, index=False)

    return engineered
