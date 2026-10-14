"""
Data Ingestion and Schema Normalization for GeoResponse MWD datasets.
"""

import os
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from src.georesponse.config import (
    COLUMN_ALIASES,
    MINIMAL_REQUIRED_MWD_COLUMNS,
    OPTIONAL_MWD_COLUMNS,
    TARGET_COLUMN,
    DATASET_LICENSE_NOTICE,
)


def normalize_mwd_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalizes column names, converts units, checks numerical consistency, and validates MWD telemetry.

    Returns:
        (normalized_df, validation_report_dict)
    """
    df_norm = df.copy()
    report: Dict[str, Any] = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "missing_required_columns": [],
        "available_optional_columns": [],
        "missing_optional_columns": [],
        "unit_conversions": [],
        "missing_values_per_col": {},
        "class_distribution": {},
        "provenance_hash": "",
        "license_accepted": True,
        "is_eligible_for_training": True,
        "warnings": [],
        "errors": [],
    }

    # 1. Map column aliases to canonical names
    mapped_columns = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        if canonical_name in df_norm.columns:
            continue
        for alias in aliases:
            if alias in df_norm.columns and alias != canonical_name:
                df_norm.rename(columns={alias: canonical_name}, inplace=True)
                mapped_columns[alias] = canonical_name
                break

    if mapped_columns:
        report["unit_conversions"].append(f"Mapped column aliases: {mapped_columns}")

    # 2. Penetration rate normalization (m/min -> m/hr)
    has_m_hr = "penetration_rate_m_hr" in df_norm.columns
    has_m_min = "penetration_rate_m_min" in df_norm.columns

    if has_m_min and not has_m_hr:
        df_norm["penetration_rate_m_hr"] = df_norm["penetration_rate_m_min"] * 60.0
        report["unit_conversions"].append("Converted 'penetration_rate_m_min' to 'penetration_rate_m_hr' (multiplied by 60.0).")
        has_m_hr = True
    elif has_m_hr and has_m_min:
        # Check numerical consistency
        calc_hr = df_norm["penetration_rate_m_min"] * 60.0
        diff = np.abs(df_norm["penetration_rate_m_hr"] - calc_hr)
        max_diff = float(diff.max())
        if max_diff > 1.0:
            err_msg = f"Conflicting penetration rate columns detected: 'penetration_rate_m_hr' and 'penetration_rate_m_min' deviate by up to {max_diff:.2f} m/hr."
            report["errors"].append(err_msg)
            report["is_eligible_for_training"] = False
            return df_norm, report

    # 3. Column existence & missing checks
    for req_col in MINIMAL_REQUIRED_MWD_COLUMNS:
        if req_col not in df_norm.columns:
            report["missing_required_columns"].append(req_col)

    if report["missing_required_columns"]:
        err_msg = f"Missing required MWD columns for model training: {report['missing_required_columns']}"
        report["errors"].append(err_msg)
        report["is_eligible_for_training"] = False

    for opt_col in OPTIONAL_MWD_COLUMNS:
        if opt_col in df_norm.columns:
            report["available_optional_columns"].append(opt_col)
        else:
            report["missing_optional_columns"].append(opt_col)

    # 4. Check missing values in present columns
    all_present_cols = [c for c in (MINIMAL_REQUIRED_MWD_COLUMNS + OPTIONAL_MWD_COLUMNS + [TARGET_COLUMN]) if c in df_norm.columns]
    for col in all_present_cols:
        n_missing = int(df_norm[col].isnull().sum())
        report["missing_values_per_col"][col] = n_missing
        if n_missing > 0:
            report["warnings"].append(f"Column '{col}' contains {n_missing} missing values ({n_missing/len(df):.1%}).")

    # 5. Target class distribution and label quality
    if TARGET_COLUMN in df_norm.columns:
        # Drop rows where target label is missing for class distribution count
        valid_targets = df_norm[TARGET_COLUMN].dropna()
        counts = valid_targets.value_counts().to_dict()
        report["class_distribution"] = {str(k): int(v) for k, v in counts.items()}
        if len(counts) < 2:
            report["warnings"].append("Dataset contains fewer than 2 distinct rock type labels.")
            report["is_eligible_for_training"] = False

    # 6. Compute provenance SHA-256 fingerprint
    present_cols_for_hash = [c for c in MINIMAL_REQUIRED_MWD_COLUMNS if c in df_norm.columns]
    if present_cols_for_hash:
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(df_norm[present_cols_for_hash], index=True).values.tobytes()
        ).hexdigest()[:16]
        report["provenance_hash"] = fingerprint

    return df_norm, report


def validate_mwd_dataset(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Wrapper around normalize_mwd_schema returning boolean eligibility flag and report.
    """
    df_norm, report = normalize_mwd_schema(df)
    is_valid = report["is_eligible_for_training"] and len(report["errors"]) == 0
    return is_valid, report


def load_and_validate_csv(filepath_or_buffer: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Reads MWD CSV file and executes schema normalization & validation report pipeline.
    """
    df = pd.read_csv(filepath_or_buffer)
    df_norm, report = normalize_mwd_schema(df)
    report["license_notice"] = DATASET_LICENSE_NOTICE
    return df_norm, report
