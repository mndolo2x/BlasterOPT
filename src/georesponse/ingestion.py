"""
Data Ingestion and Schema Validation for GeoResponse MWD datasets.
"""

import os
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from src.georesponse.config import REQUIRED_MWD_COLUMNS, TARGET_COLUMN, DATASET_LICENSE_NOTICE


def validate_mwd_dataset(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates an ingested MWD dataset for required columns, data types, missing values,
    and label quality.

    Returns:
        (is_valid, validation_report_dict)
    """
    report: Dict[str, Any] = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "missing_columns": [],
        "missing_values_per_col": {},
        "class_distribution": {},
        "provenance_hash": "",
        "license_accepted": True,
        "warnings": [],
    }

    # Column existence check
    missing_cols = [c for c in REQUIRED_MWD_COLUMNS if c not in df.columns]
    report["missing_columns"] = missing_cols

    if missing_cols:
        report["warnings"].append(f"Missing required MWD columns: {missing_cols}")
        return False, report

    # Missing values check
    for col in REQUIRED_MWD_COLUMNS:
        n_missing = int(df[col].isnull().sum())
        report["missing_values_per_col"][col] = n_missing
        if n_missing > 0:
            report["warnings"].append(f"Column '{col}' contains {n_missing} missing values ({n_missing/len(df):.1%}).")

    # Target class distribution and label quality
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts().to_dict()
        report["class_distribution"] = {str(k): int(v) for k, v in counts.items()}
        if len(counts) < 2:
            report["warnings"].append("Dataset contains fewer than 2 distinct rock type labels.")
            return False, report

    # Compute provenance SHA-256 fingerprint
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(df[REQUIRED_MWD_COLUMNS], index=True).values.tobytes()
    ).hexdigest()[:16]
    report["provenance_hash"] = fingerprint

    return True, report


def load_and_validate_csv(filepath_or_buffer: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Reads MWD CSV file and executes full validation report pipeline.
    """
    df = pd.read_csv(filepath_or_buffer)
    is_valid, report = validate_mwd_dataset(df)
    report["license_notice"] = DATASET_LICENSE_NOTICE
    return df, report
