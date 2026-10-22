"""
Data Ingestion and Schema Normalization for GeoResponse MWD datasets.
"""

import os
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from src.georesponse.config import (
    normalize_header_name,
    COLUMN_ALIASES,
    SUGGESTED_ALIASES,
    MINIMAL_REQUIRED_MWD_COLUMNS,
    OPTIONAL_MWD_COLUMNS,
    TARGET_COLUMN,
    DATASET_LICENSE_NOTICE,
)


def normalize_mwd_schema(
    df: pd.DataFrame,
    user_confirmed_mappings: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalizes column names, converts units, checks numerical consistency, and validates MWD telemetry.

    Parameters:
        df: Input DataFrame.
        user_confirmed_mappings: Dict mapping original column name -> canonical field name confirmed by user.

    Returns:
        (normalized_df, validation_report_dict)
    """
    df_norm = df.copy()
    report: Dict[str, Any] = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "original_columns": list(df.columns),
        "mapped_columns": {},
        "suggested_mappings": {},
        "unknown_columns": [],
        "duplicate_mappings": [],
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

    user_mappings = user_confirmed_mappings or {}

    # Map normalized alias list for canonical fields
    normalized_aliases: Dict[str, List[str]] = {}
    for canonical, alias_list in COLUMN_ALIASES.items():
        normalized_aliases[canonical] = [normalize_header_name(a) for a in alias_list]

    normalized_suggested: Dict[str, List[str]] = {}
    for canonical, sug_list in SUGGESTED_ALIASES.items():
        normalized_suggested[canonical] = [normalize_header_name(s) for s in sug_list]

    # Analyze headers
    canonical_to_orig: Dict[str, List[str]] = {}
    rename_dict: Dict[str, str] = {}

    for orig_col in df.columns:
        norm_col = normalize_header_name(orig_col)

        # 1. User confirmed mapping takes priority
        if orig_col in user_mappings:
            target_canonical = user_mappings[orig_col]
            rename_dict[orig_col] = target_canonical
            canonical_to_orig.setdefault(target_canonical, []).append(orig_col)
            report["mapped_columns"][orig_col] = f"{target_canonical} (User Confirmed)"
            continue

        # 2. Check unambiguous aliases (or exact normalized match)
        matched_canonical = None
        for canonical, alias_norms in normalized_aliases.items():
            if norm_col == canonical or norm_col in alias_norms:
                matched_canonical = canonical
                break

        if matched_canonical:
            rename_dict[orig_col] = matched_canonical
            canonical_to_orig.setdefault(matched_canonical, []).append(orig_col)
            report["mapped_columns"][orig_col] = matched_canonical
            continue

        # 3. Check semantically ambiguous suggestions (require confirmation)
        suggested_canonical = None
        for canonical, sug_norms in normalized_suggested.items():
            if norm_col in sug_norms:
                suggested_canonical = canonical
                break

        if suggested_canonical:
            report["suggested_mappings"][orig_col] = suggested_canonical
            report["unknown_columns"].append(orig_col)
            continue

        # 4. Unknown column
        report["unknown_columns"].append(orig_col)

    # Detect duplicate mappings (multiple original columns mapping to same canonical target)
    for canonical, orig_list in canonical_to_orig.items():
        if len(orig_list) > 1:
            # Check if values are identical
            is_identical = True
            first_vals = df[orig_list[0]]
            for other in orig_list[1:]:
                if not first_vals.equals(df[other]):
                    is_identical = False
                    break

            dup_info = {
                "canonical": canonical,
                "original_columns": orig_list,
                "is_identical": is_identical,
            }
            report["duplicate_mappings"].append(dup_info)

            if not is_identical:
                err_msg = (
                    f"Conflicting duplicate mapping detected: Columns {orig_list} all map to '{canonical}' "
                    "with conflicting values. User resolution required."
                )
                report["errors"].append(err_msg)
                report["is_eligible_for_training"] = False
            else:
                report["warnings"].append(
                    f"Duplicate columns {orig_list} map to '{canonical}' with identical values."
                )

    # Perform column renaming if no conflicting duplicate errors
    if report["is_eligible_for_training"]:
        df_norm.rename(columns=rename_dict, inplace=True)

    # Record unit conversions / header mappings
    if report["mapped_columns"]:
        report["unit_conversions"].append(f"Mapped column headers: {report['mapped_columns']}")

    # Penetration rate normalization (m/min -> m/hr)
    has_m_hr = "penetration_rate_m_hr" in df_norm.columns
    has_m_min = "penetration_rate_m_min" in df_norm.columns

    if has_m_min and not has_m_hr:
        df_norm["penetration_rate_m_hr"] = pd.to_numeric(df_norm["penetration_rate_m_min"], errors="coerce") * 60.0
        report["unit_conversions"].append("Converted 'penetration_rate_m_min' to 'penetration_rate_m_hr' (multiplied by 60.0).")
        has_m_hr = True
    elif has_m_hr and has_m_min:
        calc_hr = pd.to_numeric(df_norm["penetration_rate_m_min"], errors="coerce") * 60.0
        hr_vals = pd.to_numeric(df_norm["penetration_rate_m_hr"], errors="coerce")
        diff = np.abs(hr_vals - calc_hr)
        max_diff = float(diff.max()) if not diff.empty else 0.0
        if max_diff > 1.0:
            err_msg = f"Conflicting penetration rate columns detected: 'penetration_rate_m_hr' and 'penetration_rate_m_min' deviate by up to {max_diff:.2f} m/hr."
            report["errors"].append(err_msg)
            report["is_eligible_for_training"] = False

    # Required column checks
    for req_col in MINIMAL_REQUIRED_MWD_COLUMNS:
        if req_col not in df_norm.columns:
            report["missing_required_columns"].append(req_col)

    if report["missing_required_columns"]:
        err_msg = f"Missing required MWD columns for model training: {report['missing_required_columns']}"
        report["errors"].append(err_msg)
        report["is_eligible_for_training"] = False

    # Optional column checks
    for opt_col in OPTIONAL_MWD_COLUMNS:
        if opt_col in df_norm.columns:
            report["available_optional_columns"].append(opt_col)
        else:
            report["missing_optional_columns"].append(opt_col)

    # Missing value checks in present MWD columns
    all_present_cols = [c for c in (MINIMAL_REQUIRED_MWD_COLUMNS + OPTIONAL_MWD_COLUMNS + [TARGET_COLUMN]) if c in df_norm.columns]
    for col in all_present_cols:
        n_missing = int(df_norm[col].isnull().sum())
        report["missing_values_per_col"][col] = n_missing
        if n_missing > 0:
            report["warnings"].append(f"Column '{col}' contains {n_missing} missing values ({n_missing/len(df):.1%}).")

    # Target class distribution
    if TARGET_COLUMN in df_norm.columns:
        valid_targets = df_norm[TARGET_COLUMN].dropna()
        counts = valid_targets.value_counts().to_dict()
        report["class_distribution"] = {str(k): int(v) for k, v in counts.items()}
        if len(counts) < 2:
            report["warnings"].append("Dataset contains fewer than 2 distinct rock type labels.")
            report["is_eligible_for_training"] = False

    # Compute provenance SHA-256 fingerprint
    present_cols_for_hash = [c for c in MINIMAL_REQUIRED_MWD_COLUMNS if c in df_norm.columns]
    if present_cols_for_hash:
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(df_norm[present_cols_for_hash], index=True).values.tobytes()
        ).hexdigest()[:16]
        report["provenance_hash"] = fingerprint

    return df_norm, report


def validate_mwd_dataset(
    df: pd.DataFrame,
    user_confirmed_mappings: Optional[Dict[str, str]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Wrapper around normalize_mwd_schema returning boolean eligibility flag and report.
    """
    df_norm, report = normalize_mwd_schema(df, user_confirmed_mappings=user_confirmed_mappings)
    is_valid = report["is_eligible_for_training"] and len(report["errors"]) == 0
    return is_valid, report


def load_and_validate_csv(
    filepath_or_buffer: Any,
    user_confirmed_mappings: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Reads MWD CSV file and executes schema normalization & validation report pipeline.
    """
    df = pd.read_csv(filepath_or_buffer)
    df_norm, report = normalize_mwd_schema(df, user_confirmed_mappings=user_confirmed_mappings)
    report["license_notice"] = DATASET_LICENSE_NOTICE
    return df_norm, report
