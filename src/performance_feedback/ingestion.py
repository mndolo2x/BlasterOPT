"""
Actual Results Ingestion & Validation Pipeline for Blast Performance.

Supports CSV / Excel actual results file parsing, schema normalization, candidate column mapping,
row-level validation, duplicate/conflict detection, sample CSV/Excel template generation,
and preservation of unknown columns.
"""

import io
import re
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from src.performance_feedback.models import ActualMeasurement

# Expected target columns and their canonical aliases (supporting wide format test CSV names)
TARGET_ALIASES: Dict[str, List[str]] = {
    "d50_mm": ["d50", "d50_mm", "p50", "p50_mm", "d50_fragment_mm", "fragmentation_d50", "mean_fragment_size", "actual_p50_mm", "p50_mm"],
    "fragmentation_d80_cm": ["d80", "d80_cm", "p80", "p80_cm", "p80_mm", "fragmentation_d80", "d80_mm", "actual_p80_mm", "actual_p80_cm"],
    "ppv_mms": ["ppv", "ppv_mms", "vibration_ppv", "vibration_ppv_mms", "peak_particle_velocity", "vibration", "actual_ppv_mm_s", "actual_ppv_mms"],
    "airblast_db": ["airblast", "airblast_db", "airblast_dbl", "noise_db", "overpressure_db", "actual_airblast_db"],
    "flyrock_m": ["flyrock", "flyrock_m", "flyrock_distance", "max_flyrock_m", "actual_flyrock_m"],
    "cost_per_tonne_usd": ["cost", "cost_usd", "cost_per_tonne", "cost_per_tonne_usd", "actual_cost", "actual_cost_usd"],
}

TARGET_UNITS: Dict[str, str] = {
    "d50_mm": "mm",
    "fragmentation_d80_cm": "cm",
    "ppv_mms": "mm/s",
    "airblast_db": "dB",
    "flyrock_m": "m",
    "cost_per_tonne_usd": "$/t",
}

VALIDATION_BOUNDS: Dict[str, Tuple[float, float]] = {
    "d50_mm": (5.0, 1000.0),
    "fragmentation_d80_cm": (1.0, 150.0),
    "ppv_mms": (0.01, 100.0),
    "airblast_db": (50.0, 160.0),
    "flyrock_m": (0.0, 1000.0),
    "cost_per_tonne_usd": (0.50, 50.0),
}


def normalize_header(header: str) -> str:
    """Normalizes header string by lowercasing, stripping, and replacing spaces/hyphens with underscores."""
    clean = str(header).strip().lower()
    clean = re.sub(r"[\s\-\.]+", "_", clean)
    return clean


def generate_sample_template(target_metric: str = "d50_mm", format_type: str = "wide") -> pd.DataFrame:
    """Generates sample DataFrame template for blast register, actual results (wide/long), or predictions."""
    if target_metric == "blast_register":
        return pd.DataFrame([
            {
                "blast_id": "BLASTOPT-TEST-001",
                "site": "Jwaneng",
                "operation_type": "open_pit",
                "area": "Cut 8",
                "bench_id": "BENCH_JWA_15S",
                "blast_date": "2026-10-20T12:00:00Z",
                "rock_type": "Kimberlite",
                "geological_domain": "Kimberlite_Hard",
                "record_status": "EXECUTED",
                "notes": "Sample test blast for feedback loop",
            }
        ])
    elif target_metric == "prediction_records":
        return pd.DataFrame([
            {
                "blast_id": "BLASTOPT-TEST-001",
                "prediction_run_id": "RUN_BLASTOPT_001_D50",
                "prediction_timestamp": "2026-10-20T10:00:00Z",
                "model_name": "GA-ANN Model",
                "model_version": "v1.0",
                "target_name": "d50_mm",
                "predicted_value": 240.0,
                "unit": "mm",
                "provenance": "synthetic_test",
            }
        ])
    elif format_type == "long":
        return pd.DataFrame([
            {
                "blast_id": "BLASTOPT-TEST-001",
                "measurement_id": "MEAS_BLASTOPT_001_D50",
                "measurement_timestamp": "2026-10-20T14:00:00Z",
                "target_name": "d50_mm",
                "actual_value": 245.0,
                "unit": "mm",
                "measurement_method": "UAV Photogrammetry",
                "measurement_source": "Drone Survey Team",
                "review_status": "Awaiting Review",
            }
        ])
    else:  # wide format default
        return pd.DataFrame([
            {
                "blast_id": "BLASTOPT-TEST-001",
                "measurement_timestamp": "2026-10-20T14:00:00Z",
                "actual_p80_mm": 32.0,
                "actual_p50_mm": 245.0,
                "actual_p20_mm": 110.0,
                "actual_oversize_pct": 3.5,
                "actual_fines_pct": 12.0,
                "actual_ppv_mm_s": 4.2,
                "actual_airblast_db": 114.5,
                "actual_avg_hole_depth_m": 15.2,
                "actual_explosive_kg": 320.0,
                "actual_cost_usd": 4.85,
            }
        ])


def identify_candidate_mappings(columns: List[str]) -> Dict[str, str]:
    """
    Inspects DataFrame header names and suggests candidate target field mappings.
    Returns dict mapping input column name -> target_metric.
    """
    mappings: Dict[str, str] = {}
    for col in columns:
        norm_col = normalize_header(col)

        # Check blast_id
        if norm_col in ["blast_id", "blastid", "blast_name", "pattern_id"]:
            mappings[col] = "blast_id"
            continue

        # Check target metrics
        matched_target = None
        for target, aliases in TARGET_ALIASES.items():
            if norm_col == target or norm_col in aliases:
                matched_target = target
                break

        if matched_target:
            mappings[col] = matched_target
        elif norm_col in ["measurement_timestamp", "date", "time", "timestamp"]:
            mappings[col] = "measurement_timestamp"
        elif norm_col in ["method", "measurement_method", "instrument"]:
            mappings[col] = "measurement_method"
        elif norm_col in ["sensor_distance", "sensor_distance_m", "distance_m", "distance"]:
            mappings[col] = "sensor_distance_m"
        else:
            mappings[col] = col  # Preserve as unknown/custom column

    return mappings


def validate_actual_results_df(
    df: pd.DataFrame, column_mappings: Dict[str, str]
) -> Tuple[List[ActualMeasurement], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validates uploaded actual results DataFrame against required schema, numeric bounds, and units.

    Returns:
    --------
    Tuple[List[ActualMeasurement], List[Dict[str, Any]], List[Dict[str, Any]]]
        (valid_measurements, row_validation_errors, duplicate_conflicts)
    """
    valid_measurements: List[ActualMeasurement] = []
    row_errors: List[Dict[str, Any]] = []
    duplicate_conflicts: List[Dict[str, Any]] = []

    # Rename columns based on confirmed user mappings
    df_renamed = df.rename(columns=column_mappings)

    if "blast_id" not in df_renamed.columns:
        row_errors.append({
            "row_index": 0,
            "error_type": "MISSING_MANDATORY_COLUMN",
            "message": "Mandatory column 'blast_id' is missing from file.",
        })
        return valid_measurements, row_errors, duplicate_conflicts

    # Identify whether file is long format or wide format
    norm_cols = [normalize_header(c) for c in df_renamed.columns]
    is_long_format = ("actual_value" in norm_cols or "measured_value" in norm_cols) and ("target_name" in norm_cols or "target_metric" in norm_cols)

    # Normalize column names for direct matching
    col_norm_map = {c: normalize_header(c) for c in df_renamed.columns}

    # Map long format column names if present
    if is_long_format:
        for orig, norm in col_norm_map.items():
            if norm in ["actual_value", "measured_value", "value"]:
                df_renamed = df_renamed.rename(columns={orig: "measured_value"})
            elif norm in ["target_name", "target_metric", "target", "metric"]:
                df_renamed = df_renamed.rename(columns={orig: "target_metric"})
            elif norm in ["measurement_id", "meas_id"]:
                df_renamed = df_renamed.rename(columns={orig: "measurement_id"})
            elif norm in ["measurement_method", "method"]:
                df_renamed = df_renamed.rename(columns={orig: "measurement_method"})

    present_targets = [t for t in TARGET_ALIASES.keys() if t in df_renamed.columns]

    seen_keys: set = set()

    for idx, row in df_renamed.iterrows():
        blast_id = str(row.get("blast_id", "")).strip()
        if not blast_id or blast_id.lower() in ["nan", "null", "none", ""]:
            row_errors.append({
                "row_index": idx + 1,
                "error_type": "INVALID_BLAST_ID",
                "message": "Blast ID is blank or missing.",
            })
            continue

        timestamp = str(row.get("measurement_timestamp", datetime.now(timezone.utc).isoformat()))

        if is_long_format:
            target_raw = str(row.get("target_metric", "")).strip()
            # Map target_raw to canonical metric key if possible
            target_norm = normalize_header(target_raw)
            matched_canonical = None
            for can_k, aliases in TARGET_ALIASES.items():
                if target_norm == can_k or target_norm in aliases:
                    matched_canonical = can_k
                    break
            target_metric = matched_canonical if matched_canonical else target_raw
            val_raw = row.get("measured_value")
            targets_to_eval = [(target_metric, val_raw)]
        else:
            targets_to_eval = [(t, row.get(t)) for t in present_targets if pd.notna(row.get(t))]

        for target_metric, val_raw in targets_to_eval:

            # Check duplicate in same upload batch
            dedup_key = (blast_id, target_metric)
            if dedup_key in seen_keys:
                duplicate_conflicts.append({
                    "row_index": idx + 1,
                    "blast_id": blast_id,
                    "target_metric": target_metric,
                    "conflicting_value": val_raw,
                    "message": f"Duplicate measurement detected for blast_id={blast_id}, target={target_metric}.",
                })
                continue
            seen_keys.add(dedup_key)

            # Convert numeric value
            try:
                meas_val = float(val_raw)
            except (ValueError, TypeError):
                row_errors.append({
                    "row_index": idx + 1,
                    "blast_id": blast_id,
                    "target_metric": target_metric,
                    "error_type": "NON_NUMERIC_VALUE",
                    "message": f"Measured value '{val_raw}' for {target_metric} is not a valid number.",
                })
                continue

            # Bounds sanity check
            min_b, max_b = VALIDATION_BOUNDS.get(target_metric, (0.001, 10000.0))
            if not (min_b <= meas_val <= max_b):
                row_errors.append({
                    "row_index": idx + 1,
                    "blast_id": blast_id,
                    "target_metric": target_metric,
                    "error_type": "OUT_OF_BOUNDS_VALUE",
                    "message": f"Measured value {meas_val} for {target_metric} is outside expected physical bounds [{min_b}, {max_b}].",
                })
                continue

            # Construct ActualMeasurement
            unit = TARGET_UNITS.get(target_metric, "unit")
            method = str(row.get("measurement_method", "Field Sensor")) if pd.notna(row.get("measurement_method")) else "Field Sensor"
            inst_id = str(row.get("instrument_id", "")) if pd.notna(row.get("instrument_id")) else None

            try:
                sensor_dist = float(row.get("sensor_distance_m")) if pd.notna(row.get("sensor_distance_m")) else None
            except (ValueError, TypeError):
                sensor_dist = None

            meas = ActualMeasurement(
                measurement_id=f"MEAS_{blast_id}_{target_metric}_{int(datetime.now(timezone.utc).timestamp())}",
                blast_id=blast_id,
                measurement_timestamp=timestamp,
                target_metric=target_metric,
                measured_value=meas_val,
                unit=unit,
                measurement_method=method,
                instrument_id=inst_id,
                sensor_distance_m=sensor_dist,
                source_filename=str(row.get("source_filename", "uploaded_file.csv")),
                raw_attributes={k: str(v) for k, v in row.to_dict().items() if k not in TARGET_ALIASES and k not in ["blast_id", target_metric]},
            )
            valid_measurements.append(meas)

    return valid_measurements, row_errors, duplicate_conflicts
