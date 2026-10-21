"""
Dedicated Importers for Blast Register, Actual Results (Wide & Long), and Prediction Reference Records.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Literal
from src.performance_feedback.models import (
    BlastRecord,
    PredictionSnapshot,
    ActualMeasurement,
)
from src.performance_feedback.db import (
    save_blast_record,
    get_blast_record,
    save_prediction_snapshot,
    get_prediction_snapshots_for_blast,
    save_actual_measurement,
    save_comparison_result,
    ComparisonResult,
    DatasetReviewRecord,
    save_dataset_review,
)
from src.performance_feedback.ingestion import (
    normalize_header,
    validate_actual_results_df,
    identify_candidate_mappings,
    TARGET_UNITS,
)


def import_blast_records(
    df: pd.DataFrame,
    conflict_mode: Literal["skip", "update", "cancel"] = "skip",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parses and validates Blast Register CSV/Excel uploads.

    Expected columns:
        blast_id, site, operation_type, area, bench_id, blast_date, rock_type, geological_domain, record_status, notes
    """
    col_map = {c: normalize_header(c) for c in df.columns}
    df_norm = df.rename(columns=col_map)

    created_count = 0
    updated_count = 0
    skipped_count = 0
    rejected_count = 0
    errors: List[str] = []

    for idx, row in df_norm.iterrows():
        b_id = str(row.get("blast_id", "")).strip()
        if not b_id or b_id.lower() in ["nan", "null", "none", ""]:
            rejected_count += 1
            errors.append(f"Row {idx+1}: Missing or invalid blast_id")
            continue

        site = str(row.get("site", row.get("site_id", "Jwaneng"))).strip()
        op_type = str(row.get("operation_type", "open_pit")).strip().lower()
        if op_type not in ["open_pit", "underground"]:
            op_type = "open_pit"

        bench = str(row.get("bench_id", row.get("area", row.get("location_identifier", "BENCH_01")))).strip()
        b_date = str(row.get("blast_date", datetime.now(timezone.utc).isoformat()))
        r_type = str(row.get("rock_type", "Kimberlite"))
        g_domain = str(row.get("geological_domain", r_type))
        rec_status = str(row.get("record_status", "EXECUTED")).upper()
        if rec_status not in ["PLANNED", "EXECUTED", "MEASURED", "REVIEWED", "ARCHIVED"]:
            rec_status = "EXECUTED"

        notes = str(row.get("notes", "")) if pd.notna(row.get("notes")) else ""

        # Check conflict in DB
        kwargs = {"db_path": db_path} if db_path else {}
        existing = get_blast_record(b_id, **kwargs)

        if existing:
            if conflict_mode == "cancel":
                return {
                    "status": "cancelled",
                    "created": created_count,
                    "updated": updated_count,
                    "skipped": skipped_count,
                    "rejected": rejected_count,
                    "errors": errors + [f"Cancelled import due to conflicting existing blast_id: {b_id}"],
                }
            elif conflict_mode == "skip":
                skipped_count += 1
                continue
            elif conflict_mode == "update":
                updated_rec = existing.update_snapshot(
                    site_id=site,
                    operation_type=op_type,
                    location_identifier=bench,
                    blast_date=b_date,
                    status=rec_status,
                    geological_context={"rock_type": r_type, "geological_domain": g_domain},
                    notes=notes,
                )
                save_blast_record(updated_rec, **kwargs)
                updated_count += 1
                continue

        # Create new record
        new_rec = BlastRecord(
            blast_id=b_id,
            site_id=site,
            operation_type=op_type,
            location_identifier=bench,
            blast_date=b_date,
            status=rec_status,
            geological_context={"rock_type": r_type, "geological_domain": g_domain},
            notes=notes,
        )
        save_blast_record(new_rec, **kwargs)
        created_count += 1

    return {
        "status": "completed",
        "created": created_count,
        "updated": updated_count,
        "skipped": skipped_count,
        "rejected": rejected_count,
        "errors": errors,
    }


def import_prediction_records(
    df: pd.DataFrame,
    provenance_label: str = "synthetic_test",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parses and imports prediction reference records for Feedback Loop testing.

    Expected columns:
        blast_id, prediction_run_id, prediction_timestamp, model_name, model_version, target_name, predicted_value, unit
    """
    col_map = {c: normalize_header(c) for c in df.columns}
    df_norm = df.rename(columns=col_map)

    created_count = 0
    rejected_count = 0
    errors: List[str] = []

    kwargs = {"db_path": db_path} if db_path else {}

    for idx, row in df_norm.iterrows():
        b_id = str(row.get("blast_id", "")).strip()
        if not b_id or b_id.lower() in ["nan", "null", "none", ""]:
            rejected_count += 1
            errors.append(f"Row {idx+1}: Missing blast_id")
            continue

        p_run_id = str(row.get("prediction_run_id", f"RUN_{b_id}_{idx+1}")).strip()
        p_ts = str(row.get("prediction_timestamp", datetime.now(timezone.utc).isoformat()))
        m_name = str(row.get("model_name", "GA-ANN Model")).strip()
        m_ver = str(row.get("model_version", "v1.0")).strip()
        target = str(row.get("target_name", row.get("target_metric", "d50_mm"))).strip()

        try:
            pred_val = float(row.get("predicted_value", row.get("predicted", 0.0)))
        except (ValueError, TypeError):
            rejected_count += 1
            errors.append(f"Row {idx+1}: Non-numeric predicted_value")
            continue

        unit = str(row.get("unit", TARGET_UNITS.get(target, "unit"))).strip()

        # Ensure BlastRecord exists
        b_rec = get_blast_record(b_id, **kwargs)
        if not b_rec:
            b_rec = BlastRecord(
                blast_id=b_id,
                site_id="Jwaneng",
                operation_type="open_pit",
                location_identifier="BENCH_IMPORT",
                status="PLANNED",
            )
            save_blast_record(b_rec, **kwargs)

        snap = PredictionSnapshot(
            prediction_run_id=p_run_id,
            blast_id=b_id,
            prediction_timestamp=p_ts,
            model_name=m_name,
            model_version=m_ver,
            target_metric=target,
            predicted_value=pred_val,
            unit=unit,
            provenance=provenance_label,
        )
        save_prediction_snapshot(snap, **kwargs)
        created_count += 1

    return {
        "status": "completed",
        "created": created_count,
        "rejected": rejected_count,
        "errors": errors,
    }
