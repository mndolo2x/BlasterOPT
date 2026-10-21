"""
Reviewed Dataset Export & Model Retraining Integration.

Generates target-specific CSV exports for approved dataset records with cryptographic
ExportManifest provenance records. Ensures blast-level group partitioning during retraining
to prevent data leakage across train/validation splits.
"""

import io
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from src.performance_feedback.models import ExportManifest, DatasetReviewRecord
from src.performance_feedback.db import (
    list_dataset_reviews,
    get_actual_measurements_for_blast,
    get_blast_record,
    save_export_manifest,
)


def export_approved_dataset(
    target_metric: str,
    db_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, ExportManifest, bytes]:
    """
    Exports target-specific CSV dataset containing only approved records.

    Returns:
    --------
    Tuple[pd.DataFrame, ExportManifest, bytes]
        (export_df, manifest, csv_bytes)
    """
    kwargs = {}
    if db_path:
        kwargs["db_path"] = db_path

    reviews = list_dataset_reviews(target_metric=target_metric, status="Approved", **kwargs)
    approved_blast_ids = [r.blast_id for r in reviews if r.inclusion_status]

    rows: List[Dict[str, Any]] = []
    unit_defs: Dict[str, str] = {}

    for b_id in approved_blast_ids:
        b_rec = get_blast_record(b_id, **kwargs)
        measurements = get_actual_measurements_for_blast(b_id, target_metric=target_metric, **kwargs)

        if not b_rec or not measurements:
            continue

        actual_meas = measurements[0]

        row_dict: Dict[str, Any] = {
            "blast_id": b_id,
            "site_id": b_rec.site_id,
            "operation_type": b_rec.operation_type,
            "location_identifier": b_rec.location_identifier,
            "target_metric": target_metric,
            "target_value": actual_meas.measured_value,
            "target_unit": actual_meas.unit,
        }

        unit_defs["target_value"] = actual_meas.unit

        # Merge planned features
        if b_rec.planned_params:
            for k, v in b_rec.planned_params.items():
                row_dict[k] = v

        # Merge geological context
        if b_rec.geological_context:
            for k, v in b_rec.geological_context.items():
                row_dict[f"geo_{k}"] = v

        rows.append(row_dict)

    df_export = pd.DataFrame(rows)

    if df_export.empty:
        # Fallback empty structure
        df_export = pd.DataFrame(columns=["blast_id", "site_id", "target_metric", "target_value"])

    # Generate CSV bytes
    csv_buf = io.StringIO()
    df_export.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    manifest_hash = ExportManifest.compute_hash(csv_bytes)
    export_id = f"EXP_{target_metric.upper()}_{int(datetime.now(timezone.utc).timestamp())}"

    manifest = ExportManifest(
        export_id=export_id,
        export_timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_version="v1.0.0",
        target_metric=target_metric,
        included_blast_ids=approved_blast_ids,
        approved_record_count=len(df_export),
        feature_schema=list(df_export.columns),
        unit_definitions=unit_defs,
        filters_applied={"status": "Approved", "inclusion_status": True},
        manifest_hash=manifest_hash,
    )

    save_export_manifest(manifest, **kwargs)

    return df_export, manifest, csv_bytes


def prepare_group_split_dataset(
    df: pd.DataFrame, group_col: str = "blast_id", test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Group K-Fold / Group split ensuring observations from the same blast_id
    do not leak across train and test partitions.
    """
    if group_col not in df.columns or df[group_col].nunique() <= 1:
        # Fallback to standard random split if no distinct blast_ids
        np.random.seed(seed)
        shuffled_idx = np.random.permutation(len(df))
        n_test = int(len(df) * test_size)
        return df.iloc[shuffled_idx[n_test:]], df.iloc[shuffled_idx[:n_test]]

    unique_groups = df[group_col].unique()
    np.random.seed(seed)
    shuffled_groups = np.random.permutation(unique_groups)

    n_test_groups = max(1, int(len(unique_groups) * test_size))
    test_groups = set(shuffled_groups[:n_test_groups])

    test_mask = df[group_col].isin(test_groups)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    return train_df, test_df
