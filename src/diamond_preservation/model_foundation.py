"""
Predictive Model Foundation and Training Contract for Diamond Preservation.

Defines dataset readiness checks, independent blast group validation, and future predictor contract interfaces.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from src.diamond_preservation.models import (
    DiamondObservation,
    DiamondDatasetReadinessReport,
)


def evaluate_dataset_readiness(
    observations: List[DiamondObservation],
    db_path: Optional[str] = None,
) -> DiamondDatasetReadinessReport:
    """
    Evaluates historical diamond observations against model training contract criteria.
    Ensures independent blast IDs, label completeness, and domain coverage.
    """
    if not observations:
        return DiamondDatasetReadinessReport(
            total_observations=0,
            total_distinct_blasts=0,
            usable_independent_blasts=0,
            label_completeness_pct=0.0,
            domain_coverage={},
            stage_breakdown={},
            is_ready_for_training=False,
            readiness_warnings=["No diamond observations recorded in database."],
        )

    df_obs = pd.DataFrame([o.model_dump() if hasattr(o, "model_dump") else o.dict() for o in observations])

    total_obs = len(df_obs)
    distinct_blasts = df_obs["blast_id"].nunique()

    # Known damage label completeness
    known_labels = df_obs[df_obs["damage_type"] != "unknown"]
    usable_blasts = known_labels["blast_id"].nunique()
    label_completeness = (len(known_labels) / total_obs) * 100.0 if total_obs > 0 else 0.0

    stage_counts = df_obs["stage"].value_counts().to_dict()

    warnings: List[str] = []
    if usable_blasts < 10:
        warnings.append(f"Insufficient independent blasts with confirmed labels ({usable_blasts} < 10 required).")

    if label_completeness < 70.0:
        warnings.append(f"Low label completeness ({label_completeness:.1f}% < 70% required).")

    is_ready = len(warnings) == 0

    return DiamondDatasetReadinessReport(
        total_observations=total_obs,
        total_distinct_blasts=distinct_blasts,
        usable_independent_blasts=usable_blasts,
        label_completeness_pct=round(label_completeness, 1),
        domain_coverage={"Jwaneng Kimberlite": distinct_blasts},
        stage_breakdown=stage_counts,
        is_ready_for_training=is_ready,
        readiness_warnings=warnings,
    )
