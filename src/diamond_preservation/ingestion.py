"""
Diamond Population Ingestion and Validation Engine.

Parses CSV/Excel diamond observation files, normalizes columns, validates stone counts,
carats, stage classifications, and damage attributions, and exports validated CSV records.
"""

import io
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
from src.diamond_preservation.models import DiamondObservation

OBSERVATION_STAGES = [
    "before_blasting", "post_blast", "excavation_loading",
    "transport", "crushing", "milling", "recovery", "unknown"
]

DAMAGE_TYPES = [
    "none", "chipping", "cracking", "breakage_cleavage", "severe_fragmentation", "unknown"
]

DAMAGE_ATTRIBUTIONS = [
    "blast_induced", "crusher_induced", "natural_flaw", "unknown"
]


def generate_observation_template() -> pd.DataFrame:
    """Generates sample CSV template for diamond population observations."""
    return pd.DataFrame([
        {
            "sample_lot_id": "LOT_JWA_2026_01",
            "blast_id": "BLASTOPT-TEST-001",
            "stage": "post_blast",
            "observation_method": "UAV Optical Scanner / Manual Sort",
            "stone_size_class": "+11 DTC Gem",
            "stone_count": 4,
            "total_carats": 18.5,
            "stone_quality_category": "Gem",
            "damage_type": "chipping",
            "damage_attribution_cause": "blast_induced",
            "attribution_confidence": 0.85,
            "observer_id": "GEOLOGIST_BW_01",
            "observation_date": datetime.now(timezone.utc).isoformat(),
            "notes": "Observed corner chipping post-blast in bench muckpile sample.",
        },
        {
            "sample_lot_id": "LOT_JWA_2026_02",
            "blast_id": "BLASTOPT-TEST-001",
            "stage": "crushing",
            "observation_method": "Primary Crusher Discharge Sampler",
            "stone_size_class": "Special (>10.8ct)",
            "stone_count": 1,
            "total_carats": 14.2,
            "stone_quality_category": "Gem",
            "damage_type": "breakage_cleavage",
            "damage_attribution_cause": "crusher_induced",
            "attribution_confidence": 0.90,
            "observer_id": "METALLURGIST_BW_02",
            "observation_date": datetime.now(timezone.utc).isoformat(),
            "notes": "Cleavage plane breakage observed post-primary crushing.",
        },
    ])


def validate_diamond_observations_df(
    df: pd.DataFrame
) -> Tuple[List[DiamondObservation], List[Dict[str, Any]]]:
    """
    Validates uploaded diamond population observation records against required schema, numeric bounds, and valid categories.

    Returns:
    --------
    Tuple[List[DiamondObservation], List[Dict[str, Any]]]
        (valid_observations, row_errors)
    """
    valid_obs: List[DiamondObservation] = []
    errors: List[Dict[str, Any]] = []

    norm_cols = {c: str(c).strip().lower() for c in df.columns}
    df_renamed = df.rename(columns=norm_cols)

    for idx, row in df_renamed.iterrows():
        b_id = str(row.get("blast_id", "")).strip()
        lot_id = str(row.get("sample_lot_id", row.get("lot_id", f"LOT_{idx+1}"))).strip()

        if not b_id or b_id.lower() in ["nan", "null", "none", ""]:
            errors.append({
                "row_index": idx + 1,
                "error_type": "MISSING_BLAST_ID",
                "message": "Mandatory column 'blast_id' is missing or empty.",
            })
            continue

        stage = str(row.get("stage", "post_blast")).strip().lower()
        if stage not in OBSERVATION_STAGES:
            stage = "unknown"

        dmg_type = str(row.get("damage_type", "none")).strip().lower()
        if dmg_type not in DAMAGE_TYPES:
            dmg_type = "unknown"

        attribution = str(row.get("damage_attribution_cause", row.get("attribution", "unknown"))).strip().lower()
        if attribution not in DAMAGE_ATTRIBUTIONS:
            attribution = "unknown"

        try:
            stone_cnt = int(row.get("stone_count", 1))
            carats = float(row.get("total_carats", row.get("carats", 0.0)))
        except (ValueError, TypeError):
            errors.append({
                "row_index": idx + 1,
                "error_type": "NON_NUMERIC_METRICS",
                "message": f"Row {idx+1}: Invalid stone_count or total_carats numeric format.",
            })
            continue

        if stone_cnt <= 0 or carats <= 0.0:
            errors.append({
                "row_index": idx + 1,
                "error_type": "INVALID_NUMERIC_BOUNDS",
                "message": f"Row {idx+1}: stone_count ({stone_cnt}) and total_carats ({carats}) must be > 0.",
            })
            continue

        obs_id = str(row.get("observation_id", f"OBS_{b_id}_{lot_id}_{idx+1}"))
        method = str(row.get("observation_method", "Visual Sort"))
        size_cls = str(row.get("stone_size_class", "+11 DTC"))
        quality_cat = str(row.get("stone_quality_category", "Gem"))
        try:
            confidence = float(row.get("attribution_confidence", 0.8))
        except (ValueError, TypeError):
            confidence = 0.8

        notes = str(row.get("notes", "")) if pd.notna(row.get("notes")) else None

        valid_obs.append(DiamondObservation(
            observation_id=obs_id,
            blast_id=b_id,
            sample_lot_id=lot_id,
            stage=stage,
            observation_method=method,
            stone_size_class=size_cls,
            stone_count=stone_cnt,
            total_carats=carats,
            stone_quality_category=quality_cat,
            damage_type=dmg_type,
            damage_attribution_cause=attribution,
            attribution_confidence=confidence,
            observer_id=str(row.get("observer_id", "GEOLOGIST_BW_01")),
            notes=notes,
        ))

    return valid_obs, errors
