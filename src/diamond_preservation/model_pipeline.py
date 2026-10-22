"""
Predictive Model Pipeline and Contract Interface for Diamond Preservation.

Defines feature contract, dataset readiness validation, group-based train/test splitting,
and typed serving interface for future calibrated diamond breakage predictors.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class DiamondBreakagePredictorInterface:
    """
    Typed interface for diamond breakage predictor models.
    Returns explicit unavailable status when no calibrated model is trained/promoted.
    """

    def __init__(self, model_version: str = "v0.1-uncalibrated"):
        self.model_version = model_version
        self.is_calibrated = False

    def is_trained(self) -> bool:
        return self.is_calibrated

    def predict(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts diamond breakage risk.
        If uncalibrated, returns explicit unavailable status.
        """
        if not self.is_calibrated:
            return {
                "status": "unavailable",
                "message": "Diamond-breakage prediction unavailable — no validated model is configured.",
                "predicted_breakage_probability": None,
                "confidence_bounds": None,
                "model_version": self.model_version,
                "is_calibrated": False,
            }

        # Placeholder for calibrated model inference
        return {
            "status": "success",
            "message": "Model inference complete",
            "predicted_breakage_probability": 0.12,
            "confidence_bounds": {"lower_95": 0.08, "upper_95": 0.18},
            "model_version": self.model_version,
            "is_calibrated": True,
        }


def validate_training_contract(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates training dataset against mandatory features and outcome labels.
    """
    mandatory_features = [
        "blast_id", "powder_factor_kg_m3", "burden_m", "spacing_m",
        "hole_diameter_mm", "bench_height_m", "max_charge_per_delay_kg",
        "rock_type", "damage_type"
    ]

    missing = [c for c in mandatory_features if c not in df.columns]

    if missing:
        return {
            "valid": False,
            "missing_columns": missing,
            "message": f"Dataset violates training contract. Missing required columns: {missing}",
        }

    return {
        "valid": True,
        "missing_columns": [],
        "record_count": len(df),
        "distinct_blasts": df["blast_id"].nunique(),
        "message": "Dataset satisfies training contract schema requirements.",
    }


def prepare_grouped_training_split(
    df: pd.DataFrame, group_col: str = "blast_id", test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into train and test partitions by blast_id to prevent data leakage.
    """
    if group_col not in df.columns or df[group_col].nunique() <= 1:
        np.random.seed(seed)
        idx = np.random.permutation(len(df))
        n_test = int(len(df) * test_size)
        return df.iloc[idx[n_test:]], df.iloc[idx[:n_test]]

    unique_blasts = df[group_col].unique()
    np.random.seed(seed)
    shuffled_blasts = np.random.permutation(unique_blasts)

    n_test_blasts = max(1, int(len(unique_blasts) * test_size))
    test_blasts = set(shuffled_blasts[:n_test_blasts])

    test_mask = df[group_col].isin(test_blasts)
    return df[~test_mask].copy(), df[test_mask].copy()
