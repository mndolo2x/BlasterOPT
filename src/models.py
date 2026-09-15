"""
Machine Learning Models Module for BlastOpt Botswana.

Handles model training, multi-output regression, cross-validation evaluation,
feature importance extraction, and persistence.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

FEATURE_COLS = [
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
    "spacing_burden_ratio",
    "stiffness_ratio",
    "stemming_burden_ratio",
    "scaled_distance",
    "energy_factor_mj_m3",
]

TARGET_COLS = ["d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"]


def get_model_instance(model_type: str = "random_forest", seed: int = 42):
    """Factory function returning regressor instance."""
    if model_type == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=seed, max_depth=12, n_jobs=-1)
    elif model_type == "xgboost":
        return XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=6, random_state=seed, n_jobs=-1)
    elif model_type == "ridge":
        return Ridge(alpha=1.0)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class BlastMLPipeline:
    """
    Multi-target machine learning model suite for blasting performance predictions.
    """

    def __init__(self, model_type: str = "random_forest", seed: int = 42):
        self.model_type = model_type
        self.seed = seed
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = []

    def train_and_evaluate(
        self, df: pd.DataFrame, cv_folds: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Trains separate estimators for each target variable and computes K-Fold CV metrics.
        """
        # Ensure engineered features are present
        X = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
        self.feature_names = list(X.columns)

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)

        for target in TARGET_COLS:
            if target not in df.columns:
                continue

            y = df[target].values
            estimator = get_model_instance(self.model_type, self.seed)

            # Cross validation metrics
            r2_scores = []
            rmse_scores = []
            mae_scores = []

            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                m = get_model_instance(self.model_type, self.seed)
                m.fit(X_tr, y_tr)
                preds = m.predict(X_val)

                r2_scores.append(r2_score(y_val, preds))
                rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
                mae_scores.append(mean_absolute_error(y_val, preds))

            # Fit on full dataset
            estimator.fit(X, y)
            self.models[target] = estimator

            self.metrics[target] = {
                "R2_mean": float(np.mean(r2_scores)),
                "R2_std": float(np.std(r2_scores)),
                "RMSE_mean": float(np.mean(rmse_scores)),
                "MAE_mean": float(np.mean(mae_scores)),
            }

        return self.metrics

    def predict(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for all targets on new input features.
        """
        X = df_input[[c for c in self.feature_names if c in df_input.columns]].copy()

        # Fill missing features if any
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0

        X = X[self.feature_names]
        preds_df = pd.DataFrame(index=df_input.index)

        for target, model in self.models.items():
            preds_df[f"pred_{target}"] = model.predict(X)

        return preds_df

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Returns feature importances per target (for Random Forest and XGBoost).
        """
        importance_dict = {"feature": self.feature_names}

        for target, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importance_dict[target] = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance_dict[target] = np.abs(model.coef_)

        return pd.DataFrame(importance_dict)

    def save_models(self, dir_path: str = "models/"):
        """Saves trained models and metadata to directory."""
        os.makedirs(dir_path, exist_ok=True)
        save_dict = {
            "model_type": self.model_type,
            "seed": self.seed,
            "models": self.models,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
        }
        filepath = os.path.join(dir_path, f"blast_models_{self.model_type}.joblib")
        joblib.dump(save_dict, filepath)
        return filepath

    @classmethod
    def load_models(cls, filepath: str):
        """Loads trained models from joblib file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        data = joblib.load(filepath)
        instance = cls(model_type=data["model_type"], seed=data["seed"])
        instance.models = data["models"]
        instance.metrics = data["metrics"]
        instance.feature_names = data["feature_names"]
        return instance
