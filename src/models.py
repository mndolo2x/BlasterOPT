"""
Machine Learning Models Module for BlastOpt Botswana.

Handles model training, multi-output regression, cross-validation evaluation,
GridSearchCV hyperparameter tuning, feature importance extraction, and persistence.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object

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
    "pf_burden_interaction",
    "spacing_stemming_interaction",
]

TARGET_COLS = ["d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"]


def get_model_instance(model_type: str = "random_forest", seed: int = 42):
    """
    Factory function returning a configured machine learning regressor instance.

    Parameters:
    -----------
    model_type : str, default="random_forest"
        Type of algorithm: "random_forest", "xgboost", or "ridge".
    seed : int, default=42
        Random state seed for reproducibility.

    Returns:
    --------
    BaseEstimator
        Scikit-learn or XGBoost regressor instance.
    """
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

    def tune_and_save_fragmentation_model(
        self,
        df: pd.DataFrame,
        save_filepath: str = "models/best_fragmentation_model.pkl",
        cv_folds: int = 5,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Executes GridSearchCV hyperparameter tuning on the best model for fragmentation prediction (d50_mm)
        and saves the tuned model to disk.
        """
        X = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
        y = df["d50_mm"].values

        if self.model_type == "xgboost":
            base_model = XGBRegressor(random_state=self.seed, n_jobs=-1)
            param_grid = {
                "n_estimators": [50, 100, 150],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.08, 0.15],
            }
        else:
            base_model = RandomForestRegressor(random_state=self.seed, n_jobs=-1)
            param_grid = {
                "n_estimators": [50, 100, 150],
                "max_depth": [8, 12, 16],
                "min_samples_split": [2, 5],
            }

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring="r2",
            n_jobs=-1,
        )
        grid_search.fit(X, y)

        best_estimator = grid_search.best_estimator_
        preds = best_estimator.predict(X)

        best_metrics = {
            "R2": float(r2_score(y, preds)),
            "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
            "MAE": float(mean_absolute_error(y, preds)),
            "best_params": grid_search.best_params_,
        }

        # Update pipeline model for d50_mm
        self.models["d50_mm"] = best_estimator
        self.feature_names = list(X.columns)

        # Save to specified path
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        joblib.dump(best_estimator, save_filepath)

        return best_estimator, best_metrics

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

        # Also save fragmentation model to models/best_fragmentation_model.pkl
        if "d50_mm" in self.models:
            pkl_path = os.path.join(dir_path, "best_fragmentation_model.pkl")
            joblib.dump(self.models["d50_mm"], pkl_path)

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


if HAS_TORCH:
    class GAANNModel(nn.Module):
        """
        Multi-output GA-ANN for simultaneous prediction of fragmentation,
        ground vibration, and airblast at Jwaneng Mine.

        Architecture: 10-70-25-3
        Source: Jwaneng Mine, 120 production blasts
        Performance: R² = 0.910 (frag), 0.925 (vib), 0.967 (airblast)
        """
        def __init__(self, input_size=10):
            super().__init__()
            self.hidden1 = nn.Linear(input_size, 70)
            self.hidden2 = nn.Linear(70, 25)
            self.output = nn.Linear(25, 3)  # fragmentation, vibration, airblast
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.hidden1(x))
            x = self.relu(self.hidden2(x))
            return self.output(x)


MODEL_REGISTRY = {
    "ga_ann_jwaneng": {
        "architecture": "10-70-25-3",
        "optimizer": "genetic_algorithm",
        "outputs": ["fragmentation", "vibration", "airblast"],
        "source": "Jwaneng Mine, 120 blasts",
        "performance": {"fragmentation": 0.910, "vibration": 0.925, "airblast": 0.967}
    },
    "ann_rf_ensemble_jwaneng": {
        "architecture": "ensemble",
        "outputs": ["fragmentation", "vibration"],
        "source": "Jwaneng Mine, 120 blasts",
        "performance": {"fragmentation": 0.956, "vibration": 0.930}
    },
    "pso_ann_orapa": {
        "architecture": "7-65-30-1",
        "optimizer": "particle_swarm",
        "outputs": ["fragmentation"],
        "source": "Orapa Mine, 120 blasts",
        "performance": {"fragmentation": 0.86}
    },
    "airblast_minimizer": {
        "architecture": "ANN",
        "outputs": ["airblast"],
        "source": "Debswana open-pit",
        "key_sensitivity": {"stemming": "high", "spacing": "low"}
    }
}
