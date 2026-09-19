"""
Base Abstract Model Interface (`models.base`).
Defines standard interface and ModelMetadata schema required for all machine learning blast models.
"""

import os
import joblib
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelMetadata(BaseModel):
    """Pydantic schema for model registration metadata."""
    name: str = Field(..., description="Unique internal model identifier key")
    display_name: str = Field(..., description="User-facing display name for UI dropdowns")
    model_type: str = Field(..., description="Model family type: 'sklearn', 'pytorch', 'physics_informed', 'ensemble', 'calibration'")
    description: str = Field(..., description="Detailed summary description")
    version: str = Field(default="1.0.0", description="Model semantic version")
    author: str = Field(default="BlastOpt Botswana Team", description="Author or organization")
    input_features: List[str] = Field(
        default_factory=lambda: [
            "burden_m", "spacing_m", "hole_diameter_mm", "bench_height_m",
            "stemming_m", "subdrill_m", "powder_factor_kg_m3", "max_charge_per_delay_kg",
            "rock_factor_a", "rmr", "monitoring_distance_m", "explosive_rws"
        ],
        description="Expected input feature names"
    )
    output_features: List[str] = Field(
        default_factory=lambda: ["d50_mm", "ppv_mms", "flyrock_m", "cost_usd"],
        description="Predicted output target names"
    )
    supports_training: bool = Field(default=True, description="Whether the model supports online re-training")
    supports_pipeline_training: bool = Field(default=True, description="Whether model supports single-target cross-validation pipeline training")
    supports_uncertainty: bool = Field(default=False, description="Whether model predicts confidence intervals")
    supports_explainability: bool = Field(default=False, description="Whether model supports feature importance or SHAP explanations")
    requires_gpu: bool = Field(default=False, description="Whether GPU hardware is required")
    tags: List[str] = Field(default_factory=list, description="Categorical tags for filtering")


class BaseBlastModel(ABC):
    """
    Abstract Base Class for all ML/Physics blast prediction models in BlastOpt Botswana.
    All registered models must inherit from BaseBlastModel and implement get_metadata(), fit(), and predict().
    """

    def __init__(self, model_name: str = "BaseBlastModel", config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.is_fitted = False

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> ModelMetadata:
        """Return model metadata for registry and Streamlit UI."""
        pass

    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], **kwargs) -> "BaseBlastModel":
        """Train the model on input features X and target outputs y."""
        pass

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Predict blast outcome targets."""
        pass

    def predict_with_uncertainty(self, X: Union[pd.DataFrame, np.ndarray]) -> Optional[Union[pd.DataFrame, Dict[str, Any]]]:
        """Optional: predict outcomes with uncertainty bounds. Override if supported."""
        return None

    def explain(self, X: Union[pd.DataFrame, np.ndarray]) -> Optional[Dict[str, Any]]:
        """Optional: return feature importance or SHAP feature explanations. Override if supported."""
        return None

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], Y: Union[pd.DataFrame, np.ndarray]) -> Dict[str, float]:
        """Evaluates R2, RMSE, and MAE scores across target outputs."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        Y_arr = Y.values if isinstance(Y, pd.DataFrame) else Y

        preds_res = self.predict(X_arr)
        preds = preds_res.values if isinstance(preds_res, pd.DataFrame) else preds_res

        ss_res = np.sum((Y_arr - preds) ** 2, axis=0)
        ss_tot = np.sum((Y_arr - np.mean(Y_arr, axis=0)) ** 2, axis=0)
        ss_tot = np.where(ss_tot < 1e-5, 1.0, ss_tot)
        r2_vec = 1.0 - (ss_res / ss_tot)

        rmse_vec = np.sqrt(np.mean((Y_arr - preds) ** 2, axis=0))
        mae_vec = np.mean(np.abs(Y_arr - preds), axis=0)

        return {
            "mean_r2": float(np.mean(r2_vec)),
            "mean_rmse": float(np.mean(rmse_vec)),
            "mean_mae": float(np.mean(mae_vec)),
            "d50_r2": float(r2_vec[0]) if Y_arr.shape[1] > 0 else 0.0,
            "ppv_r2": float(r2_vec[1]) if Y_arr.shape[1] > 1 else 0.0,
        }

    def save(self, path: str) -> None:
        """Save model instance to disk path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved {self.model_name} to {path}")

    @classmethod
    def load(cls, path: str) -> "BaseBlastModel":
        """Load model instance from disk path."""
        model = joblib.load(path)
        logger.info(f"Loaded model from {path}")
        return model
