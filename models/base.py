"""
Base Abstract Model Interface (`models.base`).
Defines standard interface required for all machine learning blast models in BlastOpt Botswana.
"""

import os
import joblib
import logging
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class BaseBlastModel(ABC):
    """
    Abstract Base Class for all ML/Physics blast prediction models.
    All models registered in the ModelRegistry must implement this contract.
    """

    def __init__(self, model_name: str = "BaseBlastModel", config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "BaseBlastModel":
        """
        Trains the model on input features X and target outputs Y.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions for input features X.
        Returns predicted array with columns: [d50_mm, ppv_mms, flyrock_m, cost_usd].
        """
        pass

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """
        Evaluates R2, RMSE, and MAE scores across target outputs.
        """
        preds = self.predict(X)

        ss_res = np.sum((Y - preds) ** 2, axis=0)
        ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
        ss_tot = np.where(ss_tot < 1e-5, 1.0, ss_tot)
        r2_vec = 1.0 - (ss_res / ss_tot)

        rmse_vec = np.sqrt(np.mean((Y - preds) ** 2, axis=0))
        mae_vec = np.mean(np.abs(Y - preds), axis=0)

        return {
            "mean_r2": float(np.mean(r2_vec)),
            "mean_rmse": float(np.mean(rmse_vec)),
            "mean_mae": float(np.mean(mae_vec)),
            "d50_r2": float(r2_vec[0]) if Y.shape[1] > 0 else 0.0,
            "ppv_r2": float(r2_vec[1]) if Y.shape[1] > 1 else 0.0,
        }

    def save(self, filepath: str) -> str:
        """
        Saves model instance to file path.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved {self.model_name} to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "BaseBlastModel":
        """
        Loads model instance from file path.
        """
        model = joblib.load(filepath)
        logger.info(f"Loaded model from {filepath}")
        return model
