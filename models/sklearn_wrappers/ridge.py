"""
Ridge Regression Blast Prediction Model Wrapper (`models/sklearn_wrappers/ridge.py`).
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge
from models.base import BaseBlastModel


class RidgeBlastModel(BaseBlastModel):
    """
    Linear Ridge Regression baseline model for blast predictions.
    """

    def __init__(self, model_name: str = "RidgeBlastModel", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name=model_name, config=config)
        alpha = self.config.get("alpha", 1.0)
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeBlastModel":
        self.model.fit(X, Y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            n_samples = X.shape[0] if X.ndim > 1 else 1
            return np.tile([220.0, 4.20, 110.0, 4.80], (n_samples, 1))
        preds = self.model.predict(X)
        return np.maximum(0.01, preds)
