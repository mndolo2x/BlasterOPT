"""
Random Forest Blast Prediction Model Wrapper (`models/sklearn_wrappers/random_forest.py`).
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from models.base import BaseBlastModel


class RandomForestBlastModel(BaseBlastModel):
    """
    Multi-output Random Forest regressor for blast outcome predictions.
    """

    def __init__(self, model_name: str = "RandomForestBlastModel", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name=model_name, config=config)
        n_estimators = self.config.get("n_estimators", 100)
        max_depth = self.config.get("max_depth", 12)
        random_state = self.config.get("random_state", 42)

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RandomForestBlastModel":
        self.model.fit(X, Y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            n_samples = X.shape[0] if X.ndim > 1 else 1
            return np.tile([220.0, 4.20, 110.0, 4.80], (n_samples, 1))
        preds = self.model.predict(X)
        return np.maximum(0.01, preds)
