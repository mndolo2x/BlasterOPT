"""
Ridge Regression Blast Prediction Model Wrapper (`models/sklearn_wrappers/ridge.py`).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.linear_model import Ridge
from models.base import BaseBlastModel, ModelMetadata


class RidgeBlastModel(BaseBlastModel):
    """
    Linear Ridge Regression baseline model for blast predictions.
    """

    def __init__(self, model_name: str = "RidgeBlastModel", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name=model_name, config=config)
        alpha = self.config.get("alpha", 1.0)
        self.model = Ridge(alpha=alpha)

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="ridge",
            display_name="Ridge Linear Regression",
            model_type="sklearn",
            description="Regularized linear regression baseline model.",
            version="1.0.0",
            author="BlastOpt Botswana Team",
            tags=["linear", "baseline"]
        )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], **kwargs) -> "RidgeBlastModel":
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.DataFrame) else y
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if not self.is_fitted:
            n_samples = X_arr.shape[0] if X_arr.ndim > 1 else 1
            preds = np.tile([220.0, 4.20, 110.0, 4.80], (n_samples, 1))
        else:
            preds = np.maximum(0.01, self.model.predict(X_arr))

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(preds, columns=["d50_mm", "ppv_mms", "flyrock_m", "cost_usd"])
        return preds
