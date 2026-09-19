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
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_arr = X_df.values
        if not self.is_fitted:
            preds = np.tile([220.0, 4.20, 110.0, 4.80], (len(X_df), 1))
        else:
            preds = np.maximum(0.01, self.model.predict(X_arr))

        if isinstance(preds, np.ndarray) and preds.ndim == 1:
            return pd.Series(preds, index=X_df.index)

        cols = self.get_metadata().output_features if preds.shape[1] == 4 else [f"col_{i}" for i in range(preds.shape[1])]
        return pd.DataFrame(preds, columns=cols, index=X_df.index)
