"""
Random Forest Blast Prediction Model Wrapper (`models/sklearn_wrappers/random_forest.py`).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.ensemble import RandomForestRegressor
from models.base import BaseBlastModel, ModelMetadata


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

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="random_forest",
            display_name="Random Forest Regressor",
            model_type="sklearn",
            description="Multi-output decision tree ensemble model.",
            version="1.0.0",
            author="BlastOpt Botswana Team",
            supports_explainability=True,
            tags=["tree", "ensemble", "baseline"]
        )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], **kwargs) -> "RandomForestBlastModel":
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

    def explain(self, X: Union[pd.DataFrame, np.ndarray]) -> Optional[Dict[str, Any]]:
        if not self.is_fitted:
            return None
        importances = self.model.feature_importances_
        feature_names = self.get_metadata().input_features
        return {"feature_importances": dict(zip(feature_names[:len(importances)], importances.tolist()))}
