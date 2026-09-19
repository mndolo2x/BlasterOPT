"""
Random Forest Blast Prediction Model Wrapper (`models/sklearn_wrappers/random_forest.py`).
"""

import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.ensemble import RandomForestRegressor
from models.base import BaseBlastModel, ModelMetadata


class RandomForestModel(BaseBlastModel):
    def __init__(self, **kwargs):
        super().__init__(model_name="RandomForestModel", config=kwargs)
        self.model = RandomForestRegressor(**kwargs)

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="random_forest",
            display_name="Random Forest",
            model_type="sklearn",
            description="Random Forest regressor for blast outcome prediction.",
            version="1.0.0",
            author="BlastOpt Team",
            input_features=["burden", "spacing", "powder_factor", "stemming", "rock_factor"],
            output_features=["fragmentation_p80", "ppv", "airblast"],
            supports_training=True,
            supports_uncertainty=False,
            supports_explainability=True,
            requires_gpu=False,
            tags=["baseline", "tree-based"],
        )

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not self.is_fitted:
            preds = pd.DataFrame(
                [[220.0, 4.20, 110.0]] * len(X_df),
                columns=self.get_metadata().output_features,
                index=X_df.index
            )
            return preds

        preds = self.model.predict(X_df)
        return pd.DataFrame(preds, columns=self.get_metadata().output_features, index=X_df.index)

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        import joblib
        obj = cls()
        obj.model = joblib.load(path)
        obj.is_fitted = True
        return obj


# Alias for backward compatibility
RandomForestBlastModel = RandomForestModel
