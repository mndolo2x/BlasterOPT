"""
GeoResponse wrapper for the registry (models.custom.geo_response_model.GeoResponseModel).

Wraps the lightweight GeoResponseClassifier implemented under src/georesponse so the
app/tests can import a standard BaseBlastModel subclass.
"""
import os
import numpy as np
import pandas as pd
from typing import Any

from models.base import BaseBlastModel, ModelMetadata
from src.georesponse.models import GeoResponseClassifier
from src.georesponse.config import CONFIDENCE_THRESHOLD, ALL_FEATURE_COLUMNS


class GeoResponseModel(BaseBlastModel):
    def __init__(self, model_type: str = "random_forest", random_state: int = 42, **kwargs: Any):
        super().__init__(model_name="GeoResponseModel", config=kwargs)
        self.model_type = model_type
        self.random_state = random_state
        self.classifier = GeoResponseClassifier(model_type=self.model_type, random_state=self.random_state)
        # feature_names_ will be set when fitting the underlying classifier
        self.feature_names_ = []

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="geo_response",
            display_name="GeoResponse (MWD Rock Classification)",
            model_type="custom",
            description="MWD rock-type classification research prototype (GeoResponse).",
            version="0.1.0",
            author="BlastOpt Team",
            input_features=ALL_FEATURE_COLUMNS,
            output_features=["rock_class"],
            supports_training=True,
            supports_pipeline_training=True,
            supports_uncertainty=False,
            supports_explainability=False,
            requires_gpu=False,
            tags=["prototype", "georesponse"],
            task_type="classification",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "GeoResponseModel":
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # Delegate to the dataset-aware classifier which records fitted feature schema
        self.classifier.fit(X_df, y)
        # Save chosen feature schema for later validation
        self.feature_names_ = list(self.classifier.feature_names_)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("GeoResponseModel must be fitted before calling predict_proba()")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # ensure feature columns exist (fill missing with median/zeros gracefully)
        feat_cols = self.feature_names_
        missing = [c for c in feat_cols if c not in X_df.columns]
        if missing:
            # create missing columns filled with zeros to avoid KeyError
            for c in missing:
                X_df[c] = 0.0
        X_in = X_df[feat_cols].fillna(X_df[feat_cols].median())
        # GeoResponseClassifier stores the underlying sklearn-like model at .model
        probs = self.classifier.model.predict_proba(X_in)
        return np.asarray(probs)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("GeoResponseModel must be fitted before calling predict()")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        feat_cols = self.feature_names_
        missing = [c for c in feat_cols if c not in X_df.columns]
        if missing:
            for c in missing:
                X_df[c] = 0.0
        X_in = X_df[feat_cols].fillna(X_df[feat_cols].median())
        preds = self.classifier.predict(X_in)
        probs = self.classifier.model.predict_proba(X_in)
        max_probs = probs.max(axis=1)
        review_flag = max_probs < CONFIDENCE_THRESHOLD
        result = pd.DataFrame({
            "rock_class": preds,
            "confidence_score": max_probs,
            "review_flag": review_flag,
        }, index=X_df.index)
        return result

    def save(self, path: str) -> None:
        # fallback to BaseBlastModel.save if you want to persist wrapper state
        super().save(path)

    @classmethod
    def load(cls, path: str) -> "GeoResponseModel":
        # default load uses joblib.load in BaseBlastModel; if saved as wrapper object that works.
        return super().load(path)
