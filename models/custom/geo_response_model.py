"""
GeoResponse Model Wrapper (`models/custom/geo_response_model.py`).
Wraps GeoResponse MWD rock-type classifier implementation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from models.base import BaseBlastModel, ModelMetadata
from src.georesponse.models import GeoResponseClassifier


class GeoResponseModel(BaseBlastModel):
    def __init__(self, **kwargs):
        super().__init__(model_name="GeoResponseModel", config=kwargs)
        self.classifier = GeoResponseClassifier(**kwargs)

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="geo_response",
            display_name="GeoResponse MWD Rock-Type Classifier",
            model_type="custom",
            task_type="classification",
            description="Dataset-aware rock-type classifier trained on Measure-While-Drilling (MWD) telemetry.",
            version="0.1.0",
            author="GeoResponse Research Team",
            input_features=[
                "penetration_rate_m_hr",
                "percussion_pressure_bar",
                "feed_pressure_bar",
                "rotation_pressure_bar",
                "flushing_pressure_bar",
                "rotation_speed_rpm",
                "depth_m",
                "feed_percussion_ratio",
            ],
            output_features=["rock_class"],
            supports_training=True,
            supports_pipeline_training=True,
            supports_uncertainty=False,
            supports_explainability=False,
            requires_gpu=False,
            tags=["custom", "georesponse", "mwd", "prototype", "classifier"],
        )

    def fit(self, X, y, **kwargs):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.classifier.fit(X_df, y)
        self.is_fitted = True
        return self

    def predict(self, X) -> pd.DataFrame:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        review_res = self.classifier.predict_with_review_flags(X_df)
        preds_df = pd.DataFrame({
            "rock_class": review_res["predicted_rock_type"],
            "confidence_score": review_res["confidence_score"],
            "review_flag": review_res["requires_review_flag"],
        })
        return preds_df

    def predict_proba(self, X) -> np.ndarray:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self.classifier.predict_proba(X_df)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.classifier, path)

    @classmethod
    def load(cls, path: str) -> "GeoResponseModel":
        obj = cls()
        obj.classifier = joblib.load(path)
        obj.is_fitted = True
        return obj
