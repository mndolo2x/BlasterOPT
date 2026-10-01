"""
GA-ANN Model Wrapper (`models/custom/ga_ann.py`).
Wraps Genetic Algorithm + Neural Network (GAANNModel) implementation.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from models.base import BaseBlastModel, ModelMetadata
from src.models import GAANNModel


class GAANNBlastModel(BaseBlastModel):
    def __init__(self, **kwargs):
        super().__init__(model_name="GAANNBlastModel", config=kwargs)
        self.model = GAANNModel(**kwargs)

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="ga_ann",
            display_name="GA-ANN (Genetic Algorithm + Neural Network)",
            model_type="pytorch",
            description="Hybrid GA-ANN model for fragmentation, PPV, and airblast prediction.",
            version="1.0.0",
            author="BlastOpt Team",
            input_features=[
                "burden", "spacing", "powder_factor", "stemming",
                "rock_factor", "blastability_index", "charge_per_delay"
            ],
            output_features=["fragmentation_d80_cm", "vibration_ppv_mms", "airblast_db"],
            supports_training=True,
            supports_pipeline_training=True,
            supports_uncertainty=False,
            supports_explainability=False,
            requires_gpu=False,
            tags=["custom", "ga-ann", "core"],
        )

    def fit(self, X, y, **kwargs):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.model.fit(X_df, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        preds_df = self.model.predict(X_df)
        return preds_df

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.model = joblib.load(path)
        obj.is_fitted = True
        return obj
