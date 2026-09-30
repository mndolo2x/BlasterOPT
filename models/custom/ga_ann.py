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
            output_features=["d50_mm", "ppv_mms", "flyrock_m"],
            supports_training=True,
            supports_pipeline_training=True,
            supports_uncertainty=False,
            supports_explainability=False,
            requires_gpu=True,
            tags=["custom", "ga-ann", "core"],
        )

    def fit(self, X, y, **kwargs):
        # Untrained model requires actual PyTorch training loop on GPU/CPU.
        # Mark as not fitted if training was not executed.
        self.is_fitted = False
        raise NotImplementedError("GA-ANN model PyTorch pipeline training is not implemented on CPU. Trained PyTorch weights required.")

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model is not trained. Training required before prediction.")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        try:
            preds_raw = self.model.predict(X_df)
            preds = preds_raw[:, :3] if preds_raw.shape[1] >= 3 else preds_raw
        except Exception as exc:
            raise ValueError(f"Failed to generate predictions for GA-ANN model: {exc}") from exc

        return pd.DataFrame(preds, columns=self.get_metadata().output_features, index=X_df.index)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.model = joblib.load(path)
        obj.is_fitted = True
        return obj
