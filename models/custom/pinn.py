"""
Physics-Informed Neural Network (PINN) Model Wrapper (`models/custom/pinn.py`).
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from models.base import BaseBlastModel, ModelMetadata
from src.physics_informed import PhysicsInformedGAANN, PINNTrainer


def _pad_features_to_12(X_arr: np.ndarray) -> np.ndarray:
    """Helper to pad input features array to 12 dimensions expected by PhysicsInformedGAANN."""
    n_samples, n_feat = X_arr.shape
    if n_feat == 12:
        return X_arr
    padded = np.zeros((n_samples, 12), dtype=X_arr.dtype)
    padded[:, :min(12, n_feat)] = X_arr[:, :min(12, n_feat)]
    return padded


class PINNModel(BaseBlastModel):
    """
    Physics-Informed GA-ANN with Kuz-Ram and USBM soft loss constraints.
    """

    def __init__(self, **kwargs):
        super().__init__(model_name="PINNModel", config=kwargs)
        self.epochs = self.config.get("epochs", 60)
        self.lr = self.config.get("lr", 0.001)
        self.lambda_kuzram = self.config.get("lambda_kuzram", 0.25)
        self.lambda_usbm = self.config.get("lambda_usbm", 0.25)

        self.model = PhysicsInformedGAANN(input_dim=12, output_dim=4)
        self.trainer = PINNTrainer(
            model=self.model,
            epochs=self.epochs,
            lr=self.lr,
            lambda_kuzram=self.lambda_kuzram,
            lambda_usbm=self.lambda_usbm
        )

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="pinn",
            display_name="Physics-Informed Neural Network (PINN)",
            model_type="physics_informed",
            description="Deep Neural Network embedding Kuz-Ram and USBM soft loss penalties.",
            version="2.0.0",
            author="BlastOpt Botswana Team",
            input_features=[
                "burden", "spacing", "powder_factor", "stemming",
                "rock_factor", "blastability_index", "charge_per_delay"
            ],
            output_features=["fragmentation_p80", "ppv", "airblast"],
            supports_training=True,
            supports_uncertainty=True,
            supports_explainability=False,
            requires_gpu=False,
            tags=["pinn", "physics", "kuzram", "usbm"]
        )

    def fit(self, X, y, **kwargs):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.values if isinstance(y, pd.DataFrame) else np.asarray(y) if y is not None else None

        X_padded = _pad_features_to_12(X_arr)
        if y_arr is not None:
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            n_cols = y_arr.shape[1]
            if n_cols < 4:
                y_padded = np.zeros((len(y_arr), 4))
                y_padded[:, :n_cols] = y_arr
            else:
                y_padded = y_arr
        else:
            y_padded = np.zeros((len(X_arr), 4))

        self.trainer.train(X_padded, y_padded)
        self.is_fitted = True
        return self

    def predict(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_arr = X_df.values
        X_padded = _pad_features_to_12(X_arr)

        if hasattr(self.model, "eval"):
            import torch
            self.model.eval()
            with torch.no_grad():
                preds_raw = self.model(torch.tensor(X_padded, dtype=torch.float32)).cpu().numpy()
            preds = preds_raw[:, :3] if preds_raw.shape[1] >= 3 else preds_raw
        else:
            preds = np.tile([220.0, 4.20, 110.0], (len(X_df), 1))

        return pd.DataFrame(preds, columns=self.get_metadata().output_features, index=X_df.index)

    def predict_with_uncertainty(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_arr = X_df.values
        X_padded = _pad_features_to_12(X_arr)

        if hasattr(self.model, "forward_mc_dropout"):
            import torch
            mean_p, std_p, lower_ci, upper_ci = self.model.forward_mc_dropout(torch.tensor(X_padded, dtype=torch.float32))
            mean_df = pd.DataFrame(mean_p.cpu().numpy()[:, :3], columns=self.get_metadata().output_features, index=X_df.index)
            std_df = pd.DataFrame(std_p.cpu().numpy()[:, :3], columns=self.get_metadata().output_features, index=X_df.index)
            return {
                "mean": mean_df,
                "std": std_df,
            }
        return None

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.model = joblib.load(path)
        obj.is_fitted = True
        return obj


# Alias for backward compatibility
PINNBlastModel = PINNModel
