"""
Physics-Informed Neural Network (PINN) Model Wrapper (`models/custom/pinn.py`).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from models.base import BaseBlastModel, ModelMetadata
from src.physics_informed import PhysicsInformedGAANN, PINNTrainer, PINNEvaluator


class PINNBlastModel(BaseBlastModel):
    """
    Physics-Informed GA-ANN with Kuz-Ram and USBM soft loss constraints.
    """

    def __init__(self, model_name: str = "PINNBlastModel", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name=model_name, config=config)
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
            supports_uncertainty=True,
            tags=["pinn", "physics", "kuzram", "usbm"]
        )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], **kwargs) -> "PINNBlastModel":
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.DataFrame) else y
        self.trainer.train(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if hasattr(self.model, "eval"):
            import torch
            self.model.eval()
            with torch.no_grad():
                preds = self.model(torch.tensor(X_arr, dtype=torch.float32)).cpu().numpy()
            preds = np.maximum(0.01, preds)
        else:
            preds = self.model.predict(X_arr)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(preds, columns=["d50_mm", "ppv_mms", "flyrock_m", "cost_usd"])
        return preds

    def predict_with_uncertainty(self, X: Union[pd.DataFrame, np.ndarray]) -> Optional[Union[pd.DataFrame, Dict[str, Any]]]:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if hasattr(self.model, "forward_mc_dropout"):
            import torch
            mean_p, std_p, lower_ci, upper_ci = self.model.forward_mc_dropout(torch.tensor(X_arr, dtype=torch.float32))
            return {
                "mean": mean_p.cpu().numpy(),
                "std": std_p.cpu().numpy(),
                "lower_95_ci": lower_ci.cpu().numpy(),
                "upper_95_ci": upper_ci.cpu().numpy(),
            }
        return None
