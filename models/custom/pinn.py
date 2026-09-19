"""
Physics-Informed Neural Network (PINN) Model Wrapper (`models/custom/pinn.py`).
"""

import numpy as np
from typing import Dict, Any, Optional
from models.base import BaseBlastModel
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

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "PINNBlastModel":
        self.trainer.train(X, Y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "eval"):
            import torch
            self.model.eval()
            with torch.no_grad():
                preds = self.model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
            return np.maximum(0.01, preds)
        return self.model.predict(X)
