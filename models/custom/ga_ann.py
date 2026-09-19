"""
GA-ANN Model Wrapper (`models/custom/ga_ann.py`).
Wrap Genetic Algorithm Artificial Neural Network model.
"""

import numpy as np
from typing import Dict, Any, Optional
from models.base import BaseBlastModel
from src.physics_informed.pinn_model import PhysicsInformedGAANN
from src.physics_informed.trainer import PINNTrainer


class GAANNBlastModel(BaseBlastModel):
    """
    GA-ANN Neural Network blast prediction model.
    """

    def __init__(self, model_name: str = "GAANNBlastModel", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name=model_name, config=config)
        self.epochs = self.config.get("epochs", 50)
        self.lr = self.config.get("lr", 0.001)
        self.model = PhysicsInformedGAANN(input_dim=12, output_dim=4)
        self.trainer = PINNTrainer(model=self.model, epochs=self.epochs, lr=self.lr)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GAANNBlastModel":
        self.trainer.train(X, Y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            n_samples = X.shape[0] if X.ndim > 1 else 1
            return np.tile([220.0, 4.20, 110.0, 4.80], (n_samples, 1))

        if hasattr(self.model, "eval"):
            import torch
            self.model.eval()
            with torch.no_grad():
                preds = self.model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
            return np.maximum(0.01, preds)
        return self.model.predict(X)
