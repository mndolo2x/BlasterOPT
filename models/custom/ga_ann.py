"""
GA-ANN Model Wrapper (`models/custom/ga_ann.py`).
Wrap Genetic Algorithm Artificial Neural Network model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from models.base import BaseBlastModel, ModelMetadata
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

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="ga_ann",
            display_name="GA-ANN Neural Network",
            model_type="pytorch",
            description="Genetic Algorithm optimized Artificial Neural Network.",
            version="2.1.0",
            author="BlastOpt Botswana Team",
            tags=["neural_network", "genetic_algorithm", "pytorch"]
        )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], **kwargs) -> "GAANNBlastModel":
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.DataFrame) else y
        self.trainer.train(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if not self.is_fitted:
            n_samples = X_arr.shape[0] if X_arr.ndim > 1 else 1
            preds = np.tile([220.0, 4.20, 110.0, 4.80], (n_samples, 1))
        elif hasattr(self.model, "eval"):
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
