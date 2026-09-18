"""
Ensemble Trainer Submodule for GA-ANN Neural Network Models.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from src.ensemble_uq.models import TrainingConfig

logger = logging.getLogger(__name__)


class GAANNEnsembleMember:
    """A single neural network member of the GA-ANN ensemble."""

    def __init__(self, seed: int, hidden_layer_sizes: Tuple[int, ...] = (64, 32), activation: str = "relu"):
        self.seed = seed
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=300,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=15,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GAANNEnsembleMember":
        """Fits member model on scaled input features and target labels."""
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        self.model.fit(X_scaled, y_scaled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions and un-scales back to original units."""
        X_scaled = self.scaler_x.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        if y_pred_scaled.ndim == 1 and self.scaler_y.mean_.shape[0] > 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, self.scaler_y.mean_.shape[0])
        return self.scaler_y.inverse_transform(y_pred_scaled)


class EnsembleTrainer:
    """
    Trains N=7 GA-ANN models using distinct random seeds, varied hidden layer architectures,
    and bootstrap sub-sampling of training data.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.members: List[GAANNEnsembleMember] = []

    def train(self, X: np.ndarray, y: np.ndarray) -> List[GAANNEnsembleMember]:
        """
        Trains N ensemble members on bootstrap samples of (X, y).
        """
        self.members = []
        n_samples = X.shape[0]
        n_members = self.config.n_models

        for i in range(n_members):
            seed = self.config.seeds[i % len(self.config.seeds)] + i * 10
            arch = self.config.architectures[i % len(self.config.architectures)]

            # Bootstrap sub-sampling (80% replacement)
            np.random.seed(seed)
            boot_indices = np.random.choice(n_samples, size=int(n_samples * self.config.data_splits), replace=True)

            X_boot = X[boot_indices]
            y_boot = y[boot_indices]

            member = GAANNEnsembleMember(
                seed=seed,
                hidden_layer_sizes=arch["hidden_layer_sizes"],
                activation=arch["activation"],
            )
            member.fit(X_boot, y_boot)
            self.members.append(member)
            logger.info(f"Trained GA-ANN ensemble member #{i+1}/{n_members} (seed={seed}, arch={arch['hidden_layer_sizes']}).")

        return self.members

    def save_ensemble(self, output_dir: str = "models/ensemble_uq/") -> List[str]:
        """Saves all ensemble member models and scaler metadata to disk."""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        for idx, m in enumerate(self.members):
            p = os.path.join(output_dir, f"ga_ann_member_{idx+1}.joblib")
            joblib.dump(m, p)
            saved_paths.append(p)
        return saved_paths

    def load_ensemble(self, input_dir: str = "models/ensemble_uq/") -> List[GAANNEnsembleMember]:
        """Loads ensemble members from disk."""
        self.members = []
        if os.path.exists(input_dir):
            for f in sorted(os.listdir(input_dir)):
                if f.startswith("ga_ann_member_") and f.endswith(".joblib"):
                    p = os.path.join(input_dir, f)
                    m = joblib.load(p)
                    self.members.append(m)
        return self.members
