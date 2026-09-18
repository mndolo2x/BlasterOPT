"""
Physics-Informed GA-ANN Trainer Submodule.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

try:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from src.physics_informed.pinn_model import PhysicsInformedGAANN
from src.physics_informed.loss import CompositePhysicsLoss

logger = logging.getLogger(__name__)


class PINNTrainer:
    """
    Trains PhysicsInformedGAANN neural network balancing data MSE loss
    and Kuz-Ram / USBM physics soft constraint losses.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        lambda_kuzram: float = 0.25,
        lambda_usbm: float = 0.25,
        lr: float = 0.001,
        epochs: int = 200,
        batch_size: int = 32,
        seed: int = 42
    ):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        if HAS_TORCH:
            torch.manual_seed(seed)
            self.model = model or PhysicsInformedGAANN(input_dim=12, output_dim=4)
            self.loss_fn = CompositePhysicsLoss(lambda_kuzram=lambda_kuzram, lambda_usbm=lambda_usbm)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            self.model = model or PhysicsInformedGAANN(input_dim=12, output_dim=4)
            self.loss_fn = None
            self.optimizer = None

        self.loss_history: List[Dict[str, float]] = []

    def train(self, X_train: np.ndarray, Y_train: np.ndarray) -> Dict[str, Any]:
        """
        Executes training loop over X_train and Y_train.
        """
        self.loss_history = []

        if not HAS_TORCH:
            logger.warning("PyTorch unavailable. Returning fallback training history.")
            return {"epochs_completed": 0, "final_loss": 0.0, "loss_history": []}

        x_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(Y_train, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for epoch in range(1, self.epochs + 1):
            epoch_loss_accum = {"total": 0.0, "data": 0.0, "kuzram": 0.0, "usbm": 0.0}
            num_batches = 0

            for bx, by in dataloader:
                self.optimizer.zero_grad()
                pred_y = self.model(bx)

                total_loss, loss_comp = self.loss_fn(pred_y, by, bx)
                total_loss.backward()

                # Gradient clipping to ensure stable training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss_accum["total"] += loss_comp["total_loss"]
                epoch_loss_accum["data"] += loss_comp["data_loss"]
                epoch_loss_accum["kuzram"] += loss_comp["kuzram_physics_loss"]
                epoch_loss_accum["usbm"] += loss_comp["usbm_physics_loss"]
                num_batches += 1

            # Average loss components for epoch
            avg_loss_entry = {
                "epoch": epoch,
                "total_loss": epoch_loss_accum["total"] / max(1, num_batches),
                "data_loss": epoch_loss_accum["data"] / max(1, num_batches),
                "kuzram_physics_loss": epoch_loss_accum["kuzram"] / max(1, num_batches),
                "usbm_physics_loss": epoch_loss_accum["usbm"] / max(1, num_batches),
            }
            self.loss_history.append(avg_loss_entry)

            if epoch % 50 == 0 or epoch == 1:
                logger.info(
                    f"PINN Epoch {epoch}/{self.epochs} | "
                    f"Total Loss: {avg_loss_entry['total_loss']:.4f} | "
                    f"Data MSE: {avg_loss_entry['data_loss']:.4f} | "
                    f"Physics Kuz-Ram: {avg_loss_entry['kuzram_physics_loss']:.4f}"
                )

        return {
            "epochs_completed": self.epochs,
            "final_loss": self.loss_history[-1]["total_loss"] if self.loss_history else 0.0,
            "loss_history": self.loss_history,
        }

    def save_checkpoint(self, path: str = "models/pinn_checkpoint.pt") -> str:
        """Saves PINN model checkpoint to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if HAS_TORCH and hasattr(self.model, "state_dict"):
            torch.save(self.model.state_dict(), path)
        return path
