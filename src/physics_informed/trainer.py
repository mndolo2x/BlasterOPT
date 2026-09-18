"""
Physics-Informed GA-ANN Trainer Submodule.
Supports AdamW, CosineAnnealingLR, early stopping on validation loss/R2 score,
best checkpoint saving, and curriculum learning.
"""

import os
import logging
import copy
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

try:
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from src.physics_informed.pinn_model import PhysicsInformedGAANN
from src.physics_informed.loss import CompositePhysicsLoss

logger = logging.getLogger(__name__)


def _compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes overall mean R2 score across all output targets."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    # Avoid zero division
    ss_tot = np.where(ss_tot == 0, 1e-5, ss_tot)
    r2_per_target = 1.0 - (ss_res / ss_tot)
    return float(np.mean(r2_per_target))


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
        patience: int = 50,
        curriculum: bool = False,
        seed: int = 42
    ):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.curriculum = curriculum
        self.base_lambda_kuzram = lambda_kuzram
        self.base_lambda_usbm = lambda_usbm

        if HAS_TORCH:
            torch.manual_seed(seed)
            self.model = model or PhysicsInformedGAANN(input_dim=12, output_dim=4)
            self.loss_fn = CompositePhysicsLoss(lambda_kuzram=lambda_kuzram, lambda_usbm=lambda_usbm)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        else:
            self.model = model or PhysicsInformedGAANN(input_dim=12, output_dim=4)
            self.loss_fn = None
            self.optimizer = None
            self.scheduler = None

        self.loss_history: List[Dict[str, float]] = []

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes training loop over X_train and Y_train with optional validation early stopping and best model saving by R2 score.
        """
        self.loss_history = []

        if not HAS_TORCH:
            logger.warning("PyTorch unavailable. Returning fallback training history.")
            return {"epochs_completed": 0, "final_loss": 0.0, "best_val_r2": 0.0, "loss_history": []}

        x_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(Y_train, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and Y_val is not None
        if has_val:
            x_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

        best_val_r2 = -float("inf")
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        stopped_early = False

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            # Curriculum learning: Start physics-heavy (1.0x to 2.0x base), annealing toward target lambda values
            if self.curriculum:
                progress = (epoch - 1) / max(1, self.epochs - 1)
                # Linear schedule: decay physics weight multiplier from 2.0 to 1.0
                phys_multiplier = 2.0 - 1.0 * progress
                self.loss_fn.lambda_kuzram = self.base_lambda_kuzram * phys_multiplier
                self.loss_fn.lambda_usbm = self.base_lambda_usbm * phys_multiplier

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

            # Update CosineAnnealingLR scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Average train loss components for epoch
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.lr
            avg_loss_entry = {
                "epoch": epoch,
                "total_loss": epoch_loss_accum["total"] / max(1, num_batches),
                "data_loss": epoch_loss_accum["data"] / max(1, num_batches),
                "kuzram_physics_loss": epoch_loss_accum["kuzram"] / max(1, num_batches),
                "usbm_physics_loss": epoch_loss_accum["usbm"] / max(1, num_batches),
                "lr": current_lr,
            }

            # Validation evaluation and early stopping
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(x_val_tensor)
                    val_total_loss, val_loss_comp = self.loss_fn(val_pred, y_val_tensor, x_val_tensor)

                    val_pred_np = val_pred.cpu().numpy()
                    val_r2 = _compute_r2_score(Y_val, val_pred_np)

                val_loss_val = float(val_total_loss.item())
                avg_loss_entry["val_loss"] = val_loss_val
                avg_loss_entry["val_r2"] = val_r2

                # Save best model state dict by validation R2 score
                if val_r2 > best_val_r2 or (val_r2 == best_val_r2 and val_loss_val < best_val_loss):
                    best_val_r2 = val_r2
                    best_val_loss = val_loss_val
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (patience={self.patience}).")
                    stopped_early = True

            self.loss_history.append(avg_loss_entry)

            if epoch % 50 == 0 or epoch == 1:
                val_str = f" | Val Loss: {avg_loss_entry.get('val_loss', 0.0):.4f} | Val R2: {avg_loss_entry.get('val_r2', 0.0):.4f}" if has_val else ""
                logger.info(
                    f"PINN Epoch {epoch}/{self.epochs} | "
                    f"Total Loss: {avg_loss_entry['total_loss']:.4f} | "
                    f"Data MSE: {avg_loss_entry['data_loss']:.4f} | "
                    f"Physics Kuz-Ram: {avg_loss_entry['kuzram_physics_loss']:.4f}{val_str}"
                )

            if stopped_early:
                break

        # Restore best model weights if available
        if best_model_state is not None and HAS_TORCH:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best PINN model state dict with Best Val R2 = {best_val_r2:.4f}")

        return {
            "epochs_completed": len(self.loss_history),
            "final_loss": self.loss_history[-1]["total_loss"] if self.loss_history else 0.0,
            "best_val_r2": best_val_r2 if has_val else None,
            "stopped_early": stopped_early,
            "loss_history": self.loss_history,
        }

    def save_checkpoint(self, path: str = "models/pinn_checkpoint.pt") -> str:
        """Saves PINN model checkpoint to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if HAS_TORCH and hasattr(self.model, "state_dict"):
            torch.save(self.model.state_dict(), path)
        return path
