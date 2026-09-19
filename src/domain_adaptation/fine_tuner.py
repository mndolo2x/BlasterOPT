"""
Fine-Tuner Submodule for Transfer Learning Fine-Tuning (`domain_adaptation`).
Freezes early feature extraction layers and retrains later layers on target domain (e.g., Granite).
"""

import copy
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

logger = logging.getLogger(__name__)


class TransferFineTuner:
    """
    Performs layer-freezing fine-tuning on a PyTorch GA-ANN / PINN model using target domain data.
    """

    def __init__(
        self,
        lr: float = 0.0001,
        epochs: int = 50,
        freeze_early_layers: bool = True,
        seed: int = 42
    ):
        self.lr = lr
        self.epochs = epochs
        self.freeze_early_layers = freeze_early_layers
        self.seed = seed

    def fine_tune(
        self,
        base_model: Any,
        X_target: np.ndarray,
        Y_target: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Fine-tunes base_model on target domain data. Freezes early network layers if configured.
        """
        if not HAS_TORCH or not isinstance(base_model, torch.nn.Module):
            logger.warning("PyTorch not available or model is not a PyTorch Module. Returning model copy.")
            return copy.deepcopy(base_model), {"fine_tune_epochs": 0, "final_loss": 0.0, "loss_history": []}

        fine_tuned_model = copy.deepcopy(base_model)
        fine_tuned_model.train()

        # Freeze early layers if enabled
        if self.freeze_early_layers and hasattr(fine_tuned_model, "network"):
            total_layers = len(fine_tuned_model.network)
            freeze_cutoff = max(1, total_layers // 2)
            for idx, param in enumerate(fine_tuned_model.network.parameters()):
                if idx < freeze_cutoff:
                    param.requires_grad = False

        # Optimizer over trainable parameters only
        trainable_params = [p for p in fine_tuned_model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        x_t = torch.tensor(X_target, dtype=torch.float32)
        y_t = torch.tensor(Y_target, dtype=torch.float32)

        loss_history = []
        for ep in range(1, self.epochs + 1):
            optimizer.zero_grad()
            preds = fine_tuned_model(x_t)
            if isinstance(preds, tuple):
                preds = preds[0]
            loss = loss_fn(preds, y_t)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            loss_history.append({"epoch": ep, "fine_tune_loss": loss_val})

        fine_tuned_model.eval()

        return fine_tuned_model, {
            "fine_tune_epochs": self.epochs,
            "final_loss": loss_history[-1]["fine_tune_loss"] if loss_history else 0.0,
            "loss_history": loss_history,
        }
