"""
Fine-Tuner Submodule for Transfer Learning Fine-Tuning (`domain_adaptation`).
Freezes early feature extraction layers and retrains later layers on target domain (e.g., Granite).
Supports configurable freeze depth and from-scratch baseline comparisons.
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
        lr: float = 0.001,
        epochs: int = 60,
        freeze_depth: int = 2,
        seed: int = 42
    ):
        self.lr = lr
        self.epochs = epochs
        self.freeze_depth = freeze_depth
        self.seed = seed

    def fine_tune(
        self,
        base_model: Any,
        X_target: np.ndarray,
        Y_target: np.ndarray,
        freeze_depth: Optional[int] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Fine-tunes base_model on target domain data. Freezes `freeze_depth` early network layers.
        """
        f_depth = freeze_depth if freeze_depth is not None else self.freeze_depth

        if not HAS_TORCH or not isinstance(base_model, torch.nn.Module):
            logger.warning("PyTorch not available or model is not a PyTorch Module. Returning model copy.")
            return copy.deepcopy(base_model), {"fine_tune_epochs": 0, "final_loss": 0.0, "loss_history": []}

        fine_tuned_model = copy.deepcopy(base_model)
        fine_tuned_model.train()

        # Freeze early layers based on freeze_depth
        if hasattr(fine_tuned_model, "network") and f_depth > 0:
            param_count = 0
            for param in fine_tuned_model.network.parameters():
                if param_count < f_depth * 2:  # Each layer has weight & bias
                    param.requires_grad = False
                param_count += 1

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
            "freeze_depth": f_depth,
            "final_loss": loss_history[-1]["fine_tune_loss"] if loss_history else 0.0,
            "loss_history": loss_history,
        }

    def compare_finetune_vs_scratch(
        self,
        base_model: Any,
        X_target: np.ndarray,
        Y_target: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compares transfer learning fine-tuning performance against training a new model from scratch.
        """
        # 1. Fine-tuned model
        ft_model, ft_info = self.fine_tune(base_model, X_target, Y_target)

        # 2. From-scratch model
        from src.physics_informed import PhysicsInformedGAANN
        scratch_model = PhysicsInformedGAANN(input_dim=X_target.shape[1], output_dim=Y_target.shape[1])
        scratch_ft_tuner = TransferFineTuner(lr=self.lr, epochs=self.epochs, freeze_depth=0)
        scratch_model, scratch_info = scratch_ft_tuner.fine_tune(scratch_model, X_target, Y_target)

        return {
            "fine_tuned_final_loss": ft_info.get("final_loss", 0.0),
            "from_scratch_final_loss": scratch_info.get("final_loss", 0.0),
            "transfer_advantage_loss_reduction": float(scratch_info.get("final_loss", 0.0) - ft_info.get("final_loss", 0.0)),
        }
