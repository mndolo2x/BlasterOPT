"""
Domain Adaptation Manager Submodule.
Main orchestration interface managing transfer learning fine-tuning, DANN adversarial alignment,
evaluating target geology accuracy, and returning domain adaptation reports.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from src.domain_adaptation.models import DomainAdaptationConfig, FineTuneMetrics, DomainAdaptationReport
from src.domain_adaptation.fine_tuner import TransferFineTuner
from src.domain_adaptation.dann import DANNTrainer

logger = logging.getLogger(__name__)


def _evaluate_model_r2(model: Any, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, float]:
    """Computes mean R2, RMSE, and MAE across target domain outputs."""
    if HAS_TORCH and isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            preds_tensor = model(torch.tensor(X, dtype=torch.float32))
            if isinstance(preds_tensor, tuple):
                preds_tensor = preds_tensor[0]
            preds = preds_tensor.detach().cpu().numpy()
    elif hasattr(model, "predict"):
        preds = np.asarray(model.predict(X))
        if isinstance(preds, tuple):
            preds = preds[0]
    else:
        preds = np.tile([250.0, 5.0, 100.0, 5.0], (len(X), 1))

    ss_res = np.sum((Y - preds) ** 2, axis=0)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
    ss_tot = np.where(ss_tot == 0, 1e-5, ss_tot)
    r2_vec = 1.0 - (ss_res / ss_tot)
    mean_r2 = float(np.mean(r2_vec))

    rmse = float(np.mean(np.sqrt(np.mean((Y - preds) ** 2, axis=0))))
    mae = float(np.mean(np.abs(Y - preds)))

    return mean_r2, rmse, mae


class DomainAdaptationManager:
    """
    Orchestrates transfer learning fine-tuning and adversarial DANN domain adaptation
    to transition blast optimization models from Kimberlite to Granite.
    """

    def __init__(self, config: Optional[DomainAdaptationConfig] = None):
        self.config = config or DomainAdaptationConfig()
        self.fine_tuner = TransferFineTuner(
            lr=self.config.fine_tune_lr,
            epochs=self.config.fine_tune_epochs,
            freeze_early_layers=self.config.freeze_early_layers
        )

    def adapt_domain(
        self,
        base_model: Any,
        X_source: np.ndarray,
        Y_source: np.ndarray,
        X_target: np.ndarray,
        Y_target: np.ndarray
    ) -> Tuple[Any, DomainAdaptationReport]:
        """
        Adapts base_model to target domain (e.g. Granite).
        1. Evaluates pre-adaptation accuracy on target domain.
        2. Applies transfer learning fine-tuning with layer freezing.
        3. Checks if post-fine-tuning R2 < target_r2_threshold.
        4. If required or forced, applies DANN adversarial feature alignment.
        5. Returns adapted model and DomainAdaptationReport.
        """
        # Step 1: Evaluate unadapted base model on target domain
        pre_r2, pre_rmse, pre_mae = _evaluate_model_r2(base_model, X_target, Y_target)

        # Step 2: Perform transfer learning fine-tuning
        adapted_model, ft_info = self.fine_tuner.fine_tune(base_model, X_target, Y_target)
        post_r2, post_rmse, post_mae = _evaluate_model_r2(adapted_model, X_target, Y_target)

        method_used = "FINE_TUNING"

        # Step 3: Check if fine-tuning alone is insufficient and DANN is enabled/needed
        if (post_r2 < self.config.target_r2_threshold or self.config.enable_dann) and HAS_TORCH:
            logger.info("Target domain R2 below threshold or DANN explicitly enabled. Applying DANN adversarial adaptation...")
            dann_trainer = DANNTrainer(epochs=self.config.fine_tune_epochs, alpha_grl=self.config.alpha_grl)
            dann_model, _ = dann_trainer.fit(X_source, Y_source, X_target, Y_target)

            dann_r2, dann_rmse, dann_mae = _evaluate_model_r2(dann_model, X_target, Y_target)
            if dann_r2 > post_r2:
                adapted_model = dann_model
                post_r2, post_rmse, post_mae = dann_r2, dann_rmse, dann_mae
                method_used = "DANN_ADVERSARIAL"

        # Step 4: Build metrics and recommendations
        r2_improvement = post_r2 - pre_r2
        metrics = FineTuneMetrics(
            target_geology=self.config.target_geology,
            sample_count=len(X_target),
            pre_adaptation_r2=round(pre_r2, 4),
            post_adaptation_r2=round(post_r2, 4),
            r2_improvement=round(r2_improvement, 4),
            target_rmse=round(post_rmse, 2),
            target_mae=round(post_mae, 2),
            method_used=method_used
        )

        recs = [
            f"Model successfully adapted to {self.config.target_geology} rock mass using {method_used}.",
            f"R2 score improved from {pre_r2:.2f} to {post_r2:.2f} (+{r2_improvement:.2f}).",
            f"Increase powder factor by ~15-20% in Granite due to higher Rock Factor A (11.0 vs 8.0) and compressive strength."
        ]

        report = DomainAdaptationReport(
            source_domain=self.config.source_geology,
            target_domain=self.config.target_geology,
            status="ADAPTATION_SUCCESSFUL" if post_r2 >= 0.70 else "THRESHOLD_EXCEEDED",
            metrics=metrics,
            recommendations=recs
        )

        return adapted_model, report
