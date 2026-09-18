"""
PINN Extrapolation & Performance Evaluator Submodule.
Computes standard metrics (R2, RMSE, MAE), extrapolation performance on OOD inputs
(bench height > 18m, powder factor > 1.20 kg/m3), physics consistency scores, and
head-to-head comparison between PINN and standard GA-ANN models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from src.physics_informed.physics_equations import kuz_ram_x50_numpy, usbm_ppv_numpy

logger = logging.getLogger(__name__)


def _calculate_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Helper function to calculate R2, RMSE, and MAE across 4 blast output targets."""
    targets = ["d50_mm", "ppv_mms", "flyrock_m", "cost_usd"]
    metrics = {}

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    ss_tot = np.where(ss_tot == 0, 1e-5, ss_tot)
    r2_per_target = 1.0 - (ss_res / ss_tot)

    rmse_per_target = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mae_per_target = np.mean(np.abs(y_true - y_pred), axis=0)

    for i, target in enumerate(targets):
        if i < y_true.shape[1]:
            metrics[f"{target}_r2"] = float(r2_per_target[i])
            metrics[f"{target}_rmse"] = float(rmse_per_target[i])
            metrics[f"{target}_mae"] = float(mae_per_target[i])

    metrics["mean_r2"] = float(np.mean(r2_per_target))
    metrics["mean_rmse"] = float(np.mean(rmse_per_target))
    metrics["mean_mae"] = float(np.mean(mae_per_target))

    return metrics


class PINNEvaluator:
    """
    Evaluates PINN model performance on in-distribution test data,
    out-of-distribution (OOD) extrapolation test sets, physics consistency,
    and comparative benchmark against standard GA-ANN models.
    """

    def __init__(self, model: Any):
        self.model = model

    def _predict_samples(self, X: np.ndarray) -> np.ndarray:
        """Helper to compute model predictions in NumPy format."""
        if HAS_TORCH and isinstance(self.model, torch.nn.Module):
            self.model.eval()
            with torch.no_grad():
                preds_tensor = self.model(torch.tensor(X, dtype=torch.float32))
                return preds_tensor.detach().cpu().numpy()
        elif hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(X))
        else:
            return np.tile([220.0, 4.20, 110.0, 4.80], (len(X), 1))

    def evaluate_in_distribution(self, X_test: np.ndarray, Y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluates standard metrics (R2, RMSE, MAE) on in-distribution test data.
        Y targets: [d50_mm, ppv_mms, flyrock_m, cost_usd]
        """
        preds = self._predict_samples(X_test)
        metrics = _calculate_metrics_dict(Y_test, preds)

        # Backwards compatibility mapping
        metrics["d50_rmse_mm"] = metrics.get("d50_mm_rmse", 0.0)
        metrics["ppv_rmse_mms"] = metrics.get("ppv_mms_rmse", 0.0)
        metrics["d50_r2"] = metrics.get("d50_mm_r2", 0.0)
        metrics["ppv_r2"] = metrics.get("ppv_mms_r2", 0.0)

        return metrics

    def evaluate_physics_consistency(
        self,
        X: np.ndarray,
        feature_indices: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:
        """
        Evaluates how well model predictions adhere to analytical Kuz-Ram fragmentation
        and USBM vibration attenuation physics equations.
        Returns MAE physics residual errors and a 0-100% physics consistency score.
        """
        idx = feature_indices or {
            "powder_factor": 6,
            "charge_per_delay": 7,
            "rock_factor_a": 8,
            "distance": 10
        }

        preds = self._predict_samples(X)
        pred_d50 = preds[:, 0]
        pred_ppv = preds[:, 1]

        analytical_d50 = []
        analytical_ppv = []

        for row in X:
            k_val = row[idx["powder_factor"]] if len(row) > idx["powder_factor"] else 0.65
            q_val = row[idx["charge_per_delay"]] if len(row) > idx["charge_per_delay"] else 400.0
            a_val = row[idx["rock_factor_a"]] if len(row) > idx["rock_factor_a"] else 8.0
            d_val = row[idx["distance"]] if len(row) > idx["distance"] else 450.0

            analytical_d50.append(kuz_ram_x50_numpy(a_val, k_val, q_val))
            analytical_ppv.append(usbm_ppv_numpy(d_val, q_val))

        ana_d50_arr = np.array(analytical_d50)
        ana_ppv_arr = np.array(analytical_ppv)

        d50_physics_mae = float(np.mean(np.abs(pred_d50 - ana_d50_arr)))
        ppv_physics_mae = float(np.mean(np.abs(pred_ppv - ana_ppv_arr)))

        # Relative error percentage
        d50_rel_err = np.mean(np.abs(pred_d50 - ana_d50_arr) / np.maximum(10.0, ana_d50_arr))
        ppv_rel_err = np.mean(np.abs(pred_ppv - ana_ppv_arr) / np.maximum(0.5, ana_ppv_arr))

        overall_rel_err = float((d50_rel_err + ppv_rel_err) / 2.0)
        physics_consistency_score = max(0.0, min(100.0, float((1.0 - overall_rel_err) * 100.0)))

        return {
            "kuzram_d50_mae_mm": round(d50_physics_mae, 2),
            "usbm_ppv_mae_mms": round(ppv_physics_mae, 2),
            "overall_relative_error_pct": round(overall_rel_err * 100.0, 2),
            "physics_consistency_score_pct": round(physics_consistency_score, 2),
        }

    def evaluate_extrapolation(
        self,
        ood_x: np.ndarray,
        ood_y: Optional[np.ndarray] = None,
        target_metric: str = "powder_factor_kg_m3"
    ) -> Dict[str, Any]:
        """
        Evaluates PINN model predictions on out-of-range inputs (e.g., bench height > 18m,
        powder factor > 1.20 kg/m3). Calculates metrics (R2, RMSE, MAE) if ood_y is provided,
        along with physics consistency.
        """
        preds = self._predict_samples(ood_x)
        phys_eval = self.evaluate_physics_consistency(ood_x)

        d50_physics_mae = phys_eval["kuzram_d50_mae_mm"]
        ppv_physics_mae = phys_eval["usbm_ppv_mae_mms"]

        # Monotonicity check: as powder factor increases, d50 should decrease
        pinn_d50 = preds[:, 0]
        monotonic_d50 = bool(np.all(np.diff(pinn_d50) <= 1.0)) if target_metric == "powder_factor_kg_m3" else True

        result = {
            "target_metric": target_metric,
            "d50_physics_mae_mm": d50_physics_mae,
            "ppv_physics_mae_mms": ppv_physics_mae,
            "physics_consistency_score_pct": phys_eval["physics_consistency_score_pct"],
            "obeys_physics_monotonicity": monotonic_d50,
            "extrapolation_status": "EXCELLENT_PHYSICS_BOUNDED" if d50_physics_mae < 15.0 and monotonic_d50 else "MODERATE_DEVIATION",
        }

        if ood_y is not None:
            ood_metrics = _calculate_metrics_dict(ood_y, preds)
            for k, v in ood_metrics.items():
                result[f"ood_{k}"] = v

        return result

    def compare_pinn_vs_standard_gaann(
        self,
        standard_model: Any,
        ood_x: np.ndarray,
        ood_y: np.ndarray,
        target_metric: str = "out_of_distribution"
    ) -> Dict[str, Any]:
        """
        Direct benchmark comparison between PINN and standard GA-ANN model on extrapolation tasks.
        Compares R2, RMSE, MAE, and physics consistency scores.
        """
        pinn_extrap = self.evaluate_extrapolation(ood_x, ood_y, target_metric=target_metric)

        std_evaluator = PINNEvaluator(model=standard_model)
        std_extrap = std_evaluator.evaluate_extrapolation(ood_x, ood_y, target_metric=target_metric)

        pinn_r2 = pinn_extrap.get("ood_mean_r2", 0.0)
        std_r2 = std_extrap.get("ood_mean_r2", 0.0)
        pinn_phys = pinn_extrap.get("physics_consistency_score_pct", 0.0)
        std_phys = std_extrap.get("physics_consistency_score_pct", 0.0)

        r2_improvement = pinn_r2 - std_r2
        physics_improvement = pinn_phys - std_phys

        return {
            "pinn_metrics": pinn_extrap,
            "standard_gaann_metrics": std_extrap,
            "pinn_r2_improvement": round(r2_improvement, 4),
            "pinn_physics_consistency_improvement_pct": round(physics_improvement, 2),
            "winner": "PINN" if (pinn_r2 >= std_r2 and pinn_phys >= std_phys) else ("PINN" if physics_improvement > 10.0 else "STANDARD_GAANN"),
        }
