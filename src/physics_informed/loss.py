"""
Composite Physics-Informed Loss Function Submodule with Adaptive Loss Weighting.
L_total = L_data + lambda_1 * L_kuzram + lambda_2 * L_usbm
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

from src.physics_informed.physics_equations import kuz_ram_x50_torch, usbm_ppv_torch

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class CompositePhysicsLoss(nn.Module):
        """
        Composite loss function combining data-driven MSE loss with soft physics penalty terms:
        L_total = L_data + lambda_kuzram * L_kuzram + lambda_usbm * L_usbm

        Supports adaptive weighting: if physics loss dominates data loss, physics lambdas
        are automatically scaled down so the data fit is not overwhelmed.
        """

        def __init__(
            self,
            lambda_kuzram: float = 0.25,
            lambda_usbm: float = 0.25,
            adaptive_weighting: bool = True,
            max_physics_ratio: float = 2.0,
            feature_index_map: Optional[Dict[str, int]] = None
        ):
            super().__init__()
            self.mse_loss = nn.MSELoss()
            self.base_lambda_kuzram = lambda_kuzram
            self.base_lambda_usbm = lambda_usbm
            self.lambda_kuzram = lambda_kuzram
            self.lambda_usbm = lambda_usbm
            self.adaptive_weighting = adaptive_weighting
            self.max_physics_ratio = max_physics_ratio

            # Default 12-feature input index map
            self.idx_map = feature_index_map or {
                "burden_m": 0,
                "spacing_m": 1,
                "hole_diameter_mm": 2,
                "bench_height_m": 3,
                "stemming_m": 4,
                "subdrill_m": 5,
                "powder_factor_kg_m3": 6,
                "max_charge_per_delay_kg": 7,
                "rock_factor_a": 8,
                "rmr": 9,
                "monitoring_distance_m": 10,
                "explosive_rws": 11,
            }

        def forward(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            x_inputs: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            Computes composite loss and returns (total_loss, loss_components_dict).

            y_pred / y_true columns: [d50_mm, ppv_mms, flyrock_m, cost_per_tonne_usd]
            x_inputs: 12-feature input tensor.
            """
            # 1. Data-driven Mean Squared Error loss
            data_loss = self.mse_loss(y_pred, y_true)

            # 2. Extract physics input tensors
            a_vec = x_inputs[:, self.idx_map["rock_factor_a"]]
            k_vec = x_inputs[:, self.idx_map["powder_factor_kg_m3"]]
            q_vec = x_inputs[:, self.idx_map["max_charge_per_delay_kg"]]
            e_vec = x_inputs[:, self.idx_map["explosive_rws"]] if "explosive_rws" in self.idx_map else torch.tensor(100.0, device=x_inputs.device)
            d_vec = x_inputs[:, self.idx_map["monitoring_distance_m"]]

            # 3. Physics equation analytical targets
            kuzram_target_d50 = kuz_ram_x50_torch(a_vec, k_vec, q_vec, e_vec)
            usbm_target_ppv = usbm_ppv_torch(d_vec, q_vec)

            # 4. Soft physics residual losses
            pred_d50 = y_pred[:, 0]
            pred_ppv = y_pred[:, 1]

            kuzram_loss = self.mse_loss(pred_d50, kuzram_target_d50)
            usbm_loss = self.mse_loss(pred_ppv, usbm_target_ppv)

            # Adaptive Weighting: If physics loss dominates data_loss, reduce lambdas dynamically
            if self.adaptive_weighting and data_loss.item() > 1e-6:
                d_loss_val = data_loss.item()
                k_loss_val = kuzram_loss.item()
                u_loss_val = usbm_loss.item()

                if k_loss_val > self.max_physics_ratio * d_loss_val:
                    scale_k = (self.max_physics_ratio * d_loss_val) / max(1e-5, k_loss_val)
                    self.lambda_kuzram = self.base_lambda_kuzram * scale_k
                else:
                    self.lambda_kuzram = self.base_lambda_kuzram

                if u_loss_val > self.max_physics_ratio * d_loss_val:
                    scale_u = (self.max_physics_ratio * d_loss_val) / max(1e-5, u_loss_val)
                    self.lambda_usbm = self.base_lambda_usbm * scale_u
                else:
                    self.lambda_usbm = self.base_lambda_usbm

            # 5. Composite total loss
            total_loss = data_loss + self.lambda_kuzram * kuzram_loss + self.lambda_usbm * usbm_loss

            loss_components = {
                "total_loss": float(total_loss.item()),
                "data_loss": float(data_loss.item()),
                "kuzram_physics_loss": float(kuzram_loss.item()),
                "usbm_physics_loss": float(usbm_loss.item()),
                "effective_lambda_kuzram": float(self.lambda_kuzram),
                "effective_lambda_usbm": float(self.lambda_usbm),
            }

            return total_loss, loss_components

else:
    class CompositePhysicsLoss:
        """Fallback CompositePhysicsLoss when PyTorch is unavailable."""
        def __init__(self, lambda_kuzram: float = 0.25, lambda_usbm: float = 0.25, adaptive_weighting: bool = True):
            self.lambda_kuzram = lambda_kuzram
            self.lambda_usbm = lambda_usbm

        def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
            mse = float(np.mean((y_pred - y_true) ** 2))
            return {"total_loss": mse, "data_loss": mse, "kuzram_physics_loss": 0.0, "usbm_physics_loss": 0.0}
