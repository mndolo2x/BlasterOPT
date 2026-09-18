"""
Physics-Informed GA-ANN PyTorch Model Submodule.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class PhysicsInformedGAANN(nn.Module):
        """
        Physics-Informed GA-ANN PyTorch neural network.
        Predicts 4 blast outcomes: [d50_mm, ppv_mms, flyrock_m, cost_per_tonne_usd].
        Embedded with differentiable soft physics constraints in loss evaluation.
        """

        def __init__(self, input_dim: int = 12, hidden_dims: Tuple[int, ...] = (128, 64, 32), output_dim: int = 4):
            super().__init__()
            layers = []
            prev_dim = input_dim

            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.LayerNorm(h_dim))
                layers.append(nn.SiLU())  # Smooth differentiable activation
                layers.append(nn.Dropout(p=0.05))
                prev_dim = h_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass. Returns unscaled predicted target tensor.
            Output columns: [d50_mm, ppv_mms, flyrock_m, cost_per_tonne_usd]
            """
            raw_out = self.network(x)
            # Ensure non-negative predictions for physics metrics
            return torch.clamp(raw_out, min=0.01)

else:
    class PhysicsInformedGAANN:
        """Fallback Python class when PyTorch is not available."""
        def __init__(self, input_dim: int = 12, hidden_dims: Tuple[int, ...] = (128, 64, 32), output_dim: int = 4):
            self.input_dim = input_dim
            self.output_dim = output_dim

        def predict(self, x: np.ndarray) -> np.ndarray:
            """Fallback prediction array."""
            n_samples = x.shape[0] if x.ndim > 1 else 1
            return np.tile([220.0, 4.20, 110.0, 4.80], (n_samples, 1))
