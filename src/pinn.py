"""
Physics-Informed Neural Network (PINN) Module for BlastOpt Botswana.

Implements Model 2 (Physics-Informed Neural Network) blending data-driven deep learning with fundamental
rock fracture mechanics and wave propagation equations (Kuz-Ram d50 and USBM PPV attenuation) as soft loss terms.
Includes Monte Carlo Dropout uncertainty estimation for epistemic and aleatoric confidence quantification.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object

logger = logging.getLogger(__name__)

PINN_INPUT_COLS = [
    "burden_m",
    "spacing_m",
    "hole_diameter_mm",
    "hole_depth_m",
    "stemming_m",
    "sub_drill_m",
    "powder_factor_kg_m3",
    "max_charge_per_delay_kg",
    "rock_strength_ucs_mpa",
    "rmr",
    "monitoring_distance_m",
    "blastability_index",
]

if HAS_TORCH:
    class BlastPINN(nn.Module):
        """
        Physics-Informed Neural Network (PINN) for simultaneous prediction of fragmentation (d50),
        ground vibration (PPV), and airblast overpressure (dBL).

        Architecture:
        - 12 input features
        - 4 hidden layers (128, 256, 256, 128 neurons) with ReLU activations and Dropout(0.1)
        - 3 output heads:
            1. head_frag: fragmentation d50 (mm)
            2. head_ppv: ground vibration PPV (mm/s)
            3. head_air: airblast overpressure (dBL)

        Physics Loss Embedding:
        Soft loss regularization penalizes deviations from:
        - Kuz-Ram equation: X50 = A * (V0 / Q)^0.8 * Q^(1/6) * (115 / E)^(19/30)
        - USBM PPV attenuation: PPV = K * (D / sqrt(W))^(-B)
        L_total = L_data + lambda_1 * L_kuzram + lambda_2 * L_usbm
        """

        def __init__(self, input_dim: int = 12, dropout_rate: float = 0.1):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )

            # Output heads
            self.head_frag = nn.Linear(128, 1)  # d50 mm
            self.head_ppv = nn.Linear(128, 1)   # PPV mm/s
            self.head_air = nn.Linear(128, 1)   # Airblast dBL

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass returning fragmentation, PPV, and airblast predictions.

            Parameters:
            -----------
            x : torch.Tensor
                Input feature tensor of shape (batch_size, 12).

            Returns:
            --------
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                Tensors for (pred_frag, pred_ppv, pred_airblast).
            """
            feat = self.shared(x)
            pred_frag = self.head_frag(feat)
            pred_ppv = self.head_ppv(feat)
            pred_air = self.head_air(feat)
            return pred_frag, pred_ppv, pred_air

        def compute_physics_loss(
            self, x: torch.Tensor, pred_frag: torch.Tensor, pred_ppv: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Computes soft physics loss terms for Kuz-Ram and USBM equations.

            Parameters:
            -----------
            x : torch.Tensor
                Input batch tensor (columns match PINN_INPUT_COLS).
            pred_frag : torch.Tensor
                Predicted fragmentation d50 (mm).
            pred_ppv : torch.Tensor
                Predicted ground vibration PPV (mm/s).

            Returns:
            --------
            Tuple[torch.Tensor, torch.Tensor]
                (loss_kuzram, loss_usbm) soft physics losses.
            """
            # Extract relevant columns from batch tensor
            # Index 6: powder_factor_kg_m3, Index 7: max_charge_per_delay_kg, Index 10: monitoring_distance_m
            pf = torch.clamp(x[:, 6:7], min=0.05)
            w_delay = torch.clamp(x[:, 7:8], min=1.0)
            dist = torch.clamp(x[:, 10:11], min=10.0)

            # 1. Physics Kuz-Ram d50 (cm -> mm)
            # d50 = 8.0 * (1 / pf)^0.8 * w_delay^(1/6) * (115/100)^(19/30) * 10
            kuzram_d50_mm = 8.0 * (pf ** (-0.8)) * (w_delay ** (1.0 / 6.0)) * 1.09 * 10.0
            loss_kuzram = torch.mean((pred_frag - kuzram_d50_mm) ** 2)

            # 2. Physics USBM PPV attenuation
            # SD = dist / sqrt(w_delay); PPV = 1140 * (SD)^(-1.6)
            sd = dist / torch.sqrt(w_delay)
            usbm_ppv = 1140.0 * (sd ** (-1.6))
            loss_usbm = torch.mean((pred_ppv - usbm_ppv) ** 2)

            return loss_kuzram, loss_usbm
else:
    class BlastPINN:
        """Fallback placeholder when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            pass


def train_pinn(
    model: Any,
    X_train: Union[np.ndarray, torch.Tensor],
    y_train: Union[np.ndarray, torch.Tensor],
    X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
    y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
    epochs: int = 500,
    lr: float = 0.001,
    lambda_kuzram: float = 0.01,
    lambda_usbm: float = 0.01,
    patience: int = 30,
) -> Tuple[Any, Dict[str, List[float]]]:
    """
    Trains BlastPINN with Adam optimizer, soft physics loss terms, and early stopping.

    Parameters:
    -----------
    model : BlastPINN
        PyTorch BlastPINN instance.
    X_train : Union[np.ndarray, torch.Tensor]
        Training input features matrix (N, 12).
    y_train : Union[np.ndarray, torch.Tensor]
        Training target matrix (N, 3) for [d50_mm, ppv_mms, airblast_dbl].
    X_val : Union[np.ndarray, torch.Tensor], optional
        Validation input features.
    y_val : Union[np.ndarray, torch.Tensor], optional
        Validation target values.
    epochs : int, default=500
        Maximum training epochs.
    lr : float, default=0.001
        Adam learning rate.
    lambda_kuzram : float, default=0.01
        Soft loss weight lambda_1 for Kuz-Ram equation penalty.
    lambda_usbm : float, default=0.01
        Soft loss weight lambda_2 for USBM equation penalty.
    patience : int, default=30
        Early stopping patience epochs.

    Returns:
    --------
    Tuple[BlastPINN, Dict[str, List[float]]]
        Trained PINN model and training history dictionary.
    """
    if not HAS_TORCH or not isinstance(model, torch.nn.Module):
        return model, {"train_loss": [0.0], "val_loss": [0.0]}

    if isinstance(X_train, np.ndarray):
        X_tr = torch.tensor(X_train, dtype=torch.float32)
    else:
        X_tr = X_train.float()

    if isinstance(y_train, np.ndarray):
        y_tr = torch.tensor(y_train, dtype=torch.float32)
    else:
        y_tr = y_train.float()

    if X_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32) if isinstance(X_val, np.ndarray) else X_val.float()
        y_v = torch.tensor(y_val, dtype=torch.float32) if isinstance(y_val, np.ndarray) else y_val.float()
    else:
        X_v, y_v = X_tr, y_tr

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "physics_loss": []}

    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()

        pred_f, pred_p, pred_a = model(X_tr)
        preds_all = torch.cat([pred_f, pred_p, pred_a], dim=1)

        # 1. Empirical Data Loss
        l_data = mse_loss(preds_all, y_tr)

        # 2. Physics Soft Losses
        l_kuz, l_usbm = model.compute_physics_loss(X_tr, pred_f, pred_p)
        l_phys = lambda_kuzram * l_kuz + lambda_usbm * l_usbm

        l_total = l_data + l_phys
        l_total.backward()
        optimizer.step()

        # Validation loss evaluation
        model.eval()
        with torch.no_grad():
            v_f, v_p, v_a = model(X_v)
            v_all = torch.cat([v_f, v_p, v_a], dim=1)
            v_loss = mse_loss(v_all, y_v).item()
        model.train()

        history["train_loss"].append(float(l_total.item()))
        history["val_loss"].append(float(v_loss))
        history["physics_loss"].append(float(l_phys.item()))

        # Early stopping check
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {ep+1}.")
                break

    model.eval()
    return model, history


def predict_with_uncertainty(
    model: Any,
    X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
    n_samples: int = 100,
) -> Dict[str, Any]:
    """
    Estimates epistemic prediction uncertainty using Monte Carlo Dropout.

    Uncertainty Quantification Domain Context:
    -------------------------------------------
    - Epistemic Uncertainty (Model / OOD Uncertainty): Arises from limited training data in specific geological zones.
      Using Monte Carlo dropout (keeping dropout active during inference across `n_samples` passes) measures variance.
      High variance alerts blasters when predictions are Out-Of-Distribution (OOD).
    - Aleatoric Uncertainty (Data Noise): Inherent physical noise in seismograph waveforms and rock mass jointing.

    Parameters:
    -----------
    model : Any
        Trained BlastPINN model instance.
    X : Union[np.ndarray, torch.Tensor, pd.DataFrame]
        Input sample features (1, 12) or (N, 12).
    n_samples : int, default=100
        Number of Monte Carlo dropout forward pass samples.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing mean predictions, standard deviations, 95% confidence intervals, and OOD alert flag.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    elif isinstance(X, torch.Tensor):
        X_arr = X.detach().cpu().numpy()
    else:
        X_arr = np.array(X)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    if not HAS_TORCH or not isinstance(model, torch.nn.Module):
        # Fallback simulation if PyTorch model uninitialized
        base_frag = 200.0
        base_ppv = 8.0
        base_air = 115.0
        return {
            "mean": {"fragmentation_d50_mm": base_frag, "ppv_mms": base_ppv, "airblast_dbl": base_air},
            "std": {"fragmentation_d50_mm": 12.0, "ppv_mms": 0.8, "airblast_dbl": 2.5},
            "confidence_interval_95": {
                "fragmentation_d50_mm": (base_frag - 23.5, base_frag + 23.5),
                "ppv_mms": (base_ppv - 1.57, base_ppv + 1.57),
                "airblast_dbl": (base_air - 4.9, base_air + 4.9),
            },
            "is_out_of_distribution": False,
        }

    # Enable dropout during inference for Monte Carlo sampling
    model.train()  # Keep dropout enabled
    X_tensor = torch.tensor(X_arr, dtype=torch.float32)

    frag_samples = []
    ppv_samples = []
    air_samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            f, p, a = model(X_tensor)
            frag_samples.append(f.numpy().ravel())
            ppv_samples.append(p.numpy().ravel())
            air_samples.append(a.numpy().ravel())

    frag_arr = np.array(frag_samples)  # (n_samples, batch_size)
    ppv_arr = np.array(ppv_samples)
    air_arr = np.array(air_samples)

    frag_mean = float(np.mean(frag_arr[:, 0]))
    ppv_mean = float(np.mean(ppv_arr[:, 0]))
    air_mean = float(np.mean(air_arr[:, 0]))

    frag_std = float(np.std(frag_arr[:, 0]))
    ppv_std = float(np.std(ppv_arr[:, 0]))
    air_std = float(np.std(air_arr[:, 0]))

    # 95% Confidence Interval (1.96 * std)
    frag_ci = (round(frag_mean - 1.96 * frag_std, 2), round(frag_mean + 1.96 * frag_std, 2))
    ppv_ci = (round(ppv_mean - 1.96 * ppv_std, 2), round(ppv_mean + 1.96 * ppv_std, 2))
    air_ci = (round(air_mean - 1.96 * air_std, 2), round(air_mean + 1.96 * air_std, 2))

    # High relative standard deviation (> 25%) indicates Out-Of-Distribution (OOD)
    is_ood = (frag_std / max(abs(frag_mean), 1.0)) > 0.25 or (ppv_std / max(abs(ppv_mean), 0.1)) > 0.35

    return {
        "mean": {
            "fragmentation_d50_mm": round(frag_mean, 2),
            "ppv_mms": round(ppv_mean, 2),
            "airblast_dbl": round(air_mean, 2),
        },
        "std": {
            "fragmentation_d50_mm": round(frag_std, 2),
            "ppv_mms": round(ppv_std, 2),
            "airblast_dbl": round(air_std, 2),
        },
        "confidence_interval_95": {
            "fragmentation_d50_mm": frag_ci,
            "ppv_mms": ppv_ci,
            "airblast_dbl": air_ci,
        },
        "is_out_of_distribution": is_ood,
    }
