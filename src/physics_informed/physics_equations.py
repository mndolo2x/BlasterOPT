"""
Differentiable Physics Equations Submodule (PyTorch & NumPy).
Provides differentiable Kuz-Ram fragmentation and USBM vibration attenuation equations.
"""

import logging
import numpy as np
from typing import Union, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

logger = logging.getLogger(__name__)


def kuz_ram_x50_torch(
    rock_factor_a: torch.Tensor,
    powder_factor_k: torch.Tensor,
    charge_mass_q: torch.Tensor,
    rws_e: torch.Tensor = None
) -> torch.Tensor:
    """
    Differentiable Kuz-Ram mean fragment size X50 equation in PyTorch:
    X50 (cm) = A * (K)^(-0.8) * Q^(1/6) * (115 / E)^(19/30)

    Parameters:
    -----------
    rock_factor_a : torch.Tensor
        Rock blastability factor (A), typically 4.0 - 16.0.
    powder_factor_k : torch.Tensor
        Powder factor (K) in kg/m3.
    charge_mass_q : torch.Tensor
        Charge mass per hole (Q) in kg.
    rws_e : torch.Tensor, optional
        Relative Weight Strength (E) relative to ANFO=100 (default 100.0).

    Returns:
    --------
    torch.Tensor
        Predicted mean fragment size X50 in cm.
    """
    if rws_e is None:
        rws_e = torch.tensor(100.0, dtype=rock_factor_a.dtype, device=rock_factor_a.device)

    # Clamp tensors to valid positive domains to avoid log/pow domain errors
    k_safe = torch.clamp(powder_factor_k, min=0.10, max=5.0)
    q_safe = torch.clamp(charge_mass_q, min=1.0, max=5000.0)
    e_safe = torch.clamp(rws_e, min=50.0, max=200.0)

    # X50 (cm) = A * K^(-0.8) * Q^(1/6) * (115/E)^(19/30)
    x50_cm = rock_factor_a * torch.pow(k_safe, -0.8) * torch.pow(q_safe, 1.0 / 6.0) * torch.pow(115.0 / e_safe, 19.0 / 30.0)
    # Convert cm to mm (* 10.0)
    return x50_cm * 10.0


def usbm_ppv_torch(
    distance_d: torch.Tensor,
    max_charge_w: torch.Tensor,
    site_k: float = 1140.0,
    site_b: float = 1.60
) -> torch.Tensor:
    """
    Differentiable USBM Peak Particle Velocity (PPV) wave attenuation equation in PyTorch:
    PPV (mm/s) = K * (D / sqrt(W))^(-B)

    Parameters:
    -----------
    distance_d : torch.Tensor
        Monitoring distance (D) in meters.
    max_charge_w : torch.Tensor
        Maximum charge per delay (W) in kg.
    site_k : float, default=1140.0
        Site constant K.
    site_b : float, default=1.60
        Site exponent B.

    Returns:
    --------
    torch.Tensor
        Predicted PPV in mm/s.
    """
    d_safe = torch.clamp(distance_d, min=10.0, max=5000.0)
    w_safe = torch.clamp(max_charge_w, min=1.0, max=5000.0)

    scaled_distance = d_safe / torch.sqrt(w_safe)
    ppv_mms = site_k * torch.pow(scaled_distance, -site_b)
    return ppv_mms


def kuz_ram_x50_numpy(
    rock_factor_a: float,
    powder_factor_k: float,
    charge_mass_q: float,
    rws_e: float = 100.0
) -> float:
    """NumPy equivalent for Kuz-Ram X50 prediction in mm."""
    k_safe = max(0.10, powder_factor_k)
    q_safe = max(1.0, charge_mass_q)
    e_safe = max(50.0, rws_e)

    x50_cm = rock_factor_a * (k_safe ** -0.8) * (q_safe ** (1.0 / 6.0)) * ((115.0 / e_safe) ** (19.0 / 30.0))
    return float(x50_cm * 10.0)


def usbm_ppv_numpy(
    distance_d: float,
    max_charge_w: float,
    site_k: float = 1140.0,
    site_b: float = 1.60
) -> float:
    """NumPy equivalent for USBM PPV attenuation prediction in mm/s."""
    d_safe = max(10.0, distance_d)
    w_safe = max(1.0, max_charge_w)

    sd = d_safe / np.sqrt(w_safe)
    return float(site_k * (sd ** -site_b))
