"""
Differentiable Physics Equations Submodule (PyTorch & NumPy).
Provides differentiable Kuz-Ram fragmentation and USBM vibration attenuation equations.
"""

import logging
import numpy as np
from typing import Union, Tuple, Optional

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
    rws_e: torch.Tensor = None,
    rock_volume_v0: torch.Tensor = None
) -> torch.Tensor:
    """
    Differentiable Kuz-Ram mean fragment size X50 equation in PyTorch:
    X50 (cm) = A * (V0 / Q)^0.8 * Q^(1/6) * (115 / E)^(19/30)
    or X50 (cm) = A * (K)^(-0.8) * Q^(1/6) * (115 / E)^(19/30)

    Where V0 = rock volume per hole (m3), Q = charge weight per hole (kg),
    K = Q / V0 (powder factor kg/m3), E = relative weight strength (ANFO=100).
    """
    if rws_e is None:
        rws_e = torch.tensor(100.0, dtype=rock_factor_a.dtype, device=rock_factor_a.device)

    q_safe = torch.clamp(charge_mass_q, min=1.0, max=5000.0)
    e_safe = torch.clamp(rws_e, min=50.0, max=200.0)

    if rock_volume_v0 is not None:
        v0_safe = torch.clamp(rock_volume_v0, min=0.1, max=10000.0)
        vo_q_ratio = v0_safe / q_safe
        x50_cm = rock_factor_a * torch.pow(vo_q_ratio, 0.8) * torch.pow(q_safe, 1.0 / 6.0) * torch.pow(115.0 / e_safe, 19.0 / 30.0)
    else:
        k_safe = torch.clamp(powder_factor_k, min=0.10, max=5.0)
        x50_cm = rock_factor_a * torch.pow(k_safe, -0.8) * torch.pow(q_safe, 1.0 / 6.0) * torch.pow(115.0 / e_safe, 19.0 / 30.0)

    # Convert cm to mm (* 10.0)
    return x50_cm * 10.0


def swebrec_distribution_torch(
    x_sieve: torch.Tensor,
    x50: torch.Tensor,
    x_max: torch.Tensor,
    b_curve: float = 1.25
) -> torch.Tensor:
    """
    Differentiable Swebrec cumulative size distribution equation P(x) in PyTorch:
    P(x) = 100 / (1 + ( ln(x_max / x) / ln(x_max / x50) )^b )

    Parameters:
    -----------
    x_sieve : torch.Tensor
        Particle sieve size x (mm).
    x50 : torch.Tensor
        50% passing median size x50 (mm).
    x_max : torch.Tensor
        Maximum in-situ block/top size x_max (mm).
    b_curve : float, default=1.25
        Curve curvature parameter b.

    Returns:
    --------
    torch.Tensor
        Cumulative passing percentage P(x) in range [0, 100].
    """
    x_safe = torch.clamp(x_sieve, min=0.01)
    x50_safe = torch.clamp(x50, min=0.1)
    xmax_safe = torch.clamp(x_max, min=x50_safe + 10.0)

    # Calculate log ratio terms
    num_log = torch.log(xmax_safe / x_safe)
    den_log = torch.log(xmax_safe / x50_safe)

    ratio = torch.clamp(num_log / torch.clamp(den_log, min=1e-5), min=1e-5)
    f_term = torch.pow(ratio, b_curve)

    p_x = 100.0 / (1.0 + f_term)
    return torch.clamp(p_x, min=0.0, max=100.0)


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
    rws_e: float = 100.0,
    rock_volume_v0: Optional[float] = None
) -> float:
    """NumPy equivalent for Kuz-Ram X50 prediction in mm using V0/Q or K."""
    q_safe = max(1.0, charge_mass_q)
    e_safe = max(50.0, rws_e)

    if rock_volume_v0 is not None:
        v0_safe = max(0.1, rock_volume_v0)
        x50_cm = rock_factor_a * ((v0_safe / q_safe) ** 0.8) * (q_safe ** (1.0 / 6.0)) * ((115.0 / e_safe) ** (19.0 / 30.0))
    else:
        k_safe = max(0.10, powder_factor_k)
        x50_cm = rock_factor_a * (k_safe ** -0.8) * (q_safe ** (1.0 / 6.0)) * ((115.0 / e_safe) ** (19.0 / 30.0))

    return float(x50_cm * 10.0)


def swebrec_distribution_numpy(
    x_sieve: Union[float, np.ndarray],
    x50: float,
    x_max: float,
    b_curve: float = 1.25
) -> Union[float, np.ndarray]:
    """NumPy equivalent for Swebrec cumulative size distribution P(x)."""
    x_safe = np.maximum(0.01, x_sieve)
    x50_safe = max(0.1, x50)
    xmax_safe = max(x50_safe + 10.0, x_max)

    num_log = np.log(xmax_safe / x_safe)
    den_log = np.log(xmax_safe / x50_safe)

    ratio = np.maximum(1e-5, num_log / max(1e-5, den_log))
    f_term = np.power(ratio, b_curve)

    p_x = 100.0 / (1.0 + f_term)
    return np.clip(p_x, 0.0, 100.0)


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
