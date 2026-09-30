"""
Airblast noise overpressure prediction at receptors per Siskind et al. (1980).
"""

import math
from typing import Dict, Any


def predict_noise_overpressure(
    max_charge_per_delay_kg: float,
    distance_m: float,
    site_k_noise: float = 165.0,
    beta_noise: float = 1.1,
) -> Dict[str, Any]:
    """
    Predict airblast overpressure noise (dBL) at sensitive receptor distance per Siskind et al. (1980, USBM RI 8485).

    Scaled distance:
        SD_noise = distance_m / (max_charge_per_delay_kg ** (1/3))

    Overpressure (kPa):
        P_kPa = site_k_noise * (SD_noise ** (-beta_noise))

    Noise level (dBL):
        dBL = 20 * log10(P_kPa / (2 * 10**-8)) + 120

    Returns:
        {
            "noise_dbl": float,
            "overpressure_kpa": float,
            "scaled_distance": float,
            "exceeds_botswana_limit": bool,  # limit = 120 dBL
        }
    """
    if max_charge_per_delay_kg <= 0:
        raise ValueError(f"max_charge_per_delay_kg ({max_charge_per_delay_kg}) must be positive")
    if distance_m <= 0:
        raise ValueError(f"distance_m ({distance_m}) must be positive")

    # Formula (Siskind et al. 1980): SD = D / Q**(1/3)
    sd = distance_m / (max_charge_per_delay_kg ** (1.0 / 3.0))

    # Formula: P_kpa = K * SD**(-beta)
    p_kpa = site_k_noise * (sd ** (-beta_noise)) * 0.001

    # Formula: dBL = 20 * log10(P_pa / 2e-5)
    p_pa = p_kpa * 1000.0
    dbl = 20.0 * math.log10(max(1e-9, p_pa) / 2e-5)

    limit = 120.0  # Botswana Department of Mines threshold
    exceeds = dbl > limit

    return {
        "noise_dbl": round(dbl, 1),
        "overpressure_kpa": round(p_kpa, 4),
        "scaled_distance": round(sd, 2),
        "exceeds_botswana_limit": exceeds,
    }
