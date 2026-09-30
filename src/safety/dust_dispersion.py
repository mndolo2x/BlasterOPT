"""
Dust generation and Gaussian plume dispersion modeling (US EPA AP-42, Pasquill-Gifford 1961).
"""

import math
import numpy as np
from typing import Dict, Any


def predict_dust_dispersion(
    total_explosive_mass_kg: float,
    wind_speed_m_s: float,
    distance_m: float,
    pasquill_stability_class: str = "C",
) -> Dict[str, Any]:
    """
    Predict dust generation and PM10 concentration downwind using US EPA AP-42 & Gaussian Plume Model (Pasquill-Gifford 1961).

    US EPA AP-42 emission factor for blasting (kg / blast):
        E_pm10_kg = 0.00022 * (total_explosive_mass_kg ** 1.5)

    Gaussian Plume Dispersion (Pasquill-Gifford 1961):
        C(x) = (E / (pi * u * sigma_y * sigma_z)) * 10^6  [ug/m3]

    For stability C at distance x:
        sigma_y = 0.22 * x * (1 + 0.0001 * x)^(-0.5)
        sigma_z = 0.20 * x

    Returns:
        {
            "emission_pm10_kg": float,
            "concentration_ug_m3": float,
            "exceeds_threshold": bool,  # threshold = 150 ug/m3
            "distance_m": float,
        }
    """
    if total_explosive_mass_kg <= 0:
        raise ValueError(f"total_explosive_mass_kg ({total_explosive_mass_kg}) must be positive")
    if wind_speed_m_s <= 0:
        raise ValueError(f"wind_speed_m_s ({wind_speed_m_s}) must be positive")
    if distance_m <= 0:
        raise ValueError(f"distance_m ({distance_m}) must be positive")

    # Formula (US EPA AP-42): E = 0.00022 * Q^1.5
    e_pm10_kg = 0.00022 * (total_explosive_mass_kg ** 1.5)

    x = distance_m
    sigma_y = 0.22 * x * ((1.0 + 0.0001 * x) ** (-0.5))
    sigma_z = 0.20 * x

    # Formula (Pasquill-Gifford 1961): C = (E_sec / (pi * u * sy * sz)) * 1e6
    e_rate_g_s = (e_pm10_kg * 1000.0) / 10.0  # 10 second blast release duration
    c_ug_m3 = (e_rate_g_s / (math.pi * wind_speed_m_s * sigma_y * sigma_z)) * 1e6

    threshold = 150.0  # WHO 24-hr PM10 threshold ug/m3
    exceeds = c_ug_m3 > threshold

    return {
        "emission_pm10_kg": round(e_pm10_kg, 2),
        "concentration_ug_m3": round(c_ug_m3, 2),
        "exceeds_threshold": exceeds,
        "distance_m": distance_m,
    }
