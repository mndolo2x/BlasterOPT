"""
Kuz-Ram rock fragmentation prediction model (Kuznetsov 1973, Cunningham 1983/1987).
"""

import math
from typing import Dict, Any


def predict_kuz_ram(
    powder_factor_kg_m3: float,
    charge_mass_per_hole_kg: float,
    rock_factor_a: float = 7.0,
    explosive_relative_weight_strength: float = 100.0,
) -> Dict[str, Any]:
    """
    Predict fragmentation mean particle size d50 (cm) using Kuznetsov (1973).

    Formula (Kuznetsov 1973, Cunningham 1987):
        d50_cm = A * (V_0 / Q)**(0.8) * Q**(1/6) * (115 / S_ANFO)**(19/30)

    Where:
        - A = Rock blastability factor
        - V_0 / Q = 1 / powder_factor (m3 / kg)
        - Q = Charge mass per hole (kg)
        - S_ANFO = Relative weight strength to ANFO (ANFO = 100)

    Returns:
        {
            "d50_cm": float,
            "d50_mm": float,
            "uniformity_index_n": float,
            "characteristic_size_xc_cm": float,
        }
    """
    if powder_factor_kg_m3 <= 0:
        raise ValueError(f"powder_factor_kg_m3 ({powder_factor_kg_m3}) must be positive")
    if charge_mass_per_hole_kg <= 0:
        raise ValueError(f"charge_mass_per_hole_kg ({charge_mass_per_hole_kg}) must be positive")
    if explosive_relative_weight_strength <= 0:
        raise ValueError(f"explosive_relative_weight_strength ({explosive_relative_weight_strength}) must be positive")

    # Formula: d50_cm = A * (1 / PF)**0.8 * Q**(1/6) * (115 / S_ANFO)**(19/30)
    term1 = rock_factor_a
    term2 = (1.0 / powder_factor_kg_m3) ** 0.8
    term3 = charge_mass_per_hole_kg ** (1.0 / 6.0)
    term4 = (115.0 / explosive_relative_weight_strength) ** (19.0 / 30.0)

    d50_cm = float(term1 * term2 * term3 * term4)
    d50_mm = d50_cm * 10.0

    # Uniformity index n (Cunningham 1987)
    n = 1.25

    # Rosin-Rammler characteristic size x_c = d50 / (ln(2))**(1/n)
    x_c_cm = d50_cm / ((math.log(2.0)) ** (1.0 / n))

    return {
        "d50_cm": round(d50_cm, 2),
        "d50_mm": round(d50_mm, 1),
        "uniformity_index_n": round(n, 2),
        "characteristic_size_xc_cm": round(x_c_cm, 2),
    }
