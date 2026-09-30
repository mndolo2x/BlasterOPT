"""
Dust generation and dispersion model for open-pit blasting (Tyupin & Bolotova, 2026).
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class DustModel:
    """
    Dust generation and dispersion model for open-pit blasting.

    Reference: Tyupin, V.N. & Bolotova, Yu.N. (2026). "Predicting dust
    generation parameters in blasting borehole charges in open-pit mines."
    Russian Mining Industry, (1), 62-67.

    Key findings:
    - Specific volume of dust: (2.29–3.47) × 10⁻³ m³ per 1 m of charge
    - Specific weight of dust: 0.18–0.25 kg per 1 kg of explosive
    - Dust particle size range: 1.0 μm to 500 μm
    """

    def __init__(
        self,
        explosive_type: str = "ANFO",
        charge_mass_kg: float = 0.0,
        rock_type: str = "kimberlite",
    ) -> None:
        self.explosive_type = explosive_type.lower()
        self.charge_mass_kg = charge_mass_kg
        self.rock_type = rock_type.lower()

    def calculate_dust_mass(self, total_explosive_kg: float) -> Dict[str, Any]:
        """
        Calculate total dust mass generated.

        Formula (Tyupin & Bolotova 2026):
            dust_mass_kg = explosive_mass_kg × dust_factor

        where dust_factor ranges from 0.18 to 0.25 kg/kg for ANFO.
        For emulsion, dust_factor is approximately 0.15–0.20.

        Returns:
            {
                "dust_mass_kg": float,
                "dust_mass_range_kg": (float, float),
                "factor_used": float,
            }
        """
        if total_explosive_kg <= 0:
            raise ValueError(f"total_explosive_kg ({total_explosive_kg}) must be positive")

        if "emulsion" in self.explosive_type:
            factor_used = 0.18
            f_min, f_max = 0.15, 0.20
        else:  # ANFO default
            factor_used = 0.22
            f_min, f_max = 0.18, 0.25

        dust_mass_kg = total_explosive_kg * factor_used
        min_mass = total_explosive_kg * f_min
        max_mass = total_explosive_kg * f_max

        return {
            "dust_mass_kg": round(dust_mass_kg, 2),
            "dust_mass_range_kg": (round(min_mass, 2), round(max_mass, 2)),
            "factor_used": factor_used,
        }

    def calculate_dust_volume(self, charge_length_m: float) -> Dict[str, Any]:
        """
        Calculate specific dust volume.

        Formula (Tyupin & Bolotova 2026):
            dust_volume_m3 = charge_length_m × specific_volume

        where specific_volume = (2.29–3.47) × 10⁻³ m³/m.

        Returns:
            {
                "dust_volume_m3": float,
                "specific_volume_used": float,
            }
        """
        if charge_length_m <= 0:
            raise ValueError(f"charge_length_m ({charge_length_m}) must be positive")

        # Specific volume = 2.88e-3 m3/m (mean of 2.29e-3 and 3.47e-3)
        spec_vol = 2.88e-3
        dust_volume = charge_length_m * spec_vol

        return {
            "dust_volume_m3": round(dust_volume, 5),
            "specific_volume_used": spec_vol,
        }

    def calculate_dust_cloud_radius(
        self, charge_mass_kg: float, wind_speed_m_s: float
    ) -> Dict[str, Any]:
        """
        Estimate the radius of the dust cloud after blasting.

        Simplified model based on:
            R_cloud = k × (charge_mass_kg)^(1/3) × (1 + wind_speed/5)

        where k = 15–25 for open-pit blasting (empirical).
        Wind speed increases dispersion.

        Returns:
            {
                "radius_m": float,
                "downwind_distance_m": float,
                "settling_time_s": float,
            }
        """
        if charge_mass_kg <= 0:
            raise ValueError(f"charge_mass_kg ({charge_mass_kg}) must be positive")
        if wind_speed_m_s < 0:
            raise ValueError(f"wind_speed_m_s ({wind_speed_m_s}) cannot be negative")

        k = 20.0
        r_cloud = k * (charge_mass_kg ** (1.0 / 3.0)) * (1.0 + wind_speed_m_s / 5.0)

        # Downwind travel in 600s
        downwind_dist = wind_speed_m_s * 600.0

        # Settling time for fine dust (mean 20 micron)
        settling_time = 3600.0 / (1.0 + wind_speed_m_s * 0.2)

        return {
            "radius_m": round(r_cloud, 1),
            "downwind_distance_m": round(downwind_dist, 1),
            "settling_time_s": round(settling_time, 1),
        }

    def calculate_pm10_concentration(
        self, dust_mass_kg: float, distance_m: float, wind_speed_m_s: float
    ) -> Dict[str, Any]:
        """
        Estimate PM10 concentration at a given distance.

        Gaussian plume model (simplified):
            C(x) = Q / (2π × σ_y × σ_z × u) × exp(-0.5 × (H/σ_z)²)

        where:
            Q = source strength (g/s)
            σ_y, σ_z = dispersion coefficients (Pasquill-Gifford)
            u = wind speed (m/s)
            H = effective release height (m)

        Returns:
            {
                "pm10_concentration_mg_m3": float,
                "distance_m": float,
                "exceeds_limit": bool,  # WHO limit: 0.05 mg/m³ (24-hr)
            }
        """
        if dust_mass_kg <= 0:
            raise ValueError(f"dust_mass_kg ({dust_mass_kg}) must be positive")
        if distance_m <= 0:
            raise ValueError(f"distance_m ({distance_m}) must be positive")
        if wind_speed_m_s <= 0:
            raise ValueError(f"wind_speed_m_s ({wind_speed_m_s}) must be positive")

        # PM10 fraction is ~40% of total dust mass
        pm10_mass_g = dust_mass_kg * 0.40 * 1000.0
        q_rate_g_s = pm10_mass_g / 10.0  # 10s release duration

        x = distance_m
        sigma_y = 0.22 * x * ((1.0 + 0.0001 * x) ** (-0.5))
        sigma_z = 0.20 * x
        h_eff = 20.0  # effective plume height

        c_g_m3 = (q_rate_g_s / (2.0 * math.pi * sigma_y * sigma_z * wind_speed_m_s)) * math.exp(-0.5 * ((h_eff / sigma_z) ** 2))
        c_mg_m3 = c_g_m3 * 1000.0  # g/m3 -> mg/m3

        threshold_mg = 0.05  # WHO 24-hr limit: 0.05 mg/m3

        return {
            "pm10_concentration_mg_m3": round(c_mg_m3, 4),
            "distance_m": distance_m,
            "exceeds_limit": bool(c_mg_m3 > threshold_mg),
        }


def predict_dust_dispersion(
    total_explosive_mass_kg: float,
    wind_speed_m_s: float,
    distance_m: float,
    pasquill_stability_class: str = "C",
) -> Dict[str, Any]:
    """Legacy helper wrapper for DustModel."""
    model = DustModel(charge_mass_kg=total_explosive_mass_kg)
    dust_res = model.calculate_dust_mass(total_explosive_mass_kg)
    pm10_res = model.calculate_pm10_concentration(dust_res["dust_mass_kg"], distance_m, wind_speed_m_s)

    return {
        "emission_pm10_kg": round(dust_res["dust_mass_kg"] * 0.4, 2),
        "concentration_ug_m3": round(pm10_res["pm10_concentration_mg_m3"] * 1000.0, 2),
        "exceeds_threshold": pm10_res["exceeds_limit"],
        "distance_m": distance_m,
    }
