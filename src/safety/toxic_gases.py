"""
Toxic gas emission prediction (CO, NOx, CO2) per Rowland & Mainiero (2000)
and thermochemical equilibrium calculations per Suceska et al. (2021).
"""

import math
from typing import Dict, Any, Optional


class GasModel:
    """
    Toxic gas (CO, NOx) prediction for blasting.

    Reference: Suceska, M. et al. (2021). "Prediction of concentration of
    toxic gases produced by detonation of commercial explosives by
    thermochemical equilibrium calculations." Defence Technology.

    Also: Newgold sustainability report, Emission Factor for NOx:
        NOx (kg/blast) = kg of emulsion/blast × EF (g of NOx/kg of emulsion)
    """

    # Emission factors (g of gas per kg of explosive)
    EMISSION_FACTORS = {
        "ANFO": {"CO": 15.0, "NOx": 8.0, "CO2": 180.0},
        "Heavy ANFO": {"CO": 12.0, "NOx": 6.0, "CO2": 200.0},
        "Emulsion": {"CO": 10.0, "NOx": 5.0, "CO2": 220.0},
    }

    # Molecular weights (g/mol)
    MOLECULAR_WEIGHTS = {
        "CO": 28.01,
        "NOx": 46.00,  # Evaluated as NO2 equivalent
        "CO2": 44.01,
    }

    # Regulatory exposure limits (ppm 8-hr TWA)
    EXPOSURE_LIMITS = {
        "CO": 50.0,
        "NOx": 3.0,   # NO2 limit
        "CO2": 5000.0,
    }

    def __init__(self, explosive_type: str = "ANFO") -> None:
        if explosive_type not in self.EMISSION_FACTORS:
            valid_types = ", ".join(self.EMISSION_FACTORS.keys())
            raise ValueError(f"Unknown explosive_type '{explosive_type}'. Valid types: {valid_types}")
        self.explosive_type = explosive_type
        self.factors = self.EMISSION_FACTORS[explosive_type]

    def calculate_gas_emissions(self, explosive_mass_kg: float) -> Dict[str, float]:
        """
        Calculate total gas emissions.

        Formula:
            gas_mass_kg = explosive_mass_kg × EF_gas / 1000

        Returns:
            {
                "CO_kg": float,
                "NOx_kg": float,
                "CO2_kg": float,
                "total_toxic_kg": float,
            }
        """
        if explosive_mass_kg < 0:
            raise ValueError("explosive_mass_kg cannot be negative")

        co_kg = explosive_mass_kg * (self.factors["CO"] / 1000.0)
        nox_kg = explosive_mass_kg * (self.factors["NOx"] / 1000.0)
        co2_kg = explosive_mass_kg * (self.factors["CO2"] / 1000.0)
        total_toxic_kg = co_kg + nox_kg

        return {
            "CO_kg": round(co_kg, 4),
            "NOx_kg": round(nox_kg, 4),
            "CO2_kg": round(co2_kg, 4),
            "total_toxic_kg": round(total_toxic_kg, 4),
        }

    def calculate_gas_concentration(
        self,
        gas_mass_kg: float,
        gas_type: str,
        ventilation_rate_m3_s: float,
        volume_m3: float,
        time_s: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Calculate gas concentration in a confined space (underground).

        Formula (dilution model):
            C(t) = (Q / V) × (1 - exp(-V × t / V_vent)) -- or standard dilution decay C(t) = C0 * exp(-Q_vent * t / V)
            Where C0 (ppm) = (gas_mass_kg * 1e6 * 24.45 / (MW * volume_m3))

        Returns:
            {
                "concentration_ppm": float,
                "time_to_safe_s": float,
                "exceeds_limit": bool,  # CO limit: 50 ppm (8-hr TWA)
                                       # NO2 limit: 3 ppm (8-hr TWA)
            }
        """
        if gas_type not in self.EXPOSURE_LIMITS:
            raise ValueError(f"Unsupported gas_type '{gas_type}'")
        if volume_m3 <= 0:
            raise ValueError("volume_m3 must be positive")
        if ventilation_rate_m3_s <= 0:
            raise ValueError("ventilation_rate_m3_s must be positive")

        mw = self.MOLECULAR_WEIGHTS[gas_type]
        limit_ppm = self.EXPOSURE_LIMITS[gas_type]

        # Initial concentration C0 in ppm: (m_gas / MW) * 24.45 L/mol / (volume_m3 * 1000 L/m3) * 1e6
        # C0 = (gas_mass_kg * 1e3 g / mw g/mol) * 24.45 L/mol / (volume_m3 * 1e3 L) * 1e6
        # C0 = (gas_mass_kg * 24.45 * 1e6) / (mw * volume_m3)
        c0_ppm = (gas_mass_kg * 24.45 * 1e6) / (mw * volume_m3)

        # Concentration at time t with ventilation flushing rate k = ventilation_rate_m3_s / volume_m3
        k = ventilation_rate_m3_s / volume_m3
        conc_ppm = c0_ppm * math.exp(-k * time_s)

        # Time to reach limit: limit_ppm = C0 * exp(-k * t_safe) => t_safe = -ln(limit_ppm / C0) / k
        if c0_ppm <= limit_ppm:
            time_to_safe_s = 0.0
        else:
            time_to_safe_s = -math.log(limit_ppm / c0_ppm) / k

        return {
            "concentration_ppm": round(conc_ppm, 2),
            "time_to_safe_s": round(max(0.0, time_to_safe_s), 1),
            "exceeds_limit": conc_ppm > limit_ppm,
        }

    def calculate_reentry_time(
        self,
        gas_mass_kg: float,
        gas_type: str,
        ventilation_rate_m3_s: float,
        volume_m3: float,
        target_ppm: Optional[float] = None,
    ) -> float:
        """
        Calculate safe re-entry time after blasting.

        Returns:
            time_seconds: float
        """
        if target_ppm is None:
            target_ppm = self.EXPOSURE_LIMITS.get(gas_type, 50.0)

        mw = self.MOLECULAR_WEIGHTS[gas_type]
        c0_ppm = (gas_mass_kg * 24.45 * 1e6) / (mw * volume_m3)

        if c0_ppm <= target_ppm:
            return 0.0

        k = ventilation_rate_m3_s / volume_m3
        t_seconds = -math.log(target_ppm / c0_ppm) / k
        return round(max(0.0, t_seconds), 1)

    def calculate_ventilation_requirement(
        self,
        gas_mass_kg: float,
        gas_type: str,
        target_ppm: float,
        time_s: float,
    ) -> float:
        """
        Calculate required ventilation rate to dilute gas to target
        concentration within a specified time.

        Formula:
            V_req = (gas_mass_kg × 1e6) / (target_ppm × time_s × MW_gas)

        Returns:
            ventilation_rate_m3_s: float
        """
        mw = self.MOLECULAR_WEIGHTS.get(gas_type, 28.01)
        if target_ppm <= 0 or time_s <= 0 or mw <= 0:
            raise ValueError("target_ppm, time_s, and molecular weight must be positive")

        ventilation_rate = (gas_mass_kg * 1e6) / (target_ppm * time_s * mw)
        return round(ventilation_rate, 4)


def predict_toxic_gases(
    explosive_mass_kg: float,
    anfo_fuel_oil_pct: float = 6.0,
    confinement_quality: str = "good",
) -> Dict[str, Any]:
    """
    Predict CO and NOx emissions from blasting explosives per Rowland & Mainiero (2000).

    Ideal ANFO (6% fuel oil):
        CO = 10 L / kg explosive
        NOx = 5 L / kg explosive

    Off-spec ANFO (<5% or >7% FO) or wet ground increases CO and NOx up to 5x (Rowland & Mainiero 2000).

    Returns:
        {
            "co_emissions_liters": float,
            "nox_emissions_liters": float,
            "co_concentration_ppm_at_100m": float,
            "nox_concentration_ppm_at_100m": float,
            "hazard_level": str,
        }
    """
    if explosive_mass_kg <= 0:
        raise ValueError(f"explosive_mass_kg ({explosive_mass_kg}) must be positive")

    # Formula: base CO = 10 L/kg, base NOx = 5 L/kg
    base_co_l_per_kg = 10.0
    base_nox_l_per_kg = 5.0

    # Off-spec oxygen balance penalty multiplier
    fo_dev = abs(anfo_fuel_oil_pct - 6.0)
    mult = 1.0 + fo_dev * 0.5
    if confinement_quality.lower() == "poor":
        mult *= 2.0

    co_l = explosive_mass_kg * base_co_l_per_kg * mult
    nox_l = explosive_mass_kg * base_nox_l_per_kg * mult

    # Dilution at 100m in semi-spherical plume volume (radius=100m)
    plume_vol_m3 = (2.0 / 3.0) * 3.14159 * (100.0 ** 3)
    co_ppm = (co_l / plume_vol_m3) * 1000.0
    nox_ppm = (nox_l / plume_vol_m3) * 1000.0

    hazard = "low"
    if co_ppm > 50.0 or nox_ppm > 5.0:  # ACGIH STEL thresholds
        hazard = "high"
    elif co_ppm > 25.0 or nox_ppm > 2.5:
        hazard = "medium"

    return {
        "co_emissions_liters": round(co_l, 1),
        "nox_emissions_liters": round(nox_l, 1),
        "co_concentration_ppm_at_100m": round(co_ppm, 2),
        "nox_concentration_ppm_at_100m": round(nox_ppm, 2),
        "hazard_level": hazard,
    }
