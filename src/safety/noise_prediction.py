"""
Airblast noise overpressure prediction at receptors per Siskind et al. (1980)
and Linehan & Wiss (1980) USBM models.
"""

import math
from typing import Dict, Any, List, Optional


class NoiseModel:
    """
    Noise prediction from blasting.

    Reference: Linehan, P. & Wiss, J.F. (1980). "Vibration and airblast
    from blasting." US Bureau of Mines.
    EPA WA (2011). "Noise Prediction Modelling Report."

    Formula (Linehan & Wiss 1980):
        P = 6.31 × e^(-B) × (D / W^(1/3))^(-1.16)
        SPL = 20 × log₁₀(P) + 154

    where:
        P = peak overpressure (kPa)
        D = distance from blast to receiver (m)
        W = maximum charge weight per delay (kg)
        B = scaled depth of burial (m/kg^(1/3))
    """

    def __init__(
        self,
        charge_per_delay_kg: float,
        depth_of_burial_m: float,
        explosive_type: str = "ANFO",
    ) -> None:
        if charge_per_delay_kg <= 0:
            raise ValueError(f"charge_per_delay_kg ({charge_per_delay_kg}) must be positive")
        if depth_of_burial_m < 0:
            raise ValueError(f"depth_of_burial_m ({depth_of_burial_m}) cannot be negative")

        self.charge_per_delay_kg = charge_per_delay_kg
        self.depth_of_burial_m = depth_of_burial_m
        self.explosive_type = explosive_type

        # Scaled depth of burial B = depth / (W^(1/3))
        w_third = charge_per_delay_kg ** (1.0 / 3.0)
        self.b_scaled_burial = depth_of_burial_m / w_third if w_third > 0 else 0.0

    def calculate_peak_overpressure(self, distance_m: float) -> Dict[str, float]:
        """
        Calculate peak overpressure at a given distance.

        Formula:
            P = 6.31 * exp(-B) * (D / W^(1/3))^(-1.16)  [kPa]
            SPL = 20 * log10(P * 1000 / 2e-5) or 20 * log10(P) + 154

        Returns:
            {
                "overpressure_kpa": float,
                "spl_db": float,
                "distance_m": float,
            }
        """
        if distance_m <= 0:
            raise ValueError(f"distance_m ({distance_m}) must be positive")

        w_third = self.charge_per_delay_kg ** (1.0 / 3.0)
        scaled_dist = distance_m / w_third

        # Formula: P = 6.31 * exp(-B) * SD^(-1.16) [kPa]
        p_kpa = 6.31 * math.exp(-self.b_scaled_burial) * (scaled_dist ** -1.16)

        # Sound pressure level in dB (20*log10(P_kpa) + 154 or 20*log10(P_pa / 2e-5))
        p_pa = p_kpa * 1000.0
        spl_db = 20.0 * math.log10(max(1e-9, p_pa) / 2e-5)

        return {
            "overpressure_kpa": round(p_kpa, 6),
            "spl_db": round(spl_db, 2),
            "distance_m": float(distance_m),
        }

    def calculate_noise_at_distance(self, distance_m: float) -> Dict[str, Any]:
        """
        Calculate SPL (dB) at a given distance.

        Returns:
            {
                "spl_db": float,
                "distance_m": float,
                "exceeds_limit": bool,  # Botswana limit: 120 dB
            }
        """
        peak_res = self.calculate_peak_overpressure(distance_m)
        spl_db = peak_res["spl_db"]
        limit_db = 120.0

        return {
            "spl_db": spl_db,
            "distance_m": float(distance_m),
            "exceeds_limit": spl_db > limit_db,
        }

    def calculate_zone_of_impact(self, limit_db: float = 120.0) -> Dict[str, float]:
        """
        Calculate the distance at which noise falls below a limit.

        Returns:
            {
                "distance_to_limit_m": float,
                "limit_db": float,
            }
        """
        # Convert limit_db back to P_kpa
        # limit_db = 20 * log10(P_pa / 2e-5) => P_pa = 2e-5 * 10^(limit_db / 20)
        p_target_pa = 2e-5 * (10.0 ** (limit_db / 20.0))
        p_target_kpa = p_target_pa / 1000.0

        # P = 6.31 * exp(-B) * (D / W^(1/3))^(-1.16)
        # (D / W^(1/3))^(-1.16) = P / (6.31 * exp(-B))
        # SD = (P / (6.31 * exp(-B)))^(-1 / 1.16)
        c = 6.31 * math.exp(-self.b_scaled_burial)
        if c <= 0 or p_target_kpa <= 0:
            sd_target = 1.0
        else:
            sd_target = (p_target_kpa / c) ** (-1.0 / 1.16)

        w_third = self.charge_per_delay_kg ** (1.0 / 3.0)
        distance_m = sd_target * w_third

        return {
            "distance_to_limit_m": round(distance_m, 2),
            "limit_db": float(limit_db),
        }

    def calculate_equivalent_continuous_level(
        self,
        events: List[Dict[str, float]],
        duration_s: float,
    ) -> float:
        """
        Calculate Leq (equivalent continuous sound level).

        Formula:
            Leq = 10 × log₁₀[(1/T) × Σ(10^(0.1 × Lp_i) × t_i)]

        where:
            T = total duration (s)
            Lp_i = sound pressure level of event i (dB)
            t_i = duration of event i (s)

        Returns:
            leq_db: float
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if not events:
            return 0.0

        total_energy = 0.0
        for event in events:
            lp_i = event.get("spl_db", event.get("lp_db", 0.0))
            t_i = event.get("duration_s", 1.0)
            total_energy += (10.0 ** (0.1 * lp_i)) * t_i

        leq_db = 10.0 * math.log10(total_energy / duration_s)
        return round(leq_db, 2)


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
