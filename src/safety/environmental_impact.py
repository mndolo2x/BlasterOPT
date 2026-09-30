"""
Environmental Impact Assessment (EIA) index matrix per ISO 14001 / UN EP guidelines.
"""

import os
from typing import Dict, Any, List, Optional
from src.safety.dust import DustModel
from src.safety.toxic_gases import GasModel
from src.safety.noise_prediction import NoiseModel


class EnvironmentalAssessment:
    """
    Comprehensive environmental impact assessment for blasting.

    Evaluates five impact categories:
        1. Dust (PM10, PM2.5)
        2. Toxic gases (CO, NOx, CO2)
        3. Noise (SPL, Leq)
        4. Vibration (PPV)
        5. Flyrock

    Each impact is scored on a 1-5 scale:
        1 = negligible, 2 = minor, 3 = moderate, 4 = major, 5 = severe
    """

    def __init__(self, blast_params: Dict[str, Any], receptor_distances: Dict[str, float]) -> None:
        """
        receptor_distances: dict of receptor name -> distance in meters
            e.g., {"village": 800, "road": 450, "camp": 1200}
        """
        if not receptor_distances:
            raise ValueError("receptor_distances dictionary cannot be empty")

        self.blast_params = blast_params
        self.receptor_distances = receptor_distances
        self.closest_distance_m = min(receptor_distances.values())
        if self.closest_distance_m <= 0:
            raise ValueError("Receptor distances must be positive numbers")

    def _score_from_ratio(self, val: float, threshold_minor: float, threshold_mod: float, threshold_major: float, threshold_severe: float) -> int:
        if val >= threshold_severe:
            return 5
        elif val >= threshold_major:
            return 4
        elif val >= threshold_mod:
            return 3
        elif val >= threshold_minor:
            return 2
        return 1

    def assess_dust(self) -> Dict[str, Any]:
        """
        Returns:
            {
                "impact_score": int,  # 1-5
                "pm10_mg_m3": float,
                "exceeds_who_limit": bool,
                "mitigation_measures": list[str],
            }
        """
        charge_mass = float(self.blast_params.get("explosive_mass_kg", self.blast_params.get("charge_mass_kg", 500.0)))
        exp_type = self.blast_params.get("explosive_type", "ANFO")
        wind_speed = float(self.blast_params.get("wind_speed_m_s", 3.0))

        model = DustModel(explosive_type=exp_type)
        dust_mass_res = model.calculate_dust_mass(charge_mass)
        dust_res = model.calculate_pm10_concentration(
            dust_mass_kg=dust_mass_res["dust_mass_kg"],
            distance_m=self.closest_distance_m,
            wind_speed_m_s=wind_speed,
        )

        pm10_mg_m3 = dust_res["pm10_concentration_mg_m3"]
        exceeds_who = dust_res["exceeds_limit"]

        # Score based on WHO 0.05 mg/m3 threshold
        score = self._score_from_ratio(pm10_mg_m3, 0.02, 0.05, 0.10, 0.20)

        mitigations = []
        if score >= 3:
            mitigations.append("Water spraying / dust suppression curtains prior to firing")
            mitigations.append("Surfacing muckpiles with moisture before loading")
        if score >= 4:
            mitigations.append("Suspend blasting during high wind or unfavorable thermal inversion conditions")

        return {
            "impact_score": score,
            "pm10_mg_m3": pm10_mg_m3,
            "exceeds_who_limit": exceeds_who,
            "mitigation_measures": mitigations,
        }

    def assess_gas(self) -> Dict[str, Any]:
        """
        Returns impact score and mitigation measures for toxic gases.
        """
        charge_mass = float(self.blast_params.get("explosive_mass_kg", self.blast_params.get("charge_mass_kg", 500.0)))
        exp_type = self.blast_params.get("explosive_type", "ANFO")

        model = GasModel(explosive_type=exp_type)
        emissions = model.calculate_gas_emissions(charge_mass)
        co_kg = emissions["CO_kg"]

        conc_res = model.calculate_gas_concentration(
            gas_mass_kg=co_kg,
            gas_type="CO",
            ventilation_rate_m3_s=10.0,
            volume_m3=5000.0,
            time_s=300.0,
        )

        co_ppm = conc_res["concentration_ppm"]
        score = self._score_from_ratio(co_ppm, 10.0, 25.0, 50.0, 100.0)

        mitigations = []
        if score >= 3:
            mitigations.append("Enforce minimum 30-minute re-entry clearance delay post-blast")
            mitigations.append("Maintain optimal oxygen-balanced explosive formulation")
        if score >= 4:
            mitigations.append("Deploy portable toxic gas monitors at sensitive downwind receptors")

        return {
            "impact_score": score,
            "co_ppm": co_ppm,
            "mitigation_measures": mitigations,
        }

    def assess_noise(self) -> Dict[str, Any]:
        """
        Returns impact score and mitigation measures for noise.
        """
        max_charge_delay = float(self.blast_params.get("max_charge_per_delay_kg", 150.0))
        depth_burial = float(self.blast_params.get("depth_of_burial_m", 3.0))

        model = NoiseModel(charge_per_delay_kg=max_charge_delay, depth_of_burial_m=depth_burial)
        noise_res = model.calculate_noise_at_distance(self.closest_distance_m)

        spl_db = noise_res["spl_db"]
        score = self._score_from_ratio(spl_db, 105.0, 115.0, 120.0, 130.0)

        mitigations = []
        if score >= 3:
            mitigations.append("Use adequate stemming material and length to prevent noise venting")
            mitigations.append("Utilize electronic detonator delays to avoid airblast constructive interference")
        if score >= 4:
            mitigations.append("Construct noise attenuation berms or acoustic screening barriers")

        return {
            "impact_score": score,
            "spl_db": spl_db,
            "mitigation_measures": mitigations,
        }

    def assess_vibration(self) -> Dict[str, Any]:
        """
        Returns impact score and mitigation measures for vibration.
        """
        max_charge_delay = float(self.blast_params.get("max_charge_per_delay_kg", 150.0))
        k_vib = float(self.blast_params.get("k_vib", 1140.0))
        b_vib = float(self.blast_params.get("b_vib", 1.6))

        sd = self.closest_distance_m / (max_charge_delay ** 0.5)
        ppv_mms = k_vib * (sd ** (-b_vib))

        score = self._score_from_ratio(ppv_mms, 2.0, 5.0, 10.0, 25.0)

        mitigations = []
        if score >= 3:
            mitigations.append("Reduce maximum charge weight per delay (Q_max) by deck charging")
            mitigations.append("Implement electronic detonator timing optimization")
        if score >= 4:
            mitigations.append("Pre-split or slot cut drilling to attenuate wave propagation")

        return {
            "impact_score": score,
            "ppv_mms": round(ppv_mms, 2),
            "mitigation_measures": mitigations,
        }

    def assess_flyrock(self) -> Dict[str, Any]:
        """
        Returns impact score and mitigation measures for flyrock.
        """
        burden_m = float(self.blast_params.get("burden_m", 3.5))
        hole_diam_m = float(self.blast_params.get("hole_diameter_m", 0.165))

        # Lundborg flyrock distance estimation L = 260 * (d ** 0.67) * (B ** -1)
        flyrock_dist_m = 260.0 * (hole_diam_m ** 0.67) / (burden_m if burden_m > 0 else 1.0)
        ratio = flyrock_dist_m / self.closest_distance_m

        score = self._score_from_ratio(ratio, 0.3, 0.5, 0.8, 1.0)

        mitigations = []
        if score >= 3:
            mitigations.append("Verify minimum burden and stemming depth with 3D laser/photogrammetry scanner")
            mitigations.append("Clear clearance zone and enforce 500m exclusion perimeter")
        if score >= 4:
            mitigations.append("Use heavy blasting mats over collar charges near critical infrastructure")

        return {
            "impact_score": score,
            "estimated_flyrock_m": round(flyrock_dist_m, 1),
            "mitigation_measures": mitigations,
        }

    def calculate_total_impact(self) -> Dict[str, Any]:
        """
        Aggregates all impact scores.

        Returns:
            {
                "total_score": int,  # 5-25
                "overall_rating": str,  # 'low', 'moderate', 'high', 'severe'
                "breakdown": {
                    "dust": int,
                    "gas": int,
                    "noise": int,
                    "vibration": int,
                    "flyrock": int,
                },
                "priority_mitigations": list[str],
            }
        """
        dust = self.assess_dust()
        gas = self.assess_gas()
        noise = self.assess_noise()
        vib = self.assess_vibration()
        fly = self.assess_flyrock()

        breakdown = {
            "dust": dust["impact_score"],
            "gas": gas["impact_score"],
            "noise": noise["impact_score"],
            "vibration": vib["impact_score"],
            "flyrock": fly["impact_score"],
        }

        total_score = sum(breakdown.values())

        if total_score >= 20:
            rating = "severe"
        elif total_score >= 15:
            rating = "high"
        elif total_score >= 10:
            rating = "moderate"
        else:
            rating = "low"

        all_mitigations = []
        for res in [dust, gas, noise, vib, fly]:
            all_mitigations.extend(res["mitigation_measures"])

        # Deduplicate mitigations
        priority_mitigations = list(dict.fromkeys(all_mitigations))

        return {
            "total_score": total_score,
            "overall_rating": rating,
            "breakdown": breakdown,
            "priority_mitigations": priority_mitigations,
        }

    def generate_eia_report(self, output_path: str = "data/reports/eia_report.pdf") -> str:
        """
        Generates a text/PDF environmental impact assessment report.
        Returns the file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        impact = self.calculate_total_impact()

        report_content = f"""ENVIRONMENTAL IMPACT ASSESSMENT (EIA) REPORT
==================================================
Total Impact Score: {impact['total_score']} / 25
Overall Rating: {impact['overall_rating'].upper()}

Breakdown by Category:
- Dust Impact: {impact['breakdown']['dust']} / 5
- Toxic Gas Impact: {impact['breakdown']['gas']} / 5
- Noise Impact: {impact['breakdown']['noise']} / 5
- Vibration Impact: {impact['breakdown']['vibration']} / 5
- Flyrock Impact: {impact['breakdown']['flyrock']} / 5

Priority Mitigation Measures:
"""
        for i, mit in enumerate(impact["priority_mitigations"], 1):
            report_content += f"{i}. {mit}\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return output_path


def evaluate_environmental_impact(
    ppv_mms: float,
    airblast_dbl: float,
    flyrock_m: float,
    pm10_ug_m3: float,
    nox_ppm: float,
) -> Dict[str, Any]:
    """
    Evaluate comprehensive Environmental Impact Assessment (EIA) index across vibration, airblast, flyrock, dust, and toxic gas.

    Scoring (0-100 EIA Index):
        EIA Index = 0.25 * S_ppv + 0.25 * S_airblast + 0.20 * S_flyrock + 0.15 * S_dust + 0.15 * S_gas

    Returns:
        {
            "eia_score": float,
            "impact_class": str,  # 'low', 'moderate', 'high', 'severe'
            "sub_scores": dict,
            "compliance_summary": str,
        }
    """
    s_ppv = min(100.0, (ppv_mms / 10.0) * 100.0)
    s_air = min(100.0, (airblast_dbl / 120.0) * 100.0)
    s_fly = min(100.0, (flyrock_m / 250.0) * 100.0)
    s_dust = min(100.0, (pm10_ug_m3 / 150.0) * 100.0)
    s_gas = min(100.0, (nox_ppm / 5.0) * 100.0)

    eia = 0.25 * s_ppv + 0.25 * s_air + 0.20 * s_fly + 0.15 * s_dust + 0.15 * s_gas

    if eia > 85.0:
        cls = "severe"
        summary = "CRITICAL EIA ALARM: Multiple environmental parameters exceed statutory limits."
    elif eia > 65.0:
        cls = "high"
        summary = "HIGH EIA IMPACT: Close to environmental compliance boundaries."
    elif eia > 45.0:
        cls = "moderate"
        summary = "MODERATE EIA IMPACT: Environmental parameters within operational norms."
    else:
        cls = "low"
        summary = "LOW EIA IMPACT: Excellent environmental stewardship."

    return {
        "eia_score": round(eia, 1),
        "impact_class": cls,
        "sub_scores": {
            "vibration_score": round(s_ppv, 1),
            "airblast_score": round(s_air, 1),
            "flyrock_score": round(s_fly, 1),
            "dust_score": round(s_dust, 1),
            "toxic_gas_score": round(s_gas, 1),
        },
        "compliance_summary": summary,
    }
