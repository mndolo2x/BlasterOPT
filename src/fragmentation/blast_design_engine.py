"""
Integrated Blast Design Engine orchestrating geometry, charge, fragmentation, geology, and safety.
"""

from typing import Dict, Any, List, Optional
from src.fragmentation.kuz_ram import predict_kuz_ram
from src.fragmentation.swebrec import SwebrecModel
from src.fragmentation.kco import KCOModel
from src.fragmentation.calibration import FragmentationCalibrator
from src.safety.dust import DustModel
from src.safety.toxic_gases import GasModel
from src.safety.noise_prediction import NoiseModel
from src.safety.environmental_impact import EnvironmentalAssessment
from src.safety.risk_analysis import BlastRiskAnalyzer


class BlastDesignEngine:
    """
    Integrates Kuz-Ram, Swebrec, KCO, RSM Calibration, and Safety Assessments into unified blast design.
    """

    def __init__(self) -> None:
        self.kco = KCOModel()
        self.calibrator = FragmentationCalibrator()

    def design(self, inputs: Any) -> Dict[str, Any]:
        block_id = getattr(inputs, "block_id", "BLOCK_DEFAULT")
        frag_model_choice = getattr(inputs, "fragmentation_model", "kco").lower()

        # Input blast parameters
        b_m = getattr(inputs, "burden_m", 4.0)
        s_m = getattr(inputs, "spacing_m", 5.0)
        bh_m = getattr(inputs, "bench_height_m", 12.0)
        pf = getattr(inputs, "powder_factor_kg_m3", 0.65)
        rws = getattr(inputs, "explosive_rws", 100.0)
        rock_a = getattr(inputs, "rock_factor_a", 8.0)
        hole_diam_mm = getattr(inputs, "hole_diameter_mm", 250.0)
        hole_diam_m = hole_diam_mm / 1000.0
        exp_type = getattr(inputs, "explosive_type", "ANFO")
        stemming_m = getattr(inputs, "stemming_m", 3.0)
        q_max_kg = getattr(inputs, "max_charge_per_delay_kg", 150.0)
        receptor_dist_m = getattr(inputs, "closest_receptor_dist_m", 500.0)

        vol_per_hole = b_m * s_m * bh_m
        charge_per_hole_kg = vol_per_hole * pf
        total_exp_kg = getattr(inputs, "explosive_mass_kg", charge_per_hole_kg * 20.0)

        blast_params = {
            "burden_m": b_m,
            "spacing_m": s_m,
            "bench_height_m": bh_m,
            "hole_diameter_mm": hole_diam_mm,
            "hole_diameter_m": hole_diam_m,
            "powder_factor_kg_m3": pf,
            "explosive_rws": rws,
            "rock_factor_a": rock_a,
            "blastability_index": rock_a * 6.25,
            "explosive_type": exp_type,
            "stemming_m": stemming_m,
            "max_charge_per_delay_kg": q_max_kg,
            "explosive_mass_kg": total_exp_kg,
            "closest_receptor_dist_m": receptor_dist_m,
            "depth_of_burial_m": stemming_m,
        }

        # Fragmentation prediction
        if frag_model_choice == "kco":
            frag_result = self.kco.predict(blast_params)
        elif frag_model_choice == "swebrec":
            swe_mod = SwebrecModel(x_max=b_m * 1000.0, x_50=200.0, b=1.2)
            frag_result = {
                "d50_mm": 200.0,
                "distribution": swe_mod.compute_curve(),
                "model": "Swebrec",
            }
        else:
            kr = predict_kuz_ram(
                powder_factor_kg_m3=pf,
                charge_mass_per_hole_kg=charge_per_hole_kg,
                rock_factor_a=rock_a,
                explosive_relative_weight_strength=rws,
            )
            frag_result = {
                "d50_mm": kr["d50_mm"],
                "model": "Kuz-Ram",
            }

        # Apply rock-specific calibration if available
        if self.calibrator.get_calibration_quality()["is_reliable"]:
            cal_pred = self.calibrator.predict_calibrated_d80(blast_params)
            frag_result["calibrated_d80_cm"] = cal_pred["d80_cm"]
            frag_result["calibration_applied"] = True
        else:
            frag_result["calibration_applied"] = False

        # Safety & Environmental Assessments
        dust = DustModel(explosive_type=exp_type)
        gas = GasModel(explosive_type=exp_type)
        noise = NoiseModel(charge_per_delay_kg=q_max_kg, depth_of_burial_m=stemming_m)
        receptors = {"closest": receptor_dist_m}
        eia = EnvironmentalAssessment(blast_params=blast_params, receptor_distances=receptors)
        risk = BlastRiskAnalyzer()

        dust_mass_res = dust.calculate_dust_mass(total_exp_kg)
        dust_assessment = dust.calculate_pm10_concentration(dust_mass_res["dust_mass_kg"], receptor_dist_m, wind_speed_m_s=3.0)
        gas_assessment = gas.calculate_gas_emissions(total_exp_kg)
        noise_assessment = noise.calculate_noise_at_distance(receptor_dist_m)
        environmental_impact = eia.calculate_total_impact()
        risk_assessment = risk.calculate_overall_risk(blast_params)

        warnings: List[str] = []
        if risk_assessment["overall_risk_level"] in ["high", "critical"]:
            warnings.append("High risk detected. Review mitigations.")
        if dust_assessment.get("exceeds_limit", False):
            warnings.append("PM10 dust concentration exceeds WHO limit.")
        if noise_assessment.get("exceeds_limit", False):
            warnings.append("Noise overpressure exceeds 120 dBL limit.")

        return {
            "block_id": block_id,
            "rock_factor_a": rock_a,
            "adjusted_burden_m": round(b_m, 2),
            "fragmentation_prediction": frag_result,
            "dust_assessment": dust_assessment,
            "gas_assessment": gas_assessment,
            "noise_assessment": noise_assessment,
            "environmental_impact": environmental_impact,
            "risk_assessment": risk_assessment,
            "warnings": warnings,
        }
