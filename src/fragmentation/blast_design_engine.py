"""
Integrated Blast Design Engine orchestrating geometry, charge, fragmentation, and geology.
"""

from typing import Dict, Any, List, Optional
from src.fragmentation.kuz_ram import predict_kuz_ram
from src.fragmentation.swebrec import SwebrecModel
from src.fragmentation.kco import KCOModel
from src.fragmentation.calibration import FragmentationCalibrator


class BlastDesignEngine:
    """
    Integrates Kuz-Ram, Swebrec, KCO, and RSM Calibration into unified blast design.
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

        blast_params = {
            "burden_m": b_m,
            "spacing_m": s_m,
            "bench_height_m": bh_m,
            "hole_diameter_mm": getattr(inputs, "hole_diameter_mm", 250.0),
            "powder_factor_kg_m3": pf,
            "explosive_rws": rws,
            "rock_factor_a": rock_a,
            "blastability_index": rock_a * 6.25,
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
            vol_per_hole = b_m * s_m * bh_m
            kr = predict_kuz_ram(
                powder_factor_kg_m3=pf,
                charge_mass_per_hole_kg=vol_per_hole * pf,
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

        return {
            "block_id": block_id,
            "rock_factor_a": rock_a,
            "adjusted_burden_m": round(b_m, 2),
            "fragmentation_prediction": frag_result,
        }
