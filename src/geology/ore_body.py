"""
Ore body boundaries, dilution, and ore loss analysis for BlasterOPT.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class OreBodyModel:
    """
    Models ore/waste boundaries and predicts dilution from blasting.
    """

    def __init__(
        self,
        block_model: pd.DataFrame,
        grade_column: str = "grade",
        cutoff_grade: float = 0.1,
    ) -> None:
        if block_model.empty:
            raise ValueError("block_model DataFrame cannot be empty")
        if grade_column not in block_model.columns:
            raise ValueError(f"grade_column '{grade_column}' not found in block_model")

        self.block_model = block_model.copy()
        self.grade_column = grade_column
        self.cutoff_grade = cutoff_grade

    def classify_ore_waste(self) -> pd.DataFrame:
        """Adds 'material_type' column: 'ore', 'waste', 'marginal'."""
        df = self.block_model.copy()
        grades = df[self.grade_column]

        conditions = [
            grades >= self.cutoff_grade * 1.2,
            (grades >= self.cutoff_grade) & (grades < self.cutoff_grade * 1.2),
            grades < self.cutoff_grade,
        ]
        choices = ["ore", "marginal", "waste"]

        df["material_type"] = np.select(conditions, choices, default="waste")
        return df

    def calculate_dilution(
        self,
        blast_polygon: np.ndarray,
        blasted_tonnage: float,
        ore_tonnage_expected: float,
        overbreak_waste_tonnes: float = 0.0,
    ) -> Dict[str, Any]:
        if blasted_tonnage <= 0:
            raise ValueError(f"blasted_tonnage ({blasted_tonnage}) must be positive")

        classified = self.classify_ore_waste()
        waste_blocks = classified[classified["material_type"] == "waste"]
        waste_tonnes_base = len(waste_blocks) * 100.0  # nominal 100t per block

        planned_dil_pct = (waste_tonnes_base / max(1.0, blasted_tonnage)) * 100.0
        unplanned_dil_pct = (overbreak_waste_tonnes / max(1.0, blasted_tonnage)) * 100.0
        total_dil_pct = planned_dil_pct + unplanned_dil_pct

        waste_gain = waste_tonnes_base + overbreak_waste_tonnes
        ore_loss = max(0.0, ore_tonnage_expected - (blasted_tonnage - waste_gain))
        grade_dil = total_dil_pct * 0.8  # grade reduction factor

        recs = self.recommend_selective_blasting(blast_polygon)["recommendations"]

        return {
            "planned_dilution_pct": round(planned_dil_pct, 2),
            "unplanned_dilution_pct": round(unplanned_dil_pct, 2),
            "total_dilution_pct": round(total_dil_pct, 2),
            "ore_loss_tonnes": round(ore_loss, 1),
            "waste_gain_tonnes": round(waste_gain, 1),
            "grade_dilution_pct": round(grade_dil, 2),
            "recommendations": recs,
        }

    def recommend_selective_blasting(self, blast_polygon: np.ndarray) -> Dict[str, Any]:
        recs = [
            "Use smaller diameter trim holes (115-165 mm) along ore/waste contact boundaries.",
            "Reduce powder factor by 15-25% in transition zone holes to limit throw dilution.",
            "Implement air decking or decoupled explosive charges near high-grade ore boundary.",
            "Directional initiation sequence parallel to ore contact to minimize displacement into waste.",
        ]
        return {
            "selective_blasting_required": True,
            "recommendations": recs,
        }


def calculate_ore_dilution(
    planned_ore_tonnes: float,
    actual_mucked_tonnes: float,
    waste_rock_tonnes: float,
    planned_grade_g_t: float,
    mucked_grade_g_t: float,
) -> Dict[str, Any]:
    if planned_ore_tonnes <= 0:
        raise ValueError(f"planned_ore_tonnes ({planned_ore_tonnes}) must be positive")
    if actual_mucked_tonnes <= 0:
        raise ValueError(f"actual_mucked_tonnes ({actual_mucked_tonnes}) must be positive")

    dilution_pct = (waste_rock_tonnes / actual_mucked_tonnes) * 100.0
    recovered_ore = max(0.0, actual_mucked_tonnes - waste_rock_tonnes)
    ore_loss_pct = max(0.0, (planned_ore_tonnes - recovered_ore) / planned_ore_tonnes) * 100.0
    grade_recon_pct = (mucked_grade_g_t / max(0.01, planned_grade_g_t)) * 100.0

    return {
        "dilution_pct": round(dilution_pct, 2),
        "ore_loss_pct": round(ore_loss_pct, 2),
        "grade_reconciliation_pct": round(grade_recon_pct, 2),
    }
