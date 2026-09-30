"""
Geotechnical domain spatial management and blast design derivation.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from src.geology.rmr import RMRCalculator
from src.geology.q_system import QSystemCalculator
from src.geology.structures import StructuralModel


class GeotechnicalDomain:
    """
    A 3D spatial domain with unified geotechnical properties.
    Used to assign properties to blast blocks.
    """

    def __init__(self, domain_id: str, polygon_3d: np.ndarray) -> None:
        if polygon_3d.size == 0:
            raise ValueError("polygon_3d array cannot be empty")

        self.domain_id = domain_id
        self.polygon_3d = polygon_3d
        self.rmr: Optional[dict] = None
        self.q_system: Optional[dict] = None
        self.joint_sets: Optional[List[dict]] = None
        self.structures: Optional[StructuralModel] = None

    def assign_from_borehole(self, borehole_data: dict) -> None:
        """Populate domain properties from a borehole log."""
        ucs = float(borehole_data.get("ucs_mpa", 100.0))
        rqd = float(borehole_data.get("rqd_pct", 75.0))
        spacing = float(borehole_data.get("joint_spacing_m", 0.5))

        rmr_calc = RMRCalculator()
        self.rmr = rmr_calc.calculate_total(
            ucs_mpa=ucs,
            rqd_pct=rqd,
            joint_spacing_m=spacing,
            roughness=borehole_data.get("roughness", "slightly_rough"),
        )

        q_calc = QSystemCalculator()
        self.q_system = q_calc.calculate_q(
            rqd_pct=rqd,
            num_joint_sets=borehole_data.get("num_joint_sets", 3),
            joint_type=borehole_data.get("joint_type", "rough_planar"),
            alteration=borehole_data.get("alteration", "slightly_altered"),
            water_condition=borehole_data.get("water_condition", "dry"),
            stress_condition=borehole_data.get("stress_condition", "medium_stress"),
        )

        self.joint_sets = borehole_data.get("joint_sets", [])

    def get_blast_design_parameters(self) -> Dict[str, Any]:
        """
        Returns blast design parameters derived from the geotechnical data:
            {
                "rock_factor_a": float,  # for Kuz-Ram
                "blastability_index": float,
                "powder_factor_range": (float, float),
                "stemming_multiplier": float,
                "burden_multiplier": float,
                "warnings": list[str],
            }
        """
        warnings: List[str] = []

        if self.rmr is not None:
            rmr_val = float(self.rmr["rmr"])
        else:
            rmr_val = 50.0

        rock_factor_a = float(max(2.0, min(16.0, 0.12 * (rmr_val - 10.0))))
        blastability_index = float(round(rock_factor_a * 6.25, 1))

        if rock_factor_a > 10.0:
            pf_range = (0.75, 1.20)
            stemming_mult = 1.15
            burden_mult = 0.90
            warnings.append("Hard/massive rock mass (Rock Factor A > 10): reduce burden by 10% and increase powder factor.")
        elif rock_factor_a < 5.0:
            pf_range = (0.35, 0.60)
            stemming_mult = 0.85
            burden_mult = 1.10
            warnings.append("Soft/weak rock mass (Rock Factor A < 5): expand burden by 10% to prevent over-fines.")
        else:
            pf_range = (0.55, 0.85)
            stemming_mult = 1.0
            burden_mult = 1.0

        return {
            "rock_factor_a": round(rock_factor_a, 2),
            "blastability_index": blastability_index,
            "powder_factor_range": pf_range,
            "stemming_multiplier": stemming_mult,
            "burden_multiplier": burden_mult,
            "warnings": warnings,
        }

    def render(self, plotter: Any = None) -> Dict[str, Any]:
        if plotter is not None and hasattr(plotter, "add_mesh"):
            import pyvista as pv
            mesh = pv.PolyData(self.polygon_3d)
            plotter.add_mesh(mesh, color="cyan", opacity=0.4, label=f"Domain {self.domain_id}")

        return {
            "domain_id": self.domain_id,
            "polygon_3d": self.polygon_3d,
        }
