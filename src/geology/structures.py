"""
Geological structures: Faults, dykes, and lithological contacts.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.render3d.hole_pattern import Hole


class StructuralFeature:
    def __init__(
        self,
        feature_id: str,
        feature_type: str,
        dip_deg: float,
        dip_dir_deg: float,
        thickness_m: float = 1.0,
        competency_factor: float = 0.5,
    ) -> None:
        valid_types = ["fault", "dyke", "contact"]
        if feature_type not in valid_types:
            raise ValueError(f"feature_type '{feature_type}' must be one of {valid_types}")
        if not (0.0 <= dip_deg <= 90.0):
            raise ValueError(f"dip_deg ({dip_deg}) must be between 0 and 90")

        self.feature_id = feature_id
        self.feature_type = feature_type
        self.dip_deg = dip_deg
        self.dip_dir_deg = dip_dir_deg
        self.thickness_m = thickness_m
        self.competency_factor = competency_factor

    def get_blast_interaction(self, blast_direction_deg: float) -> Dict[str, Any]:
        angle_diff = abs((self.dip_dir_deg - blast_direction_deg + 180.0) % 360.0 - 180.0)

        if angle_diff < 30.0:
            interaction = "Blasting across structure — high risk of gases venting through fracture zone and overbreak."
            risk_level = "high"
        elif angle_diff > 60.0:
            interaction = "Blasting parallel to structure — favorable shearing along weakness plane."
            risk_level = "low"
        else:
            interaction = "Oblique structural intersection — moderate backbreak and overbreak risk."
            risk_level = "medium"

        return {
            "feature_id": self.feature_id,
            "feature_type": self.feature_type,
            "angle_difference_deg": round(angle_diff, 1),
            "interaction_assessment": interaction,
            "risk_level": risk_level,
        }


class StructuralModel:
    def __init__(self) -> None:
        self.faults: List[dict] = []
        self.dykes: List[dict] = []
        self.contacts: List[dict] = []

    def add_fault(
        self,
        fault_id: str,
        polyline_3d: np.ndarray,
        dip_deg: float,
        thickness_m: float,
        infilling: str,
    ) -> None:
        if thickness_m <= 0:
            raise ValueError(f"thickness_m ({thickness_m}) must be positive")
        self.faults.append(
            {
                "fault_id": fault_id,
                "polyline": polyline_3d,
                "dip_deg": dip_deg,
                "thickness_m": thickness_m,
                "infilling": infilling,
            }
        )

    def add_dyke(
        self,
        dyke_id: str,
        polyline_3d: np.ndarray,
        dip_deg: float,
        thickness_m: float,
        rock_type: str,
    ) -> None:
        if thickness_m <= 0:
            raise ValueError(f"thickness_m ({thickness_m}) must be positive")
        self.dykes.append(
            {
                "dyke_id": dyke_id,
                "polyline": polyline_3d,
                "dip_deg": dip_deg,
                "thickness_m": thickness_m,
                "rock_type": rock_type,
            }
        )

    def add_contact(
        self,
        contact_id: str,
        polyline_3d: np.ndarray,
        rock_a: str,
        rock_b: str,
    ) -> None:
        self.contacts.append(
            {
                "contact_id": contact_id,
                "polyline": polyline_3d,
                "rock_a": rock_a,
                "rock_b": rock_b,
            }
        )

    def find_intersections(self, holes: List[Hole]) -> Dict[str, List[dict]]:
        results: Dict[str, List[dict]] = {}

        for h in holes:
            h_intersections: List[dict] = []

            for f in self.faults:
                poly = f["polyline"]
                if len(poly) > 0:
                    avg_f_z = float(np.mean(poly[:, 2]))
                    intersect_depth = h.z - avg_f_z
                    if 0.0 <= intersect_depth <= h.depth_m:
                        h_intersections.append(
                            {
                                "structure_id": f["fault_id"],
                                "type": "fault",
                                "depth_m": round(intersect_depth, 2),
                                "infilling": f["infilling"],
                                "effect": "Potential loss of explosive gases and overbreak.",
                            }
                        )

            for d in self.dykes:
                poly = d["polyline"]
                if len(poly) > 0:
                    avg_d_z = float(np.mean(poly[:, 2]))
                    intersect_depth = h.z - avg_d_z
                    if 0.0 <= intersect_depth <= h.depth_m:
                        h_intersections.append(
                            {
                                "structure_id": d["dyke_id"],
                                "type": "dyke",
                                "depth_m": round(intersect_depth, 2),
                                "rock_type": d["rock_type"],
                                "effect": "Hard rock zone requires charge adjustment.",
                            }
                        )

            for c in self.contacts:
                poly = c["polyline"]
                if len(poly) > 0:
                    avg_c_z = float(np.mean(poly[:, 2]))
                    intersect_depth = h.z - avg_c_z
                    if 0.0 <= intersect_depth <= h.depth_m:
                        h_intersections.append(
                            {
                                "structure_id": c["contact_id"],
                                "type": "contact",
                                "depth_m": round(intersect_depth, 2),
                                "rock_a": c["rock_a"],
                                "rock_b": c["rock_b"],
                                "effect": "Lithological boundary requires deck placement review.",
                            }
                        )

            results[h.hole_id] = h_intersections

        return results

    def calculate_blast_adjustment(self, intersections: Dict[str, List[dict]]) -> Dict[str, Any]:
        adjustments: Dict[str, dict] = {}

        for hole_id, inters in intersections.items():
            h_adj = {
                "charge_multiplier": 1.0,
                "stemming_multiplier": 1.0,
                "decking_recommended": False,
                "notes": [],
            }

            for item in inters:
                stype = item["type"]
                if stype == "fault":
                    inf = item.get("infilling", "").lower()
                    if "clay" in inf or inf == "soft":
                        h_adj["charge_multiplier"] *= 0.70
                        h_adj["stemming_multiplier"] *= 1.20
                        h_adj["notes"].append("Fault with clay infilling: reduced charge 30%, increased stemming 20%.")
                elif stype == "dyke":
                    h_adj["charge_multiplier"] *= 1.15
                    h_adj["notes"].append("Dyke intersection: increased charge 15% for hard rock breakage.")
                elif stype == "contact":
                    h_adj["decking_recommended"] = True
                    h_adj["notes"].append("Contact intersection within charge column: air/decking deck placement recommended.")

            adjustments[hole_id] = h_adj

        return {
            "hole_adjustments": adjustments,
            "total_holes_affected": sum(1 for adj in adjustments.values() if adj["notes"]),
        }
