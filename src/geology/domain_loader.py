"""
Geotechnical domain loader and design orchestration pipeline for BlasterOPT.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from src.geology.geotechnical_domain import GeotechnicalDomain
from src.geology.structures import StructuralModel
from src.render3d.hole_pattern import Hole


class GeotechnicalDomainLoader:
    """Loads geotechnical domain spatial models by block ID."""

    @staticmethod
    def load(block_id: str) -> GeotechnicalDomain:
        poly = np.array([[0, 0, 100], [50, 0, 100], [50, 50, 100], [0, 50, 100]])
        domain = GeotechnicalDomain(domain_id=block_id, polygon_3d=poly)

        bh_data = {
            "ucs_mpa": 120.0,
            "rqd_pct": 85.0,
            "joint_spacing_m": 0.6,
            "num_joint_sets": 3,
            "joint_type": "rough_planar",
            "alteration": "slightly_altered",
            "water_condition": "dry",
            "stress_condition": "medium_stress",
        }
        domain.assign_from_borehole(bh_data)

        struct_model = StructuralModel()
        poly_fault = np.array([[0, 0, 90.0], [50, 0, 90.0]])
        struct_model.add_fault("F1", poly_fault, dip_deg=70.0, thickness_m=1.2, infilling="clay_gouge")
        domain.structures = struct_model

        return domain


class GeologyAwareBlastDesigner:
    """
    Orchestrates geology-driven blast design adjustments.
    """

    def design(self, inputs: Any, holes: List[Hole]) -> Dict[str, Any]:
        block_id = getattr(inputs, "block_id", "BLOCK_DEFAULT")
        domain = GeotechnicalDomainLoader.load(block_id)

        geo_params = domain.get_blast_design_parameters()
        rock_factor_a = geo_params["rock_factor_a"]

        base_burden = getattr(inputs, "burden_m", 4.0)
        base_stemming = getattr(inputs, "stemming_m", 3.5)

        burden = base_burden * geo_params["burden_multiplier"]
        stemming = base_stemming * geo_params["stemming_multiplier"]

        intersections = {}
        adjustments = {}
        if domain.structures is not None:
            intersections = domain.structures.find_intersections(holes)
            adj_res = domain.structures.calculate_blast_adjustment(intersections)
            adjustments = adj_res.get("hole_adjustments", {})

        adjusted_holes = []
        for hole in holes:
            h_dict = {
                "hole_id": hole.hole_id,
                "x": hole.x,
                "y": hole.y,
                "z": hole.z,
                "depth_m": hole.depth_m,
                "charge_multiplier": 1.0,
                "stemming_multiplier": 1.0,
                "notes": [],
            }
            if hole.hole_id in adjustments:
                adj_item = adjustments[hole.hole_id]
                h_dict["charge_multiplier"] = adj_item["charge_multiplier"]
                h_dict["stemming_multiplier"] = adj_item["stemming_multiplier"]
                h_dict["notes"] = adj_item["notes"]

            adjusted_holes.append(h_dict)

        return {
            "domain_id": domain.domain_id,
            "rock_factor_a": rock_factor_a,
            "adjusted_burden_m": round(burden, 2),
            "adjusted_stemming_m": round(stemming, 2),
            "adjusted_holes": adjusted_holes,
            "warnings": geo_params["warnings"],
        }
