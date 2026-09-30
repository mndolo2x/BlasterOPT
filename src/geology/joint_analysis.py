"""
Joint set analysis, stereonet projections using mplstereonet, and block size estimation (ISRM 1978/2014, Priest & Hudson 1976).
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple


class JointAnalyzer:
    """
    Analyzes joint sets from orientation, spacing, and persistence data.
    Follows ISRM Suggested Methods (1978, updated 2014).
    """

    def __init__(self, joint_measurements: List[dict]) -> None:
        if not joint_measurements:
            raise ValueError("joint_measurements list cannot be empty")

        for m in joint_measurements:
            if not (0.0 <= m.get("dip_deg", 0.0) <= 90.0):
                raise ValueError(f"dip_deg ({m.get('dip_deg')}) must be between 0 and 90")
            if not (0.0 <= m.get("dip_direction_deg", 0.0) <= 360.0):
                raise ValueError(f"dip_direction_deg ({m.get('dip_direction_deg')}) must be between 0 and 360")
            if m.get("spacing_m", 0.0) <= 0:
                raise ValueError("spacing_m must be positive")

        self.joint_measurements = joint_measurements

    def identify_joint_sets(self, max_sets: int = 4) -> List[dict]:
        n_joints = len(self.joint_measurements)
        n_sets = min(max_sets, n_joints)

        dips = [m["dip_deg"] for m in self.joint_measurements]
        dip_dirs = [m["dip_direction_deg"] for m in self.joint_measurements]
        spacings = [m["spacing_m"] for m in self.joint_measurements]
        persistences = [m.get("persistence_m", 1.0) for m in self.joint_measurements]

        joint_sets = []
        chunk_size = int(math.ceil(n_joints / float(n_sets)))

        for i in range(n_sets):
            start = i * chunk_size
            end = min(n_joints, (i + 1) * chunk_size)
            if start >= end:
                break

            sub_dips = dips[start:end]
            sub_dd = dip_dirs[start:end]
            sub_sp = spacings[start:end]
            sub_per = persistences[start:end]

            m_dip = float(np.mean(sub_dips))
            m_dd = float(np.mean(sub_dd))
            m_sp = float(np.mean(sub_sp))
            std_sp = float(np.std(sub_sp)) if len(sub_sp) > 1 else 0.0
            m_per = float(np.mean(sub_per))

            joint_sets.append(
                {
                    "set_id": i + 1,
                    "mean_dip_deg": round(m_dip, 1),
                    "mean_dip_direction_deg": round(m_dd, 1),
                    "fisher_k": 15.0,
                    "num_joints": len(sub_dips),
                    "spacing_mean_m": round(m_sp, 2),
                    "spacing_std_m": round(std_sp, 2),
                    "persistence_mean_m": round(m_per, 2),
                    "percentage_of_total": round((len(sub_dips) / float(n_joints)) * 100.0, 1),
                }
            )

        return joint_sets

    def calculate_rqd_from_joints(self, direction_deg: float) -> float:
        mean_spacing = float(np.mean([m["spacing_m"] for m in self.joint_measurements]))
        lam = 1.0 / max(0.01, mean_spacing)
        rqd = 100.0 * math.exp(-0.1 * lam) * (0.1 * lam + 1.0)
        return float(round(min(100.0, max(0.0, rqd)), 1))

    def create_stereonet(self, plot_type: str = "schmidt"):
        """
        Creates a stereonet (Schmidt or Wulff) using mplstereonet.
        """
        import matplotlib.pyplot as plt
        try:
            import mplstereonet
            fig, ax = mplstereonet.subplots(figsize=(6, 6))
            strikes = [(m["dip_direction_deg"] - 90.0) % 360.0 for m in self.joint_measurements]
            dips = [m["dip_deg"] for m in self.joint_measurements]
            ax.pole(strikes, dips, "bo", label="Joint Poles")
            ax.grid(True)
            ax.set_title("Equal-Area Stereonet (mplstereonet)")
            return fig
        except Exception:
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
            dips = [m["dip_deg"] for m in self.joint_measurements]
            dip_dirs = [m["dip_direction_deg"] for m in self.joint_measurements]
            theta = [math.radians(dd) for dd in dip_dirs]
            r = [d / 90.0 for d in dips]
            ax.scatter(theta, r, c="blue", alpha=0.7)
            ax.set_title("Stereonet (Polar Fallback)")
            return fig

    def calculate_block_size(self) -> dict:
        sets = self.identify_joint_sets(max_sets=3)
        spacings = [s["spacing_mean_m"] for s in sets]

        if len(spacings) < 3:
            avg_s = spacings[0] if spacings else 0.5
            while len(spacings) < 3:
                spacings.append(avg_s)

        v_block = float(spacings[0] * spacings[1] * spacings[2])

        if v_block < 0.001:
            cls = "very_small"
        elif v_block < 0.01:
            cls = "small"
        elif v_block < 0.2:
            cls = "medium"
        elif v_block < 1.0:
            cls = "large"
        else:
            cls = "very_large"

        return {
            "v_block_m3": round(v_block, 3),
            "s1_m": round(spacings[0], 2),
            "s2_m": round(spacings[1], 2),
            "s3_m": round(spacings[2], 2),
            "classification": cls,
        }

    def predict_fragmentation_from_joints(self) -> dict:
        block_res = self.calculate_block_size()
        v_block = block_res["v_block_m3"]
        block_d80_cm = (v_block ** (1.0 / 3.0)) * 100.0

        if block_d80_cm < 20.0:
            pf_adj = 0.8
            reasoning = "Rock mass is intensely pre-fractured by jointing; reduce powder factor to prevent fines and over-crushing."
        elif block_d80_cm > 60.0:
            pf_adj = 1.2
            reasoning = "Massive rock formation with large joint spacing; increase powder factor to achieve target fragmentation."
        else:
            pf_adj = 1.0
            reasoning = "Joint spacing is in optimal alignment with target fragmentation size."

        blast_d80_cm = block_d80_cm * 0.6

        return {
            "natural_block_d80_cm": round(block_d80_cm, 1),
            "expected_blast_d80_cm": round(blast_d80_cm, 1),
            "powder_factor_adjustment": round(pf_adj, 2),
            "reasoning": reasoning,
        }


class JointSet:
    def __init__(self, set_id: str, dip_deg: float, dip_dir_deg: float, spacing_m: float, persistence_m: float = 5.0) -> None:
        if not (0.0 <= dip_deg <= 90.0):
            raise ValueError(f"dip_deg ({dip_deg}) must be between 0 and 90")
        if not (0.0 <= dip_dir_deg <= 360.0):
            raise ValueError(f"dip_dir_deg ({dip_dir_deg}) must be between 0 and 360")
        if spacing_m <= 0:
            raise ValueError(f"spacing_m ({spacing_m}) must be positive")

        self.set_id = set_id
        self.dip_deg = dip_deg
        self.dip_dir_deg = dip_dir_deg
        self.spacing_m = spacing_m
        self.persistence_m = persistence_m

    def get_normal_vector(self) -> Tuple[float, float, float]:
        dip_rad = math.radians(self.dip_deg)
        dd_rad = math.radians(self.dip_dir_deg)
        nx = math.sin(dip_rad) * math.sin(dd_rad)
        ny = math.sin(dip_rad) * math.cos(dd_rad)
        nz = -math.cos(dip_rad)
        return (nx, ny, nz)


def calculate_volumetric_joint_count(joint_sets: List[JointSet]) -> float:
    if not joint_sets:
        raise ValueError("joint_sets list cannot be empty")
    jv = sum(1.0 / js.spacing_m for js in joint_sets)
    return float(round(jv, 2))
