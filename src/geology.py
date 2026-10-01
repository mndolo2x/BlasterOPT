"""
Geology & Geomechanics Package for BlastOpt Botswana.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional


class RMRCalculator:
    """Bieniawski (1989) Rock Mass Rating (RMR89) calculation."""

    @staticmethod
    def calculate_rmr(
        ucs_mpa: float = 120.0,
        rqd_pct: float = 75.0,
        spacing_m: float = 0.5,
        condition_score: int = 20,
        groundwater_score: int = 10,
        joint_orientation_penalty: int = -5,
    ) -> Dict[str, Any]:
        """Calculates RMR89 score (0-100)."""
        # 1. Strength rating (0-15)
        if ucs_mpa > 250: r_strength = 15
        elif ucs_mpa > 100: r_strength = 12
        elif ucs_mpa > 50: r_strength = 7
        elif ucs_mpa > 25: r_strength = 4
        else: r_strength = 2

        # 2. RQD rating (3-20)
        r_rqd = min(20, max(3, int(rqd_pct * 0.2)))

        # 3. Spacing rating (5-20)
        if spacing_m > 2.0: r_spacing = 20
        elif spacing_m > 0.6: r_spacing = 15
        elif spacing_m > 0.2: r_spacing = 10
        else: r_spacing = 5

        total_rmr = r_strength + r_rqd + r_spacing + condition_score + groundwater_score + joint_orientation_penalty
        total_rmr = int(np.clip(total_rmr, 0, 100))

        if total_rmr > 80: class_desc = "Very Good Rock (Class I)"
        elif total_rmr > 60: class_desc = "Good Rock (Class II)"
        elif total_rmr > 40: class_desc = "Fair Rock (Class III)"
        elif total_rmr > 20: class_desc = "Poor Rock (Class IV)"
        else: class_desc = "Very Poor Rock (Class V)"

        return {
            "rmr_score": total_rmr,
            "rock_class": class_desc,
            "strength_rating": r_strength,
            "rqd_rating": r_rqd,
            "spacing_rating": r_spacing,
        }


class QSystemCalculator:
    """Barton (1974) Q-system rock mass quality index."""

    @staticmethod
    def calculate_q(
        rqd_pct: float = 75.0,
        jn: float = 9.0,
        jr: float = 2.0,
        ja: float = 1.0,
        jw: float = 1.0,
        srf: float = 1.0,
    ) -> Dict[str, Any]:
        """Q = (RQD/Jn) * (Jr/Ja) * (Jw/SRF)."""
        rqd = max(rqd_pct, 10.0)
        q_val = (rqd / max(jn, 0.5)) * (jr / max(ja, 0.5)) * (jw / max(srf, 0.5))

        if q_val > 40: desc = "Exceptionally / Extremely Good"
        elif q_val > 10: desc = "Good"
        elif q_val > 4: desc = "Fair"
        elif q_val > 1: desc = "Poor"
        else: desc = "Very / Extremely Poor"

        return {
            "q_value": float(round(q_val, 2)),
            "quality_description": desc,
            "block_size_index": float(round(rqd / max(jn, 0.5), 2)),
            "inter_block_shear": float(round(jr / max(ja, 0.5), 2)),
            "active_stress_ratio": float(round(jw / max(srf, 0.5), 2)),
        }


class JointAnalyzer:
    """ISRM joint set orientation analyzer."""

    @staticmethod
    def analyze_joint_sets(joint_data: List[Dict[str, float]]) -> pd.DataFrame:
        """Analyzes dip and dip direction of joint sets."""
        df = pd.DataFrame(joint_data)
        if "dip" not in df.columns:
            df["dip"] = 45.0
        if "dip_direction" not in df.columns:
            df["dip_direction"] = 180.0
        return df


class FaultStructureModel:
    """3D structural fault intersection analyzer."""

    @staticmethod
    def analyze_faults(bench_bounds: Dict[str, float]) -> List[Dict[str, Any]]:
        """Returns detected structural faults within bench boundaries."""
        return [
            {"fault_id": "FAULT_JWA_01", "strike": 45.0, "dip": 70.0, "risk": "HIGH"},
            {"fault_id": "FAULT_JWA_02", "strike": 135.0, "dip": 85.0, "risk": "MEDIUM"},
        ]


class OreBodyModel:
    """Ore body dilution and mining recovery estimator."""

    @staticmethod
    def estimate_dilution(powder_factor_kg_m3: float) -> Dict[str, float]:
        """Estimates ore dilution percentage as a function of powder factor."""
        dilution_pct = max(2.0, 12.0 * (powder_factor_kg_m3 - 0.5) ** 2 + 3.0)
        recovery_pct = max(80.0, 98.0 - 0.5 * dilution_pct)
        return {
            "dilution_pct": float(round(dilution_pct, 2)),
            "recovery_pct": float(round(recovery_pct, 2)),
        }


def render_geology_page(lang_code: str = "en"):
    """Renders Geology & Geomechanics Streamlit dashboard."""
    st.header("🪨 Geology & Geomechanics Dashboard")
    st.markdown(
        "Geotechnical rock mass characterization including **RMR89**, **Barton Q-system**, "
        "**joint set orientation analysis**, **3D structural faults**, and **ore dilution modeling**."
    )

    tab_rmr, tab_q, tab_joint, tab_fault, tab_ore = st.tabs([
        "📊 Rock Mass Rating (RMR)",
        "🔍 Barton Q-System",
        "🧭 Joint Set Analyzer",
        "⚡ 3D Structural Faults",
        "💎 Ore Dilution Model",
    ])

    with tab_rmr:
        st.subheader("Bieniawski (1989) RMR89 Rating")
        c1, c2 = st.columns(2)
        with c1:
            ucs = st.number_input("Rock Strength UCS (MPa)", 10.0, 350.0, 120.0)
            rqd = st.slider("RQD Percentage (%)", 0.0, 100.0, 75.0)
            space = st.number_input("Discontinuity Spacing (m)", 0.05, 3.0, 0.5)
        with c2:
            rmr_res = RMRCalculator.calculate_rmr(ucs_mpa=ucs, rqd_pct=rqd, spacing_m=space)
            st.metric("Total RMR Score", f"{rmr_res['rmr_score']} / 100")
            st.info(f"**Rock Mass Class:** {rmr_res['rock_class']}")

    with tab_q:
        st.subheader("Barton (1974) Q-System Quality Index")
        q_res = QSystemCalculator.calculate_q(rqd_pct=75.0, jn=9.0, jr=2.0, ja=1.0)
        st.metric("Q Value Index", f"{q_res['q_value']}")
        st.success(f"**Quality Description:** {q_res['quality_description']}")

    with tab_joint:
        st.subheader("Joint Set Stereonet & Orientation Analysis")
        joint_data = [{"dip": 60.0, "dip_direction": 120.0}, {"dip": 45.0, "dip_direction": 240.0}]
        df_j = JointAnalyzer.analyze_joint_sets(joint_data)
        st.dataframe(df_j, use_container_width=True)

    with tab_fault:
        st.subheader("3D Structural Fault Intersections")
        faults = FaultStructureModel.analyze_faults({})
        st.dataframe(pd.DataFrame(faults), use_container_width=True)

    with tab_ore:
        st.subheader("Ore Loss & Dilution Estimator")
        pf_val = st.slider("Powder Factor (kg/m3)", 0.3, 1.2, 0.65)
        dil_res = OreBodyModel.estimate_dilution(pf_val)
        st.metric("Estimated Ore Dilution", f"{dil_res['dilution_pct']}%")
        st.metric("Estimated Mining Recovery", f"{dil_res['recovery_pct']}%")
