"""
Streamlit UI module for Geology Integration in BlasterOPT / BlastOpt Botswana.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any

from src.geology.rmr import RMRCalculator
from src.geology.q_system import QSystemCalculator
from src.geology.joint_analysis import JointAnalyzer
from src.geology.structures import StructuralModel
from src.geology.ore_body import OreBodyModel
from src.geology.renderer import GeologyRenderer
try:
    from src.render3d.bench_model import BenchModel
    from src.render3d.hole_pattern import HolePattern
except Exception:
    class BenchModel:
        def __init__(self, bench_id: str = "B1", crest_elevation_m: float = 100.0, toe_elevation_m: float = 85.0, face_angle_deg: float = 75.0) -> None:
            self.bench_id = bench_id
            self.crest_elevation_m = crest_elevation_m
            self.toe_elevation_m = toe_elevation_m
            self.face_angle_deg = face_angle_deg
        def get_face_surface(self, resolution: int = 20) -> np.ndarray:
            return np.zeros((resolution, 3))
    class HolePattern:
        def __init__(self, burden_m: float, spacing_m: float, bench: Any, pattern_type: str = "square") -> None:
            pass
        def generate_holes(self, n_rows: int = 2, n_per_row: int = 3) -> list:
            return []


def render_geology_integration_page():
    st.header("🪨 Geological & Structural Rock Mass Integration")
    st.caption(
        "International standard rock mass classification (Bieniawski RMR89, Barton Q-system), "
        "ISRM joint set stereonets, 3D structural fault/dyke intersection modeling, and ore dilution reconciliation."
    )

    t1, t2, t3, t4, t5 = st.tabs([
        "1. Rock Mass Classification",
        "2. Joint Analysis & Stereonets",
        "3. Geological Structures",
        "4. Ore Body & Dilution",
        "5. 3D Geology Integration",
    ])

    with t1:
        st.subheader("RMR89 (Bieniawski 1989) & Q-System (Barton 1974)")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### RMR89 Parameters")
            ucs = st.number_input("Intact Rock Strength UCS (MPa)", 1.0, 500.0, 120.0, key="rmr_ucs")
            rqd = st.slider("RQD (%)", 0.0, 100.0, 85.0, key="rmr_rqd")
            sp = st.number_input("Discontinuity Spacing (m)", 0.01, 10.0, 0.6, key="rmr_sp")
            roughness = st.selectbox("Roughness", ["very_rough", "rough", "slightly_rough", "smooth", "slickensided"], index=2, key="rmr_rough")
            weathering = st.selectbox("Weathering", ["unweathered", "slightly_weathered", "highly_weathered"], index=1, key="rmr_weath")

            rmr_calc = RMRCalculator()
            rmr_res = rmr_calc.calculate_total(
                ucs_mpa=ucs,
                rqd_pct=rqd,
                joint_spacing_m=sp,
                roughness=roughness,
                weathering=weathering,
            )

            st.metric("RMR89 Score", f"{rmr_res['rmr']} / 100")
            st.info(f"**Rock Class:** {rmr_res['class'].replace('_', ' ').upper()}")
            st.markdown(f"**Excavation Support:** {rmr_res['excavation_support']}")
            st.markdown(f"**Max Span:** {rmr_res['max_span_m']} m | **Stand-Up Time:** {rmr_res['stand_up_time']}")

        with c2:
            st.markdown("### Barton Q-System Parameters")
            num_sets = st.slider("Number of Joint Sets (Jn)", 1, 5, 3, key="q_sets")
            j_type = st.selectbox("Joint Type (Jr)", ["rough_undulating", "smooth_undulating", "rough_planar", "smooth_planar"], index=2, key="q_jr")
            alt = st.selectbox("Alteration (Ja)", ["tight_unaltered", "slightly_altered", "clay_coated", "clay_filled_thick"], index=1, key="q_ja")
            water = st.selectbox("Water Condition (Jw)", ["dry", "damp", "wet", "dripping", "flowing"], index=0, key="q_jw")

            q_calc = QSystemCalculator()
            q_res = q_calc.calculate_q(
                rqd_pct=rqd,
                num_joint_sets=num_sets,
                joint_type=j_type,
                alteration=alt,
                water_condition=water,
                stress_condition="medium_stress",
            )

            st.metric("Q-System Value", f"{q_res['q_value']:.2f}")
            st.info(f"**Category (Class {q_res['class_roman']}):** {q_res['description']}")
            st.markdown(f"**Support Recommendation:** {q_res['support_recommendation']}")

    with t2:
        st.subheader("Joint Measurements & Stereonet Projection (ISRM 1978/2014)")
        measurements = [
            {"dip_deg": 60.0, "dip_direction_deg": 180.0, "spacing_m": 0.4, "persistence_m": 4.0},
            {"dip_deg": 45.0, "dip_direction_deg": 90.0, "spacing_m": 0.3, "persistence_m": 5.0},
            {"dip_deg": 75.0, "dip_direction_deg": 270.0, "spacing_m": 0.6, "persistence_m": 3.0},
        ]

        analyzer = JointAnalyzer(measurements)
        sets = analyzer.identify_joint_sets(max_sets=3)
        block = analyzer.calculate_block_size()
        frag = analyzer.predict_fragmentation_from_joints()

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("In-Situ Block Volume (V_block)", f"{block['v_block_m3']} m³")
            st.metric("Block Size Class", block['classification'].upper())
            st.metric("Natural Block D80", f"{frag['natural_block_d80_cm']} cm")
            st.metric("Expected Blast D80", f"{frag['expected_blast_d80_cm']} cm")
            st.markdown(f"**Powder Factor Multiplier:** x{frag['powder_factor_adjustment']}")
            st.info(frag['reasoning'])

        with c2:
            fig_st = analyzer.create_stereonet(plot_type="schmidt")
            st.pyplot(fig_st)
            st.subheader("Identified Joint Sets")
            st.dataframe(pd.DataFrame(sets), use_container_width=True)

    with t3:
        st.subheader("3D Faults, Dykes & Per-Hole Intersections")
        struct_model = StructuralModel()

        poly_fault = np.array([[0, 0, 90.0], [50, 0, 90.0]])
        struct_model.add_fault("F_MAIN_01", poly_fault, dip_deg=70.0, thickness_m=1.2, infilling="clay_gouge")

        poly_dyke = np.array([[0, 10, 85.0], [50, 10, 85.0]])
        struct_model.add_dyke("D_DOLERITE_01", poly_dyke, dip_deg=80.0, thickness_m=2.0, rock_type="Dolerite_Hard")

        bench = BenchModel("B1", 100.0, 85.0, 75.0)
        pattern = HolePattern(burden_m=4.0, spacing_m=5.0, bench=bench, pattern_type="square")
        holes = pattern.generate_holes(n_rows=2, n_per_row=3)

        inters = struct_model.find_intersections(holes)
        adj = struct_model.calculate_blast_adjustment(inters)

        st.markdown(f"**Total Holes Affected by Structures:** {adj['total_holes_affected']}")
        for h_id, h_adj in adj["hole_adjustments"].items():
            if h_adj["notes"]:
                st.warning(f"**Hole {h_id}:** {', '.join(h_adj['notes'])}")

    with t4:
        st.subheader("Block Model Grade Reconciliation & Dilution")
        df_bm = pd.DataFrame({
            "x": [0, 10, 20, 30],
            "y": [0, 0, 0, 0],
            "z": [100, 100, 100, 100],
            "grade": [0.5, 0.11, 0.05, 0.02],
        })

        ore_model = OreBodyModel(df_bm, grade_column="grade", cutoff_grade=0.1)
        classified = ore_model.classify_ore_waste()

        poly = np.array([[0, 0, 100], [50, 0, 100]])
        dil = ore_model.calculate_dilution(poly, blasted_tonnage=1000.0, ore_tonnage_expected=800.0, overbreak_waste_tonnes=50.0)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Planned Dilution", f"{dil['planned_dilution_pct']}%")
            st.metric("Unplanned Overbreak Dilution", f"{dil['unplanned_dilution_pct']}%")
            st.metric("Total Dilution", f"{dil['total_dilution_pct']}%")
            st.metric("Ore Loss", f"{dil['ore_loss_tonnes']} t")
        with c2:
            st.markdown("**Block Classification:**")
            st.dataframe(classified, use_container_width=True)
            st.markdown("**Selective Blasting Recommendations:**")
            for rec in dil["recommendations"]:
                st.write(f"- {rec}")

    with t5:
        st.subheader("Unified 3D Geology Bench Overlay")
        bench = BenchModel("B1", 100.0, 85.0, 75.0)
        g_renderer = GeologyRenderer()
        fig_g3d = g_renderer.render_all(bench=bench)
        st.plotly_chart(fig_g3d, use_container_width=True)
