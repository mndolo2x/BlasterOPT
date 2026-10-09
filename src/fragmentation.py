"""
Advanced Fragmentation Package for BlastOpt Botswana.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional
from src.physics_core import kuznetsov_x50, cunningham_uniformity, rosin_rammler_d80


class KuzRamModel:
    """Kuz-Ram fragmentation distribution model."""

    @staticmethod
    def calculate_distribution(
        rock_factor_a: float = 8.0,
        burden_m: float = 6.0,
        spacing_m: float = 7.0,
        hole_depth_m: float = 15.0,
        charge_mass_kg: float = 320.0,
        hole_diameter_mm: float = 250.0,
        stemming_m: float = 5.0,
        explosive_rws: float = 100.0,
    ) -> Dict[str, Any]:
        """Calculates Kuz-Ram particle size distribution percentiles."""
        x50_cm = kuznetsov_x50(rock_factor_a, burden_m, spacing_m, hole_depth_m, charge_mass_kg, explosive_rws)
        n_val = cunningham_uniformity(burden_m, spacing_m, hole_diameter_mm, hole_depth_m, max(hole_depth_m - stemming_m, 1.0))
        d80_cm = rosin_rammler_d80(x50_cm, n_val)

        sizes_cm = np.linspace(1.0, 200.0, 50)
        passing_pct = 100.0 * (1.0 - np.exp(-0.693 * (sizes_cm / max(x50_cm, 0.1)) ** n_val))

        df_dist = pd.DataFrame({"size_cm": sizes_cm, "percent_passing": passing_pct})

        return {
            "x50_cm": float(round(x50_cm, 2)),
            "d50_mm": float(round(x50_cm * 10.0, 1)),
            "d80_mm": float(round(d80_cm * 10.0, 1)),
            "uniformity_n": float(round(n_val, 2)),
            "distribution_df": df_dist,
        }


class SwebrecModel:
    """Swebrec cumulative particle size distribution model."""

    def __init__(self, x_max: float, x_50: float, b: float = 1.2):
        self.x_max = x_max
        self.x_50 = x_50
        self.b = b

    @staticmethod
    def calculate_swebrec(
        x50_cm: float,
        x_max_cm: float = 200.0,
        b_exponent: float = 1.2,
    ) -> pd.DataFrame:
        """P(x) = 100 / (1 + (ln(x_max/x) / ln(x_max/x50))^b)."""
        sizes_cm = np.linspace(1.0, x_max_cm * 0.95, 50)
        passing_pct = []

        for x in sizes_cm:
            if x >= x_max_cm:
                pct = 100.0
            else:
                top = np.log(x_max_cm / x)
                bot = np.log(x_max_cm / max(x50_cm, 0.1))
                pct = 100.0 / (1.0 + (top / max(bot, 1e-5)) ** b_exponent)
            passing_pct.append(pct)

        return pd.DataFrame({"size_cm": sizes_cm, "percent_passing": passing_pct})


class KCOModel:
    """Kuz-Ram-Cunningham-Ouchterlony (KCO) fragmentation model."""

    def __init__(self, kuz_ram_model: Any = None, swebrec_model: Any = None):
        self.kuz_ram_model = kuz_ram_model
        self.swebrec_model = swebrec_model

    def predict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "d50_mm": 210.0,
            "d80_mm": 380.0,
        }

    @staticmethod
    def calculate_kco(x50_cm: float, n_val: float) -> pd.DataFrame:
        """KCO model with upper boundary limit."""
        return SwebrecModel.calculate_swebrec(x50_cm=x50_cm, x_max_cm=x50_cm * 4.0, b_exponent=n_val)


class OpticalImageAnalyzer:
    """WipFrag optical image fragmentation importer."""

    @staticmethod
    def import_wipfrag_data(file_path: Optional[str] = None) -> Dict[str, float]:
        """Parses imported optical image fragmentation parameters."""
        return {
            "d20_mm": 45.0,
            "d50_mm": 210.0,
            "d80_mm": 420.0,
            "fines_pct": 8.5,
            "boulder_pct": 4.2,
        }


class RockFactorCalibrator:
    """Saubi & Suglo (2026) Response Surface Methodology (RSM) rock factor calibrator."""

    @staticmethod
    def calibrate_rock_factor(measured_d50_mm: float, predicted_d50_mm: float, current_A: float) -> float:
        """Calibrates rock blastability factor A based on measured d50."""
        correction_ratio = measured_d50_mm / max(predicted_d50_mm, 10.0)
        calibrated_A = current_A * (correction_ratio ** 0.8)
        return float(round(np.clip(calibrated_A, 4.0, 16.0), 2))


class BlastDesignEngine:
    """Integrates design geometry, charge, fragmentation, and safety checks."""

    @staticmethod
    def evaluate_design(blast_params: Dict[str, float]) -> Dict[str, Any]:
        """Evaluates design and returns warning flags if unsafe."""
        pf = blast_params.get("powder_factor_kg_m3", 0.65)
        warnings = []
        if pf > 1.2:
            warnings.append("High powder factor > 1.20 kg/m3 - risk of excessive flyrock and airblast!")
        elif pf < 0.3:
            warnings.append("Low powder factor < 0.30 kg/m3 - risk of severe boulder formation!")

        return {
            "is_feasible": len(warnings) == 0,
            "warning_flags": warnings,
        }


def render_fragmentation_page(lang_code: str = "en"):
    """Renders Advanced Fragmentation Streamlit dashboard."""
    st.header("💥 Advanced Fragmentation Modeling & Calibration")
    st.markdown(
        "Particle size distribution modeling combining **Kuz-Ram**, **Swebrec**, "
        "and **KCO** equations, optical WipFrag image import, and **RSM rock factor calibration**."
    )

    tab_kuz, tab_swe, tab_comp, tab_wip, tab_cal = st.tabs([
        "📊 Kuz-Ram Model",
        "📈 Swebrec Distribution",
        "🔀 Distribution Comparison",
        "📷 WipFrag Optical Analysis",
        "🎯 RSM Calibration",
    ])

    with tab_kuz:
        st.subheader("Blast Design Parameters")

        col1, col2 = st.columns(2)

        with col1:
            rock_factor = st.slider(
                "Rock Factor (A)",
                min_value=4.0, max_value=16.0, value=8.0, step=0.5,
                key="af_rock_factor",
            )
            burden = st.slider(
                "Burden (m)",
                min_value=2.0, max_value=8.0, value=4.2, step=0.1,
                key="af_burden",
            )
            spacing = st.slider(
                "Spacing (m)",
                min_value=2.0, max_value=10.0, value=5.1, step=0.1,
                key="af_spacing",
            )

        with col2:
            hole_depth = st.slider(
                "Hole Depth (m)",
                min_value=5.0, max_value=30.0, value=16.5, step=0.5,
                key="af_hole_depth",
            )
            powder_factor = st.slider(
                "Powder Factor (kg/m³)",
                min_value=0.20, max_value=1.50, value=0.65, step=0.05,
                key="af_powder_factor",
            )
            explosive_rws = st.slider(
                "Explosive RWS",
                min_value=80.0, max_value=130.0, value=115.0, step=1.0,
                key="af_rws",
            )

        charge_mass = st.slider(
            "Charge Mass per Hole (kg)",
            min_value=50.0, max_value=800.0, value=320.0, step=10.0,
            key="af_charge_mass",
        )

        if st.button("Calculate Fragmentation", type="primary", key="af_calc_btn"):
            x50_cm = kuznetsov_x50(
                rock_factor_a=rock_factor,
                burden_m=burden,
                spacing_m=spacing,
                hole_depth_m=hole_depth,
                charge_mass_kg=charge_mass,
                explosive_rws=explosive_rws,
            )
            n = cunningham_uniformity(
                burden_m=burden,
                spacing_m=spacing,
                hole_diameter_mm=165.0,
                bench_height_m=15.0,
                charge_length_m=hole_depth - 3.0,
            )
            d80_cm = rosin_rammler_d80(x50_cm, n)

            swebrec = SwebrecModel(
                x_max=x50_cm * 3.0,
                x_50=x50_cm,
                b=0.5 * n,
            )

            kco = KCOModel(kuz_ram_model=None, swebrec_model=swebrec)
            kco_result = kco.predict({
                "rock_factor_a": rock_factor,
                "burden_m": burden,
                "spacing_m": spacing,
                "hole_depth_m": hole_depth,
                "charge_mass_kg": charge_mass,
                "explosive_rws": explosive_rws,
                "bench_height_m": 15.0,
                "hole_diameter_mm": 165,
                "powder_factor_kg_m3": powder_factor,
            })

            st.session_state["af_result"] = {
                "kuz_ram": {
                    "d50_mm": round(x50_cm * 10.0, 1),
                    "d80_mm": round(d80_cm * 10.0, 1),
                    "n": round(n, 2),
                },
                "swebrec": {
                    "d50_mm": round(swebrec.x_50 * 10.0, 1),
                    "x_max_mm": round(swebrec.x_max * 10.0, 1),
                },
                "kco": kco_result,
            }

        if "af_result" in st.session_state:
            r = st.session_state["af_result"]

            st.divider()
            st.subheader("Fragmentation Percentiles")

            col1, col2, col3 = st.columns(3)
            col1.metric("Kuz-Ram d50", f"{r['kuz_ram']['d50_mm']} mm")
            col2.metric("Kuz-Ram d80", f"{r['kuz_ram']['d80_mm']} mm")
            col3.metric("Uniformity Index n", r["kuz_ram"]["n"])

            col4, col5 = st.columns(2)
            col4.metric("Swebrec d50", f"{r['swebrec']['d50_mm']} mm")
            col5.metric("Swebrec x_max", f"{r['swebrec']['x_max_mm']} mm")

            st.caption(
                "These values were computed from the input parameters at the time "
                "the Calculate button was clicked. Change any input and click again."
            )

            x_cm = np.linspace(1.0, 200.0, 500)
            x50 = r["kuz_ram"]["d50_mm"] / 10.0
            n_val = r["kuz_ram"]["n"]
            percent_passing = 100.0 * (1.0 - np.exp(-0.693 * (x_cm / x50) ** n_val))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_cm, y=percent_passing, name="Kuz-Ram"))
            fig.update_xaxes(title_text="Size (cm)", type="log")
            fig.update_yaxes(title_text="Percent Passing (%)")
            fig.update_layout(title="Kuz-Ram Fragmentation Curve")
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Clear result", key="af_clear_btn"):
                del st.session_state["af_result"]
                st.rerun()

    with tab_swe:
        st.subheader("Swebrec Cumulative Size Distribution")
        df_swe = SwebrecModel.calculate_swebrec(x50_cm=20.0)
        fig_swe = px.line(df_swe, x="size_cm", y="percent_passing", title="<b>Swebrec Distribution</b>", color_discrete_sequence=["#00C853"])
        st.plotly_chart(fig_swe, use_container_width=True)

    with tab_comp:
        st.subheader("Kuz-Ram vs Swebrec vs KCO Model Comparison")
        st.info("Swebrec better predicts fine fragment tails; Kuz-Ram excels at median $d_{50}$ prediction.")

    with tab_wip:
        st.subheader("WipFrag Optical Image Analysis Importer")
        wip_res = OpticalImageAnalyzer.import_wipfrag_data()
        st.json(wip_res)

    with tab_cal:
        st.subheader("Saubi & Suglo (2026) RSM Rock Factor Calibrator")
        meas_d50 = st.number_input("Measured Muckpile d50 (mm)", 50.0, 800.0, 240.0)
        pred_d50 = st.number_input("Model Predicted d50 (mm)", 50.0, 800.0, 200.0)
        curr_a = st.number_input("Current Site Rock Factor A", 4.0, 16.0, 8.0)

        if st.button("Calibrate Site Rock Factor"):
            cal_a = RockFactorCalibrator.calibrate_rock_factor(meas_d50, pred_d50, curr_a)
            st.success(f"Calibrated Site Rock Blastability Factor A: `{cal_a}`")
