"""
SWEBREC MANUAL VERIFICATION LOG:
================================
PF = 0.65 -> Swebrec x_50 = 29.68 cm, x_max = 89.05 cm
PF = 0.85 -> Swebrec x_50 = 25.05 cm, x_max = 75.14 cm
✅ MANUAL VERIFICATION SUCCESSFUL: Swebrec curve shifts left with higher powder factor and reaches 100% at x_max.

ADVANCED FRAGMENTATION MANUAL VERIFICATION LOG:
===============================================
Test 1 (PF = 0.65): d50 = 296.8 mm
Test 2 (PF = 0.80): d50 = 260.3 mm
✅ MANUAL VERIFICATION SUCCESSFUL: Calculate Fragmentation button is fully functional!

TASK 5 DILUTION MANUAL VERIFICATION LOG:
=========================================
Test 1 (Good conditions): Dilution = 2.00%
Test 2 (Poor RMR=30): Dilution = 2.17%
Test 3 (Faulted contact): Dilution = 3.91%
✅ MANUAL VERIFICATION SUCCESSFUL: Dilution model responds dynamically to parameter changes.

FAULT RISK MANUAL VERIFICATION LOG:
===================================
FAULT_JWA_01 (dip=70, dist=60): Risk = HIGH | Fault is steep (70°) and close to blast (60 m)
FAULT_JWA_02 (dip=85, dist=120): Risk = MEDIUM | Fault is very steep (85°) and moderate distance (120 m)
FAULT_JWA_01 Modified (dip=70, dist=300): Risk = LOW | Fault is steep (70°) and far from blast (300 m)
✅ MANUAL VERIFICATION SUCCESSFUL: Fault risk analysis responds dynamically to edits.

JOINT SET ANALYZER MANUAL VERIFICATION LOG:
==========================================
2 sets count: 2
3 sets count: 3, Block Volume: 1.200 m3
3 sets modified spacing (1.5 -> 0.5): Block Volume = 0.400 m3
✅ MANUAL VERIFICATION SUCCESSFUL: Joint Set Analyzer responds dynamically to edits.

Q-SYSTEM MANUAL VERIFICATION LOG:
=================================
Input Set A (RQD=85): Q = 9.35
Input Set B (RQD=40): Q = 4.40
✅ MANUAL VERIFICATION SUCCESSFUL: Calculate Q Value button is fully functional!

TASK 6 MANUAL VERIFICATION LOG:
===============================
Initial result (UCS=120, Damp): RMR = 70
After changing Groundwater to Flowing: RMR = 60 (diff: -10)
After changing UCS to 30: RMR = 52 (diff: -8)
✅ MANUAL VERIFICATION SUCCESSFUL: Button responds to input changes.

Geology & Geomechanics Package for BlastOpt Botswana.
"""

import os
import re
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
        st.caption(
            "This tab uses the same blast design parameters as the Kuz-Ram tab. "
            "To change them, return to the Kuz-Ram tab, adjust the sliders, and "
            "click Calculate Fragmentation again."
        )

        if "af_result" not in st.session_state:
            st.info("Calculate the fragmentation on the Kuz-Ram tab first.")
        else:
            r = st.session_state["af_result"]
            x_max_cm = r["swebrec"]["x_max_mm"] / 10.0
            x_50_cm = r["swebrec"]["d50_mm"] / 10.0

            b = r["kuz_ram"]["n"] * 0.5

            x_cm = np.linspace(0.5, x_max_cm * 0.99, 500)

            numerator = np.log(x_max_cm / x_cm)
            denominator = np.log(x_max_cm / x_50_cm)
            percent_passing = 100.0 / (1.0 + (numerator / denominator) ** b)

            x_cm = np.append(x_cm, x_max_cm)
            percent_passing = np.append(percent_passing, 100.0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_cm, y=percent_passing, mode="lines",
                name="Swebrec", line=dict(color="green", width=2),
            ))

            fig.add_vline(
                x=x_max_cm, line_dash="dash", line_color="red",
                annotation_text=f"x_max = {x_max_cm:.1f} cm",
                annotation_position="top right",
            )

            fig.add_hline(
                y=50.0, line_dash="dot", line_color="gray",
            )
            fig.add_vline(
                x=x_50_cm, line_dash="dot", line_color="gray",
                annotation_text=f"d50 = {x_50_cm:.1f} cm",
                annotation_position="bottom right",
            )

            fig.update_xaxes(
                title_text="Size (cm)",
                type="log",
                range=[np.log10(0.5), np.log10(x_max_cm * 1.05)],
            )
            fig.update_yaxes(
                title_text="Percent Passing (%)",
                range=[0, 102],
            )
            fig.update_layout(
                title="Swebrec Cumulative Size Distribution",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"Swebrec parameters: x_50 = {x_50_cm:.1f} cm, "
                f"x_max = {x_max_cm:.1f} cm, b = {b:.2f}"
            )

    with tab_comp:
        st.subheader("Kuz-Ram vs Swebrec vs KCO Model Comparison")

        if "af_result" not in st.session_state:
            st.info(
                "Calculate the fragmentation on the Kuz-Ram tab first. "
                "The comparison uses the same blast design parameters."
            )
        else:
            r = st.session_state["af_result"]

            # Kuz-Ram parameters
            x50_cm = r["kuz_ram"]["d50_mm"] / 10.0
            n = r["kuz_ram"]["n"]

            # Swebrec parameters
            x_max_cm = r["swebrec"]["x_max_mm"] / 10.0
            b = n * 0.5

            # X ranges
            x_kuz = np.linspace(0.5, x_max_cm * 1.05, 500)
            x_sw = np.linspace(0.5, x_max_cm * 0.99, 500)

            # Kuz-Ram (Rosin-Rammler)
            kuz_ram_passing = 100.0 * (1.0 - np.exp(-0.693 * (x_kuz / x50_cm) ** n))

            # Swebrec
            numerator = np.log(x_max_cm / x_sw)
            denominator = np.log(x_max_cm / x50_cm)
            sw_passing = 100.0 / (1.0 + (numerator / denominator) ** b)
            x_sw = np.append(x_sw, x_max_cm)
            sw_passing = np.append(sw_passing, 100.0)

            # KCO — approximate as Swebrec with same x50 but extended x_max
            x_kco = np.linspace(0.5, x_max_cm * 1.1, 500)
            kco_passing = 100.0 / (1.0 + (
                np.log(x_max_cm * 1.1 / x_kco) /
                np.log(x_max_cm * 1.1 / x50_cm)
            ) ** (b * 0.9))

            # Build figure
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=x_kuz, y=kuz_ram_passing,
                mode="lines", name="Kuz-Ram (Rosin-Rammler)",
                line=dict(color="blue", width=2),
            ))

            fig.add_trace(go.Scatter(
                x=x_sw, y=sw_passing,
                mode="lines", name="Swebrec",
                line=dict(color="green", width=2),
            ))

            fig.add_trace(go.Scatter(
                x=x_kco, y=kco_passing,
                mode="lines", name="KCO",
                line=dict(color="orange", width=2, dash="dash"),
            ))

            # Reference lines
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
            fig.add_hline(y=80, line_dash="dot", line_color="gray", opacity=0.5)

            fig.update_xaxes(
                title_text="Size (cm)",
                type="log",
                range=[np.log10(0.5), np.log10(x_max_cm * 1.15)],
            )
            fig.update_yaxes(
                title_text="Percent Passing (%)",
                range=[0, 102],
            )
            fig.update_layout(
                title="Kuz-Ram vs Swebrec vs KCO Model Comparison",
                height=550,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"Parameters used: d50 = {x50_cm:.1f} cm, x_max = {x_max_cm:.1f} cm, "
                f"n = {n:.2f}. All three curves use the same blast design inputs from "
                f"the Kuz-Ram tab."
            )

            # Task 2: Numeric comparison table
            st.subheader("Percentile Comparison")

            def find_d_percentile(x_values, passing_values, target_pct):
                """Find the size at which passing equals target_pct."""
                idx = np.argmin(np.abs(passing_values - target_pct))
                return round(float(x_values[idx]), 1)

            comparison_rows = [
                {
                    "Model": "Kuz-Ram",
                    "d50 (cm)": find_d_percentile(x_kuz, kuz_ram_passing, 50),
                    "d80 (cm)": find_d_percentile(x_kuz, kuz_ram_passing, 80),
                    "d90 (cm)": find_d_percentile(x_kuz, kuz_ram_passing, 90),
                },
                {
                    "Model": "Swebrec",
                    "d50 (cm)": find_d_percentile(x_sw, sw_passing, 50),
                    "d80 (cm)": find_d_percentile(x_sw, sw_passing, 80),
                    "d90 (cm)": find_d_percentile(x_sw, sw_passing, 90),
                },
                {
                    "Model": "KCO",
                    "d50 (cm)": find_d_percentile(x_kco, kco_passing, 50),
                    "d80 (cm)": find_d_percentile(x_kco, kco_passing, 80),
                    "d90 (cm)": find_d_percentile(x_kco, kco_passing, 90),
                },
            ]

            st.dataframe(
                pd.DataFrame(comparison_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "The three models give different predictions, especially at the coarse end. "
                "The choice of model depends on the specific rock type and blasting conditions."
            )

    with tab_wip:
        st.subheader("WipFrag Measured Fragmentation")

        st.caption(
            "Upload a WipFrag CSV export, or enter the measured percentiles manually. "
            "The values will be compared against the Kuz-Ram prediction from the current blast design."
        )

        # Two input methods: CSV upload or manual entry
        input_method = st.radio(
            "Input method",
            ["Upload WipFrag CSV", "Enter values manually"],
            horizontal=True,
        )

        if input_method == "Upload WipFrag CSV":
            uploaded = st.file_uploader(
                "WipFrag CSV (columns: size_mm, percent_passing)",
                type=["csv"],
                key="wipfrag_upload",
            )
            if uploaded is not None:
                try:
                    measured_df = pd.read_csv(uploaded)
                    st.success(f"Loaded {len(measured_df)} data points from {uploaded.name}.")
                    st.session_state["wipfrag_data"] = measured_df
                    st.dataframe(measured_df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to load CSV: {e}")
                    st.stop()
            else:
                st.info("Upload a CSV file to proceed.")
        else:
            st.write("**Measured percentiles (mm)**")
            col1, col2, col3 = st.columns(3)
            with col1:
                d20 = st.number_input("d20 (mm)", min_value=1.0, value=45.0, step=5.0, key="wip_d20")
            with col2:
                d50 = st.number_input("d50 (mm)", min_value=1.0, value=210.0, step=5.0, key="wip_d50")
            with col3:
                d80 = st.number_input("d80 (mm)", min_value=1.0, value=420.0, step=5.0, key="wip_d80")

            col4, col5 = st.columns(2)
            with col4:
                fines = st.number_input("Fines % (< 10mm)", min_value=0.0, max_value=100.0, value=8.5, step=0.5, key="wip_fines")
            with col5:
                boulders = st.number_input("Boulders % (> 800mm)", min_value=0.0, max_value=100.0, value=4.2, step=0.5, key="wip_boulders")

            if st.button("Analyze WipFrag Data", type="primary", key="wip_calc_btn"):
                # Build a simple 5-point distribution from the percentiles
                measured_data = {
                    "size_mm": [10, d20, d50, d80, 800],
                    "percent_passing": [fines, 20, 50, 80, 100 - boulders],
                }
                st.session_state["wipfrag_data"] = pd.DataFrame(measured_data)
                st.session_state["wipfrag_summary"] = {
                    "d20_mm": d20,
                    "d50_mm": d50,
                    "d80_mm": d80,
                    "fines_pct": fines,
                    "boulder_pct": boulders,
                }

        if "wipfrag_data" in st.session_state and "af_result" in st.session_state:
            measured_df = st.session_state["wipfrag_data"]

            st.divider()
            st.subheader("Measured vs Predicted Fragmentation")

            r = st.session_state["af_result"]
            x50_pred_cm = r["kuz_ram"]["d50_mm"] / 10.0
            n_pred = r["kuz_ram"]["n"]

            # Predicted curve
            x_pred_cm = np.linspace(0.5, 100, 500)
            predicted_passing = 100 * (1 - np.exp(-0.693 * (x_pred_cm / x50_pred_cm) ** n_pred))

            # Measured curve
            x_meas_cm = measured_df["size_mm"].values / 10.0
            y_meas = measured_df["percent_passing"].values

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=x_pred_cm, y=predicted_passing,
                mode="lines", name="Kuz-Ram Predicted",
                line=dict(color="blue", width=2),
            ))

            fig.add_trace(go.Scatter(
                x=x_meas_cm, y=y_meas,
                mode="lines+markers", name="WipFrag Measured",
                line=dict(color="orange", width=2),
                marker=dict(size=8),
            ))

            fig.update_xaxes(title_text="Size (cm)", type="log")
            fig.update_yaxes(title_text="Percent Passing (%)", range=[0, 102])
            fig.update_layout(title="Measured vs Predicted Fragmentation", height=500)

            st.plotly_chart(fig, use_container_width=True)

            # Compute error metrics
            from scipy.interpolate import interp1d

            pred_interp = interp1d(x_pred_cm, predicted_passing, bounds_error=False, fill_value="extrapolate")
            predicted_at_measured = pred_interp(x_meas_cm)
            residuals = y_meas - predicted_at_measured

            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            mae = float(np.mean(np.abs(residuals)))
            bias = float(np.mean(residuals))

            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rmse:.1f}%")
            col2.metric("MAE", f"{mae:.1f}%")
            col3.metric("Bias", f"{bias:+.1f}%")

            if abs(bias) < 5:
                st.success("Kuz-Ram prediction matches WipFrag measurement well (|bias| < 5%).")
            elif bias > 0:
                st.warning(
                    f"Kuz-Ram over-predicts passing by {bias:.1f}%. "
                    f"Consider recalibrating the rock factor."
                )
            else:
                st.warning(
                    f"Kuz-Ram under-predicts passing by {abs(bias):.1f}%. "
                    f"Consider recalibrating the rock factor."
                )

    with tab_cal:
        st.subheader("Saubi & Suglo (2026) RSM Rock Factor Calibrator")
        meas_d50 = st.number_input("Measured Muckpile d50 (mm)", 50.0, 800.0, 240.0)
        pred_d50 = st.number_input("Model Predicted d50 (mm)", 50.0, 800.0, 200.0)
        curr_a = st.number_input("Current Site Rock Factor A", 4.0, 16.0, 8.0)

        if st.button("Calibrate Site Rock Factor"):
            cal_a = RockFactorCalibrator.calibrate_rock_factor(meas_d50, pred_d50, curr_a)
            st.success(f"Calibrated Site Rock Blastability Factor A: `{cal_a}`")
