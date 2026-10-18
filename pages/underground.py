"""
Underground Mining Intelligence Module — BlastOpt Streamlit Page.
Provides research prototype modeling for Stope Overbreak, Stope Dilution (ELOS), and Underground PPV Vibration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_underground_page():
    st.header("⛏️ Underground Mining Intelligence")
    st.markdown(
        "Research prototype suite for underground hard-rock stoping operations. "
        "Evaluates **Stope Overbreak**, **Stope Dilution (ELOS)**, and **Underground Vibration (PPV)** "
        "using empirical geotechnical heuristics."
    )

    tab1, tab2, tab3 = st.tabs([
        "Stope Overbreak — HESM-RF-GBM",
        "Stope Dilution — GA-ANN",
        "Underground Vibration — PSO-SVR",
    ])

    # ---------------------------------------------------------
    # TAB 1: STOPE OVERBREAK (HESM-RF-GBM)
    # ---------------------------------------------------------
    with tab1:
        st.subheader("Stope Overbreak / Underbreak Prediction")
        st.info(
            "⚠️ **RESEARCH PROTOTYPE — Simulated / Heuristic Calculation.**  \n"
            "Predicts overbreak percentage based on drilling geometry, charging energy, "
            "breakthrough status, and rock quality (RQD)."
        )

        col_ob1, col_ob2 = st.columns([1, 2])

        with col_ob1:
            ring_seq = st.selectbox(
                "Ring / Slot Sequence",
                ["Primary Slot Ring", "Secondary Ring", "Perimeter / Hangingwall Ring", "Footwall Ring"],
                index=2,
                help="Position of ring within the stope extraction sequence."
            )
            hole_dip = st.slider("Hole Dip Angle (°)", -90, 90, -75, step=5, help="Negative for down-holes, positive for uppers.")
            breakthrough = st.checkbox("Hole Breakthrough Confirmed", value=True, help="Whether the blast hole breaks through into lower extraction drive.")
            pf_ob = st.number_input("Powder Factor (kg/m³)", 0.20, 2.50, 0.85, step=0.05)
            rqd_ob = st.slider("Rock Quality Designation (RQD %)", 10, 100, 65, step=5)
            dev_ob = st.number_input("Drillhole Deviation (m)", 0.0, 3.0, 0.4, step=0.1)

            if st.button("Calculate Stope Overbreak 🚀", type="primary"):
                # Heuristic deterministic calculation
                base_ob = 5.0
                if "Perimeter" in ring_seq:
                    base_ob += 4.0
                elif "Slot" in ring_seq:
                    base_ob += 2.0

                dip_penalty = (abs(hole_dip) / 90.0) * 2.0
                break_bonus = -1.5 if breakthrough else 3.0
                pf_factor = (pf_ob - 0.7) * 8.0
                rqd_factor = (100.0 - rqd_ob) * 0.12
                dev_factor = dev_ob * 5.0

                calc_overbreak = max(1.0, min(45.0, base_ob + dip_penalty + break_bonus + pf_factor + rqd_factor + dev_factor))
                calc_underbreak = max(0.5, min(20.0, 8.0 - (calc_overbreak * 0.25)))

                st.session_state["ob_results"] = {
                    "overbreak_pct": calc_overbreak,
                    "underbreak_pct": calc_underbreak,
                    "ring_seq": ring_seq,
                    "pf_ob": pf_ob,
                    "rqd_ob": rqd_ob,
                    "dev_ob": dev_ob,
                }

        with col_ob2:
            st.markdown("### 📊 Overbreak Analysis Results")
            res_ob = st.session_state.get("ob_results")

            if res_ob:
                m1, m2 = st.columns(2)
                m1.metric("Predicted Overbreak", f"{res_ob['overbreak_pct']:.2f}%")
                m2.metric("Estimated Underbreak", f"{res_ob['underbreak_pct']:.2f}%")

                if res_ob['overbreak_pct'] > 15.0:
                    st.error("⚠️ **HIGH OVERBREAK RISK:** Predicted overbreak exceeds 15% threshold. High risk of wall sloughing and hangingwall instability.")
                else:
                    st.success("✅ **ACCEPTABLE OVERBREAK:** Overbreak is within acceptable geotechnical limits (≤ 15%).")

                # Plotly comparison
                df_chart = pd.DataFrame({
                    "Metric": ["Predicted Overbreak %", "Estimated Underbreak %", "Geotechnical Threshold %"],
                    "Percentage (%)": [res_ob["overbreak_pct"], res_ob["underbreak_pct"], 15.0],
                    "Color": ["#E63946", "#F4A261", "#2A9D8F"],
                })
                fig_ob = px.bar(
                    df_chart,
                    x="Metric",
                    y="Percentage (%)",
                    color="Metric",
                    color_discrete_sequence=["#E63946", "#F4A261", "#2A9D8F"],
                    title="<b>Stope Overbreak vs Threshold</b>",
                    text_auto=".1f",
                )
                fig_ob.update_layout(template="plotly_white", showlegend=False, height=350)
                st.plotly_chart(fig_ob, use_container_width=True)
            else:
                st.info("Click **Calculate Stope Overbreak** to run heuristic prediction.")

    # ---------------------------------------------------------
    # TAB 2: STOPE DILUTION (GA-ANN)
    # ---------------------------------------------------------
    with tab2:
        st.subheader("Stope Dilution (ELOS) Estimation")
        st.info(
            "⚠️ **RESEARCH PROTOTYPE — Simulated / Heuristic Calculation.**  \n"
            "Calculates **Equivalent Linear Overbreak Slough (ELOS)** in meters based on "
            "Mathews-Potvin Modified Stability Graph parameters."
        )

        col_dil1, col_dil2 = st.columns([1, 2])

        with col_dil1:
            hr_val = st.number_input("Hydraulic Radius HR (m)", 2.0, 20.0, 6.5, step=0.5, help="Stope wall surface area divided by perimeter.")
            n_prime = st.number_input("Modified Stability Number (N')", 0.1, 100.0, 15.0, step=1.0, help="N' = Q' * A * B * C")
            blast_damage_factor = st.slider("Blast Damage Factor (BD)", 0.5, 2.0, 1.0, step=0.1, help="1.0 = standard smooth blasting, >1.2 = heavy unconfined blasting.")
            stope_width = st.number_input("Stope Width (m)", 0.5, 30.0, 3.0, step=0.5, help="Designed stope extraction width.")
            pf_dil = st.number_input("Stope Powder Factor (kg/m³)", 0.20, 2.50, 0.90, step=0.05, key="pf_dil_input")

            if st.button("Estimate Stope Dilution (ELOS) 🚀", type="primary"):
                # Deterministic ELOS equation: ELOS = (HR * 0.22) - (N_prime * 0.012) + (BD * 0.35)
                elos_calc = max(0.05, (hr_val * 0.22) - (n_prime * 0.012) + (blast_damage_factor * 0.35))
                dilution_pct = min(100.0, (elos_calc / stope_width) * 100.0)

                st.session_state["dil_results"] = {
                    "elos_m": elos_calc,
                    "dilution_pct": dilution_pct,
                    "hr": hr_val,
                    "n_prime": n_prime,
                    "bd": blast_damage_factor,
                    "stope_width": stope_width,
                    "pf": pf_dil,
                }

        with col_dil2:
            st.markdown("### 📊 Dilution (ELOS) Output")
            res_dil = st.session_state.get("dil_results")

            if res_dil:
                m_d1, m_d2 = st.columns(2)
                m_d1.metric("Equivalent Linear Overbreak (ELOS)", f"{res_dil['elos_m']:.2f} m")
                m_d2.metric("Volumetric Dilution %", f"{res_dil['dilution_pct']:.1f}%")
                st.caption("Powder Factor is collected for context but is not used by this specific ELOS heuristic.")

                if res_dil["pf"] > 1.20:
                    st.warning("⚠️ **HIGH POWDER FACTOR WARNING:** Powder factor > 1.20 kg/m³ increases wall blast damage and hangingwall sloughing risk.")

                if res_dil["elos_m"] > 1.0:
                    st.error("⚠️ **UNSTABLE WALL:** ELOS > 1.0 m indicates significant wall sloughing. Cable bolting or destress blasting recommended.")
                else:
                    st.success("✅ **STABLE WALL:** ELOS ≤ 1.0 m indicates manageable dilution.")

                # Sensitivity bar chart
                df_elos = pd.DataFrame({
                    "Parameter": ["HR Impact", "N' Stability", "Blast Damage", "Resulting ELOS (m)"],
                    "Value": [res_dil["hr"] * 0.22, -(res_dil["n_prime"] * 0.012), res_dil["bd"] * 0.35, res_dil["elos_m"]],
                })
                fig_elos = px.bar(
                    df_elos,
                    x="Parameter",
                    y="Value",
                    color="Parameter",
                    title="<b>ELOS Component Contribution Analysis</b>",
                    text_auto=".2f",
                )
                fig_elos.update_layout(template="plotly_white", showlegend=False, height=350)
                st.plotly_chart(fig_elos, use_container_width=True)
            else:
                st.info("Click **Estimate Stope Dilution (ELOS)** to compute results.")

    # ---------------------------------------------------------
    # TAB 3: UNDERGROUND VIBRATION (PSO-SVR)
    # ---------------------------------------------------------
    with tab3:
        st.subheader("Underground Peak Particle Velocity (PPV) Attenuation")
        st.warning("⚠️ **UNCALIBRATED SITE ATTENUATION NOTICE:** Uses standard USBM scaled distance equation. Site-specific calibration required for critical underground infrastructure.")

        col_vib1, col_vib2 = st.columns([1, 2])

        with col_vib1:
            charge_ug = st.number_input("Max Charge per Delay Q (kg)", 1.0, 500.0, 45.0, step=5.0)
            dist_ug = st.number_input("Distance to Underground Infrastructure D (m)", 5.0, 500.0, 60.0, step=5.0)
            infra_type = st.selectbox(
                "Critical Infrastructure Type",
                ["Main Shaft / Winder Chamber", "Underground Crusher Station", "Haulage Shaft Lining", "Active Extraction Drive / Stope"],
                index=2,
            )

            if st.button("Predict Underground PPV 🚀", type="primary"):
                # USBM Equation: PPV = 1140 * (D / sqrt(Q)) ** (-1.6)
                scaled_distance = dist_ug / np.sqrt(charge_ug)
                calc_ppv_ug = 1140.0 * (scaled_distance ** -1.6)

                st.session_state["vib_ug_results"] = {
                    "ppv_mms": calc_ppv_ug,
                    "scaled_distance": scaled_distance,
                    "charge": charge_ug,
                    "dist": dist_ug,
                    "infra": infra_type,
                }

        with col_vib2:
            st.markdown("### 📊 Underground Vibration Results")
            res_vib = st.session_state.get("vib_ug_results")

            if res_vib:
                m_v1, m_v2 = st.columns(2)
                m_v1.metric("Predicted Underground PPV", f"{res_vib['ppv_mms']:.2f} mm/s")
                m_v2.metric("Scaled Distance", f"{res_vib['scaled_distance']:.2f} m/kg^0.5")

                if res_vib["ppv_mms"] > 15.0:
                    st.error(f"❌ **EXCEEDS VIBRATION LIMIT:** Predicted PPV ({res_vib['ppv_mms']:.2f} mm/s) exceeds maximum safe threshold (15 mm/s) for {res_vib['infra']}.")
                else:
                    st.success(f"✅ **SAFE PPV LEVEL:** Predicted PPV ({res_vib['ppv_mms']:.2f} mm/s) is within safe operational limits for {res_vib['infra']}.")

                # Sensitivity curve (PPV vs Distance)
                distances = np.linspace(10, 200, 50)
                sd_curve = distances / np.sqrt(res_vib["charge"])
                ppv_curve = 1140.0 * (sd_curve ** -1.6)

                fig_vib = go.Figure()
                fig_vib.add_trace(go.Scatter(x=distances, y=ppv_curve, mode="lines", name="PPV Attenuation", line=dict(color="#2962FF", width=3)))
                fig_vib.add_trace(go.Scatter(x=[res_vib["dist"]], y=[res_vib["ppv_mms"]], mode="markers", name="Current Design", marker=dict(size=12, color="#FF0000")))
                fig_vib.add_hline(y=15.0, line_dash="dash", line_color="red", annotation_text="15 mm/s Threshold")

                fig_vib.update_layout(
                    title="<b>Underground PPV vs Distance Curve</b>",
                    xaxis_title="Distance (m)",
                    yaxis_title="PPV (mm/s)",
                    template="plotly_white",
                    height=350,
                )
                st.plotly_chart(fig_vib, use_container_width=True)
            else:
                st.info("Click **Predict Underground PPV** to compute vibration level.")

if __name__ == "__main__":
    render_underground_page()
