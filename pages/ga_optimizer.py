import os
import streamlit as st
import pandas as pd
import numpy as np
from src.components.model_selector import render_page_model_selector
from src.optimize import BlastOptimizer
from src.report import generate_pdf
from src.visualize import plot_optimization_convergence
from src.explainability import create_explanation_panel

st.title("Genetic Algorithm Parameter Optimizer")

model, model_key = render_page_model_selector("ga_optimizer")
if model is None:
    st.stop()

st.markdown(
    "Find optimal **Burden**, **Spacing**, **Stemming**, and **Powder Factor** "
    "to minimize cost subject to vibration & flyrock safety limits."
)

col_opt1, col_opt2 = st.columns([1, 2])

with col_opt1:
    st.subheader("Optimization Constraints")
    max_ppv = st.number_input("Max Allowed PPV (mm/s)", 1.0, 50.0, 10.0, step=1.0)
    max_flyrock = st.number_input("Max Allowed Flyrock (m)", 20.0, 300.0, 120.0, step=10.0)
    d50_min, d50_max = st.slider("Target d50 Fragmentation Range (mm)", 50, 600, (120, 320))

    st.subheader("Fixed Site Conditions")
    rock_A_opt = st.number_input("Site Rock Factor (A)", 4.0, 16.0, 8.0, key="opt_A")
    bench_h_opt = st.number_input("Bench Height (m)", 5.0, 30.0, 12.0, key="opt_h")
    hole_d_opt = st.number_input("Hole Diameter (mm)", 80.0, 380.0, 250.0, key="opt_d")
    dist_opt = st.number_input("Distance to Structure (m)", 50.0, 3000.0, 400.0, key="opt_dist")

    if st.button("Run GA Optimization", type="primary", key="run_ga_opt_btn"):
        fixed_params = {
            "rock_factor_A": rock_A_opt,
            "bench_height_m": bench_h_opt,
            "hole_diameter_mm": hole_d_opt,
            "monitoring_distance_m": dist_opt,
        }

        with st.spinner("Executing Differential Evolution optimization..."):
            optimizer = BlastOptimizer(
                fixed_parameters=fixed_params,
                max_ppv_limit_mms=max_ppv,
                max_flyrock_limit_m=max_flyrock,
                target_d50_range_mm=(d50_min, d50_max),
                ml_pipeline=None,
            )
            res = optimizer.optimize(popsize=12, maxiter=30)
            st.session_state["opt_res"] = res
            st.session_state["opt_constraints"] = {
                "max_ppv": max_ppv,
                "max_flyrock": max_flyrock,
                "d50_range": (d50_min, d50_max),
            }

with col_opt2:
    if "opt_res" in st.session_state:
        res = st.session_state["opt_res"]
        st.success("Optimization Completed!")

        st.subheader("Top Recommended Blast Design (#1 Best)")
        opt_p = res["optimized_parameters"]
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        col_res1.metric("Burden (m)", f"{opt_p['burden_m']:.2f}")
        col_res2.metric("Spacing (m)", f"{opt_p['spacing_m']:.2f}")
        col_res3.metric("Stemming (m)", f"{opt_p['stemming_m']:.2f}")
        col_res4.metric("Powder Factor", f"{opt_p['powder_factor_kg_m3']:.3f} kg/m3")

        st.subheader("Predicted Outcomes for Best Design")
        out_p = res["predicted_outputs"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted d50", f"{out_p['d50_mm']:.1f} mm")
        c2.metric("Predicted PPV", f"{out_p['ppv_mms']:.2f} mm/s")
        c3.metric("Predicted Flyrock", f"{out_p['flyrock_m']:.1f} m")
        c4.metric("Optimized Cost", f"${out_p['cost_per_tonne_usd']:.2f} / t")

        st.markdown("---")
        st.subheader("Top 5 Recommended Blast Designs & SHAP Explainability")
        top_5 = res.get("top_5_designs", [])

        table_rows = []
        for rank, item in enumerate(top_5, 1):
            p = item["parameters"]
            o = item["outputs"]
            table_rows.append({
                "Rank": f"#{rank}",
                "Burden (m)": p["burden_m"],
                "Spacing (m)": p["spacing_m"],
                "Stemming (m)": p["stemming_m"],
                "Powder Factor (kg/m3)": p["powder_factor_kg_m3"],
                "d50 (mm)": o["d50_mm"],
                "PPV (mm/s)": o["ppv_mms"],
                "Flyrock (m)": o["flyrock_m"],
                "Cost ($/t)": o["cost_per_tonne_usd"],
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        st.subheader("🔍 SHAP Explanation per Pareto Design")
        for rank, item in enumerate(top_5, 1):
            p = item["parameters"]
            o = item["outputs"]
            with st.expander(f"🔎 Explain Design #{rank} (Cost: ${o['cost_per_tonne_usd']:.2f}/t, PPV: {o['ppv_mms']:.2f} mm/s)"):
                design_payload = {
                    "rock_factor_A": float(rock_A_opt),
                    "bench_height_m": float(bench_h_opt),
                    "hole_diameter_mm": float(hole_d_opt),
                    "burden_m": float(p["burden_m"]),
                    "spacing_m": float(p["spacing_m"]),
                    "stemming_m": float(p["stemming_m"]),
                    "powder_factor_kg_m3": float(p["powder_factor_kg_m3"]),
                    "charge_mass_per_hole_kg": 320.0,
                    "max_charge_per_delay_kg": 640.0,
                    "monitoring_distance_m": float(dist_opt),
                    "explosive_rws": 100.0,
                }

                exp_outcome = st.selectbox(
                    f"Target Outcome to Explain (Design #{rank})",
                    ["d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"],
                    key=f"opt_exp_target_{rank}",
                )

                constraints_info = {
                    "metric": exp_outcome,
                    "limit": max_ppv if exp_outcome == "ppv_mms" else (max_flyrock if exp_outcome == "flyrock_m" else d50_max),
                    "d50_min_mm": d50_min,
                    "d50_max_mm": d50_max,
                    "ppv_max_mm_s": max_ppv,
                    "unit": "mm/s" if exp_outcome == "ppv_mms" else ("m" if exp_outcome == "flyrock_m" else "mm"),
                }

                panel = create_explanation_panel(
                    model=model,
                    input_data=pd.DataFrame([design_payload]),
                    prediction=o.get(exp_outcome, 10.0),
                    constraints=constraints_info,
                )

                st.info(f"**Natural Language Explanation:** {panel.get('natural_language', '')}")

                waterfall_fig = panel.get("shap", {}).get("waterfall_plot")
                if waterfall_fig is not None:
                    st.plotly_chart(waterfall_fig, use_container_width=True, key=f"opt_waterfall_{rank}")

        os.makedirs("data/processed", exist_ok=True)
        pdf_path = generate_pdf(
            designs=top_5,
            filename="data/processed/blast_optimization_report.pdf",
            constraints_info=st.session_state.get("opt_constraints", {}),
        )

        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            st.download_button(
                label="📄 Download Optimization PDF Report",
                data=pdf_bytes,
                file_name="BlastOpt_Botswana_Optimization_Report.pdf",
                mime="application/pdf",
                type="primary",
                key="download_pdf_btn",
            )

        fig_conv = plot_optimization_convergence(res["convergence_history"])
        st.plotly_chart(fig_conv, use_container_width=True)
    else:
        st.info("Click 'Run GA Optimization' to find the optimal blast geometry and generate report.")
