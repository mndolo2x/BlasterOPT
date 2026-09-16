"""
Streamlit Web Application for BlastOpt Botswana.
Main Dashboard and Interactive Mining Analytics Suite.
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

from src.synthetic_data import generate_synthetic_blast_data
from src.data_ingestion import prepare_ingested_dataset, load_real_blast_data, clean_and_preprocess, engineer_features
from src.models import BlastMLPipeline, MODEL_REGISTRY
from src.predict import predict_single_blast, total_cost_per_tonne
from src.recommender import find_similar_blasts
from src.mwd_ingestion import parse_mwd_message, MWD_HISTORY
import plotly.express as px
import plotly.graph_objects as go
from src.optimize import BlastOptimizer
from src.report import generate_pdf
from src.visualize import (
    plot_kuz_ram_curve,
    plot_ppv_attenuation,
    plot_feature_importance,
    plot_optimization_convergence,
    plot_2d_blast_pattern,
)
from src.explainability import (
    get_feature_contributions,
    plot_feature_contributions_waterfall,
    create_explanation_panel,
)

# Page configuration
st.set_page_config(
    page_title="BlastOpt Botswana | AI Blast Optimization",
    page_icon="💥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "dataset" not in st.session_state:
    # Default initial dataset
    synth_df = generate_synthetic_blast_data(num_samples=300, seed=42)
    st.session_state["dataset"] = engineer_features(synth_df)

if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = None

if "last_predict_inputs" not in st.session_state:
    st.session_state["last_predict_inputs"] = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "charge_mass_per_hole_kg": 320.0,
        "max_charge_per_delay_kg": 640.0,
        "monitoring_distance_m": 450.0,
        "explosive_rws": 100.0,
    }

if "last_predict_results" not in st.session_state:
    st.session_state["last_predict_results"] = predict_single_blast(
        st.session_state["last_predict_inputs"]
    )


# Title and Header Banner
st.title("🇧🇼 BlastOpt Botswana")
st.markdown(
    "**AI-Driven Drilling & Blasting Design, Fragmentation Modeling & Genetic Algorithm Optimizer**"
)

# Navigation Sidebar
st.sidebar.image("https://img.icons8.com/color/96/diamond.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "📊 Dashboard & Data Explorer",
        "⚙️ Data Ingestion & Generator",
        "🤖 ML Model Manager",
        "🔬 Model Comparison",
        "🎯 Predictor & Kuz-Ram Curve",
        "⚡ Genetic Algorithm Optimizer",
        "💰 Economic Dashboard",
        "👥 Similar Blasts Recommender",
        "📡 Real-Time MWD Monitoring",
        "📐 2D Blast Pattern & Delays",
        "📈 Visualize",
    ],
)

# --- MODULE 1: DASHBOARD & DATA EXPLORER ---
if page == "📊 Dashboard & Data Explorer":
    st.header("📊 Mining & Blasting Data Dashboard")

    df = st.session_state["dataset"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Blast Records", len(df))
    with col2:
        st.metric("Avg d50 Fragmentation", f"{df['d50_mm'].mean():.1f} mm")
    with col3:
        st.metric("Avg Ground Vibration (PPV)", f"{df['ppv_mms'].mean():.2f} mm/s")
    with col4:
        st.metric("Avg D&B Cost", f"${df['cost_per_tonne_usd'].mean():.2f} / t")

    st.markdown("---")
    st.subheader("Historical Blast Logs")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T, use_container_width=True)


# --- MODULE 2: DATA INGESTION & GENERATOR ---
elif page == "⚙️ Data Ingestion & Generator":
    st.header("⚙️ Data Ingestion & Synthetic Generator")

    tab1, tab2 = st.tabs(["⚡ Generate Synthetic Blast Logs", "📁 Upload Custom Blast CSV"])

    with tab1:
        st.subheader("Physics-Guided Synthetic Blast Data Generator")
        col_gen1, col_gen2 = st.columns(2)

        with col_gen1:
            num_samples = st.slider("Number of Blast Logs", 50, 2000, 400, step=50)
            seed_val = st.number_input("Random Seed", value=42)

        with col_gen2:
            rock_factor_min, rock_factor_max = st.slider(
                "Rock Blastability Factor (A) Range", 4.0, 16.0, (6.0, 12.0)
            )

        if st.button("Generate Synthetic Dataset", type="primary"):
            new_df = generate_synthetic_blast_data(
                num_samples=num_samples,
                seed=int(seed_val),
                rock_factor_range=(rock_factor_min, rock_factor_max),
            )
            processed_df = engineer_features(clean_and_preprocess(new_df))
            st.session_state["dataset"] = processed_df
            st.success(f"Generated and loaded {len(processed_df)} synthetic blast records!")
            st.dataframe(processed_df.head(10), use_container_width=True)

    with tab2:
        st.subheader("Upload Real Mine Production Data vs Synthetic Data")
        st.markdown(
            "Upload production blast logs (e.g., from Debswana Jwaneng/Orapa pits when under a data-sharing agreement) "
            "to calibrate models on site-specific geology, structural discontinuities, and actual measured fragmentation."
        )

        data_mode = st.radio(
            "Select Data Pathway for Model Training",
            ["Real Mine Data (Uploaded CSV)", "Synthetic Physics Data Generator"],
            index=0,
            help="Keep real mine data and synthetic empirical data pathways separate."
        )

        if data_mode == "Real Mine Data (Uploaded CSV)":
            uploaded_file = st.file_uploader("Upload Real Mine CSV File", type=["csv"])

            if uploaded_file is not None:
                try:
                    temp_path = "data/raw/uploaded_real_blast_data.csv"
                    os.makedirs("data/raw", exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.write("Uploaded CSV Raw Preview:")
                    st.dataframe(pd.read_csv(temp_path).head(5), use_container_width=True)

                    if st.button("Process & Load Real Mine Data", type="primary"):
                        real_processed_df = load_real_blast_data(
                            filepath=temp_path,
                            anomaly_log_path="data/processed/data_anomalies.log"
                        )
                        st.session_state["dataset"] = real_processed_df
                        st.session_state["data_source_mode"] = "real"
                        st.success("Successfully validated, cleaned, and loaded real mine production dataset!")
                        st.info("💡 Anomaly report generated at `data/processed/data_anomalies.log`.")
                        st.dataframe(real_processed_df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing real mine dataset: {e}")
        else:
            st.info("Active Pathway: Synthetic Physics Data Generator (Tab 1). Use Tab 1 controls to configure synthetic dataset parameters.")


# --- MODULE 3: ML MODEL MANAGER ---
elif page == "🤖 ML Model Manager":
    st.header("🤖 Machine Learning Model Training & Evaluation")

    df = st.session_state["dataset"]

    col_m1, col_m2 = st.columns([1, 2])

    with col_m1:
        model_type = st.selectbox(
            "Select Algorithm", ["random_forest", "xgboost", "ridge"]
        )
        cv_folds = st.slider("Cross Validation Folds", 3, 10, 5)

        if st.button("Train Models", type="primary"):
            with st.spinner("Training models across all target metrics..."):
                pipeline = BlastMLPipeline(model_type=model_type, seed=42)
                metrics = pipeline.train_and_evaluate(df, cv_folds=cv_folds)
                pipeline.save_models()
                st.session_state["pipeline"] = pipeline
                st.success(f"Successfully trained {model_type.upper()} models!")

    with col_m2:
        if st.session_state["pipeline"] is not None:
            pipeline = st.session_state["pipeline"]
            st.subheader("Model Evaluation Metrics (Cross Validation)")

            metrics_df = pd.DataFrame(pipeline.metrics).T
            st.dataframe(metrics_df.style.highlight_max(axis=0, color="#C8E6C9"), use_container_width=True)

            # Feature Importances
            st.subheader("Feature Importances")
            importances = pipeline.get_feature_importances()
            target_to_plot = st.selectbox(
                "Select Target for Importance Plot",
                ["d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"],
            )
            fig_imp = plot_feature_importance(importances, target=target_to_plot)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Train a model using the options on the left to view evaluation metrics.")


# --- MODULE: MODEL COMPARISON ---
elif page == "🔬 Model Comparison":
    st.header("🔬 Debswana Research Models Comparison")
    st.markdown(
        "Benchmarking production-validated research models trained on **Jwaneng** and **Orapa** mine datasets."
    )

    # Performance Table
    rows = []
    for model_key, meta in MODEL_REGISTRY.items():
        perf = meta.get("performance", {})
        r2_frag = perf.get("fragmentation_r2", perf.get("fragmentation", perf.get("fragmentation_optimal_pct", "N/A")))
        r2_vib = perf.get("vibration_r2", perf.get("vibration", "N/A"))
        r2_air = perf.get("airblast_r2", perf.get("airblast", perf.get("min_airblast_db", "N/A")))
        rows.append({
            "Model Key": model_key,
            "Architecture": meta.get("architecture", "N/A"),
            "Optimizer": meta.get("optimizer", "N/A"),
            "Source Data": meta.get("source", "N/A"),
            "Fragmentation Performance (R²)": f"{r2_frag:.3f}" if isinstance(r2_frag, (float, int)) else str(r2_frag),
            "Vibration Performance (R²)": f"{r2_vib:.3f}" if isinstance(r2_vib, (float, int)) else str(r2_vib),
            "Airblast Performance (R²)": f"{r2_air:.3f}" if isinstance(r2_air, (float, int)) else str(r2_air),
            "Reference Citation": meta.get("reference", "N/A"),
        })

    df_registry = pd.DataFrame(rows)
    st.subheader("Model Performance Summary Table")
    st.dataframe(df_registry, use_container_width=True)

    st.markdown("---")
    st.subheader("Select Model for Sensitivity & Key Drivers Analysis")
    selected_key = st.selectbox("Select Model", list(MODEL_REGISTRY.keys()))

    if selected_key:
        model_info = MODEL_REGISTRY[selected_key]
        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Model Metadata & Source")
            st.info(f"**Architecture:** `{model_info.get('architecture')}`")
            st.info(f"**Outputs:** `{', '.join(model_info.get('outputs', []))}`")
            st.info(f"**Data Source:** `{model_info.get('source')}`")
            st.success(f"**Literature Citation:** {model_info.get('reference')}")

        with c2:
            st.subheader("Key Drivers & Sensitivity Analysis")
            if "key_drivers" in model_info:
                st.write("**Key Driving Features:**")
                st.json(model_info["key_drivers"])
            elif "key_sensitivity" in model_info:
                st.write("**Key Parameter Sensitivity:**")
                st.json(model_info["key_sensitivity"])

            if "inverse_design" in model_info:
                st.write("**Inverse Design Targets:**")
                st.json(model_info["inverse_design"])


# --- MODULE 4: PREDICTOR & KUZ-RAM CURVE ---
elif page == "🎯 Predictor & Kuz-Ram Curve":
    st.header("🎯 Single Blast Design Predictor & Fragmentation Curve")

    col_p1, col_p2 = st.columns([1, 2])

    with col_p1:
        st.subheader("Input Blast Parameters")

        last_in = st.session_state["last_predict_inputs"]
        rock_A = st.number_input("Rock Factor (A)", 4.0, 16.0, float(last_in["rock_factor_A"]), step=0.5)
        bench_h = st.number_input("Bench Height (m)", 5.0, 30.0, float(last_in["bench_height_m"]), step=0.5)
        hole_d = st.number_input("Hole Diameter (mm)", 80.0, 380.0, float(last_in["hole_diameter_mm"]), step=10.0)
        burden = st.number_input("Burden (m)", 2.0, 12.0, float(last_in["burden_m"]), step=0.2)
        spacing = st.number_input("Spacing (m)", 2.0, 15.0, float(last_in["spacing_m"]), step=0.2)
        stemming = st.number_input("Stemming (m)", 1.0, 10.0, float(last_in["stemming_m"]), step=0.2)
        pf = st.number_input("Powder Factor (kg/m3)", 0.2, 2.5, float(last_in["powder_factor_kg_m3"]), step=0.05)
        charge_per_hole = st.number_input("Charge Mass per Hole (kg)", 10.0, 1500.0, float(last_in["charge_mass_per_hole_kg"]), step=10.0)
        max_charge_delay = st.number_input("Max Charge per Delay (kg)", 10.0, 3000.0, float(last_in["max_charge_per_delay_kg"]), step=20.0)
        dist = st.number_input("Distance to Structure (m)", 50.0, 3000.0, float(last_in["monitoring_distance_m"]), step=25.0)

        input_payload = {
            "rock_factor_A": rock_A,
            "bench_height_m": bench_h,
            "hole_diameter_mm": hole_d,
            "burden_m": burden,
            "spacing_m": spacing,
            "stemming_m": stemming,
            "powder_factor_kg_m3": pf,
            "charge_mass_per_hole_kg": charge_per_hole,
            "max_charge_per_delay_kg": max_charge_delay,
            "monitoring_distance_m": dist,
            "explosive_rws": 100.0,
        }

        st.session_state["last_predict_inputs"] = input_payload

    with col_p2:
        st.subheader("Predicted Blast Outcomes")

        pipeline = st.session_state.get("pipeline", None)
        predictions = predict_single_blast(input_payload, model_pipeline=pipeline)
        st.session_state["last_predict_results"] = predictions

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("d50 Fragment Size", f"{predictions['d50_mm']:.1f} mm")
        m2.metric("Ground PPV", f"{predictions['ppv_mms']:.2f} mm/s")
        m3.metric("Flyrock Distance", f"{predictions['flyrock_m']:.1f} m")
        m4.metric("D&B Cost", f"${predictions['cost_per_tonne_usd']:.2f} / t")

        st.markdown("---")
        with st.expander("🔍 Why this prediction?", expanded=True):
            target_explain = st.selectbox(
                "Select Outcome to Explain",
                ["d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"],
                key="explain_target_select",
            )

            # Generate full explanation panel (SHAP, LIME, Natural Language, Top 5)
            target_model = None
            if pipeline is not None and target_explain in pipeline.models:
                target_model = pipeline.models[target_explain]

            target_units = {
                "d50_mm": "mm",
                "ppv_mms": "mm/s",
                "flyrock_m": "m",
                "cost_per_tonne_usd": "$/t",
            }
            constraints_info = {
                "metric": target_explain,
                "limit": 10.0 if target_explain == "ppv_mms" else (250.0 if target_explain == "d50_mm" else 150.0),
                "unit": target_units.get(target_explain, ""),
            }

            exp_panel = create_explanation_panel(
                model=target_model if target_model is not None else None,
                input_data=pd.DataFrame([input_payload]),
                prediction=predictions.get(target_explain, 10.0),
                constraints=constraints_info,
            )

            st.subheader("💬 Natural Language Summary")
            st.info(exp_panel.get("natural_language", "Explanation summary generated."))

            st.subheader("🔝 Top 5 Feature Contributions")
            shap_dict = exp_panel.get("shap", {}).get("feature_contributions", {})
            sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

            top5_rows = []
            for fn, sv in sorted_feats:
                direction = "Pushes Higher ⬆️" if sv >= 0 else "Pushes Lower ⬇️"
                top5_rows.append({
                    "Feature Name": fn,
                    "SHAP Impact Value": f"{sv:+.2f}",
                    "Direction": direction,
                })
            st.dataframe(pd.DataFrame(top5_rows), use_container_width=True)

            st.subheader("📊 Interactive SHAP Force Plot")
            force_fig = exp_panel.get("shap", {}).get("force_plot")
            if force_fig is not None:
                st.plotly_chart(force_fig, use_container_width=True)

            st.subheader("📉 Interactive SHAP Waterfall Plot")
            waterfall_fig = exp_panel.get("shap", {}).get("waterfall_plot")
            if waterfall_fig is not None:
                st.plotly_chart(waterfall_fig, use_container_width=True)

            st.subheader("🍋 LIME Local Explanation Weights")
            lime_weights = exp_panel.get("lime", {}).get("feature_weights", {})
            st.json(lime_weights)

        st.markdown("---")
        fig_kuz = plot_kuz_ram_curve(predictions["d50_mm"], n_uniformity=1.2)
        st.plotly_chart(fig_kuz, use_container_width=True)

        fig_ppv = plot_ppv_attenuation(max_charge_delay)
        st.plotly_chart(fig_ppv, use_container_width=True)


# --- MODULE 5: GENETIC ALGORITHM OPTIMIZER ---
elif page == "⚡ Genetic Algorithm Optimizer":
    st.header("⚡ Genetic Algorithm Parameter Optimizer")

    st.markdown("Find optimal **Burden**, **Spacing**, **Stemming**, and **Powder Factor** to minimize cost subject to vibration & flyrock safety limits.")

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

        if st.button("Run GA Optimization", type="primary"):
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
                    ml_pipeline=st.session_state.get("pipeline", None),
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

            # Per-design SHAP Explainability
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

                    pipeline = st.session_state.get("pipeline", None)
                    target_model = None
                    if pipeline is not None and exp_outcome in pipeline.models:
                        target_model = pipeline.models[exp_outcome]

                    constraints_info = {
                        "metric": exp_outcome,
                        "limit": max_ppv if exp_outcome == "ppv_mms" else (max_flyrock if exp_outcome == "flyrock_m" else 250.0),
                        "unit": "mm/s" if exp_outcome == "ppv_mms" else ("m" if exp_outcome == "flyrock_m" else "mm"),
                    }

                    panel = create_explanation_panel(
                        model=target_model,
                        input_data=pd.DataFrame([design_payload]),
                        prediction=o.get(exp_outcome, 10.0),
                        constraints=constraints_info,
                    )

                    st.info(f"**Natural Language Explanation:** {panel.get('natural_language', '')}")

                    # SHAP Waterfall Chart
                    waterfall_fig = panel.get("shap", {}).get("waterfall_plot")
                    if waterfall_fig is not None:
                        st.plotly_chart(waterfall_fig, use_container_width=True, key=f"opt_waterfall_{rank}")

            # Generate PDF Report Download Button
            pdf_path = generate_pdf(
                designs=top_5,
                filename="data/processed/blast_optimization_report.pdf",
                constraints_info=st.session_state.get("opt_constraints", {}),
            )

            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            st.download_button(
                label="📄 Download Optimization PDF Report",
                data=pdf_bytes,
                file_name="BlastOpt_Botswana_Optimization_Report.pdf",
                mime="application/pdf",
                type="primary",
            )

            fig_conv = plot_optimization_convergence(res["convergence_history"])
            st.plotly_chart(fig_conv, use_container_width=True)
        else:
            st.info("Click 'Run GA Optimization' to find the optimal blast geometry and generate report.")


# --- MODULE: ECONOMIC DASHBOARD ---
elif page == "💰 Economic Dashboard":
    st.header("💰 Economic & Mine-to-Mill Cost Breakdown Dashboard")
    st.markdown(
        "Real-time Mine-to-Mill total cost analysis per tonne ($/t) across drilling, explosives, loading/digging, hauling, crushing, and milling."
    )

    c_econ1, c_econ2 = st.columns([1, 2])

    with c_econ1:
        st.subheader("⚙️ Configurable Unit Cost Parameters")

        drilling_rate = st.slider(
            "Drilling Rate ($/m)", 5.0, 30.0, 12.0, step=0.5,
            help="Base drilling rate per linear meter."
        )
        exp_price = st.slider(
            "Explosive Price ($/kg)", 0.5, 4.0, 1.5, step=0.1,
            help="Bulk explosive product unit price per kg."
        )
        dig_base = st.slider(
            "Shovel Digging Base ($/t)", 0.1, 2.0, 0.40, step=0.05,
            help="Excavator/shovel loading cost base rate per tonne."
        )
        haul_base = st.slider(
            "Haulage Base ($/t)", 0.2, 3.0, 0.80, step=0.05,
            help="Haul truck transport base rate per tonne."
        )
        crush_base = st.slider(
            "Primary Crushing Base ($/t)", 0.1, 2.0, 0.30, step=0.05,
            help="Primary gyratory/jaw crushing energy & liner wear rate."
        )
        mill_base = st.slider(
            "Milling Grinding Base ($/t)", 0.5, 10.0, 2.50, step=0.25,
            help="SAG/ball mill specific energy consumption base rate."
        )

        st.subheader("Blast Design Parameters")
        last_in = st.session_state.get("last_predict_inputs", {})
        last_out = st.session_state.get("last_predict_results", {})

        pf_econ = st.number_input("Powder Factor (kg/m3)", 0.2, 2.5, float(last_in.get("powder_factor_kg_m3", 0.65)), step=0.05)
        d50_econ = st.number_input("Mean Fragment Size d50 (mm)", 20.0, 1000.0, float(last_out.get("d50_mm", 220.0)), step=10.0)

        blast_params_econ = {
            "powder_factor_kg_m3": pf_econ,
            "bench_height_m": float(last_in.get("bench_height_m", 12.0)),
            "hole_diameter_mm": float(last_in.get("hole_diameter_mm", 250.0)),
            "burden_m": float(last_in.get("burden_m", 6.0)),
            "spacing_m": float(last_in.get("spacing_m", 7.0)),
            "d50_mm": d50_econ,
        }

        unit_costs_config = {
            "drilling_rate_usd_m": drilling_rate,
            "explosive_price_usd_kg": exp_price,
            "digging_base_usd_t": dig_base,
            "hauling_base_usd_t": haul_base,
            "crushing_base_usd_t": crush_base,
            "milling_base_usd_t": mill_base,
        }

    with c_econ2:
        st.subheader("📊 Real-Time Mine-to-Mill Cost Breakdown")

        # Compute cost breakdown in real time
        cost_breakdown = total_cost_per_tonne(blast_params_econ, unit_costs=unit_costs_config)

        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Total Mine-to-Mill Cost", f"${cost_breakdown['total_cost_usd_t']:.2f} / t")
        m_c2.metric("Drill & Blast Share", f"${cost_breakdown['drilling_cost_usd_t'] + cost_breakdown['explosive_cost_usd_t']:.2f} / t")
        m_c3.metric("Load & Haul Share", f"${cost_breakdown['digging_cost_usd_t'] + cost_breakdown['hauling_cost_usd_t']:.2f} / t")
        m_c4.metric("Comminution Share", f"${cost_breakdown['crushing_cost_usd_t'] + cost_breakdown['milling_cost_usd_t']:.2f} / t")

        st.markdown("---")

        # Donut Chart Breakdown
        df_costs = pd.DataFrame([
            {"Stage": "Drilling", "Cost ($/t)": cost_breakdown["drilling_cost_usd_t"]},
            {"Stage": "Explosives", "Cost ($/t)": cost_breakdown["explosive_cost_usd_t"]},
            {"Stage": "Loading (Digging)", "Cost ($/t)": cost_breakdown["digging_cost_usd_t"]},
            {"Stage": "Hauling", "Cost ($/t)": cost_breakdown["hauling_cost_usd_t"]},
            {"Stage": "Crushing", "Cost ($/t)": cost_breakdown["crushing_cost_usd_t"]},
            {"Stage": "Milling (Grinding)", "Cost ($/t)": cost_breakdown["milling_cost_usd_t"]},
        ])

        fig_pie = px.pie(
            df_costs,
            values="Cost ($/t)",
            names="Stage",
            title="<b>Mine-to-Mill Cost Distribution Breakdown ($/tonne)</b>",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_traces(textinfo="label+percent+value")
        fig_pie.update_layout(template="plotly_white", height=420)

        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Detailed Cost Components Table")
        st.dataframe(df_costs, use_container_width=True)


# --- MODULE: SIMILAR BLASTS RECOMMENDER ---
elif page == "👥 Similar Blasts Recommender":
    st.header("👥 Similar Blast Recommender & Knowledge Transfer")
    st.markdown(
        "Empowers junior blasters and newly rotated mining engineers to query historical blast logs, "
        "learn from past blast outcomes ($d_{50}$, PPV, flyrock, cost), and review historical lessons learned."
    )

    c_rec1, c_rec2 = st.columns([1, 2])

    with c_rec1:
        st.subheader("Current Blast Parameters")
        last_in = st.session_state.get("last_predict_inputs", {})

        rock_A_rec = st.number_input("Rock Factor (A)", 4.0, 16.0, float(last_in.get("rock_factor_A", 8.0)), step=0.5, key="rec_rock")
        bench_h_rec = st.number_input("Bench Height (m)", 5.0, 30.0, float(last_in.get("bench_height_m", 12.0)), step=0.5, key="rec_h")
        hole_d_rec = st.number_input("Hole Diameter (mm)", 80.0, 380.0, float(last_in.get("hole_diameter_mm", 250.0)), step=10.0, key="rec_d")
        burden_rec = st.number_input("Burden (m)", 2.0, 12.0, float(last_in.get("burden_m", 6.0)), step=0.2, key="rec_b")
        spacing_rec = st.number_input("Spacing (m)", 2.0, 15.0, float(last_in.get("spacing_m", 7.0)), step=0.2, key="rec_s")
        stemming_rec = st.number_input("Stemming (m)", 1.0, 10.0, float(last_in.get("stemming_m", 5.0)), step=0.2, key="rec_stem")
        pf_rec = st.number_input("Powder Factor (kg/m3)", 0.2, 2.5, float(last_in.get("powder_factor_kg_m3", 0.65)), step=0.05, key="rec_pf")
        charge_rec = st.number_input("Charge Mass per Hole (kg)", 10.0, 1500.0, float(last_in.get("charge_mass_per_hole_kg", 320.0)), step=10.0, key="rec_charge")
        dist_rec = st.number_input("Monitoring Distance (m)", 50.0, 3000.0, float(last_in.get("monitoring_distance_m", 450.0)), step=25.0, key="rec_dist")

        query_payload = {
            "rock_factor_A": rock_A_rec,
            "bench_height_m": bench_h_rec,
            "hole_diameter_mm": hole_d_rec,
            "burden_m": burden_rec,
            "spacing_m": spacing_rec,
            "stemming_m": stemming_rec,
            "powder_factor_kg_m3": pf_rec,
            "charge_mass_per_hole_kg": charge_rec,
            "max_charge_per_delay_kg": charge_rec * 2.0,
            "monitoring_distance_m": dist_rec,
            "explosive_rws": 100.0,
        }

        top_k_val = st.slider("Number of Similar Blasts to Retrieve", 3, 10, 5)

    with c_rec2:
        st.subheader("Top Matching Historical Blasts")
        hist_df = st.session_state["dataset"]

        similar_df = find_similar_blasts(query_payload, hist_df, top_k=top_k_val)

        if not similar_df.empty:
            display_cols = [
                c for c in ["similarity_distance", "burden_m", "spacing_m", "stemming_m", "powder_factor_kg_m3", "d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"]
                if c in similar_df.columns
            ]
            st.dataframe(similar_df[display_cols], use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Outcomes Comparison: Current Design vs Historical Blasts")

            # Predicted outcomes for current proposed design
            curr_pred = predict_single_blast(query_payload, model_pipeline=st.session_state.get("pipeline", None))

            metrics_comp = ["d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"]
            labels_comp = ["d50 (mm)", "PPV (mm/s)", "Flyrock (m)", "Cost ($/t)"]

            curr_vals = [curr_pred.get(m, 0.0) for m in metrics_comp]
            hist_avg_vals = [similar_df[m].mean() if m in similar_df.columns else 0.0 for m in metrics_comp]

            fig_bar = go.Figure(data=[
                go.Bar(name="Current Proposed Design", x=labels_comp, y=curr_vals, marker_color="#2962FF"),
                go.Bar(name="Top Similar Blasts Avg", x=labels_comp, y=hist_avg_vals, marker_color="#00C853"),
            ])
            fig_bar.update_layout(
                barmode="group",
                title="<b>Outcome Metric Comparison</b>",
                template="plotly_white",
                height=380,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")
            st.subheader("💡 Lessons Learned & Historical Observations")
            for idx, (_, row) in enumerate(similar_df.iterrows(), 1):
                dist_val = row.get("similarity_distance", 0.0)
                d50_val = row.get("d50_mm", 200.0)
                ppv_val = row.get("ppv_mms", 5.0)

                # Simulated domain lessons learned based on historical outcome physics
                if ppv_val > 10.0:
                    lesson = "High ground vibration observed. Recommend increasing electronic delay intervals or reducing maximum charge per delay."
                elif d50_val > 300.0:
                    lesson = "Coarse fragmentation produced. Increasing powder factor or reducing burden/spacing ratio improved digging rates."
                else:
                    lesson = "Optimal blast performance recorded. Good muckpile displacement and balanced fragmentation achieved."

                with st.expander(f"Blast Log #{idx} (Similarity Distance: {dist_val:.2f}) - d50: {d50_val:.1f} mm | PPV: {ppv_val:.2f} mm/s"):
                    st.write(f"**Parameters:** Burden: `{row.get('burden_m', 6.0):.2f}m`, Spacing: `{row.get('spacing_m', 7.0):.2f}m`, Stemming: `{row.get('stemming_m', 5.0):.2f}m`, PF: `{row.get('powder_factor_kg_m3', 0.65):.3f} kg/m³`")
                    st.info(f"**Historical Lesson Learned:** {lesson}")


# --- MODULE: REAL-TIME MWD MONITORING ---
elif page == "📡 Real-Time MWD Monitoring":
    st.header("📡 Real-Time Measure-While-Drilling (MWD) Telemetry & Closed-Loop Control")
    st.markdown(
        "Live MQTT stream ingestion of drill rig telemetry (penetration rate, torque, weight-on-bit, vibration). "
        "Closed-loop analytics automatically compare **as-drilled** vs **as-designed** parameters and adapt charging plans."
    )

    col_mwd1, col_mwd2 = st.columns([1, 2])

    with col_mwd1:
        st.subheader("Simulate Live MWD Telemetry Stream")
        mwd_hole_id = st.text_input("Drill Hole ID", value="HOLE_JWA_204")
        mwd_depth = st.number_input("Measured Depth (m)", 1.0, 30.0, 15.0, step=0.5)
        mwd_rop = st.number_input("Rate of Penetration (m/hr)", 5.0, 120.0, 38.0, step=2.0)
        mwd_torque = st.number_input("Drill Torque (N·m)", 200.0, 5000.0, 1350.0, step=50.0)
        mwd_wob = st.number_input("Weight on Bit (kg)", 1000.0, 25000.0, 9200.0, step=200.0)
        mwd_rock = st.selectbox("In-Situ Rock Strata", ["Kimberlite_Hard", "Granite_Competent", "Sandstone_Soft", "Void_Fractured"])

        if st.button("Transmit MWD Telemetry Sample", type="primary"):
            sample_payload = {
                "hole_id": mwd_hole_id,
                "depth_m": mwd_depth,
                "penetration_rate_m_hr": mwd_rop,
                "torque_nm": mwd_torque,
                "weight_on_bit_kg": mwd_wob,
                "rock_type": mwd_rock,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
            parsed_sample = parse_mwd_message(sample_payload)
            MWD_HISTORY.append(parsed_sample)
            st.success(f"Simulated MWD sample transmitted for {mwd_hole_id}!")

    with col_mwd2:
        st.subheader("📊 Live MWD Telemetry Stream & Specific Energy")

        # Mock telemetry feed if history is empty
        if not MWD_HISTORY:
            default_samples = [
                parse_mwd_message({"hole_id": f"HOLE_{i:03d}", "depth_m": 12.0 + i*0.5, "penetration_rate_m_hr": 35.0 + i*2, "torque_nm": 1200.0 + i*50, "weight_on_bit_kg": 8500.0})
                for i in range(1, 6)
            ]
            df_mwd = pd.DataFrame(default_samples)
        else:
            df_mwd = pd.DataFrame(MWD_HISTORY[-15:])

        st.dataframe(df_mwd, use_container_width=True)

        st.markdown("---")
        st.subheader("📐 As-Drilled vs As-Designed Geometry Comparison")

        # As-designed targets vs As-drilled MWD measured averages
        designed_depth = 15.0
        designed_burden = 6.0
        measured_depth = float(df_mwd["depth_m"].iloc[-1]) if not df_mwd.empty else 15.0

        fig_mwd = go.Figure(data=[
            go.Bar(name="As-Designed Target", x=["Depth (m)", "Burden (m)"], y=[designed_depth, designed_burden], marker_color="#2962FF"),
            go.Bar(name="As-Drilled Measured (MWD)", x=["Depth (m)", "Burden (m)"], y=[measured_depth, designed_burden * (1 + (measured_depth - designed_depth)*0.05)], marker_color="#FF6D00"),
        ])
        fig_mwd.update_layout(barmode="group", title="<b>Geometry Compliance Check</b>", template="plotly_white", height=350)
        st.plotly_chart(fig_mwd, use_container_width=True)

        st.markdown("---")
        st.subheader("🚨 Real-Time Closed-Loop Adaptive Charging Alerts")

        depth_diff = measured_depth - designed_depth
        if abs(depth_diff) > 1.0:
            st.error(f"⚠️ **GEOMETRY DEVIATION ALERT:** Hole depth deviates by {depth_diff:+.2f} m from design target ({designed_depth:.1f} m).")
            st.warning("⚡ **ADAPTIVE CHARGING PLAN:** Automatically adjusting sub-drilling stemming length and bulk explosive density to prevent flyrock and toe accumulation.")
        else:
            st.success("✅ **GEOMETRY COMPLIANT:** As-drilled hole dimensions are within ±1.0 m tolerance bounds of design specifications.")


# --- MODULE 6: 2D BLAST PATTERN & DELAYS ---
elif page == "📐 2D Blast Pattern & Delays":
    st.header("📐 2D Blast Pattern & Initiation Timing Layout")

    col_pat1, col_pat2 = st.columns([1, 3])

    with col_pat1:
        st.subheader("Grid Parameters")
        num_rows = st.slider("Number of Rows", 2, 10, 4)
        holes_per_row = st.slider("Holes per Row", 4, 20, 8)
        b_pat = st.number_input("Burden (m)", 2.0, 10.0, 6.0, key="pat_b")
        s_pat = st.number_input("Spacing (m)", 2.0, 12.0, 7.0, key="pat_s")

        st.subheader("Timing Sequence")
        row_delay = st.number_input("Inter-Row Delay (ms)", 0, 100, 42)
        hole_delay = st.number_input("Inter-Hole Delay (ms)", 0, 50, 17)

    with col_pat2:
        fig_pattern = plot_2d_blast_pattern(
            num_rows=num_rows,
            holes_per_row=holes_per_row,
            burden_m=b_pat,
            spacing_m=s_pat,
            row_delay_ms=row_delay,
            hole_delay_ms=hole_delay,
        )
        st.plotly_chart(fig_pattern, use_container_width=True)


# --- MODULE 7: VISUALIZE ---
elif page == "📈 Visualize":
    st.header("📈 Interactive Fragmentation Curve & Sensitivity Visualizer")
    st.markdown(
        "Explore cumulative rock fragmentation size distributions ($P(x)$ vs. $x$) based on parameters from the Predictor module."
    )

    pred_res = st.session_state.get("last_predict_results", {"d50_mm": 220.0})
    base_d50 = float(pred_res.get("d50_mm", 220.0))

    # Base characteristic size calculation (d50 / (ln 2)^(1/n)) with default n=1.2
    default_xc = base_d50 / (np.log(2.0) ** (1.0 / 1.2))

    col_v1, col_v2 = st.columns([1, 2])

    with col_v1:
        st.subheader("Kuz-Ram Model Tuning Sliders")
        st.info(f"**Base Predicted d50:** `{base_d50:.1f} mm`")

        n_uniformity = st.slider(
            "Uniformity Index (n)",
            min_value=0.5,
            max_value=2.5,
            value=1.2,
            step=0.05,
            help="Higher n indicates a more uniform fragment size distribution (fewer boulders and fines).",
        )

        use_custom_xc = st.checkbox("Override Derived Characteristic Size (x_c)", value=False)
        if use_custom_xc:
            xc_val = st.slider(
                "Characteristic Size x_c (mm)",
                min_value=10.0,
                max_value=1000.0,
                value=float(np.round(default_xc, 1)),
                step=5.0,
                help="Sieve size through which 63.2% of blasted rock mass passes.",
            )
        else:
            xc_val = None

    with col_v2:
        fig_frag = plot_kuz_ram_curve(
            d50_mm=base_d50,
            n_uniformity=n_uniformity,
            xc_custom_mm=xc_val,
            label="Current Blast Design",
        )
        st.plotly_chart(fig_frag, use_container_width=True)
