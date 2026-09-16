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
from src.realtime_adaptive import adjust_charging_plan, risk_controller, audit_log
from src.digital_twin import (
    build_digital_twin,
    simulate_fragmentation,
    link_to_downstream,
    FragmentationModel,
    DownstreamModel,
    MineToMillTwin,
    OreTracker,
    ScenarioAnalyzer,
)
from src.drill_connectivity import connect_to_sandvik, connect_to_epiroc, sync_design_to_drill
from src.detonator_integration import (
    upload_timing_sequence,
    download_firing_confirmation,
    validate_sequence,
    FIRING_CONFIRMATIONS_DB,
)
from src.offline_sync import WriteAheadLog, SyncManager, resolve_conflicts
from src.regulatory import load_regulatory_limits, check_compliance, generate_compliance_report
from src.i18n import get_translation
from src.integrations import connect_to_sap, connect_to_deswik, connect_to_surpac, push_to_sap
from src.pinn import BlastPINN, predict_with_uncertainty, PINN_INPUT_COLS
from src.pareto_optimizer import run_nsga2, select_best_design, generate_trade_off_explanation, plot_pareto_front
from src.model_cards import generate_model_card
from src.explainability_audit import log_explanation, get_recent_explanations, get_explanation_history
from src.ensemble_uncertainty import EnsembleUQ, train_ensemble, predict_with_uncertainty as predict_ensemble_uq, plot_uncertainty_decomposition
from src.agent.guardrails import get_guardrail_trips
from src.agent.voice_interface import process_voice_turn, start_voice_session
from src.agent.agent_ui import render_agent_chat, render_guided_mode, render_expert_mode, render_voice_mode
from src.agent.audit import get_interaction_history, get_decision_history, export_audit_log_json
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
    get_shap_explanation,
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

# Language Selector Sidebar
st.sidebar.image("https://img.icons8.com/color/96/diamond.png", width=64)
selected_lang_label = st.sidebar.selectbox(
    "🌐 Language / Puo",
    ["English 🇬🇧", "Setswana 🇧🇼"],
    index=0,
)
lang_code = "tn" if "Setswana" in selected_lang_label else "en"

# Voice Mode Toggle Sidebar
st.sidebar.markdown("---")
voice_mode_active = st.sidebar.toggle("🎤 Voice Interaction Mode", value=False, help="Enable bilingual voice input/output interaction mode.")

if voice_mode_active:
    st.sidebar.subheader("🎙️ Voice Agent Assistant")
    voice_audio_input = st.sidebar.file_uploader("Upload or Record Voice Audio (.wav / .mp3)", type=["wav", "mp3", "ogg"])

    if voice_audio_input is not None:
        audio_bytes = voice_audio_input.read()
        st.sidebar.audio(audio_bytes, format="audio/wav")

        if st.sidebar.button("Process Voice Command 🚀", type="primary"):
            with st.spinner("Processing speech-to-text and agent reasoning..."):
                voice_res = process_voice_turn(audio_bytes=audio_bytes, user_id="SIDEBAR_VOICE_USER")

                st.sidebar.success(f"**Recognized ({voice_res['language'].upper()}):** {voice_res['transcription']}")
                st.sidebar.info(f"**Agent Response:** {voice_res['text']}")

                # Play synthesized speech response
                st.sidebar.audio(voice_res["audio"], format="audio/wav", autoplay=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        get_translation("nav_dashboard", lang_code),
        get_translation("nav_ingestion", lang_code),
        get_translation("nav_ml_manager", lang_code),
        get_translation("nav_comparison", lang_code),
        get_translation("nav_predictor", lang_code),
        get_translation("nav_optimizer", lang_code),
        get_translation("nav_pareto", lang_code),
        get_translation("nav_economic", lang_code),
        get_translation("nav_recommender", lang_code),
        get_translation("nav_mwd", lang_code),
        get_translation("nav_digital_twin", lang_code),
        get_translation("nav_connectivity", lang_code),
        get_translation("nav_detonator", lang_code),
        get_translation("nav_sync", lang_code),
        get_translation("nav_regulatory", lang_code),
        get_translation("nav_pinn", lang_code),
        get_translation("nav_integrations", lang_code),
        get_translation("nav_model_cards", lang_code),
        get_translation("nav_ensemble_uq", lang_code),
        get_translation("nav_agent", lang_code),
        get_translation("nav_pattern", lang_code),
        get_translation("nav_guardrail_log", lang_code),
        get_translation("nav_audit_log", lang_code),
        get_translation("nav_visualize", lang_code),
    ],
)

# Normalize page string matching across languages
page_keys = {
    get_translation("nav_dashboard", "en"): "dashboard",
    get_translation("nav_dashboard", "tn"): "dashboard",
    get_translation("nav_ingestion", "en"): "ingestion",
    get_translation("nav_ingestion", "tn"): "ingestion",
    get_translation("nav_ml_manager", "en"): "ml_manager",
    get_translation("nav_ml_manager", "tn"): "ml_manager",
    get_translation("nav_comparison", "en"): "comparison",
    get_translation("nav_comparison", "tn"): "comparison",
    get_translation("nav_predictor", "en"): "predictor",
    get_translation("nav_predictor", "tn"): "predictor",
    get_translation("nav_optimizer", "en"): "optimizer",
    get_translation("nav_optimizer", "tn"): "optimizer",
    get_translation("nav_pareto", "en"): "pareto",
    get_translation("nav_pareto", "tn"): "pareto",
    get_translation("nav_economic", "en"): "economic",
    get_translation("nav_economic", "tn"): "economic",
    get_translation("nav_recommender", "en"): "recommender",
    get_translation("nav_recommender", "tn"): "recommender",
    get_translation("nav_mwd", "en"): "mwd",
    get_translation("nav_mwd", "tn"): "mwd",
    get_translation("nav_digital_twin", "en"): "digital_twin",
    get_translation("nav_digital_twin", "tn"): "digital_twin",
    get_translation("nav_connectivity", "en"): "connectivity",
    get_translation("nav_connectivity", "tn"): "connectivity",
    get_translation("nav_detonator", "en"): "detonator",
    get_translation("nav_detonator", "tn"): "detonator",
    get_translation("nav_sync", "en"): "sync",
    get_translation("nav_sync", "tn"): "sync",
    get_translation("nav_regulatory", "en"): "regulatory",
    get_translation("nav_regulatory", "tn"): "regulatory",
    get_translation("nav_pinn", "en"): "pinn",
    get_translation("nav_pinn", "tn"): "pinn",
    get_translation("nav_integrations", "en"): "integrations",
    get_translation("nav_integrations", "tn"): "integrations",
    get_translation("nav_model_cards", "en"): "model_cards",
    get_translation("nav_model_cards", "tn"): "model_cards",
    get_translation("nav_ensemble_uq", "en"): "ensemble_uq",
    get_translation("nav_ensemble_uq", "tn"): "ensemble_uq",
    get_translation("nav_agent", "en"): "agent",
    get_translation("nav_agent", "tn"): "agent",
    get_translation("nav_pattern", "en"): "pattern",
    get_translation("nav_pattern", "tn"): "pattern",
    get_translation("nav_guardrail_log", "en"): "guardrail_log",
    get_translation("nav_guardrail_log", "tn"): "guardrail_log",
    get_translation("nav_audit_log", "en"): "audit_log",
    get_translation("nav_audit_log", "tn"): "audit_log",
    get_translation("nav_visualize", "en"): "visualize",
    get_translation("nav_visualize", "tn"): "visualize",
}

active_module = page_keys.get(page, "dashboard")

# --- MODULE 1: DASHBOARD & DATA EXPLORER ---
if active_module == "dashboard":
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
elif active_module == "ingestion":
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
elif active_module == "ml_manager":
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
elif active_module == "comparison":
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
elif active_module == "predictor":
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
elif active_module == "optimizer":
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
elif active_module == "economic":
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
elif active_module == "recommender":
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
elif active_module == "mwd":
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

        st.markdown("---")
        st.subheader("⚡ Model 1: Dynamic Adaptive Charging & Risk Controller")

        col_ad1, col_ad2 = st.columns(2)

        curr_design_mwd = st.session_state.get("last_predict_inputs", {"powder_factor_kg_m3": 0.65, "stemming_m": 5.0})
        latest_mwd_sample = df_mwd.iloc[-1].to_dict() if not df_mwd.empty else {"hole_id": "HOLE_001", "penetration_rate": 20.0, "torque": 1900.0}

        with col_ad1:
            if st.button("Recommend In-Flight Charging Adjustment", type="primary"):
                adjusted_plan = adjust_charging_plan(mwd_data=latest_mwd_sample, current_design=curr_design_mwd)
                st.session_state["active_adjusted_plan"] = adjusted_plan

                # Immutably log recommendation to SQLite database
                audit_log(
                    action="RECOMMEND_ADJUSTMENT",
                    user_id="REALTIME_ADAPTIVE_ENGINE",
                    original_value=curr_design_mwd,
                    new_value=adjusted_plan,
                    reason_code="MWD_ROP_LOW_TORQUE_HIGH",
                )
                st.success(f"Dynamic adjustment recommended for {adjusted_plan.get('adjusted_for_hole', 'hole')}!")

        if "active_adjusted_plan" in st.session_state:
            adj_plan = st.session_state["active_adjusted_plan"]
            st.json(adj_plan)

            risk_eval = risk_controller(adj_plan)

            if risk_eval["is_safe"]:
                st.success("🛡️ **RISK CONTROLLER:** Adjusted design is SAFE and complies with all regulatory limits.")
            else:
                st.error("🛡️ **RISK CONTROLLER WARNING:** Predicted limit violation in adjusted design!")
                for v in risk_eval["violations"]:
                    st.write(f"- ⚠️ {v}")
                for r in risk_eval["recommendations"]:
                    st.info(f"💡 {r}")


# --- MODULE: DIGITAL TWIN OF THE BENCH ---
elif active_module == "digital_twin":
    st.header("💎 3D Digital Twin of the Bench & Mine-to-Mill Value Simulator")
    st.markdown(
        "Interactive 3D spatial digital twin connecting geological block models, as-drilled geometry, "
        "and structural jointing to downstream digger productivity, truck payload, primary crusher throughput, and ore tracking."
    )

    tab_dt1, tab_dt2, tab_dt3 = st.tabs(["🧊 3D Bench & Fragmentation Simulation", "📈 What-If Scenario Sensitivity", "🗺️ Bench-to-Mill Ore Tracking"])

    with tab_dt1:
        c_dt1, c_dt2 = st.columns([1, 2])

        with c_dt1:
            st.subheader("⚙️ Digital Twin Bench Inputs")
            bench_id_input = st.text_input("Bench ID", value="BENCH_JWA_15S")
            rock_type_input = st.selectbox("In-Situ Rock Strata", ["Kimberlite_Hard", "Waste_Granite_Hard", "Kimberlite_Soft", "Sandstone_Medium"])
            rock_A_dt = st.slider("Rock Blastability Factor (A)", 4.0, 16.0, 8.5, step=0.5)

            st.subheader("Blast Design Parameters")
            pf_dt = st.number_input("Powder Factor (kg/m3)", 0.2, 2.5, 0.65, step=0.05, key="dt_pf")
            b_dt = st.number_input("Burden (m)", 2.0, 12.0, 6.0, step=0.2, key="dt_b")
            s_dt = st.number_input("Spacing (m)", 2.0, 15.0, 7.0, step=0.2, key="dt_s")

            geo_params = {
                "rock_type": rock_type_input,
                "rock_factor_A": rock_A_dt,
                "density_t_m3": 2.65,
                "hardness_index": 12.5,
            }

            dt_obj = build_digital_twin(bench_id=bench_id_input, geological_data=geo_params)
            blast_payload_dt = {
                "powder_factor_kg_m3": pf_dt,
                "burden_m": b_dt,
                "spacing_m": s_dt,
                "stemming_m": 5.0,
                "bench_height_m": 15.0,
                "charge_mass_per_hole_kg": pf_dt * b_dt * s_dt * 15.0,
            }

            twin_obj = MineToMillTwin(rock_factor_A=rock_A_dt, ore_hardness_wi=12.5)
            twin_sim = twin_obj.simulate(blast_payload_dt)

        with c_dt2:
            st.subheader("🧊 Interactive 3D Bench Block & As-Drilled Holes")
            as_drilled_df = dt_obj["as_drilled_data"]

            fig_3d = go.Figure()
            fig_3d.add_trace(go.Scatter3d(
                x=as_drilled_df["x_m"],
                y=as_drilled_df["y_m"],
                z=as_drilled_df["z_m"],
                mode="markers+text",
                name="As-Drilled Collars",
                marker=dict(size=8, color="#D50000", symbol="circle"),
                text=as_drilled_df["hole_id"],
            ))

            for _, hole in as_drilled_df.iterrows():
                fig_3d.add_trace(go.Scatter3d(
                    x=[hole["x_m"], hole["x_m"]],
                    y=[hole["y_m"], hole["y_m"]],
                    z=[hole["z_m"], hole["z_m"] - hole["depth_m"]],
                    mode="lines",
                    line=dict(color="#FF6D00", width=4),
                    showlegend=False,
                ))

            fig_3d.update_layout(
                title="<b>3D Bench Spatial Geometry & As-Drilled Trajectories</b>",
                scene=dict(xaxis_title="Easting (m)", yaxis_title="Northing (m)", zaxis_title="Elevation (m)"),
                template="plotly_white",
                height=420,
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            st.markdown("---")
            st.subheader("💥 Fragmentation Size Distribution & Percentiles")

            frag_data = twin_sim["fragmentation"]
            col_f1, col_f2, col_f3 = st.columns(3)
            col_f1.metric("D50 Fragment Size", f"{frag_data['d50_mm']:.1f} mm")
            col_f2.metric("D80 Fragment Size", f"{frag_data['d80_mm']:.1f} mm")
            col_f3.metric("Total Mine-to-Mill Cost", f"${twin_sim['cost']:.2f} / t")

            df_dist = frag_data["distribution_df"]
            fig_swebrec = px.line(
                df_dist,
                x="size_cm",
                y="percent_passing",
                title="<b>Swebrec Cumulative Size Distribution Curve P(x)</b>",
                labels={"size_cm": "Sieve Particle Size (cm)", "percent_passing": "Cumulative Passing (%)"},
                color_discrete_sequence=["#2962FF"],
            )
            fig_swebrec.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_swebrec, use_container_width=True)

            st.markdown("---")
            st.subheader("🏗️ Predicted Downstream Mine-to-Mill KPIs")
            downstream_kpis = twin_sim["downstream"]

            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            col_d1.metric("Digger Fill Factor", f"{downstream_kpis['digger_fill_factor']*100:.0f}%")
            col_d2.metric("Truck Payload", f"{downstream_kpis['truck_payload_t']:.1f} t")
            col_d3.metric("Crusher Throughput", f"{downstream_kpis['crusher_throughput_tph']:.0f} t/h")
            col_d4.metric("Specific Energy", f"{downstream_kpis['specific_energy_kwh_t']:.2f} kWh/t")

    with tab_dt2:
        st.subheader("💡 What-If Scenario & Parameter Sensitivity Analyzer")

        param_to_sweep = st.selectbox(
            "Select Parameter for What-If Sensitivity Sweep",
            ["powder_factor_kg_m3", "burden_m", "spacing_m", "stemming_m"],
            index=0,
        )

        min_range = 0.30 if param_to_sweep == "powder_factor_kg_m3" else 2.0
        max_range = 1.20 if param_to_sweep == "powder_factor_kg_m3" else 10.0

        analyzer = ScenarioAnalyzer(twin=MineToMillTwin(rock_factor_A=8.5, ore_hardness_wi=12.5))
        base_payload = {
            "powder_factor_kg_m3": 0.65,
            "burden_m": 6.0,
            "spacing_m": 7.0,
            "stemming_m": 5.0,
            "bench_height_m": 15.0,
        }

        fig_sens = analyzer.sensitivity_sweep(
            blast_params=base_payload,
            param_to_vary=param_to_sweep,
            range_min=min_range,
            range_max=max_range,
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        df_whatif = analyzer.twin.what_if(
            blast_params=base_payload,
            param_to_vary=param_to_sweep,
            range_min=min_range,
            range_max=max_range,
            steps=10,
        )
        st.dataframe(df_whatif, use_container_width=True)

    with tab_dt3:
        st.subheader("🗺️ Bench-to-Mill Ore Block Graph Tracking")
        st.markdown(
            "Traces ore blocks from bench origin polygon to primary crusher and milling batches using GPS, RFID, or ML telematics."
        )

        tracker = OreTracker()
        tracker.add_blast("BLAST_JWA_2026_08", coordinates=(-24.52, 25.83, 1150.0), timestamp="2026-09-16T10:00:00")
        tracker.add_ore_block("BLOCK_CUT8_15S_01", blast_id="BLAST_JWA_2026_08", coordinates=(-24.521, 25.832, 1150.0))
        tracker.add_processing_batch("BATCH_MILL_402", block_ids=["BLOCK_CUT8_15S_01"], timestamp="2026-09-16T14:30:00")

        query_block_id = st.text_input("Query Ore Block ID to Trace Path", value="BLOCK_CUT8_15S_01")

        if st.button("Trace Ore Path from Bench to Mill"):
            path_nodes = tracker.trace_ore(query_block_id)

            st.success(f"Successfully traced ore block `{query_block_id}` across {len(path_nodes)} graph nodes!")

            for idx, node in enumerate(path_nodes, 1):
                st.info(f"**Step {idx} [{node['node_type']}]:** {node['data']}")


# --- MODULE: DRILL CONNECTIVITY ---
elif active_module == "connectivity":
    st.header("🚜 Direct-to-Drill Telematics & ISO 15143-3 Integration")
    st.markdown(
        "Direct API connectivity to **Sandvik (My Sandvik)** and **Epiroc (Certiq)** smart drill rigs via ISO 15143-3 (AEMP 2.0). "
        "Eliminates manual paper/USB data entry by pushing 3D drill patterns directly to drill rig cabin displays."
    )

    c_dc1, c_dc2 = st.columns([1, 2])

    with c_dc1:
        st.subheader("🔌 Vendor Telematics Status")
        sandvik_conn = connect_to_sandvik()
        epiroc_conn = connect_to_epiroc()

        st.info(f"**Sandvik Status:** `{sandvik_conn['status'].upper()}` (ISO 15143-3 Compliant)")
        st.info(f"**Epiroc Status:** `{epiroc_conn['status'].upper()}` (ISO 15143-3 Compliant)")

        st.subheader("🚀 Push Pattern to Drill Rig")
        selected_vendor = st.selectbox("Select Drill Vendor", ["Sandvik", "Epiroc"])

        if selected_vendor == "Sandvik":
            rig_list = [r["drill_id"] for r in sandvik_conn["fleet"]]
        else:
            rig_list = [r["drill_id"] for r in epiroc_conn["fleet"]]

        selected_rig = st.selectbox("Select Target Drill Rig", rig_list)
        pattern_id = st.text_input("Pattern Design ID", value="PATTERN_CUT8_BENCH15S")
        num_holes_push = st.number_input("Number of Holes in Pattern", 4, 200, 24)

        if st.button("Push Design File to Rig", type="primary"):
            design_payload = {"design_id": pattern_id, "num_holes": int(num_holes_push)}
            sync_res = sync_design_to_drill(
                design_file=design_payload,
                drill_id=selected_rig,
                vendor=selected_vendor.lower(),
            )
            st.success(sync_res["message"])

    with c_dc2:
        st.subheader("📡 Connected Drill Rig Fleets")

        all_fleet = sandvik_conn["fleet"] + epiroc_conn["fleet"]
        df_fleet = pd.DataFrame(all_fleet)
        st.dataframe(df_fleet, use_container_width=True)

        st.markdown("---")
        st.subheader("📐 As-Drilled vs As-Designed Telematics Compliance")

        # As-designed vs As-drilled compliance table
        compliance_data = [
            {"Hole ID": "Hole_01", "Design Depth (m)": 15.0, "Drilled Depth (m)": 15.2, "Deviation (m)": "+0.20", "Status": "PASS"},
            {"Hole ID": "Hole_02", "Design Depth (m)": 15.0, "Drilled Depth (m)": 14.8, "Deviation (m)": "-0.20", "Status": "PASS"},
            {"Hole ID": "Hole_03", "Design Depth (m)": 15.0, "Drilled Depth (m)": 16.4, "Deviation (m)": "+1.40", "Status": "WARN (Overdrilled)"},
            {"Hole ID": "Hole_04", "Design Depth (m)": 15.0, "Drilled Depth (m)": 15.1, "Deviation (m)": "+0.10", "Status": "PASS"},
        ]
        st.dataframe(pd.DataFrame(compliance_data), use_container_width=True)


# --- MODULE: ELECTRONIC DETONATOR INTEGRATION ---
elif active_module == "detonator":
    st.header("⚡ Electronic Detonator Field-to-Cloud Integration")
    st.markdown(
        "Direct integration with major electronic initiation systems in Botswana (**AEL IntelliShot**, **BME AXXIS**, **Orica i-kon III**). "
        "Upload millisecond-precision timing sequences, validate regulatory compliance, and download firing confirmations."
    )

    c_det1, c_det2 = st.columns([1, 2])

    with c_det1:
        st.subheader("⚙️ System Selection & Sequence Upload")
        det_system = st.selectbox("Select Electronic Detonator System", ["AEL IntelliShot", "BME AXXIS", "Orica i-kon III"])
        blast_id_det = st.text_input("Blast Pattern ID", value="BLAST_JWA_2024_08")

        hole_delay_det = st.number_input("Inter-Hole Delay (ms)", 1.0, 100.0, 17.0, step=1.0)
        row_delay_det = st.number_input("Inter-Row Delay (ms)", 5.0, 200.0, 42.0, step=1.0)
        charge_delay_det = st.number_input("Max Charge per Delay (kg)", 50.0, 5000.0, 640.0, step=50.0)

        seq_payload = {
            "blast_id": blast_id_det,
            "hole_delay_ms": hole_delay_det,
            "row_delay_ms": row_delay_det,
            "max_charge_per_delay_kg": charge_delay_det,
            "predicted_ppv_mms": 8.5,
            "predicted_airblast_dbl": 115.0,
        }

        # Real-time Sequence Regulatory Validation
        is_valid_seq, seq_violations = validate_sequence(seq_payload)

        if is_valid_seq:
            st.success("✅ **REGULATORY COMPLIANT:** Sequence satisfies Botswana vibration and airblast thresholds.")
        else:
            for v in seq_violations:
                st.error(v)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("Upload Timing Sequence", type="primary"):
                up_res = upload_timing_sequence(det_system, seq_payload, blast_id=blast_id_det)
                st.success(f"Uploaded to {up_res['detonator_system']}!")

        with col_b2:
            if st.button("Download Confirmation"):
                conf_res = download_firing_confirmation(det_system, blast_id=blast_id_det)
                st.success(f"Downloaded confirmation for {conf_res['blast_id']}!")

    with c_det2:
        st.subheader("📊 Firing Confirmations & Field Diagnostics")

        if not FIRING_CONFIRMATIONS_DB:
            # Seed mock default confirmation if database is empty
            download_firing_confirmation("AEL IntelliShot", blast_id="BLAST_JWA_2024_01")
            download_firing_confirmation("BME AXXIS", blast_id="BLAST_ORA_2024_03")

        df_conf = pd.DataFrame(FIRING_CONFIRMATIONS_DB)
        st.dataframe(df_conf, use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Electronic System Vendor Architecture")
        st.info("**AEL IntelliShot:** Uses Commander control boxes and Tagger handheld devices with smart lead wire auto-tagging.")
        st.info("**BME AXXIS:** AXXIS Titanium / Gii dual-capacitor architecture with sub-millisecond firing window accuracy.")
        st.info("**Orica i-kon III:** High-capacity Logger/Blaster suite supporting up to 4,800 caps per blast with encrypted telemetry.")


# --- MODULE: SYNC STATUS & WRITE-AHEAD LOG ---
elif active_module == "sync":
    st.header("🔄 Offline-First Sync Status & Write-Ahead Log (WAL) Manager")
    st.markdown(
        "Ensures zero data loss in remote Botswana open-pit benches (Jwaneng, Orapa, Karowe) "
        "by queuing field actions in a persistent **Write-Ahead Log (WAL)** and replaying queue with exponential backoff retries upon connection."
    )

    c_sync1, c_sync2 = st.columns([1, 2])

    # Instantiate persistent WriteAheadLog & SyncManager
    wal_instance = WriteAheadLog("data/processed/write_ahead_log.json")
    sync_manager = SyncManager(wal=wal_instance)

    with c_sync1:
        st.subheader("⚙️ Connection & Queue Status")
        is_online_sim = st.toggle("Simulate Network Connection", value=True, help="Toggle between Online and Offline pit floor connection.")

        pending_wal = wal_instance.get_pending()

        status_color = "green" if is_online_sim else "red"
        st.markdown(f"**Network Connectivity:** :{status_color}[{'ONLINE (Connected)' if is_online_sim else 'OFFLINE (Remote Pit)'}]")
        st.metric("Pending WAL Actions", len(pending_wal))
        st.metric("Last Sync Timestamp", sync_manager.last_sync_timestamp if sync_manager.last_sync_timestamp else "Not Synced Yet")

        st.markdown("---")
        st.subheader("Simulate Offline Action Entry")
        sim_action_type = st.selectbox("Action Type", ["FIELD_LOG", "DESIGN_UPDATE", "MWD_TELEMETRY"])
        sim_hole = st.text_input("Hole ID", value="HOLE_WAL_101")
        sim_val = st.number_input("Measured Depth / Value", 1.0, 50.0, 15.2)

        if st.button("Enqueue Action into WAL", type="primary"):
            entry = wal_instance.append(sim_action_type, {"hole_id": sim_hole, "measured_value": sim_val, "role": "blaster"})
            st.success(f"Action enqueued to WAL! ID: {entry['wal_id']}")

        if st.button("Force Sync Now 🚀"):
            sync_res = sync_manager.sync_all(online_check_fn=lambda: is_online_sim)
            if sync_res["status"] == "offline":
                st.error("Cannot sync: Network device is currently offline.")
            else:
                st.success(f"Sync Execution Complete! Processed {sync_res['synced_count']} WAL items with exponential backoff retry support.")

    with c_sync2:
        st.subheader("📋 Persistent Write-Ahead Log (WAL) Pending Queue")

        pending_items = wal_instance.get_pending()
        if pending_items:
            df_wal = pd.DataFrame(pending_items)
            st.dataframe(df_wal, use_container_width=True)
        else:
            st.info("Write-Ahead Log (WAL) is empty. All field actions are synchronized with cloud server.")

        st.markdown("---")
        st.subheader("⚔️ Conflict Resolution Engine Test")
        st.markdown("Tests timestamp-based `last_write_wins` or `merge_conservative` conflict resolution for simultaneous field/cloud edits.")

        loc_depth = st.number_input("Local Field Depth Edit (m)", 1.0, 30.0, 15.5)
        rem_depth = st.number_input("Remote Server Depth (m)", 1.0, 30.0, 15.0)
        strategy_sel = st.selectbox("Conflict Resolution Strategy", ["last_write_wins", "remote_wins", "local_wins", "merge_conservative"])

        if st.button("Resolve Conflict Example"):
            loc_rec = {"hole_id": "HOLE_01", "depth_m": loc_depth, "timestamp": "2026-09-16T12:00:00"}
            rem_rec = {"hole_id": "HOLE_01", "depth_m": rem_depth, "timestamp": "2026-09-16T10:00:00"}
            res_rec = resolve_conflicts(loc_rec, rem_rec, strategy=strategy_sel)
            st.json(res_rec)


# --- MODULE: REGULATORY COMPLIANCE ---
elif active_module == "regulatory":
    st.header("📜 Botswana Mining Regulatory Compliance & Audit Module")
    st.markdown(
        "Automated compliance evaluation under the **Mines, Quarries, Works and Machinery Act (Cap. 44:02)** "
        "and **Data Protection Act of Botswana**. Evaluates ground vibration (PPV), airblast noise overpressure (dBL), "
        "flyrock safety boundaries, and stemming confinement."
    )

    c_reg1, c_reg2 = st.columns([1, 2])

    reg_limits = load_regulatory_limits()

    with c_reg1:
        st.subheader("⚖️ Active Regulatory Threshold Limits")
        st.info(f"**Legislative Framework:** {reg_limits.get('act')}")
        st.metric("Max Ground Vibration (PPV)", f"{reg_limits.get('max_ppv_mms', 10.0):.1f} mm/s")
        st.metric("Max Airblast Overpressure", f"{reg_limits.get('max_airblast_dbl', 120.0):.1f} dBL")
        st.metric("Max Flyrock Boundary Range", f"{reg_limits.get('max_flyrock_m', 250.0):.1f} m")
        st.metric("Min Stemming Confinement", f"{reg_limits.get('min_stemming_m', 2.5):.1f} m")

    with c_reg2:
        st.subheader("🔍 Proposed Blast Design Compliance Evaluation")

        last_in = st.session_state.get("last_predict_inputs", {})
        last_out = st.session_state.get("last_predict_results", {})

        comp_eval = check_compliance(blast_params=last_in, predictions=last_out, custom_limits=reg_limits)

        if comp_eval["is_compliant"]:
            st.success("✅ **FULLY COMPLIANT:** Proposed blast design satisfies all Botswana Department of Mines environmental & safety regulations.")
        else:
            st.error("❌ **NON-COMPLIANT:** Detected regulatory threshold violations in proposed design!")
            st.subheader("Detected Violations:")
            for v in comp_eval["violations"]:
                st.write(f"- ⚠️ {v}")

            st.subheader("Actionable Engineering Recommendations:")
            for r in comp_eval["recommendations"]:
                st.info(f"💡 {r}")

        st.markdown("---")
        st.subheader("📄 Export Official Regulatory Submission Report")

        report_pdf_path = generate_compliance_report(
            blast_params=last_in,
            predictions=last_out,
            output_path="data/processed/botswana_regulatory_compliance_report.pdf",
        )

        with open(report_pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        st.download_button(
            label="📄 Download Official Compliance PDF Report",
            data=pdf_bytes,
            file_name="Botswana_Mining_Regulatory_Compliance_Report.pdf",
            mime="application/pdf",
            type="primary",
        )


# --- MODULE: DEBSWANA ENTERPRISE INTEGRATIONS ---
elif active_module == "integrations":
    st.header("🔗 Debswana Enterprise Systems Integration")
    st.markdown(
        "Direct API integration with Debswana's core enterprise systems (**SAP ERP**, **Deswik CAD**, **GEOVIA Surpac**) "
        "to eliminate manual spreadsheet exports and data silos across procurement, mine planning, and geology."
    )

    c_int1, c_int2 = st.columns([1, 2])

    with c_int1:
        st.subheader("⚙️ System API Status & Verification")

        st.markdown("### 🏢 1. SAP ERP (Procurement & Costing)")
        if st.button("Test SAP Connection", type="primary"):
            sap_res = connect_to_sap()
            st.success(f"Status: {sap_res['status'].upper()} | Sync: {sap_res['last_sync']}")
            st.json(sap_res["retrieved_costs"])

        st.markdown("### 📐 2. Deswik CAD (Mine Planning)")
        if st.button("Test Deswik Connection"):
            deswik_res = connect_to_deswik()
            st.success(f"Status: {deswik_res['status'].upper()} | Sync: {deswik_res['last_sync']}")
            st.json(deswik_res["retrieved_mine_plan"])

        st.markdown("### 🪨 3. GEOVIA Surpac (Geology Block Model)")
        if st.button("Test Surpac Connection"):
            surpac_res = connect_to_surpac()
            st.success(f"Status: {surpac_res['status'].upper()} | Sync: {surpac_res['last_sync']}")
            st.json(surpac_res["retrieved_geology"])

    with c_int2:
        st.subheader("📊 Enterprise Data Flow Summary")

        df_flow = pd.DataFrame([
            {"System": "SAP ERP", "Inbound Data": "Explosive $/kg, Drilling $/m, Budget Limits", "Outbound Data": "Actual post-blast $/t expenditure", "Protocol": "REST / OData"},
            {"System": "Deswik CAD", "Inbound Data": "Bench boundaries, 3D collar targets", "Outbound Data": "Optimized pattern geometry", "Protocol": "REST / API"},
            {"System": "GEOVIA Surpac", "Inbound Data": "3D Block model, RQD, Bond Work Index", "Outbound Data": "Predicted d50 size overlays", "Protocol": "API / File Exchange"},
        ])
        st.dataframe(df_flow, use_container_width=True)

        st.markdown("---")
        st.subheader("📤 Push Actual Blast Costs to SAP")

        blast_id_push = st.text_input("Blast ID for SAP Cost Accounting", value="BLAST_JWA_2024_08")
        cost_push_val = st.number_input("Actual Calculated Cost ($/t)", 0.1, 100.0, 5.25)

        if st.button("Push Cost Record to SAP Cost Center"):
            push_res = push_to_sap({"blast_id": blast_id_push, "total_cost_per_tonne_usd": cost_push_val})
            st.success(push_res["message"])


# --- MODULE: PINN PREDICTION & UNCERTAINTY ---
elif active_module == "pinn":
    st.header("🧠 Physics-Informed Neural Network (PINN) & Uncertainty Quantification")
    st.markdown(
        "Model 2 (Physics-Informed Neural Network) embeds Kuz-Ram fragmentation and USBM PPV wave attenuation equations "
        "as soft loss terms ($L_{\\text{total}} = L_{\\text{data}} + \\lambda_1 L_{\\text{kuzram}} + \\lambda_2 L_{\\text{usbm}}$). "
        "Monte Carlo Dropout estimates epistemic uncertainty and alerts blasters to Out-Of-Distribution (OOD) risks."
    )

    c_pinn1, c_pinn2 = st.columns([1, 2])

    with c_pinn1:
        st.subheader("⚙️ PINN 12 Input Parameters")
        last_in = st.session_state.get("last_predict_inputs", {})

        b_pinn = st.slider("Burden (m)", 2.0, 12.0, float(last_in.get("burden_m", 6.0)), step=0.2)
        s_pinn = st.slider("Spacing (m)", 2.0, 15.0, float(last_in.get("spacing_m", 7.0)), step=0.2)
        d_pinn = st.slider("Hole Diameter (mm)", 80.0, 380.0, float(last_in.get("hole_diameter_mm", 250.0)), step=10.0)
        h_pinn = st.slider("Hole Depth (m)", 5.0, 35.0, float(last_in.get("bench_height_m", 15.0)), step=0.5)
        stem_pinn = st.slider("Stemming Length (m)", 1.0, 10.0, float(last_in.get("stemming_m", 5.0)), step=0.2)
        sub_pinn = st.slider("Sub-drill (m)", 0.5, 3.0, 1.5, step=0.1)
        pf_pinn = st.slider("Powder Factor (kg/m3)", 0.2, 2.5, float(last_in.get("powder_factor_kg_m3", 0.65)), step=0.05)
        w_pinn = st.slider("Max Charge per Delay (kg)", 10.0, 2000.0, float(last_in.get("max_charge_per_delay_kg", 640.0)), step=20.0)
        ucs_pinn = st.slider("Rock Strength UCS (MPa)", 20.0, 300.0, 120.0, step=5.0)
        rmr_pinn = st.slider("Rock Mass Rating RMR", 20.0, 90.0, 65.0, step=1.0)
        dist_pinn = st.slider("Monitoring Distance (m)", 50.0, 3000.0, float(last_in.get("monitoring_distance_m", 450.0)), step=25.0)
        bi_pinn = st.slider("Blastability Index BI", 10.0, 100.0, 55.0, step=1.0)

        pinn_feature_vec = [
            b_pinn, s_pinn, d_pinn, h_pinn, stem_pinn, sub_pinn,
            pf_pinn, w_pinn, ucs_pinn, rmr_pinn, dist_pinn, bi_pinn
        ]

        mc_samples = st.slider("Monte Carlo Dropout Pass Samples", 20, 300, 100, step=20)

    with c_pinn2:
        st.subheader("📊 PINN Predictions & 95% Confidence Intervals")

        pinn_model = BlastPINN(input_dim=12)
        uncertainty_res = predict_with_uncertainty(pinn_model, np.array([pinn_feature_vec]), n_samples=mc_samples)

        means = uncertainty_res["mean"]
        cis = uncertainty_res["ci_95"]
        is_high_unc = uncertainty_res.get("high_uncertainty", False)

        m_p1, m_p2, m_p3 = st.columns(3)
        m_p1.metric(
            "Fragmentation (D80)",
            f"{means['fragmentation']:.1f} mm",
            f"95% CI: [{cis['fragmentation'][0]:.1f}, {cis['fragmentation'][1]:.1f}]",
        )
        m_p2.metric(
            "Ground Vibration (PPV)",
            f"{means['ppv']:.2f} mm/s",
            f"95% CI: [{cis['ppv'][0]:.2f}, {cis['ppv'][1]:.2f}]",
        )
        m_p3.metric(
            "Airblast Overpressure",
            f"{means['airblast']:.1f} dB",
            f"95% CI: [{cis['airblast'][0]:.1f}, {cis['airblast'][1]:.1f}]",
        )

        st.markdown("---")
        st.subheader("🚨 Epistemic Uncertainty & OOD Risk Assessment")

        if is_high_unc:
            st.error("⚠️ **HIGH UNCERTAINTY / OOD WARNING:** Input features are Out-Of-Distribution (OOD) relative to training pit data.")
            st.warning("💡 **RECOMMENDATION:** High prediction variance detected across Monte Carlo passes. Verify rock mass jointing in field log or run conservative physics bounds.")
        else:
            st.success("✅ **CONFIDENT PREDICTION:** Low epistemic variance detected across Monte Carlo dropout passes.")

        st.markdown("---")
        st.subheader("🔍 DeepSHAP Feature Contribution Explanation for PINN")

        pinn_df_input = pd.DataFrame([pinn_feature_vec], columns=PINN_INPUT_COLS)
        pinn_shap = get_shap_explanation(
            model=pinn_model,
            input_data=pinn_df_input,
            feature_names=PINN_INPUT_COLS,
        )

        waterfall_fig = pinn_shap.get("waterfall_plot")
        if waterfall_fig is not None:
            st.plotly_chart(waterfall_fig, use_container_width=True)

        force_fig = pinn_shap.get("force_plot")
        if force_fig is not None:
            st.plotly_chart(force_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📐 Physics Soft Loss Terms Embedding")
        st.info("**Kuz-Ram Soft Penalty ($L_{\\text{kuzram}}$):** Penalizes predictions deviating from $X_{50} = A \\cdot K^{-0.8} \\cdot Q^{1/6} \\cdot (115/E)^{19/30}$.")
        st.info("**USBM PPV Soft Penalty ($L_{\\text{usbm}}$):** Penalizes predictions deviating from $PPV = 1140 \\cdot (D / \\sqrt{W})^{-1.6}$.")


# --- MODULE: MULTI-OBJECTIVE PARETO OPTIMIZER ---
elif active_module == "pareto":
    st.header("⚡ Model 3: Multi-Objective NSGA-II Pareto Optimizer")
    st.markdown(
        "Discovers the non-dominated **Pareto Frontier** across 5 competing blast design objectives: "
        "minimizing fragmentation ($D_{80}$), minimizing ground vibration ($PPV$), minimizing airblast ($dB$), "
        "minimizing cost ($/t), and maximizing primary crusher throughput ($t/h$)."
    )

    c_par1, c_par2 = st.columns([1, 2])

    with c_par1:
        st.subheader("⚖️ Objective Importance Weights")
        w_frag = st.slider("Minimizing D80 Fragmentation Weight", 0.0, 1.0, 0.25, step=0.05)
        w_vib = st.slider("Minimizing Ground PPV Weight", 0.0, 1.0, 0.25, step=0.05)
        w_air = st.slider("Minimizing Airblast Overpressure Weight", 0.0, 1.0, 0.15, step=0.05)
        w_cost = st.slider("Minimizing Unit Cost Weight", 0.0, 1.0, 0.20, step=0.05)
        w_tph = st.slider("Maximizing Crusher Throughput Weight", 0.0, 1.0, 0.15, step=0.05)

        weights_dict = {
            "weight_fragmentation": w_frag,
            "weight_vibration": w_vib,
            "weight_airblast": w_air,
            "weight_cost": w_cost,
            "weight_throughput": w_tph,
        }

        n_gen = st.slider("NSGA-II Generations", 20, 300, 100, step=20)
        pop_size = st.slider("Population Size", 20, 150, 50, step=10)

        if st.button("Run Multi-Objective NSGA-II", type="primary"):
            with st.spinner("Calculating non-dominated Pareto Frontier across 5 objectives..."):
                df_pareto = run_nsga2(n_gen=n_gen, pop_size=pop_size, seed=42)
                st.session_state["pareto_front_df"] = df_pareto
                st.success(f"Discovered {len(df_pareto)} non-dominated Pareto-optimal designs!")

    with c_par2:
        if "pareto_front_df" in st.session_state and not st.session_state["pareto_front_df"].empty:
            df_p = st.session_state["pareto_front_df"]

            st.subheader("📊 Interactive Pareto Front Scatter Plot")
            c_p_x, c_p_y = st.columns(2)
            with c_p_x:
                obj_x = st.selectbox("X-Axis Objective", ["d80_mm", "ppv_mms", "airblast_dbl", "cost_per_tonne_usd", "crusher_throughput_tph"], index=0)
            with c_p_y:
                obj_y = st.selectbox("Y-Axis Objective", ["ppv_mms", "d80_mm", "airblast_dbl", "cost_per_tonne_usd", "crusher_throughput_tph"], index=0)

            fig_p = plot_pareto_front(df_p, x_objective=obj_x, y_objective=obj_y)
            st.plotly_chart(fig_p, use_container_width=True)

            st.markdown("---")
            st.subheader("📋 Pareto-Optimal Candidate Designs Table")
            st.dataframe(df_p, use_container_width=True)

            st.markdown("---")
            st.subheader("🌟 Select Design & View Trade-Off Explanation")

            selected_design = select_best_design(df_p, weights_dict)

            if st.button("Select Recommended Design"):
                st.session_state["active_selected_pareto"] = selected_design

            if "active_selected_pareto" in st.session_state:
                sel_d = st.session_state["active_selected_pareto"]

                st.json(sel_d)
                explanation_str = generate_trade_off_explanation(df_p, sel_d)
                st.info(f"💡 **Trade-Off Explanation:** {explanation_str}")

            st.markdown("---")
            st.subheader("⚔️ Compare Two Pareto Designs Side-by-Side")

            c_cmp1, c_cmp2 = st.columns(2)
            with c_cmp1:
                idx_1 = st.selectbox("Select Design #1 (Row Index)", list(df_p.index), index=0, key="pareto_cmp_1")
                design_1 = df_p.loc[idx_1].to_dict()
                st.write("**Design #1 Parameters & Outcomes:**")
                st.json(design_1)

            with c_cmp2:
                default_idx_2 = min(1, len(df_p) - 1)
                idx_2 = st.selectbox("Select Design #2 (Row Index)", list(df_p.index), index=default_idx_2, key="pareto_cmp_2")
                design_2 = df_p.loc[idx_2].to_dict()
                st.write("**Design #2 Parameters & Outcomes:**")
                st.json(design_2)

            df_side_by_side = pd.DataFrame([design_1, design_2], index=["Design #1", "Design #2"]).T
            st.subheader("Side-by-Side Metrics Comparison Table")
            st.dataframe(df_side_by_side, use_container_width=True)

        else:
            st.info("Adjust objective weights and click 'Run Multi-Objective NSGA-II' to compute Pareto front.")


# --- MODULE: MODEL CARDS & EXPLAINABILITY AUDIT ---
elif active_module == "model_cards":
    st.header("📋 Model Cards Governance & Explanation Audit Trail")
    st.markdown(
        "Standardized model documentation cards and immutable SQLite audit trail recording "
        "every AI prediction explanation generated for regulatory compliance under the **Mines, Quarries, Works and Machinery Act (Cap. 44:02)**."
    )

    tab_mc1, tab_mc2 = st.tabs(["📄 Model Cards Viewer", "📜 Immutable Explanation Audit Log"])

    with tab_mc1:
        st.subheader("Select Model Card")

        # Map of available models in registry & pipeline
        available_models = {
            "GA-ANN Jwaneng Multi-Output Model": "ga_ann_jwaneng",
            "ANN-RF Ensemble Jwaneng Predictor": "ann_rf_ensemble_jwaneng",
            "PSO-ANN Orapa Fragmentation Model": "pso_ann_orapa",
            "Debswana Open-Pit Airblast Minimizer": "airblast_minimizer",
        }

        selected_card_label = st.selectbox("Select Model", list(available_models.keys()))
        selected_key = available_models[selected_card_label]

        model_meta = MODEL_REGISTRY.get(selected_key, {})

        # Generate / ensure model card Markdown file exists
        col_mc_btn1, col_mc_btn2 = st.columns([1, 3])

        with col_mc_btn1:
            if st.button("Generate Model Card 📄", type="primary"):
                card_filepath = generate_model_card(
                    model_name=selected_card_label,
                    model=None,
                    training_data={"source": model_meta.get("source", "Debswana Open-Pit Mine"), "size": "120 production blasts"},
                    performance_metrics=model_meta.get("performance", {}),
                    research_reference=model_meta.get("reference", "Debswana Mining Series"),
                    output_dir="models/cards/",
                    version="1.0.0",
                )
                st.session_state["last_generated_card"] = card_filepath
                st.success(f"Generated model card: `{card_filepath}`")

        card_path_to_display = st.session_state.get("last_generated_card", f"models/cards/{selected_key.lower()}_v1.0.0.md")

        if not os.path.exists(card_path_to_display):
            card_path_to_display = generate_model_card(
                model_name=selected_card_label,
                model=None,
                training_data={"source": model_meta.get("source", "Debswana Open-Pit Mine"), "size": "120 production blasts"},
                performance_metrics=model_meta.get("performance", {}),
                output_dir="models/cards/",
                version="1.0.0",
            )

        if os.path.exists(card_path_to_display):
            with open(card_path_to_display, "r", encoding="utf-8") as f:
                card_md_content = f.read()

            st.markdown(card_md_content)

            st.download_button(
                label="📄 Download Model Card (.md)",
                data=card_md_content,
                file_name=os.path.basename(card_path_to_display),
                mime="text/markdown",
            )
        else:
            st.error("Model card Markdown file could not be generated.")
        with col_mc_btn2:
            pass

    with tab_mc2:
        st.subheader("📜 Recent Logged Explanations (Immutable SQLite Audit Trail)")
        st.markdown(
            "Every prediction explanation (SHAP feature values, LIME weights, natural language summary, user ID, timestamp) "
            "is automatically recorded in an append-only database table to ensure full post-blast forensic accountability."
        )

        col_aud1, col_aud2 = st.columns([1, 2])

        with col_aud1:
            st.markdown("### 🧪 Simulate Logging an Explanation")
            sim_pred_id = st.text_input("Prediction / Blast ID", value="PRED_JWA_2026_08")
            sim_user_id = st.text_input("Blaster / Engineer User ID", value="CHIEF_BLASTER_BOTSWANA_01")
            sim_nl = st.text_area("Natural Language Summary", value="Ground PPV is predicted to meet safety constraints at 8.50 mm/s (limit: 10.0 mm/s). Primary driver is maximum charge per delay.")

            if st.button("Log Explanation to Audit Database", type="primary"):
                log_res = log_explanation(
                    prediction_id=sim_pred_id,
                    shap_values={"powder_factor_kg_m3": -12.4, "max_charge_per_delay_kg": 18.2, "burden_m": 4.1},
                    lime_weights={"powder_factor_kg_m3": -0.15, "max_charge_per_delay_kg": 0.22},
                    natural_language=sim_nl,
                    user_id=sim_user_id,
                )
                st.success(f"Successfully logged explanation record! Row ID: {log_res['row_id']}")

        with col_aud2:
            st.markdown("### 📋 Audit Trail Log Records")
            audit_history = get_explanation_history(limit=25)

            if audit_history:
                df_audit = pd.DataFrame(audit_history)
                st.dataframe(df_audit, use_container_width=True)
            else:
                st.info("No audit logs recorded yet. Use the simulation tool on the left to record sample explanations.")


# --- MODULE: ENSEMBLE UNCERTAINTY QUANTIFICATION ---
elif active_module == "ensemble_uq":
    st.header("🛡️ Model 5: Ensemble Uncertainty Quantification (UQ)")
    st.markdown(
        "Combines four base model architectures (**ANN**, **XGBoost**, **Random Forest**, **PINN**) "
        "trained on 80% bootstrap sub-samples. Explicitly decomposes uncertainty into **aleatoric** (data noise) "
        "and **epistemic** (model knowledge gap) components with 95% confidence intervals."
    )

    c_uq1, c_uq2 = st.columns([1, 2])

    with c_uq1:
        st.subheader("⚙️ Blast Input Parameters")
        last_in = st.session_state.get("last_predict_inputs", {})

        b_uq = st.slider("Burden (m)", 2.0, 12.0, float(last_in.get("burden_m", 6.0)), step=0.2, key="uq_b")
        s_uq = st.slider("Spacing (m)", 2.0, 15.0, float(last_in.get("spacing_m", 7.0)), step=0.2, key="uq_s")
        stem_uq = st.slider("Stemming (m)", 1.0, 10.0, float(last_in.get("stemming_m", 5.0)), step=0.2, key="uq_stem")
        pf_uq = st.slider("Powder Factor (kg/m3)", 0.2, 2.5, float(last_in.get("powder_factor_kg_m3", 0.65)), step=0.05, key="uq_pf")
        charge_uq = st.slider("Max Charge per Delay (kg)", 10.0, 2000.0, float(last_in.get("max_charge_per_delay_kg", 640.0)), step=20.0, key="uq_w")
        dist_uq = st.slider("Monitoring Distance (m)", 50.0, 3000.0, float(last_in.get("monitoring_distance_m", 450.0)), step=25.0, key="uq_dist")

        uq_input_vec = [b_uq, s_uq, 250.0, 15.0, stem_uq, 1.5, pf_uq, charge_uq, 120.0, 65.0, dist_uq, 55.0]

        n_members = st.slider("Bagging Ensemble Members per Family", 3, 20, 5, step=1)

        if st.button("Retrain Ensemble 🔄", type="primary"):
            with st.spinner("Retraining multi-architecture bagging ensemble across bootstrap sub-samples..."):
                df_curr = st.session_state["dataset"]
                feature_cols_present = [c for c in FEATURE_COLS if c in df_curr.columns]
                X_mat = df_curr[feature_cols_present].values if feature_cols_present else np.random.randn(100, 12)
                y_mat = df_curr[["d50_mm", "ppv_mms", "flyrock_m"]].values if "d50_mm" in df_curr.columns else np.random.randn(100, 3)

                ens_fitted = train_ensemble(X_mat, y_mat, n_models=n_members, seed=42)
                st.session_state["active_ensemble_uq"] = ens_fitted
                st.success("Ensemble retraining complete!")

    with c_uq2:
        st.subheader("📊 Ensemble Predictions & 95% Confidence Intervals")

        ens_obj = st.session_state.get("active_ensemble_uq", EnsembleUQ(n_models=5))
        uq_res = predict_ensemble_uq(ens_obj, np.array([uq_input_vec]))

        means = uq_res["mean"]
        cis = uq_res["ci_95"]
        al_dict = uq_res["aleatoric"]
        ep_dict = uq_res["epistemic"]

        m_u1, m_u2, m_u3 = st.columns(3)
        m_u1.metric(
            "Fragmentation (D80)",
            f"{means['fragmentation']:.1f} mm",
            f"95% CI: [{cis['fragmentation'][0]:.1f}, {cis['fragmentation'][1]:.1f}]",
        )
        m_u2.metric(
            "Ground Vibration (PPV)",
            f"{means['ppv']:.2f} mm/s",
            f"95% CI: [{cis['ppv'][0]:.2f}, {cis['ppv'][1]:.2f}]",
        )
        m_u3.metric(
            "Airblast Overpressure",
            f"{means['airblast']:.1f} dB",
            f"95% CI: [{cis['airblast'][0]:.1f}, {cis['airblast'][1]:.1f}]",
        )

        st.markdown("---")
        st.subheader("🚨 Epistemic Uncertainty & Out-Of-Distribution Risk Assessment")

        if uq_res.get("high_uncertainty", False):
            st.error("⚠️ **HIGH EPISTEMIC UNCERTAINTY ALERT:** Input blast parameters represent an Out-Of-Distribution (OOD) extrapolation.")
            st.warning("💡 **RECOMMENDATION:** High variance between ANN, XGBoost, RF, and PINN ensemble members. Collect field logs or apply conservative safety factors.")
        else:
            st.success("✅ **CONFIDENT PREDICTION:** High agreement between ensemble members across all 4 base architectures.")

        st.markdown("---")
        st.subheader("📉 Stacked Uncertainty Decomposition Chart")
        fig_uq = plot_uncertainty_decomposition(uq_res)
        st.plotly_chart(fig_uq, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Detailed Variance Score Table")
        df_var = pd.DataFrame({
            "Target Metric": ["Fragmentation (D80)", "Ground Vibration (PPV)", "Airblast (dB)"],
            "Mean Prediction": [means["fragmentation"], means["ppv"], means["airblast"]],
            "Aleatoric Variance (Data Noise)": [al_dict["fragmentation"], al_dict["ppv"], al_dict["airblast"]],
            "Epistemic Variance (Model Knowledge)": [ep_dict["fragmentation"], ep_dict["ppv"], ep_dict["airblast"]],
            "95% Confidence Interval": [str(cis["fragmentation"]), str(cis["ppv"]), str(cis["airblast"])],
        })
        st.dataframe(df_var, use_container_width=True)


# --- MODULE: CONVERSATIONAL AGENT ASSISTANT ---
elif active_module == "agent":
    st.header("🤖 BlasterOPT Conversational Agent Assistant")
    st.markdown(
        "Interactive AI decision support agent for open-pit diamond mining operations. "
        "Understands natural language intent, evaluates hard-coded safety guardrails, invokes physics/ML tools, "
        "and presents recommendations in plain English or Setswana."
    )

    c_ag1, c_ag2 = st.columns([1, 3])

    with c_ag1:
        st.subheader("⚙️ Agent Controls & Context")

        user_role_sel = st.selectbox(
            "User Operating Role",
            ["engineer", "blaster", "supervisor", "operator"],
            index=0,
            help="Adapts explanation complexity and available UI controls."
        )

        bench_ctx_sel = st.selectbox(
            "Active Mine Bench Context",
            ["BENCH_JWA_15S", "BENCH_JWA_12N", "BENCH_ORA_15S", "BENCH_KAR_08W"],
            index=0,
        )

        mode_sel = st.radio(
            "Interface Mode",
            ["Interactive Chat", "Guided Workflow (Non-Expert)", "Expert Technical Deep-Dive", "Voice Assistant (Offline)"],
            index=0,
        )

        st.markdown("---")
        demo_mode = st.toggle("🧪 Demo Mode (Pre-load Sample Conversation)", value=False)

        if demo_mode:
            st.session_state["chat_messages"] = [
                {"role": "user", "content": "Help me design an optimal blast for bench BENCH_JWA_15S with 10000 tonnes target."},
                {"role": "assistant", "content": "📋 **Blast Design Summary for BENCH_JWA_15S:**\n- **Powder Factor:** 0.65 kg/m³\n- **Burden x Spacing:** 6.0m x 7.0m\n- **Predicted D50:** 220 mm\n- **Predicted PPV:** 4.20 mm/s (Compliant <= 5.0 mm/s)\n- **Cost:** $4.80 / t\n\n👉 **Next Step:** Would you like me to route this design to your certified blaster for review?"},
            ]
            st.info("Sample conversation pre-loaded!")

    with c_ag2:
        if mode_sel == "Interactive Chat":
            render_agent_chat(user_role=user_role_sel, bench_id=bench_ctx_sel)
        elif mode_sel == "Guided Workflow (Non-Expert)":
            render_guided_mode(bench_id=bench_ctx_sel)
        elif mode_sel == "Expert Technical Deep-Dive":
            render_expert_mode(bench_id=bench_ctx_sel)
        elif mode_sel == "Voice Assistant (Offline)":
            render_voice_mode(user_id=f"AGENT_USER_{user_role_sel.upper()}")


# --- MODULE: AGENT AUDIT LOG ---
elif active_module == "audit_log":
    st.header("📜 Conversational Agent Regulatory Audit Log")
    st.markdown(
        "Append-only immutable audit trail recording every agent interaction, tool invocation, "
        "and operator decision/override. Mandated for 7-year regulatory retention under the "
        "**Mines, Quarries, Works and Machinery Act (Cap. 44:02)**."
    )

    tab_a1, tab_a2 = st.tabs(["💬 Conversation Interactions Log", "⚙️ Engineering Decisions & Overrides"])

    col_flt1, col_flt2 = st.columns(2)
    with col_flt1:
        filter_sess = st.text_input("Filter by Session ID", value="")
    with col_flt2:
        filter_user = st.text_input("Filter by User ID", value="")

    with tab_a1:
        st.subheader("Recent Agent Conversation Interactions")
        interactions = get_interaction_history(
            session_id=filter_sess if filter_sess else None,
            user_id=filter_user if filter_user else None,
            limit=100,
        )

        if interactions:
            df_inter = pd.DataFrame(interactions)
            st.dataframe(df_inter, use_container_width=True)

            csv_data = df_inter.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Export Interactions CSV",
                data=csv_data,
                file_name="BlasterOPT_Agent_Interactions_Audit.csv",
                mime="text/csv",
            )
        else:
            st.info("No interaction logs recorded for current filter criteria.")

    with tab_a2:
        st.subheader("Recent Engineering Decisions & Overrides")
        decisions = get_decision_history(
            session_id=filter_sess if filter_sess else None,
            user_id=filter_user if filter_user else None,
            limit=100,
        )

        if decisions:
            df_dec = pd.DataFrame(decisions)
            st.dataframe(df_dec, use_container_width=True)

            csv_dec_data = df_dec.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Export Decisions CSV",
                data=csv_dec_data,
                file_name="BlasterOPT_Agent_Decisions_Audit.csv",
                mime="text/csv",
            )
        else:
            st.info("No decision override logs recorded for current filter criteria.")

    st.markdown("---")
    st.subheader("🏛️ Regulatory Submission Package Export")
    if st.button("Generate Regulatory Audit Package (JSON Format) 🚀", type="primary"):
        export_file = export_audit_log_json()
        with open(export_file, "r", encoding="utf-8") as f:
            export_content = f.read()

        st.success(f"Audit package generated successfully at `{export_file}`!")
        st.download_button(
            label="📄 Download Official Regulatory Audit Package (.json)",
            data=export_content,
            file_name="Botswana_Mines_Act_Cap4402_Agent_Audit_Package.json",
            mime="application/json",
        )


# --- MODULE: GUARDRAIL LOG ---
elif active_module == "guardrail_log":
    st.header("🛡️ Hard-Coded Safety Guardrail Trip Log")
    st.markdown(
        "Immutable audit log recording every hard-coded safety guardrail trip event. "
        "These guardrails enforce non-negotiable safety rules (preventing unauthorized detonation, regulatory limit bypass, "
        "invented measurements, and blaster sign-off bypass)."
    )

    all_trips = get_guardrail_trips(trip_type_filter="ALL", limit=200)

    # Metrics overview
    c_g1, c_g2, c_g3, c_g4 = st.columns(4)
    c_g1.metric("Total Guardrail Trips", len(all_trips))

    type_counts = {}
    for t in all_trips:
        ttype = t.get("trip_type", "UNKNOWN")
        type_counts[ttype] = type_counts.get(ttype, 0) + 1

    c_g2.metric("Fire Blast Blocked", type_counts.get("FIRE_BLAST_REQUEST", 0))
    c_g3.metric("Limit Bypass Blocked", type_counts.get("BYPASS_LIMITS", 0))
    c_g4.metric("Regulatory Limit Violations", type_counts.get("REGULATORY_LIMIT_EXCEEDED", 0))

    st.markdown("---")
    st.subheader("📋 Guardrail Trips Table & Filtering")

    filter_options = ["ALL"] + sorted(list(type_counts.keys()))
    selected_trip_type = st.selectbox("Filter by Trip Type", filter_options)

    filtered_trips = get_guardrail_trips(trip_type_filter=selected_trip_type, limit=100)

    if filtered_trips:
        df_trips = pd.DataFrame(filtered_trips)
        st.dataframe(df_trips, use_container_width=True)
    else:
        st.info("No guardrail trips recorded for the selected filter.")


# --- MODULE 6: 2D BLAST PATTERN & DELAYS ---
elif active_module == "pattern":
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
elif active_module == "visualize":
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
