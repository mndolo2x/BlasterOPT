"""
GeoResponse v0.1 Streamlit Module: MWD Ingestion & Rock-Type Classification Benchmark.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.components.model_selector import render_page_model_selector
from src.georesponse.config import DATASET_LICENSE_NOTICE, REQUIRED_MWD_COLUMNS
from src.georesponse.ingestion import load_and_validate_csv
from src.georesponse.features import prepare_feature_matrix
from src.georesponse.models import GeoResponseClassifier
from src.georesponse.evaluation import evaluate_rock_type_classifier

st.title("🪨 GeoResponse v0.1 — MWD Rock-Type Classifier")

st.warning("⚠️ **RESEARCH BENCHMARK NOTICE:** GeoResponse operates in research mode. Predictions do NOT automatically modify operational blast designs.")

tab1, tab2, tab3 = st.tabs(["📁 Dataset Ingestion & Validation", "📊 Group CV Model Evaluation", "🔍 Prediction & Review Flags"])

# --- TAB 1: INGESTION & VALIDATION ---
with tab1:
    st.subheader("1. Ingest MWD Telemetry Dataset")
    st.caption("Upload an MWD CSV dataset containing drilling telemetry (penetration rate, torque, WOB, pressure, rock_type_label).")

    uploaded = st.file_uploader("Upload MWD CSV File", type=["csv"], key="mwd_csv_upload")

    if uploaded is not None:
        try:
            df_mwd, report = load_and_validate_csv(uploaded)
            st.session_state["georesponse_mwd_df"] = df_mwd
            st.session_state["georesponse_report"] = report

            st.success(f"✅ Loaded {report['num_rows']} rows × {report['num_cols']} columns. Provenance SHA-256: `{report['provenance_hash']}`")
        except Exception as e:
            st.error(f"Failed to load MWD dataset: {e}")

    if "georesponse_mwd_df" in st.session_state:
        df_mwd = st.session_state["georesponse_mwd_df"]
        report = st.session_state["georesponse_report"]

        st.divider()
        st.subheader("Automated Validation Report")

        st.text(report["license_notice"])

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Rows Ingested", report["num_rows"])
        col_r2.metric("Required Columns", f"{len(REQUIRED_MWD_COLUMNS) - len(report['missing_columns'])} / {len(REQUIRED_MWD_COLUMNS)}")
        col_r3.metric("Distinct Rock Classes", len(report["class_distribution"]))

        if report["warnings"]:
            for w in report["warnings"]:
                st.warning(w)

        st.write("**Class Distribution:**")
        st.json(report["class_distribution"])

        st.dataframe(df_mwd.head(10), use_container_width=True)


# --- TAB 2: TRAINING & EVALUATION ---
with tab2:
    st.subheader("2. Model Benchmark & Group-Based Validation")
    st.caption("Executes GroupKFold cross-validation grouped by tunnel/blast ID to prevent observation leakage across splits.")

    if "georesponse_mwd_df" not in st.session_state:
        st.info("Upload an MWD dataset in Tab 1 to run model benchmarking.")
    else:
        df_mwd = st.session_state["georesponse_mwd_df"]

        group_col_options = [c for c in ["tunnel_id", "blast_id", "hole_id"] if c in df_mwd.columns]
        selected_group_col = st.selectbox("Group Identifier for CV Split", options=group_col_options if group_col_options else [None])

        if st.button("Run Group Cross-Validation Benchmark", type="primary"):
            with st.spinner("Engineering features and evaluating GroupKFold model performance..."):
                X, feat_cols = prepare_feature_matrix(df_mwd)
                df_eval = df_mwd.copy()
                for fc in feat_cols:
                    df_eval[fc] = X[fc]

                eval_results = evaluate_rock_type_classifier(
                    df=df_eval,
                    feature_cols=feat_cols,
                    group_col=selected_group_col,
                    n_splits=5,
                )
                st.session_state["georesponse_eval_results"] = eval_results

                # Train full Random Forest model for inference
                clf = GeoResponseClassifier(model_type="random_forest")
                clf.fit(X, df_mwd["rock_type_label"])
                st.session_state["georesponse_trained_clf"] = clf
                st.session_state["georesponse_feature_cols"] = feat_cols

        if "georesponse_eval_results" in st.session_state:
            res = st.session_state["georesponse_eval_results"]

            st.divider()
            st.subheader("Validation Results")
            st.info(f"**Validation Strategy:** `{res['validation_strategy']}` | **Samples:** {res['n_samples']}")

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Random Forest Macro-F1", f"{res['random_forest']['macro_f1']:.4f}")
            col_m2.metric("Majority Baseline Macro-F1", f"{res['majority_baseline']['macro_f1']:.4f}")
            col_m3.metric("F1 Delta vs Baseline", f"+{res['f1_improvement_over_baseline']:.4f}")

            st.write("**Per-Class Performance Breakdown:**")
            st.dataframe(pd.DataFrame(res["random_forest"]["per_class"]).T, use_container_width=True)

            st.write("**Confusion Matrix:**")
            cm_df = pd.DataFrame(
                res["random_forest"]["confusion_matrix"],
                index=[f"True: {c}" for c in res["class_labels"]],
                columns=[f"Pred: {c}" for c in res["class_labels"]],
            )
            st.dataframe(cm_df, use_container_width=True)


# --- TAB 3: PREDICTION REVIEW ---
with tab3:
    st.subheader("3. MWD Prediction & Review Warning Flags")
    st.caption("Review rock-type predictions on new MWD samples. Predictions below 70% confidence trigger review warnings.")

    if "georesponse_trained_clf" not in st.session_state:
        st.info("Run model cross-validation benchmark in Tab 2 to enable prediction review.")
    else:
        clf = st.session_state["georesponse_trained_clf"]
        feat_cols = st.session_state["georesponse_feature_cols"]

        st.subheader("Sample Input Telemetry")
        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            rop = st.number_input("Penetration Rate (m/hr)", 1.0, 120.0, 35.0)
            torque = st.number_input("Torque (N·m)", 100.0, 5000.0, 1200.0)
        with col_i2:
            wob = st.number_input("Weight on Bit (kg)", 500.0, 25000.0, 8000.0)
            percussion = st.number_input("Percussion Pressure (bar)", 50.0, 300.0, 180.0)
        with col_i3:
            flushing = st.number_input("Flushing Pressure (bar)", 2.0, 30.0, 10.0)

        sample_raw = pd.DataFrame([{
            "penetration_rate_m_hr": rop,
            "torque_nm": torque,
            "weight_on_bit_kg": wob,
            "percussion_pressure_bar": percussion,
            "flushing_pressure_bar": flushing,
        }])

        sample_X, _ = prepare_feature_matrix(sample_raw)

        if st.button("Predict Rock Type"):
            review_df = clf.predict_with_review_flags(sample_X)
            st.session_state["sample_review_df"] = review_df

        if "sample_review_df" in st.session_state:
            rdf = st.session_state["sample_review_df"]
            p_class = rdf["predicted_rock_type"].iloc[0]
            conf = rdf["confidence_score"].iloc[0]
            warn = rdf["review_warning"].iloc[0]

            st.divider()
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Predicted Rock Type", p_class)
            col_res2.metric("Model Confidence", f"{conf*100:.1f}%")

            if conf < 0.70:
                st.warning(warn)
            else:
                st.success(warn)

            st.caption("⚠️ **Operational Notice:** Predictions must be reviewed by the certified pit geologist before adjusting blast pattern charging.")
