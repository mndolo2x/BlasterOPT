"""
GeoResponse v0.1 Streamlit Module: MWD Ingestion & Rock-Type Classification Benchmark.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.georesponse.config import DATASET_LICENSE_NOTICE, MINIMAL_REQUIRED_MWD_COLUMNS, OPTIONAL_MWD_COLUMNS
from src.georesponse.ingestion import load_and_validate_csv, normalize_mwd_schema
from src.georesponse.features import prepare_feature_matrix, compute_mwd_features
from src.georesponse.models import GeoResponseClassifier
from src.georesponse.evaluation import evaluate_rock_type_classifier

st.title("🪨 GeoResponse v0.1 — MWD Rock-Type Classifier")

st.warning("⚠️ **RESEARCH PROTOTYPE — Trained on Norwegian tunnel drilling data. Not calibrated for Botswana open-pit geology. Do not use for operational decisions.**")

tab1, tab2, tab3 = st.tabs(["📁 Dataset Ingestion & Validation", "📊 Group CV Model Evaluation", "🔍 Prediction & Review Flags"])

# --- TAB 1: INGESTION & VALIDATION ---
with tab1:
    st.subheader("1. Ingest MWD Telemetry Dataset")
    st.caption("Upload an MWD CSV dataset containing drilling telemetry (penetration rate, torque, WOB, pressure, rock_type_label).")

    uploaded = st.file_uploader("Upload MWD CSV File", type=["csv"], key="mwd_csv_upload")

    if uploaded is not None:
        try:
            df_mwd, report = load_and_validate_csv(uploaded)
            df_mwd, feat_report = compute_mwd_features(df_mwd)
            report["feature_engineering"] = feat_report

            st.session_state["georesponse_mwd_df"] = df_mwd
            st.session_state["georesponse_report"] = report

            if report["is_eligible_for_training"]:
                st.success(f"✅ Loaded {report['num_rows']} rows × {report['num_cols']} columns. Provenance SHA-256: `{report['provenance_hash']}`")
            else:
                st.error("❌ Dataset failed schema validation for model training. Review validation details below.")
        except Exception as e:
            st.error(f"Failed to process MWD dataset: {e}")

    if "georesponse_mwd_df" in st.session_state:
        df_mwd = st.session_state["georesponse_mwd_df"]
        report = st.session_state["georesponse_report"]

        st.divider()
        st.subheader("Automated Validation & Schema Report")

        st.text(report["license_notice"])

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Rows Ingested", report["num_rows"])
        col_r2.metric("Training Eligible", "YES ✅" if report["is_eligible_for_training"] else "NO ❌")
        col_r3.metric("Distinct Rock Classes", len(report["class_distribution"]))

        st.write("**Accepted Column & Unit Schema:**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.write("• **Minimal Required Columns:**", MINIMAL_REQUIRED_MWD_COLUMNS)
            st.write("• **Missing Required Columns:**", report["missing_required_columns"] if report["missing_required_columns"] else "None ✅")
        with col_s2:
            st.write("• **Available Optional Telemetry:**", report["available_optional_columns"])
            st.write("• **Missing Optional Telemetry:**", report["missing_optional_columns"])

        if report.get("unit_conversions"):
            st.info("ℹ️ **Unit Conversions & Column Mapping:**\n" + "\n".join([f"- {uc}" for uc in report["unit_conversions"]]))

        feat_rep = report.get("feature_engineering", {})
        if feat_rep:
            st.write("**Derived MWD Feature Engineering Status:**")
            if feat_rep.get("computed_features"):
                st.success(f"Calculated Features: `{feat_rep['computed_features']}`")
            if feat_rep.get("skipped_features"):
                st.warning(f"Skipped Features: `{feat_rep['skipped_features']}`")
                st.json(feat_rep.get("skip_reasons", {}))

        if report["errors"]:
            for err in report["errors"]:
                st.error(f"❌ {err}")

        if report["warnings"]:
            for w in report["warnings"]:
                st.warning(f"⚠️ {w}")

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
        report = st.session_state.get("georesponse_report", {})

        if not report.get("is_eligible_for_training", True):
            st.error("❌ The loaded MWD dataset is not eligible for model training due to schema validation errors. Resolve dataset errors in Tab 1.")
        else:
            group_col_options = [c for c in ["tunnel_id", "blast_id", "hole_id"] if c in df_mwd.columns]
            selected_group_col = st.selectbox("Group Identifier for CV Split", options=group_col_options if group_col_options else [None])

            target_col = "rock_type_label" if "rock_type_label" in df_mwd.columns else ("rock_class" if "rock_class" in df_mwd.columns else None)

            if st.button("Run Group Cross-Validation Benchmark", type="primary"):
                try:
                    with st.spinner("Engineering features and evaluating GroupKFold model performance..."):
                        X, feat_cols, feat_rep = prepare_feature_matrix(df_mwd)
                        df_eval = df_mwd.copy()
                        for fc in feat_cols:
                            df_eval[fc] = X[fc]

                        eval_results = evaluate_rock_type_classifier(
                            df=df_eval,
                            feature_cols=feat_cols,
                            target_col=target_col,
                            group_col=selected_group_col,
                            n_splits=5,
                        )
                        st.session_state["georesponse_eval_results"] = eval_results

                        # Train full Random Forest model for inference
                        clf = GeoResponseClassifier(model_type="random_forest")
                        clf.fit(X, df_mwd[target_col])
                        st.session_state["georesponse_trained_clf"] = clf
                        st.session_state["georesponse_feature_cols"] = feat_cols
                except Exception as e:
                    st.error(f"Error during model cross-validation benchmark: {e}")

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
        st.caption(f"Required features for active trained model: `{feat_cols}`")

        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            rop = st.number_input("Penetration Rate (m/hr)", 1.0, 120.0, 35.0)
            torque = st.number_input("Torque (N·m)", 100.0, 5000.0, 1200.0) if "torque_nm" in feat_cols else None
        with col_i2:
            wob = st.number_input("Weight on Bit (kg)", 500.0, 25000.0, 8000.0) if "weight_on_bit_kg" in feat_cols else None
            percussion = st.number_input("Percussion Pressure (bar)", 50.0, 300.0, 180.0) if "percussion_pressure_bar" in feat_cols else None
        with col_i3:
            flushing = st.number_input("Flushing Pressure (bar)", 2.0, 30.0, 10.0) if "flushing_pressure_bar" in feat_cols else None

        sample_input_dict = {"penetration_rate_m_hr": rop}
        if torque is not None:
            sample_input_dict["torque_nm"] = torque
        if wob is not None:
            sample_input_dict["weight_on_bit_kg"] = wob
        if percussion is not None:
            sample_input_dict["percussion_pressure_bar"] = percussion
        if flushing is not None:
            sample_input_dict["flushing_pressure_bar"] = flushing

        sample_raw = pd.DataFrame([sample_input_dict])

        if st.button("Predict Rock Type"):
            try:
                sample_X, _, _ = prepare_feature_matrix(sample_raw, required_features=feat_cols)
                review_df = clf.predict_with_review_flags(sample_X)
                st.session_state["sample_review_df"] = review_df
            except Exception as e:
                st.error(f"Prediction failed: {e}")

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
