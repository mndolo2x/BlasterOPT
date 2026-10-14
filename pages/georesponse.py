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
from src.ui.geo_response_predict import (
    REQUIRED_FEATURE_ORDER,
    DEFAULT_MWD_INPUTS,
    FIELD_DISPLAY_NAMES,
    FEATURE_RANGES,
    validate_mwd_input_ranges,
    build_ordered_feature_dataframe,
    get_or_create_default_classifier,
)

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

    # Initialize default input values in session state
    for k, v in DEFAULT_MWD_INPUTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Retrieve or create active classifier
    clf = st.session_state.get("georesponse_trained_clf")
    if clf is None:
        clf = get_or_create_default_classifier()
        st.caption("ℹ️ *Using default Norwegian MWD rock-type classifier (run Tab 2 Group CV to re-train on custom dataset).*")

    st.subheader("Sample Input Telemetry")
    st.caption(f"Required features for active trained model: `{REQUIRED_FEATURE_ORDER}`")

    # Row 1
    col_r1_1, col_r1_2, col_r1_3 = st.columns(3)
    with col_r1_1:
        rop = st.number_input(
            "Penetration Rate (m/hr)",
            min_value=1.0,
            max_value=120.0,
            value=float(st.session_state["penetration_rate_m_hr"]),
            step=1.0,
            key="input_penetration_rate_m_hr",
        )
    with col_r1_2:
        percussion = st.number_input(
            "Percussion Pressure (bar)",
            min_value=50.0,
            max_value=300.0,
            value=float(st.session_state["percussion_pressure_bar"]),
            step=5.0,
            key="input_percussion_pressure_bar",
        )
    with col_r1_3:
        flushing = st.number_input(
            "Flushing Pressure (bar)",
            min_value=2.0,
            max_value=30.0,
            value=float(st.session_state["flushing_pressure_bar"]),
            step=1.0,
            key="input_flushing_pressure_bar",
        )

    # Row 2
    col_r2_1, col_r2_2, col_r2_3 = st.columns(3)
    with col_r2_1:
        feed_press = st.number_input(
            "Feed Pressure (bar)",
            min_value=10.0,
            max_value=250.0,
            value=float(st.session_state["feed_pressure_bar"]),
            step=5.0,
            key="input_feed_pressure_bar",
        )
    with col_r2_2:
        rot_press = st.number_input(
            "Rotation Pressure (bar)",
            min_value=10.0,
            max_value=250.0,
            value=float(st.session_state["rotation_pressure_bar"]),
            step=5.0,
            key="input_rotation_pressure_bar",
        )
    with col_r2_3:
        rot_speed = st.number_input(
            "Rotation Speed (rpm)",
            min_value=50,
            max_value=400,
            value=int(st.session_state["rotation_speed_rpm"]),
            step=5,
            key="input_rotation_speed_rpm",
        )

    # Row 3
    col_r3_1, col_r3_2, _ = st.columns(3)
    with col_r3_1:
        depth = st.number_input(
            "Depth (m)",
            min_value=0.1,
            max_value=20.0,
            value=float(st.session_state["depth_m"]),
            step=0.1,
            key="input_depth_m",
        )
    with col_r3_2:
        feed_perc_ratio = st.number_input(
            "Feed Percussion Ratio",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state["feed_percussion_ratio"]),
            step=0.05,
            key="input_feed_percussion_ratio",
        )

    # Map current input values
    current_inputs = {
        "penetration_rate_m_hr": float(rop),
        "percussion_pressure_bar": float(percussion),
        "feed_pressure_bar": float(feed_press),
        "rotation_pressure_bar": float(rot_press),
        "flushing_pressure_bar": float(flushing),
        "rotation_speed_rpm": float(rot_speed),
        "depth_m": float(depth),
        "feed_percussion_ratio": float(feed_perc_ratio),
    }

    # Synchronize session state
    for k, v in current_inputs.items():
        st.session_state[k] = v

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("Reset to Defaults"):
            for k, v in DEFAULT_MWD_INPUTS.items():
                st.session_state[k] = v
                if f"input_{k}" in st.session_state:
                    st.session_state[f"input_{k}"] = v
            st.rerun()

    if st.button("Predict Rock Type", type="primary"):
        # Validate ranges
        range_warnings = validate_mwd_input_ranges(current_inputs)
        for w in range_warnings:
            st.warning(w)

        try:
            sample_X = build_ordered_feature_dataframe(current_inputs)
            review_df = clf.predict_with_review_flags(sample_X)
            st.session_state["sample_review_df"] = review_df
            st.session_state["sample_inputs_used"] = current_inputs
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if "sample_review_df" in st.session_state and "sample_inputs_used" in st.session_state:
        rdf = st.session_state["sample_review_df"]
        inputs_used = st.session_state["sample_inputs_used"]
        p_class = rdf["predicted_rock_type"].iloc[0]
        conf = rdf["confidence_score"].iloc[0]
        review_flag = rdf["requires_review_flag"].iloc[0]
        warn_msg = rdf["review_warning"].iloc[0]

        st.divider()
        st.subheader("📋 MWD Prediction Summary Card")

        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Predicted Rock Type", p_class)
        col_res2.metric("Model Confidence", f"{conf * 100:.1f}%")
        col_res3.metric("Review Flag Status", "TRIGGERED ⚠️" if review_flag else "NOT TRIGGERED ✅")

        if review_flag:
            st.warning(warn_msg)
        else:
            st.success(warn_msg)

        st.write("**Eight Input Values Used for Prediction:**")
        summary_df = pd.DataFrame([
            {
                "Feature Name": k,
                "Display Label": FIELD_DISPLAY_NAMES.get(k, k),
                "Input Value": v,
                "Expected Range": f"{FEATURE_RANGES[k][0]} – {FEATURE_RANGES[k][1]}",
            }
            for k, v in inputs_used.items()
        ])
        st.dataframe(summary_df, use_container_width=True)

        st.caption("⚠️ **Operational Notice:** Predictions must be reviewed by the certified pit geologist before adjusting blast pattern charging.")
