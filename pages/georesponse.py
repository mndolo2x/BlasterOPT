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

# Default MWD Input Constants
DEFAULT_INPUTS = {
    "penetration_rate_m_hr": 35.0,
    "percussion_pressure_bar": 180.0,
    "feed_pressure_bar": 90.0,
    "rotation_pressure_bar": 95.0,
    "flushing_pressure_bar": 10.0,
    "rotation_speed_rpm": 180,
    "depth_m": 3.5,
    "feed_percussion_ratio": 0.50,
}

# Physical Range Specifications for Validation
RANGE_SPECS = {
    "penetration_rate_m_hr": (5.0, 80.0, "Penetration Rate (m/hr)"),
    "percussion_pressure_bar": (100.0, 250.0, "Percussion Pressure (bar)"),
    "feed_pressure_bar": (40.0, 150.0, "Feed Pressure (bar)"),
    "rotation_pressure_bar": (40.0, 160.0, "Rotation Pressure (bar)"),
    "flushing_pressure_bar": (3.0, 20.0, "Flushing Pressure (bar)"),
    "rotation_speed_rpm": (80, 300, "Rotation Speed (rpm)"),
    "depth_m": (0.5, 10.0, "Depth (m)"),
    "feed_percussion_ratio": (0.2, 0.9, "Feed Percussion Ratio"),
}

# Exact feature order expected by trained model
EXPECTED_FEATURE_ORDER = [
    "penetration_rate_m_hr",
    "percussion_pressure_bar",
    "feed_pressure_bar",
    "rotation_pressure_bar",
    "flushing_pressure_bar",
    "rotation_speed_rpm",
    "depth_m",
    "feed_percussion_ratio",
]


# Initialize or Reset Prediction Session Inputs
def init_predict_inputs(reset=False):
    for key, val in DEFAULT_INPUTS.items():
        session_key = f"mwd_in_{key}"
        if reset or session_key not in st.session_state:
            st.session_state[session_key] = val


init_predict_inputs()

# --- TAB 1: INGESTION & VALIDATION ---
with tab1:
    st.subheader("1. Ingest MWD Telemetry Dataset")
    st.caption("Upload an MWD CSV dataset containing drilling telemetry (penetration rate, torque, WOB, pressure, rock_type_label).")

    uploaded = st.file_uploader("Upload MWD CSV File", type=["csv"], key="mwd_csv_upload")

    if uploaded is not None:
        try:
            raw_upload_df = pd.read_csv(uploaded)
            st.session_state["georesponse_raw_upload_df"] = raw_upload_df

            confirmed_maps = st.session_state.get("georesponse_confirmed_mappings", {})
            df_mwd, report = normalize_mwd_schema(raw_upload_df, user_confirmed_mappings=confirmed_maps)
            df_mwd, feat_report = compute_mwd_features(df_mwd)
            report["feature_engineering"] = feat_report
            report["license_notice"] = DATASET_LICENSE_NOTICE

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

        # Column Mapping Confirmation Interface
        sug_maps = report.get("suggested_mappings", {})
        missing_reqs = report.get("missing_required_columns", [])

        if sug_maps or missing_reqs or report.get("duplicate_mappings"):
            st.warning("⚠️ **Column Mapping Attention Required**")
            st.markdown("Some columns require explicit confirmation or user mapping to fulfill GeoResponse canonical requirements.")

            confirmed_mappings = st.session_state.get("georesponse_confirmed_mappings", {})
            user_mapping_edits = {}

            if sug_maps:
                st.write("**Suggested Unambiguous/Ambiguous Mappings:**")
                for orig_c, sug_c in sug_maps.items():
                    opt_choice = st.selectbox(
                        f"Map original column `{orig_c}` to:",
                        options=["-- Do Not Map --", sug_c] + [c for c in MINIMAL_REQUIRED_MWD_COLUMNS + OPTIONAL_MWD_COLUMNS if c != sug_c],
                        index=1,
                        key=f"mapping_select_{orig_c}"
                    )
                    if opt_choice != "-- Do Not Map --":
                        user_mapping_edits[orig_c] = opt_choice

            if missing_reqs and not sug_maps:
                st.write("**Missing Required Canonical Fields:**", missing_reqs)
                unmapped_cols = report.get("unknown_columns", [])
                if unmapped_cols:
                    for req_field in missing_reqs:
                        opt_c = st.selectbox(
                            f"Select unmapped column for required field `{req_field}`:",
                            options=["-- None --"] + unmapped_cols,
                            key=f"map_missing_{req_field}"
                        )
                        if opt_c != "-- None --":
                            user_mapping_edits[opt_c] = req_field

            if st.button("Apply & Re-validate Column Mappings", type="primary"):
                try:
                    raw_df = st.session_state.get("georesponse_raw_upload_df")
                    if raw_df is not None:
                        re_df, re_report = normalize_mwd_schema(raw_df, user_confirmed_mappings=user_mapping_edits)
                        re_df, feat_report = compute_mwd_features(re_df)
                        re_report["feature_engineering"] = feat_report

                        st.session_state["georesponse_mwd_df"] = re_df
                        st.session_state["georesponse_report"] = re_report
                        st.session_state["georesponse_confirmed_mappings"] = user_mapping_edits
                        st.success("Updated column mappings and re-validated dataset successfully!")
                        st.rerun()
                except Exception as ex:
                    st.error(f"Re-validation error: {ex}")

        st.write("**Accepted Column & Unit Schema:**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.write("• **Minimal Required Columns:**", MINIMAL_REQUIRED_MWD_COLUMNS)
            st.write("• **Missing Required Columns:**", report["missing_required_columns"] if report["missing_required_columns"] else "None ✅")
        with col_s2:
            st.write("• **Available Optional Telemetry:**", report["available_optional_columns"])
            st.write("• **Missing Optional Telemetry:**", report["missing_optional_columns"])

        if report.get("mapped_columns"):
            st.info("ℹ️ **Automatic Header Mappings:**\n" + "\n".join([f"- `{k}` → `{v}`" for k, v in report["mapped_columns"].items()]))

        if report.get("unknown_columns"):
            st.caption(f"Unmapped / Preserved Original Columns ({len(report['unknown_columns'])}): `{report['unknown_columns']}`")

        if report.get("unit_conversions"):
            st.info("ℹ️ **Unit Conversions & Operations:**\n" + "\n".join([f"- {uc}" for uc in report["unit_conversions"]]))

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
        # Provide default trained classifier for research benchmark preview if none trained in Tab 2
        default_clf = GeoResponseClassifier(model_type="random_forest")
        # Fit on dummy Norwegian research benchmark split
        dummy_X = pd.DataFrame([
            {"penetration_rate_m_hr": 35.0, "percussion_pressure_bar": 180.0, "feed_pressure_bar": 90.0, "rotation_pressure_bar": 95.0, "flushing_pressure_bar": 10.0, "rotation_speed_rpm": 180, "depth_m": 3.5, "feed_percussion_ratio": 0.50},
            {"penetration_rate_m_hr": 15.0, "percussion_pressure_bar": 220.0, "feed_pressure_bar": 130.0, "rotation_pressure_bar": 140.0, "flushing_pressure_bar": 15.0, "rotation_speed_rpm": 120, "depth_m": 8.0, "feed_percussion_ratio": 0.75},
            {"penetration_rate_m_hr": 60.0, "percussion_pressure_bar": 120.0, "feed_pressure_bar": 60.0, "rotation_pressure_bar": 60.0, "flushing_pressure_bar": 5.0, "rotation_speed_rpm": 240, "depth_m": 1.5, "feed_percussion_ratio": 0.30},
        ])
        dummy_y = ["Medium", "Hard", "Soft"]
        default_clf.fit(dummy_X[EXPECTED_FEATURE_ORDER], dummy_y)

        st.session_state["georesponse_trained_clf"] = default_clf
        st.session_state["georesponse_feature_cols"] = EXPECTED_FEATURE_ORDER

    clf = st.session_state["georesponse_trained_clf"]
    feat_cols = st.session_state.get("georesponse_feature_cols", EXPECTED_FEATURE_ORDER)

    st.subheader("Sample MWD Telemetry Inputs (All 8 Features)")

    # Row 1 Inputs
    r1_col1, r1_col2, r1_col3 = st.columns(3)
    with r1_col1:
        v_rop = st.number_input(
            "Penetration Rate (m/hr)",
            min_value=0.0, max_value=200.0,
            value=float(st.session_state.get("mwd_in_penetration_rate_m_hr", DEFAULT_INPUTS["penetration_rate_m_hr"])),
            step=1.0, key="input_rop"
        )
    with r1_col2:
        v_percussion = st.number_input(
            "Percussion Pressure (bar)",
            min_value=0.0, max_value=500.0,
            value=float(st.session_state.get("mwd_in_percussion_pressure_bar", DEFAULT_INPUTS["percussion_pressure_bar"])),
            step=5.0, key="input_percussion"
        )
    with r1_col3:
        v_flushing = st.number_input(
            "Flushing Pressure (bar)",
            min_value=0.0, max_value=100.0,
            value=float(st.session_state.get("mwd_in_flushing_pressure_bar", DEFAULT_INPUTS["flushing_pressure_bar"])),
            step=0.5, key="input_flushing"
        )

    # Row 2 Inputs
    r2_col1, r2_col2, r2_col3 = st.columns(3)
    with r2_col1:
        v_feed = st.number_input(
            "Feed Pressure (bar)",
            min_value=0.0, max_value=500.0,
            value=float(st.session_state.get("mwd_in_feed_pressure_bar", DEFAULT_INPUTS["feed_pressure_bar"])),
            step=5.0, key="input_feed"
        )
    with r2_col2:
        v_rotation = st.number_input(
            "Rotation Pressure (bar)",
            min_value=0.0, max_value=500.0,
            value=float(st.session_state.get("mwd_in_rotation_pressure_bar", DEFAULT_INPUTS["rotation_pressure_bar"])),
            step=5.0, key="input_rotation_p"
        )
    with r2_col3:
        v_rpm = st.number_input(
            "Rotation Speed (rpm)",
            min_value=0, max_value=1000,
            value=int(st.session_state.get("mwd_in_rotation_speed_rpm", DEFAULT_INPUTS["rotation_speed_rpm"])),
            step=5, key="input_rpm"
        )

    # Row 3 Inputs
    r3_col1, r3_col2, r3_col3 = st.columns(3)
    with r3_col1:
        v_depth = st.number_input(
            "Depth (m)",
            min_value=0.0, max_value=50.0,
            value=float(st.session_state.get("mwd_in_depth_m", DEFAULT_INPUTS["depth_m"])),
            step=0.5, key="input_depth"
        )
    with r3_col2:
        v_fpr = st.number_input(
            "Feed Percussion Ratio",
            min_value=0.0, max_value=5.0,
            value=float(st.session_state.get("mwd_in_feed_percussion_ratio", DEFAULT_INPUTS["feed_percussion_ratio"])),
            step=0.05, key="input_fpr"
        )
    with r3_col3:
        st.write("")
        st.write("")
        if st.button("Reset to Defaults"):
            init_predict_inputs(reset=True)
            st.rerun()

    current_input_dict = {
        "penetration_rate_m_hr": v_rop,
        "percussion_pressure_bar": v_percussion,
        "feed_pressure_bar": v_feed,
        "rotation_pressure_bar": v_rotation,
        "flushing_pressure_bar": v_flushing,
        "rotation_speed_rpm": v_rpm,
        "depth_m": v_depth,
        "feed_percussion_ratio": v_fpr,
    }

    # Non-blocking Range Validation
    range_warnings = []
    for k, val in current_input_dict.items():
        min_v, max_v, label_name = RANGE_SPECS[k]
        if val < min_v or val > max_v:
            range_warnings.append(f"Warning: {label_name} ({val}) is outside the typical range [{min_v}, {max_v}] for the training data. Predictions may be unreliable.")

    if range_warnings:
        for w in range_warnings:
            st.warning(w)

    st.divider()

    if st.button("Predict Rock Type", type="primary"):
        # Task 4: Confirm Feature Order Matches the Model
        # CRITICAL: Build input DataFrame using the exact expected feature order below:
        # ['penetration_rate_m_hr', 'percussion_pressure_bar', 'feed_pressure_bar',
        #  'rotation_pressure_bar', 'flushing_pressure_bar', 'rotation_speed_rpm',
        #  'depth_m', 'feed_percussion_ratio']
        # The model's Scikit-learn estimator relies on exact column order alignment during matrix evaluation.
        input_data_ordered = {col: [current_input_dict[col]] for col in EXPECTED_FEATURE_ORDER}
        sample_X = pd.DataFrame(input_data_ordered)

        try:
            review_df = clf.predict_with_review_flags(sample_X)
            st.session_state["sample_review_df"] = review_df
            st.session_state["sample_used_inputs"] = current_input_dict
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Task 5: Display the Input Summary Card After Prediction
    if "sample_review_df" in st.session_state and "sample_used_inputs" in st.session_state:
        rdf = st.session_state["sample_review_df"]
        used_inputs = st.session_state["sample_used_inputs"]

        p_class = rdf["predicted_rock_type"].iloc[0]
        conf = rdf["confidence_score"].iloc[0]
        review_flag = bool(rdf["requires_review_flag"].iloc[0])
        warn = rdf["review_warning"].iloc[0]

        st.subheader("🎯 Prediction Results & Input Summary Card")

        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Predicted Rock Class", p_class)
        col_res2.metric("Confidence Score", f"{conf * 100:.1f}%")
        col_res3.metric("Review Flag Status", "TRIGGERED ⚠️" if review_flag else "CLEAR ✅")

        if conf < 0.70:
            st.warning(warn)
        else:
            st.success(warn)

        st.markdown("### 📋 Inputs Used for Model Prediction")
        df_summary = pd.DataFrame([used_inputs]).T.reset_index()
        df_summary.columns = ["MWD Feature Name", "Value Used"]
        st.dataframe(df_summary, use_container_width=True)

        st.caption("⚠️ **Operational Notice:** Predictions must be reviewed by the certified pit geologist before adjusting blast pattern charging.")
