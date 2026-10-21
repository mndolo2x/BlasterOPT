"""
Blast Performance Feedback Streamlit Page.

Provides 5 connected workflow sections for certified mining engineers:
1. Blast Register — Search, filter, inspect, and create blast records (open-pit & underground).
2. Actual Results — Download templates, upload CSV/Excel files, map headers, validate, resolve conflicts, and persist measurements.
3. Prediction Comparison — Compare original prediction snapshots vs measured actuals with MAE, RMSE, Bias, 1:1 scatter, residual plots, and timelines.
4. Historical Analytics — Interactive performance trends, error distributions, model version comparisons, and sample sufficiency warnings.
5. Dataset Review — Review workflow (Uploaded -> Validated -> Awaiting Review -> Approved/Rejected), target-specific CSV export with manifest, and model retraining integration.
"""

import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone

import src.performance_feedback as pf

# Page CSS Override for Enterprise Dark Navy / Gold Styling
st.markdown("""
<style>
    [data-baseweb="tab-highlight"] {
        background-color: #F2A900 !important;
    }
    [data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    [aria-selected="true"] {
        color: #ffffff !important;
    }
    .stButton > button {
        background-color: #002B5B !important;
        color: #F2A900 !important;
        border: 1px solid #F2A900 !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        background-color: #F2A900 !important;
        color: #002B5B !important;
    }
    [data-testid="stDataFrame"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="color: #ffffff; font-weight: 700; margin-bottom: 0;">💥 Blast Performance Feedback Loop</h1>
<p style="color: #94a3b8; font-size: 16px; margin-top: 4px;">Record executed blasts, upload actual post-blast measurements, compare against original predictions, evaluate accuracy, and review data for model retraining.</p>
<hr style="border-color: #1e293b; margin: 16px 0 24px 0;">
""", unsafe_allow_html=True)

# 5 Main Connected Workflow Tabs
tab_reg, tab_upload, tab_comp, tab_analytics, tab_review = st.tabs([
    "📋 1. Blast Register",
    "📤 2. Actual Results Upload",
    "📊 3. Prediction Comparison",
    "📈 4. Historical Analytics",
    "✅ 5. Dataset Review & Retraining",
])


# ==============================================================================
# SECTION 1: BLAST REGISTER
# ==============================================================================
with tab_reg:
    st.subheader("📋 Blast Register (Open-Pit & Underground Operations)")
    st.markdown("Search, filter, manage, and inspect immutable blast records across mining sites.")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        site_filter = st.selectbox("Site Filter", ["ALL", "Jwaneng", "Orapa", "Morupule", "Karowe"], index=0, key="reg_site_f")
    with col_f2:
        op_filter = st.selectbox("Operation Type", ["ALL", "open_pit", "underground"], index=0, key="reg_op_f")
    with col_f3:
        status_filter = st.selectbox("Status Filter", ["ALL", "PLANNED", "EXECUTED", "MEASURED", "REVIEWED"], index=0, key="reg_status_f")
    with col_f4:
        search_query = st.text_input("Search Blast ID / Location", value="", key="reg_search")

    records = pf.list_blast_records(
        site_id=site_filter,
        operation_type=op_filter,
        status=status_filter,
    )

    if search_query.strip():
        sq = search_query.strip().lower()
        records = [r for r in records if sq in r.blast_id.lower() or sq in r.location_identifier.lower()]

    st.markdown(f"**Found {len(records)} blast record(s)**")

    if records:
        reg_table_data = []
        for r in records:
            reg_table_data.append({
                "Blast ID": r.blast_id,
                "Site": r.site_id,
                "Operation": r.operation_type,
                "Location / Bench": r.location_identifier,
                "Execution Date": r.blast_date[:10],
                "Status": r.status,
                "Engineer": r.engineer_id,
                "Design ID": r.design_id or "N/A",
            })
        st.dataframe(pd.DataFrame(reg_table_data), use_container_width=True, hide_index=True)
    else:
        st.info("No blast records found matching criteria. Use the form below to create a new record.")

    st.markdown("---")
    with st.expander("➕ Register New Blast Execution Record", expanded=False):
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            new_blast_id = st.text_input("Unique Blast ID", value=f"BLAST_JWA_2026_{int(datetime.now(timezone.utc).timestamp()) % 1000:03d}")
            new_site = st.selectbox("Site", ["Jwaneng", "Orapa", "Morupule", "Karowe"])
            new_op = st.selectbox("Operation Type", ["open_pit", "underground"])
            new_location = st.text_input("Location / Bench / Stope ID", value="BENCH_JWA_15S")
        with c_r2:
            new_date = st.date_input("Blast Execution Date", value=datetime.now(timezone.utc))
            new_engineer = st.text_input("Engineer ID", value="ENGINEER_BOTSWANA_01")
            new_notes = st.text_area("Operational Notes", value="Standard production blast in medium Kimberlite.")

        if st.button("Save Blast Record", type="primary", key="btn_save_blast"):
            rec_obj = pf.BlastRecord(
                blast_id=new_blast_id,
                site_id=new_site,
                operation_type=new_op,
                location_identifier=new_location,
                blast_date=new_date.isoformat(),
                engineer_id=new_engineer,
                status="PLANNED",
                planned_params=st.session_state.get("last_predict_inputs", {}),
                geological_context={"rock_type": "Kimberlite", "rock_factor_A": 8.0, "RMR": 65},
                notes=new_notes,
            )
            pf.save_blast_record(rec_obj)
            st.success(f"✅ Blast record `{new_blast_id}` created and persisted!")
            st.rerun()


# ==============================================================================
# SECTION 2: ACTUAL RESULTS UPLOAD WORKFLOW
# ==============================================================================
with tab_upload:
    st.subheader("📤 Post-Blast Actual Results Upload & Ingestion Pipeline")
    st.markdown("Upload field observations (photogrammetry P50/P80, seismograph PPV, airblast overpressure, costs, MWD telemetry).")

    col_u1, col_u2 = st.columns([1, 2])

    with col_u1:
        st.subheader("1. Download Templates")
        template_target = st.selectbox("Select Target Metric Template", ["d50_mm", "fragmentation_d80_cm", "ppv_mms", "airblast_db", "cost_per_tonne_usd"])
        sample_tmpl_df = pf.generate_sample_template(target_metric=template_target)

        csv_tmpl_buf = io.StringIO()
        sample_tmpl_df.to_csv(csv_tmpl_buf, index=False)
        st.download_button(
            label=f"📄 Download {template_target} CSV Template",
            data=csv_tmpl_buf.getvalue(),
            file_name=f"Actual_Results_Template_{template_target}.csv",
            mime="text/csv",
        )

    with col_u2:
        st.subheader("2. Upload & Map Actual Results File")
        uploaded_actuals_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

        if uploaded_actuals_file is not None:
            try:
                if uploaded_actuals_file.name.endswith(".xlsx"):
                    raw_up_df = pd.read_excel(uploaded_actuals_file)
                else:
                    raw_up_df = pd.read_csv(uploaded_actuals_file)

                st.write("**Uploaded File Preview:**")
                st.dataframe(raw_up_df.head(5), use_container_width=True, hide_index=True)

                st.subheader("3. Inspect & Confirm Field Mappings")
                candidate_maps = pf.identify_candidate_mappings(list(raw_up_df.columns))

                confirmed_maps: Dict[str, str] = {}
                m_c1, m_c2 = st.columns(2)
                for i, (orig_col, suggested_col) in enumerate(candidate_maps.items()):
                    with (m_c1 if i % 2 == 0 else m_c2):
                        confirmed_maps[orig_col] = st.text_input(
                            f"Column '{orig_col}' ->",
                            value=suggested_col,
                            key=f"map_col_{orig_col}",
                        )

                st.subheader("4. Validate & Process Results")
                if st.button("Validate Actual Results", type="primary", key="btn_val_actuals"):
                    valid_meass, row_errs, dup_conflicts = pf.validate_actual_results_df(
                        df=raw_up_df, column_mappings=confirmed_maps
                    )

                    st.session_state["pending_valid_meass"] = valid_meass
                    st.session_state["pending_row_errs"] = row_errs
                    st.session_state["pending_dup_conflicts"] = dup_conflicts

                if "pending_valid_meass" in st.session_state:
                    valid_meass = st.session_state["pending_valid_meass"]
                    row_errs = st.session_state.get("pending_row_errs", [])
                    dup_conflicts = st.session_state.get("pending_dup_conflicts", [])

                    st.info(f"✅ Valid Measurements: {len(valid_meass)} | ⚠️ Validation Errors: {len(row_errs)} | ⚔️ Duplicate Conflicts: {len(dup_conflicts)}")

                    if row_errs:
                        st.error("Row-Level Validation Errors Detected:")
                        st.dataframe(pd.DataFrame(row_errs), use_container_width=True, hide_index=True)

                    if dup_conflicts:
                        st.warning("Duplicate / Conflicting Records Detected:")
                        st.dataframe(pd.DataFrame(dup_conflicts), use_container_width=True, hide_index=True)

                    if valid_meass:
                        if st.button("Confirm Import & Persist to Database 🚀", type="primary", key="btn_confirm_import"):
                            for m in valid_meass:
                                pf.save_actual_measurement(m)
                                # Update blast record status to MEASURED
                                b_rec = pf.get_blast_record(m.blast_id)
                                if b_rec:
                                    pf.save_blast_record(b_rec.update_snapshot(status="MEASURED"))

                                # Create / Update Awaiting Review Dataset Review Record
                                rev_record = pf.DatasetReviewRecord(
                                    review_id=f"REV_{m.blast_id}_{m.target_metric}",
                                    blast_id=m.blast_id,
                                    target_metric=m.target_metric,
                                    status="Awaiting Review",
                                    inclusion_status=False,
                                )
                                pf.save_dataset_review(rev_record)

                                # Calculate comparison if original prediction snapshot exists
                                pred_snaps = pf.get_prediction_snapshots_for_blast(m.blast_id, target_metric=m.target_metric)
                                if pred_snaps:
                                    snap = pred_snaps[0]
                                    signed_e = snap.predicted_value - float(m.measured_value)
                                    abs_e = abs(signed_e)
                                    pct_e = (abs_e / abs(float(m.measured_value))) * 100.0 if abs(float(m.measured_value)) > 1e-6 else None

                                    comp_res = pf.ComparisonResult(
                                        comparison_id=f"COMP_{m.blast_id}_{m.target_metric}",
                                        blast_id=m.blast_id,
                                        prediction_run_id=snap.prediction_run_id,
                                        measurement_id=m.measurement_id,
                                        target_metric=m.target_metric,
                                        predicted_value=snap.predicted_value,
                                        measured_value=float(m.measured_value),
                                        signed_error=round(signed_e, 4),
                                        absolute_error=round(abs_e, 4),
                                        percentage_error=round(pct_e, 2) if pct_e is not None else None,
                                        unit=m.unit,
                                        model_name=snap.model_name,
                                        model_version=snap.model_version,
                                    )
                                    pf.save_comparison_result(comp_res)

                            st.success(f"Successfully imported and persisted {len(valid_meass)} actual measurement records!")
                            del st.session_state["pending_valid_meass"]
                            st.rerun()

            except Exception as exc:
                st.error(f"Error processing actual results file: {exc}")


# ==============================================================================
# SECTION 3: PREDICTION COMPARISON
# ==============================================================================
with tab_comp:
    st.subheader("📊 Original Prediction vs Actual Result Comparison")
    st.markdown("Directly evaluate model error ($e_i = \\hat{y}_i - y_i$), MAE, RMSE, and Bias across comparable blast targets.")

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        comp_target = st.selectbox("Comparison Target Metric", ["ALL", "d50_mm", "fragmentation_d80_cm", "ppv_mms", "airblast_db", "cost_per_tonne_usd"], index=1, key="comp_target_s")
    with col_c2:
        comp_site = st.selectbox("Site Filter", ["ALL", "Jwaneng", "Orapa", "Morupule"], index=0, key="comp_site_s")
    with col_c3:
        comp_model = st.selectbox("Model Filter", ["ALL", "GA-ANN Model", "ANN-RF Ensemble", "PINN Model"], index=0, key="comp_model_s")

    comparisons = pf.list_comparison_results(
        site_id=comp_site,
        target_metric=comp_target,
        model_name=comp_model,
    )

    metrics_eval = pf.evaluate_comparisons(comparisons)

    # Metric KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Comparable Blasts", metrics_eval["count"])
    k2.metric("MAE", f"{metrics_eval['mae']:.2f}" if metrics_eval['mae'] is not None else "N/A")
    k3.metric("RMSE", f"{metrics_eval['rmse']:.2f}" if metrics_eval['rmse'] is not None else "N/A")
    k4.metric("Bias (Signed Error)", f"{metrics_eval['bias']:+.2f}" if metrics_eval['bias'] is not None else "N/A")
    k5.metric("MAPE", f"{metrics_eval['mape']:.1f}%" if metrics_eval['mape'] is not None else "N/A")

    st.markdown(f"**Model Health Warning Status:** `{metrics_eval['health_status']}`")

    if comparisons:
        comp_df = pd.DataFrame([c.model_dump() if hasattr(c, "model_dump") else c.dict() for c in comparisons])

        st.markdown("---")
        ch_col1, ch_col2 = st.columns(2)

        with ch_col1:
            st.subheader("🎯 Predicted vs Measured Scatter Plot (1:1 Line)")
            fig_scatter = px.scatter(
                comp_df,
                x="measured_value",
                y="predicted_value",
                color="model_name",
                hover_data=["blast_id", "signed_error"],
                title="Predicted vs Measured Outcomes (Ideal 1:1 Parity Line)",
                labels={"measured_value": f"Measured Actual ({comp_df['unit'].iloc[0]})", "predicted_value": f"Predicted Value ({comp_df['unit'].iloc[0]})"},
            )
            min_val = min(comp_df["measured_value"].min(), comp_df["predicted_value"].min())
            max_val = max(comp_df["measured_value"].max(), comp_df["predicted_value"].max())
            fig_scatter.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"))
            fig_scatter.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with ch_col2:
            st.subheader("📉 Error Residual Plot (Predicted - Measured)")
            fig_res = px.scatter(
                comp_df,
                x="measured_value",
                y="signed_error",
                color="model_name",
                hover_data=["blast_id"],
                title="Signed Error Residuals vs Measured Value",
                labels={"measured_value": "Measured Actual", "signed_error": "Signed Error e_i"},
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_res, use_container_width=True)

        st.subheader("📋 Individual Blast Comparison Details")
        st.dataframe(comp_df[["blast_id", "target_metric", "predicted_value", "measured_value", "signed_error", "absolute_error", "percentage_error", "unit", "model_name", "compared_at"]], use_container_width=True, hide_index=True)
    else:
        st.info("No paired prediction vs measurement comparison records found for selected filters.")


# ==============================================================================
# SECTION 4: HISTORICAL ANALYTICS
# ==============================================================================
with tab_analytics:
    st.subheader("📈 Historical Performance Analytics & Trend Investigation")
    st.markdown("Investigate model error distributions and accuracy trends across mine sites, benches, rock types, and time ranges.")

    all_comps = pf.list_comparison_results()
    if all_comps:
        all_comp_df = pd.DataFrame([c.model_dump() if hasattr(c, "model_dump") else c.dict() for c in all_comps])

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.subheader("📊 Signed Error Distribution Histogram")
            fig_hist = px.histogram(
                all_comp_df,
                x="signed_error",
                color="target_metric",
                nbins=20,
                title="Error Distribution (Zero Center indicates Zero Bias)",
                labels={"signed_error": "Signed Error (Predicted - Measured)"},
            )
            fig_hist.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_a2:
            st.subheader("📅 Model Performance Trend Over Time")
            all_comp_df["compared_at_dt"] = pd.to_datetime(all_comp_df["compared_at"])
            all_comp_df_sorted = all_comp_df.sort_values("compared_at_dt")

            fig_trend = px.line(
                all_comp_df_sorted,
                x="compared_at_dt",
                y="absolute_error",
                color="target_metric",
                markers=True,
                title="Absolute Error Trend Over Time",
                labels={"compared_at_dt": "Date", "absolute_error": "Absolute Error"},
            )
            fig_trend.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No historical comparison data available for trend analytics.")


# ==============================================================================
# SECTION 5: DATASET REVIEW & RETRAINING
# ==============================================================================
with tab_review:
    st.subheader("✅ Structured Data Review, Approved Export & Retraining Pipeline")
    st.markdown("Review candidate actual measurements through the workflow: `Uploaded -> Validated -> Awaiting Review -> Approved or Rejected -> Eligible for Export`.")

    rev_target = st.selectbox("Target Metric for Review Pipeline", ["d50_mm", "fragmentation_d80_cm", "ppv_mms", "airblast_db", "cost_per_tonne_usd"], index=0, key="rev_target_s")

    reviews = pf.list_dataset_reviews(target_metric=rev_target)

    st.markdown(f"**Found {len(reviews)} candidate record(s) for target `{rev_target}`**")

    if reviews:
        col_rw1, col_rw2 = st.columns([2, 1])

        with col_rw1:
            rev_data = []
            for r in reviews:
                rev_data.append({
                    "Review ID": r.review_id,
                    "Blast ID": r.blast_id,
                    "Target": r.target_metric,
                    "Status": r.status,
                    "Approved for Retraining": "Yes ✅" if r.inclusion_status else "No ❌",
                    "Exclusion Reason": r.exclusion_reason or "N/A",
                    "Reviewer": r.reviewer_id,
                })
            st.dataframe(pd.DataFrame(rev_data), use_container_width=True, hide_index=True)

        with col_rw2:
            st.subheader("Review Decision Tool")
            selected_rev_id = st.selectbox("Select Review Record", [r.review_id for r in reviews], key="sel_rev_id")
            selected_rev = next((r for r in reviews if r.review_id == selected_rev_id), None)

            if selected_rev:
                decision_action = st.radio("Review Decision", ["Approved", "Rejected"], index=0)
                excl_reason = st.text_input("Exclusion Reason (if Rejected)", value="")
                rev_comments = st.text_area("Review Comments", value="Verified data quality and field log.")

                if st.button("Record Review Decision", type="primary", key="btn_save_review"):
                    updated_rev = pf.DatasetReviewRecord(
                        review_id=selected_rev.review_id,
                        blast_id=selected_rev.blast_id,
                        target_metric=selected_rev.target_metric,
                        status=decision_action,
                        inclusion_status=(decision_action == "Approved"),
                        exclusion_reason=excl_reason if decision_action == "Rejected" else None,
                        reviewer_id="CHIEF_MINING_ENGINEER",
                        review_timestamp=datetime.now(timezone.utc).isoformat(),
                        comments=rev_comments,
                    )
                    pf.save_dataset_review(updated_rev)
                    st.success(f"Decision recorded for `{selected_rev.blast_id}` as {decision_action}!")
                    st.rerun()

    st.markdown("---")
    st.subheader("📦 Approved Training Dataset Export & Retraining Trigger")

    if st.button("Generate Approved Dataset Export Package 🚀", type="primary", key="btn_export_pkg"):
        export_df, manifest, csv_bytes = pf.export_approved_dataset(target_metric=rev_target)

        st.success(f"✅ Generated Export Package `{manifest.export_id}` containing {manifest.approved_record_count} approved records!")
        st.caption(f"Manifest Hash: `{manifest.manifest_hash}`")

        st.json(manifest.model_dump() if hasattr(manifest, "model_dump") else manifest.dict())

        st.download_button(
            label=f"📄 Download Approved {rev_target} Retraining CSV",
            data=csv_bytes,
            file_name=f"Approved_Retraining_Dataset_{rev_target}_{manifest.export_id}.csv",
            mime="text/csv",
        )
