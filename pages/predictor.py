import streamlit as st
import pandas as pd
import numpy as np
from src.components.model_selector import render_page_model_selector
from src.visualize import plot_kuz_ram_curve, plot_ppv_attenuation
from src.explainability import create_explanation_panel
from src.domain.safety_checks import evaluate_safety, SafetyReport
from src.domain.blast_design import BlastDesignVersion
from src.services.approval_service import submit_for_approval

st.title("Predictor & Kuz-Ram Curve")

model, model_key = render_page_model_selector("predictor")
if model is None:
    st.stop()

st.divider()

col_p1, col_p2 = st.columns([1, 2])

with col_p1:
    st.subheader("Input Blast Parameters")

    last_in = st.session_state.get("last_predict_inputs", {
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
    })

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

    X = pd.DataFrame([input_payload])
    raw_preds = model.predict(X)

    if isinstance(raw_preds, pd.DataFrame):
        preds_dict = raw_preds.iloc[0].to_dict()
    elif isinstance(raw_preds, np.ndarray):
        if raw_preds.ndim == 2:
            preds_dict = {f"output_{i}": float(v) for i, v in enumerate(raw_preds[0])}
        else:
            preds_dict = {"output_0": float(raw_preds[0])}
    else:
        preds_dict = {"output": float(raw_preds)}

    d80_cm = float(preds_dict.get("fragmentation_d80_cm", preds_dict.get("d50_mm", 220.0 / 10.0)))
    d50_mm = float(preds_dict.get("d50_mm", d80_cm * 10.0))
    ppv = float(preds_dict.get("vibration_ppv_mms", preds_dict.get("ppv_mms", 4.5)))
    airblast = float(preds_dict.get("airblast_db", 115.0))
    flyrock = float(preds_dict.get("flyrock_m", 120.0))
    cost = float(preds_dict.get("cost_per_tonne_usd", 4.8))

    predictions = {
        "d50_mm": d50_mm,
        "fragmentation_d80_cm": d80_cm,
        "ppv_mms": ppv,
        "vibration_ppv_mms": ppv,
        "airblast_db": airblast,
        "flyrock_m": flyrock,
        "cost_per_tonne_usd": cost,
    }

    safety_rep = evaluate_safety(predictions, blast_params=input_payload)
    safety_dict = safety_rep.model_dump() if hasattr(safety_rep, "model_dump") else safety_rep.dict()
    predictions["safety_report"] = safety_dict

    st.session_state["last_predict_results"] = predictions

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("d50 Fragment Size", f"{d50_mm:.1f} mm")
    m2.metric("Ground PPV", f"{ppv:.2f} mm/s")
    m3.metric("Flyrock Distance", f"{flyrock:.1f} m")
    m4.metric("D&B Cost", f"${cost:.2f} / t")

    ov_status = safety_dict.get("overall_status", "SAFE")

    st.markdown("---")
    st.subheader("🛡️ Step 3: Safety Report & Limit Check")

    if ov_status == "SAFE":
        st.success("✅ **OVERALL STATUS: SAFE** — All 95% upper confidence bounds comply strictly with site and regulatory thresholds.")
    elif ov_status == "REQUIRES_REVIEW":
        st.warning("⚠️ **OVERALL STATUS: REQUIRES REVIEW** — Mean predictions comply, but 95% upper confidence bounds cross threshold limits. Certified blaster review and mandatory override reasoning required.")
    else:
        st.error("🚨 **OVERALL STATUS: UNSAFE / INFEASIBLE** — Design violates maximum allowed safety limits. Export and transmission disabled.")

    checks_df = pd.DataFrame(safety_dict.get("checks", []))
    if not checks_df.empty:
        st.dataframe(checks_df[["check_name", "predicted_value", "lower_95", "upper_95", "limit", "status", "reasoning"]], use_container_width=True)

    st.markdown("---")
    st.subheader("📝 Step 4 & 5: Submit for Certified Blaster Approval")
    sub_user_id = st.text_input("Engineer / Submitter User ID", value="ENGINEER_BOTSWANA_01")
    pattern_req_id = st.text_input("Pattern ID for Submission", value="PATTERN_CUT8_BENCH15S")

    ack_review_checkbox = False
    if ov_status == "REQUIRES_REVIEW":
        ack_review_checkbox = st.checkbox(
            "I have reviewed the uncertainty margin and accept responsibility.",
            value=False,
        )

    submit_btn_disabled = (ov_status == "UNSAFE") or (ov_status == "REQUIRES_REVIEW" and not ack_review_checkbox)

    if st.button("Submit for Approval 🚀", disabled=submit_btn_disabled, type="primary"):
        s_report = SafetyReport(**safety_dict)
        req = submit_for_approval(
            design=input_payload,
            safety_report=s_report,
            user_id=sub_user_id,
            design_id=pattern_req_id,
        )
        st.success(f"Approval request created and routed to certified blaster! Design ID: `{req.design_id}`")

    st.markdown("---")
    st.subheader("📜 Design Version History")
    current_s_report = SafetyReport(**safety_dict)
    design_ver_1 = BlastDesignVersion.create(
        design_id=pattern_req_id,
        design_data=input_payload,
        safety_report=current_s_report,
        created_by=sub_user_id,
        version=1,
        approval_status="PENDING_APPROVAL" if ov_status != "UNSAFE" else "DRAFT",
    )

    ver_df = pd.DataFrame([{
        "Design ID": design_ver_1.design_id,
        "Version": design_ver_1.version,
        "Parent Version": str(design_ver_1.parent_version),
        "Created By": design_ver_1.created_by,
        "Created At": str(design_ver_1.created_at),
        "Approval Status": design_ver_1.approval_status,
        "Content Hash (SHA-256)": design_ver_1.content_hash[:16] + "...",
    }])
    st.dataframe(ver_df, use_container_width=True)

    st.markdown("---")
    with st.expander("🔍 Why this prediction?", expanded=True):
        target_explain = st.selectbox(
            "Select Outcome to Explain",
            ["d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"],
            key="explain_target_select",
        )

        constraints_info = {
            "metric": target_explain,
            "limit": 10.0 if target_explain == "ppv_mms" else (320.0 if target_explain == "d50_mm" else 150.0),
            "d50_min_mm": 120.0,
            "d50_max_mm": 320.0,
            "ppv_max_mm_s": 10.0,
            "unit": "mm/s" if target_explain == "ppv_mms" else ("m" if target_explain == "flyrock_m" else "mm"),
        }

        exp_panel = create_explanation_panel(
            model=model,
            input_data=X,
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
    fig_kuz = plot_kuz_ram_curve(d50_mm, n_uniformity=1.2)
    st.plotly_chart(fig_kuz, use_container_width=True)

    fig_ppv = plot_ppv_attenuation(max_charge_delay)
    st.plotly_chart(fig_ppv, use_container_width=True)
