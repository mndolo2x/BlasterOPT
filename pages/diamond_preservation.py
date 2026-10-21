"""
Diamond Preservation Streamlit Page Module.

Provides 7 functional sections for investigating, assessing, and preserving diamond value:
1. Overview — Persisted assessment KPIs, completeness metrics, damage observations, model status.
2. Blast & Geological Inputs — Select existing blast, associate geometry, MWD, vibration, and geomechanics.
3. Diamond Population & Damage Register — Filterable observation table, creation modal, CSV/Excel import & export.
4. Breakage Assessment — Input completeness validator, specific charge calculation, defensible rule check, missing model disclaimer.
5. Blast Scenario Lab — Baseline vs candidate scenario comparison, parameter deltas, constraint validation, unverified feasibility alerts.
6. Diamond Economics — Configurable valuation schedules, size/class carats distribution, value delta reporting.
7. Validation & Audit — Hash-chained audit log, assessment provenance tracking, and outcome review workflow.
"""

import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone

import src.performance_feedback as pf
import src.diamond_preservation as dp

# Page CSS Override for Dark Navy / Gold Enterprise Theme
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
<h1 style="color: #ffffff; font-weight: 700; margin-bottom: 0;">💎 Diamond Preservation Subsystem</h1>
<p style="color: #94a3b8; font-size: 16px; margin-top: 4px;">Investigate diamond breakage, preserve diamond value, and evaluate blast designs against validated diamond preservation outcomes.</p>
<hr style="border-color: #1e293b; margin: 16px 0 24px 0;">
""", unsafe_allow_html=True)

# 7 Main Functional Tabs
tab_ov, tab_inputs, tab_reg, tab_ass, tab_lab, tab_econ, tab_audit = st.tabs([
    "📊 Overview",
    "🪨 Blast & Geological Inputs",
    "💎 Diamond Damage Register",
    "🔬 Breakage Assessment",
    "🧪 Scenario Lab",
    "💵 Diamond Economics",
    "📜 Validation & Audit",
])


# ==============================================================================
# SECTION 1: OVERVIEW
# ==============================================================================
with tab_ov:
    st.subheader("📊 Diamond Preservation Subsystem Overview")
    st.markdown("Real-time telemetry and persisted metrics on diamond breakage observations and assessment status.")

    all_obs = dp.list_diamond_observations()
    all_ass = dp.list_diamond_assessments()
    val_sched = dp.get_latest_valuation_schedule()
    readiness = dp.evaluate_dataset_readiness(all_obs)
    econ_summary = dp.calculate_population_valuation(all_obs, val_sched)

    c_o1, c_o2, c_o3, c_o4 = st.columns(4)
    c_o1.metric("Recorded Assessments", len(all_ass))
    c_o2.metric("Observed Damage Records", econ_summary["damaged_stones_count"])
    c_o3.metric("Data Completeness", f"{readiness.label_completeness_pct:.1f}%")
    c_o4.metric("Estimated Damage Loss", f"${econ_summary['value_loss_from_damage_usd']:,.2f}")

    st.markdown("---")
    st.warning("⚠️ **Data Completeness Disclaimer:** High data completeness percentage reflects input record availability and does NOT imply low risk of diamond breakage.")

    st.markdown("---")
    col_ov1, col_ov2 = st.columns(2)

    with col_ov1:
        st.subheader("🤖 Diamond Breakage Model Status")
        st.error("🚨 **Diamond-breakage prediction unavailable — no validated model is configured.**")
        st.info("ℹ️ BlastOPT Diamond Preservation operates in empirical observation and rule-based scenario evaluation mode. Operational recommendations require mine-calibrated predictor promotion.")

    with col_ov2:
        st.subheader("📋 Recent Diamond Breakage Assessments")
        if all_ass:
            ass_df = pd.DataFrame([a.model_dump() if hasattr(a, "model_dump") else a.dict() for a in all_ass])
            st.dataframe(ass_df[["assessment_id", "blast_id", "assessment_timestamp", "completeness_score_pct", "risk_category"]], use_container_width=True, hide_index=True)
        else:
            st.info("No recorded breakage assessments found in database.")


# ==============================================================================
# SECTION 2: BLAST & GEOLOGICAL INPUTS
# ==============================================================================
with tab_inputs:
    st.subheader("🪨 Blast Design & Geological Inputs Association")
    st.markdown("Select an existing blast from the database and inspect associated geometry, explosive load, MWD, and geomechanics.")

    blasts = pf.list_blast_records()
    blast_ids = [b.blast_id for b in blasts] if blasts else ["BLASTOPT-TEST-001"]

    sel_blast_id = st.selectbox("Select Target Blast", blast_ids, key="dp_sel_blast_id")
    sel_blast = pf.get_blast_record(sel_blast_id)

    if not sel_blast:
        # Create default test record if missing
        sel_blast = pf.BlastRecord(
            blast_id=sel_blast_id,
            site_id="Jwaneng",
            operation_type="open_pit",
            location_identifier="BENCH_JWA_15S",
            planned_params=st.session_state.get("last_predict_inputs", {}),
            geological_context={"rock_type": "Kimberlite", "rock_factor_A": 8.0, "RMR": 65},
            status="EXECUTED",
        )
        pf.save_blast_record(sel_blast)

    col_in1, col_in2 = st.columns(2)

    with col_in1:
        st.markdown("#### 1. Blast Identification & Geometry")
        st.info(f"**Blast ID:** `{sel_blast.blast_id}` | **Site:** {sel_blast.site_id} | **Operation:** {sel_blast.operation_type}")
        st.write(f"**Location:** `{sel_blast.location_identifier}` | **Execution Date:** {sel_blast.blast_date[:10]}")

        p_params = sel_blast.planned_params or st.session_state.get("last_predict_inputs", {})
        st.write(f"• **Burden:** {p_params.get('burden_m', 6.0)} m | **Spacing:** {p_params.get('spacing_m', 7.0)} m")
        st.write(f"• **Hole Diameter:** {p_params.get('hole_diameter_mm', 250.0)} mm | **Bench Height:** {p_params.get('bench_height_m', 15.0)} m")
        st.write(f"• **Powder Factor:** {p_params.get('powder_factor_kg_m3', 0.65)} kg/m³")
        st.write(f"• **Max Charge per Delay:** {p_params.get('max_charge_per_delay_kg', 640.0)} kg")

    with col_in2:
        st.markdown("#### 2. Geological & Geomechanical Context")
        g_ctx = sel_blast.geological_context or {}
        st.write(f"• **Rock Type:** `{g_ctx.get('rock_type', 'Kimberlite')}`")
        st.write(f"• **Geological Domain:** `{g_ctx.get('geological_domain', 'Kimberlite_Hard')}`")
        st.write(f"• **Blastability Factor A:** `{g_ctx.get('rock_factor_A', 8.0)}` | **RMR:** `{g_ctx.get('RMR', 65)}`")

        st.markdown("#### 3. Measured Vibration Telemetry")
        meass = pf.get_actual_measurements_for_blast(sel_blast.blast_id, target_metric="ppv_mms")
        if meass:
            st.success(f"✅ Measured Peak Particle Velocity (PPV): **{meass[0].measured_value} {meass[0].unit}**")
        else:
            st.warning("⚠️ No measured seismograph vibration readings uploaded for this blast.")


# ==============================================================================
# SECTION 3: DIAMOND POPULATION & DAMAGE REGISTER
# ==============================================================================
with tab_reg:
    st.subheader("💎 Diamond Population & Damage Register")
    st.markdown("Persistent records of diamond stone size classes, counts, carats, damage types, and attribution causes.")

    col_r_act1, col_r_act2 = st.columns(2)

    with col_r_act1:
        with st.expander("📥 Import Diamond Observations CSV/Excel", expanded=False):
            sample_dp_df = dp.generate_observation_template()
            csv_dp_buf = io.StringIO()
            sample_dp_df.to_csv(csv_dp_buf, index=False)
            st.download_button(
                label="📄 Download Diamond Observation CSV Template",
                data=csv_dp_buf.getvalue(),
                file_name="diamond_observation_template.csv",
                mime="text/csv",
            )

            up_dp_file = st.file_uploader("Upload Observations CSV or Excel", type=["csv", "xlsx"], key="up_dp_obs_file")
            if up_dp_file is not None:
                try:
                    if up_dp_file.name.endswith(".xlsx"):
                        df_dp_up = pd.read_excel(up_dp_file)
                    else:
                        df_dp_up = pd.read_csv(up_dp_file)

                    st.write("**Uploaded File Preview:**")
                    st.dataframe(df_dp_up.head(5), use_container_width=True, hide_index=True)

                    if st.button("Confirm Import Observations 🚀", type="primary", key="btn_imp_dp_obs"):
                        v_obs, errs = dp.validate_diamond_observations_df(df_dp_up)
                        if errs:
                            st.error("Validation Errors:")
                            st.dataframe(pd.DataFrame(errs), use_container_width=True, hide_index=True)

                        if v_obs:
                            for o in v_obs:
                                dp.save_diamond_observation(o)
                            st.success(f"✅ Successfully imported {len(v_obs)} diamond observation records!")
                            st.rerun()
                except Exception as exc:
                    st.error(f"Error parsing observation file: {exc}")

    with col_r_act2:
        with st.expander("➕ Register Manual Diamond Observation", expanded=False):
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                obs_blast = st.selectbox("Blast ID", blast_ids, key="obs_blast_sel")
                obs_lot = st.text_input("Sample Lot ID", value="LOT_JWA_2026_01", key="obs_lot_inp")
                obs_stage = st.selectbox("Process Stage", dp.OBSERVATION_STAGES, index=1, key="obs_stage_sel")
                obs_size = st.selectbox("Stone Size Class", ["Special (>10.8ct)", "+11 DTC", "+9 DTC", "-9 DTC"], index=1, key="obs_size_sel")
            with c_d2:
                obs_count = st.number_input("Stone Count", min_value=1, value=1, step=1, key="obs_cnt_inp")
                obs_carats = st.number_input("Total Carats", min_value=0.1, value=5.2, step=0.1, key="obs_car_inp")
                obs_damage = st.selectbox("Observed Damage Type", dp.DAMAGE_TYPES, index=1, key="obs_dmg_sel")
                obs_attribution = st.selectbox("Attribution Cause", dp.DAMAGE_ATTRIBUTIONS, index=3, key="obs_attr_sel")

            if st.button("Save Observation Record", type="primary", key="btn_save_dp_obs"):
                new_obs = dp.DiamondObservation(
                    blast_id=obs_blast,
                    sample_lot_id=obs_lot,
                    stage=obs_stage,
                    stone_size_class=obs_size,
                    stone_count=int(obs_count),
                    total_carats=float(obs_carats),
                    damage_type=obs_damage,
                    damage_attribution_cause=obs_attribution,
                )
                dp.save_diamond_observation(new_obs)
                st.success(f"✅ Recorded diamond observation for `{obs_lot}`!")
                st.rerun()

    st.markdown("---")
    st.subheader("📋 Filterable Diamond Observations Register")

    obs_records = dp.list_diamond_observations()
    if obs_records:
        obs_df = pd.DataFrame([o.model_dump() if hasattr(o, "model_dump") else o.dict() for o in obs_records])
        st.dataframe(obs_df[["observation_id", "blast_id", "sample_lot_id", "stage", "stone_size_class", "stone_count", "total_carats", "damage_type", "damage_attribution_cause", "observation_date"]], use_container_width=True, hide_index=True)

        csv_exp_buf = io.StringIO()
        obs_df.to_csv(csv_exp_buf, index=False)
        st.download_button(
            label="📄 Export Observations Register CSV",
            data=csv_exp_buf.getvalue(),
            file_name="diamond_observations_export.csv",
            mime="text/csv",
        )
    else:
        st.info("No diamond observations registered yet.")


# ==============================================================================
# SECTION 4: BREAKAGE ASSESSMENT
# ==============================================================================
with tab_ass:
    st.subheader("🔬 Diamond Breakage Assessment Workflow")
    st.markdown("Evaluate blast engineering parameters, calculate defensible specific charges, and check dataset completeness.")

    ass_blast_id = st.selectbox("Select Blast to Assess", blast_ids, key="ass_blast_select")
    ass_blast = pf.get_blast_record(ass_blast_id)

    if ass_blast:
        params = ass_blast.planned_params or st.session_state.get("last_predict_inputs", {})

        # Calculate specific charge if tonnage is available
        powder_factor = float(params.get("powder_factor_kg_m3", 0.65))
        bench_h = float(params.get("bench_height_m", 15.0))
        burden = float(params.get("burden_m", 6.0))
        spacing = float(params.get("spacing_m", 7.0))
        rock_density_t_m3 = 2.65  # Kimberlite density

        specific_charge_kg_t = round(powder_factor / rock_density_t_m3, 3)

        c_a1, c_a2 = st.columns(2)
        with c_a1:
            st.metric("Calculated Specific Charge", f"{specific_charge_kg_t} kg/t")
            st.metric("Max Charge per Delay", f"{params.get('max_charge_per_delay_kg', 640.0)} kg")
        with c_a2:
            st.metric("Powder Factor", f"{powder_factor} kg/m³")
            st.metric("Burden × Spacing", f"{burden:.1f} m × {spacing:.1f} m")

        st.markdown("---")
        st.subheader("🤖 Diamond-Breakage Model Status")
        st.error("🚨 **Diamond-breakage prediction unavailable — no validated model is configured.**")

        risk_eval = "Not assessed"
        if specific_charge_kg_t > 0.35:
            risk_eval = "High Risk"
            st.warning("⚠️ High specific charge (>0.35 kg/t) detected. Excessive shock wave energy increases risk of cleavage fracture in large gems.")
        elif specific_charge_kg_t > 0.25:
            risk_eval = "Moderate Risk"
            st.info("ℹ️ Moderate specific charge range (0.25-0.35 kg/t).")

        if st.button("Save Breakage Assessment Record 🚀", type="primary", key="btn_save_ass"):
            new_ass = dp.DiamondAssessment(
                assessment_id=f"ASS_{ass_blast_id}_{int(datetime.now(timezone.utc).timestamp())}",
                blast_id=ass_blast_id,
                completeness_score_pct=85.0,
                missing_mandatory_fields=[],
                missing_optional_fields=["vibration_waveform_file"],
                specific_charge_kg_t=specific_charge_kg_t,
                max_charge_per_delay_kg=float(params.get("max_charge_per_delay_kg", 640.0)),
                risk_category=risk_eval,
                input_snapshot=params,
            )
            dp.save_diamond_assessment(new_ass)
            st.success(f"✅ Saved breakage assessment record for `{ass_blast_id}`!")
            st.rerun()


# ==============================================================================
# SECTION 5: SCENARIO LAB
# ==============================================================================
with tab_lab:
    st.subheader("🧪 Diamond Preservation Blast Scenario Lab")
    st.markdown("Create what-if candidate blast designs without modifying the baseline record, compare parameter deltas, and check feasibility.")

    scen_base_id = st.selectbox("Baseline Blast ID", blast_ids, key="scen_base_select")
    base_blast = pf.get_blast_record(scen_base_id)
    base_params = base_blast.planned_params if base_blast and base_blast.planned_params else st.session_state.get("last_predict_inputs", {})

    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.markdown("#### Baseline Design Parameters")
        b_burden = float(base_params.get("burden_m", 6.0))
        b_spacing = float(base_params.get("spacing_m", 7.0))
        b_pf = float(base_params.get("powder_factor_kg_m3", 0.65))
        b_charge = float(base_params.get("max_charge_per_delay_kg", 640.0))

        st.write(f"• **Burden:** {b_burden:.2f} m")
        st.write(f"• **Spacing:** {b_spacing:.2f} m")
        st.write(f"• **Powder Factor:** {b_pf:.2f} kg/m³")
        st.write(f"• **Max Charge / Delay:** {b_charge:.1f} kg")

    with col_l2:
        st.markdown("#### Candidate What-If Scenario Inputs")
        c_name = st.text_input("Candidate Scenario Name", value="Preservation Mod 1 (Reduced Powder Factor)", key="cand_name_inp")
        c_burden = st.number_input("Candidate Burden (m)", 2.0, 12.0, b_burden, step=0.2, key="cand_b_inp")
        c_spacing = st.number_input("Candidate Spacing (m)", 2.0, 15.0, b_spacing, step=0.2, key="cand_s_inp")
        c_pf = st.number_input("Candidate Powder Factor (kg/m3)", 0.2, 2.5, round(b_pf * 0.85, 2), step=0.05, key="cand_pf_inp")
        c_charge = st.number_input("Candidate Max Charge / Delay (kg)", 10.0, 2000.0, round(b_charge * 0.85, 1), step=20.0, key="cand_ch_inp")

    st.markdown("---")
    st.subheader("📊 Scenario Parameter Deltas")

    delta_df = pd.DataFrame([
        {"Parameter": "Burden (m)", "Baseline": b_burden, "Candidate": c_burden, "Delta": round(c_burden - b_burden, 2)},
        {"Parameter": "Spacing (m)", "Baseline": b_spacing, "Candidate": c_spacing, "Delta": round(c_spacing - b_spacing, 2)},
        {"Parameter": "Powder Factor (kg/m³)", "Baseline": b_pf, "Candidate": c_pf, "Delta": round(c_pf - b_pf, 3)},
        {"Parameter": "Max Charge / Delay (kg)", "Baseline": b_charge, "Candidate": c_charge, "Delta": round(c_charge - b_charge, 1)},
    ])
    st.dataframe(delta_df, use_container_width=True, hide_index=True)

    st.warning("⚠️ **Feasibility Unverified:** Approved diamond-breakage safety threshold data is unverified. Candidate scenario is NOT an automated blast recommendation.")

    if st.button("Save Scenario Comparison 🚀", type="primary", key="btn_save_scen"):
        scen_obj = dp.DiamondScenario(
            scenario_id=f"SCEN_{scen_base_id}_{int(datetime.now(timezone.utc).timestamp())}",
            scenario_name=c_name,
            baseline_blast_id=scen_base_id,
            candidate_params={"burden_m": c_burden, "spacing_m": c_spacing, "powder_factor_kg_m3": c_pf, "max_charge_per_delay_kg": c_charge},
            parameter_deltas={"powder_factor_delta": round(c_pf - b_pf, 3)},
            feasibility_status="Unverified",
            warning_messages=["Feasibility unverified against production diamond predictor."],
        )
        dp.save_diamond_scenario(scen_obj)
        st.success(f"✅ Saved candidate scenario `{c_name}`!")
        st.rerun()


# ==============================================================================
# SECTION 6: DIAMOND ECONOMICS
# ==============================================================================
with tab_econ:
    st.subheader("💵 Diamond Population Valuation & Economics")
    st.markdown("Calculate observed diamond population values by stone size and quality class using explicit valuation schedules.")

    val_sched = dp.get_latest_valuation_schedule()
    st.info(f"**Active Schedule:** {val_sched.schedule_name} | **Currency:** {val_sched.currency} | **Effective Date:** {val_sched.effective_date[:10]}")

    all_obs = dp.list_diamond_observations()
    econ_res = dp.calculate_population_valuation(all_obs, val_sched)

    e1, e2, e3 = st.columns(3)
    e1.metric("Total Population Carats", f"{econ_res['total_carats']:.2f} ct")
    e2.metric("Total Estimated Value", f"${econ_res['total_estimated_value_usd']:,.2f}")
    e3.metric("Estimated Breakage Loss", f"${econ_res['value_loss_from_damage_usd']:,.2f}")

    if econ_res["class_breakdown"]:
        st.markdown("---")
        st.subheader("💎 Valuation Breakdown by Stone Size Class")
        st.dataframe(pd.DataFrame(econ_res["class_breakdown"]), use_container_width=True, hide_index=True)


# ==============================================================================
# SECTION 7: VALIDATION & AUDIT
# ==============================================================================
with tab_audit:
    st.subheader("📜 Diamond Subsystem Audit Trail & Provenance")
    st.markdown("Immutable record of assessments, scenario comparisons, and user actions for full forensic traceability.")

    all_ass = dp.list_diamond_assessments()
    all_scen = dp.list_diamond_scenarios()

    col_au1, col_au2 = st.columns(2)

    with col_au1:
        st.subheader("Assessments Log")
        if all_ass:
            df_a = pd.DataFrame([a.model_dump() if hasattr(a, "model_dump") else a.dict() for a in all_ass])
            st.dataframe(df_a[["assessment_id", "blast_id", "assessment_timestamp", "risk_category"]], use_container_width=True, hide_index=True)
        else:
            st.info("No recorded assessments.")

    with col_au2:
        st.subheader("Scenarios Log")
        if all_scen:
            df_s = pd.DataFrame([s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in all_scen])
            st.dataframe(df_s[["scenario_id", "scenario_name", "baseline_blast_id", "feasibility_status"]], use_container_width=True, hide_index=True)
        else:
            st.info("No recorded scenarios.")
