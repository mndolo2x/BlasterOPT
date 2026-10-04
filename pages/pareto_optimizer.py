import streamlit as st
import pandas as pd
from datetime import datetime
from src.components.model_selector import render_page_model_selector
from src.pareto_optimizer import run_nsga2, plot_pareto_front, is_physically_valid, validate_design

st.title("Multi-Objective Pareto Optimizer (NSGA-II)")

PARETO_VERSION = 5  # increment every time the optimizer bounds or logic change

if st.session_state.get("pareto_version") != PARETO_VERSION:
    for key in ["pareto_front_df", "pareto_raw_front_df", "pareto_front_timestamp", "pareto_front_hash"]:
        st.session_state.pop(key, None)
    st.session_state["pareto_version"] = PARETO_VERSION
    st.info("Optimizer code updated. Cache cleared. Click Run to regenerate.")

model, model_key = render_page_model_selector("pareto_optimizer")
if model is None:
    st.stop()

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Optimization Settings")
    st.write(f"**Active model:** `{model_key}`")
    st.write(f"**Model type:** `{type(model).__name__}`")
    n_gen = st.slider("Generations", 10, 300, 40, step=10)
    pop_size = st.slider("Population Size", 20, 200, 100, step=10)
    max_ppv_limit = st.number_input("Max PPV Regulatory Limit (mm/s)", 1.0, 50.0, 5.0, step=0.5)
    max_air_limit = st.number_input("Max Airblast Limit (dB)", 90.0, 140.0, 120.0, step=1.0)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_opt_btn = st.button("Run Pareto Optimizer 🚀", type="primary")
    with col_btn2:
        if st.button("Clear cached results"):
            for key in ["pareto_front_df", "pareto_raw_front_df", "pareto_front_timestamp", "pareto_front_hash"]:
                st.session_state.pop(key, None)
            st.rerun()

# Run only when the button is clicked
if run_opt_btn:
    constraints_info = {"max_ppv": max_ppv_limit, "max_airblast": max_air_limit}
    with st.spinner("Running NSGA-II Multi-Objective Optimization..."):
        raw_front = run_nsga2(
            model=model,
            n_gen=n_gen,
            pop_size=pop_size,
            constraints_info=constraints_info,
        )

        st.session_state["pareto_raw_front_df"] = raw_front

        # Filter out invalid designs
        valid_front = raw_front[raw_front["valid"] == True].reset_index(drop=True)

        if len(valid_front) == 0:
            st.error(
                "❌ The optimizer could not find any valid designs. "
                "This means the bounds, constraints, or model are inconsistent. "
                "Check the violations below."
            )
            st.write("Sample violations from rejected designs:")
            rejected_designs = raw_front[raw_front["valid"] == False]
            if not rejected_designs.empty:
                st.dataframe(rejected_designs[["burden_m", "spacing_m", "powder_factor_kg_m3", "violations"]].head(10))
            st.stop()

        front_hash = str(pd.util.hash_pandas_object(valid_front).sum())
        valid_front.to_csv("pareto_front.csv", index=False)
        st.session_state["pareto_front_df"] = valid_front
        st.session_state["pareto_front_timestamp"] = datetime.now().isoformat()
        st.session_state["pareto_front_hash"] = front_hash

# Load from cache for display
df = st.session_state.get("pareto_front_df", None)
raw_df = st.session_state.get("pareto_raw_front_df", df)

if df is None:
    st.info("Click **Run Optimization** to generate the Pareto front.")
    st.stop()

if df is not None and not df.empty:
    st.caption(
        f"Burden: {df['burden_m'].min():.2f} – {df['burden_m'].max():.2f} m  |  "
        f"Spacing: {df['spacing_m'].min():.2f} – {df['spacing_m'].max():.2f} m  |  "
        f"PF: {df['powder_factor_kg_m3'].min():.2f} – {df['powder_factor_kg_m3'].max():.2f} kg/m³  |  "
        f"Airblast max: {df['airblast_dbl'].max():.1f} dB  |  "
        f"Run at: {st.session_state.get('pareto_front_timestamp', 'unknown')}"
    )

    with st.expander("Debug info"):
        st.write(f"Cache present: {'pareto_front_df' in st.session_state}")
        st.write(f"Cache timestamp: {st.session_state.get('pareto_front_timestamp')}")
        st.write(f"Version: {st.session_state.get('pareto_version')}")
        st.write(f"Model key: {model_key}")
        st.write(f"Burden range: {df['burden_m'].min():.2f} – {df['burden_m'].max():.2f}")
        st.write(f"PF range: {df['powder_factor_kg_m3'].min():.2f} – {df['powder_factor_kg_m3'].max():.2f}")
        st.write(f"Airblast range: {df['airblast_dbl'].min():.1f} – {df['airblast_dbl'].max():.1f}")

    with col2:
        st.subheader("Physical & Regulatory Constraint Verification")

        total = len(raw_df) if raw_df is not None else len(df)
        valid_count = raw_df["valid"].sum() if (raw_df is not None and "valid" in raw_df.columns) else len(df)
        invalid_count = total - valid_count

        if invalid_count == 0:
            st.success(f"✅ {valid_count} designs passed all constraints.")
        else:
            st.warning(
                f"⚠️ {invalid_count} of {total} designs were rejected by the validator. "
                f"Only the {valid_count} valid designs are shown."
            )

        with st.expander("Show rejected designs"):
            if raw_df is not None and "valid" in raw_df.columns:
                rejected = raw_df[raw_df["valid"] == False]
                if len(rejected) > 0:
                    st.dataframe(rejected[["burden_m", "spacing_m", "powder_factor_kg_m3", "airblast_dbl", "violations"]])
                else:
                    st.write("No rejected designs.")
            else:
                st.write("No rejected designs.")

        ppv_col = "ppv_mms" if "ppv_mms" in df.columns else "vibration_ppv_mms"
        fig_pareto = plot_pareto_front(df, x_objective="d80_mm", y_objective=ppv_col)
        st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Pareto Front Design Candidates")
    st.dataframe(df, use_container_width=True)
