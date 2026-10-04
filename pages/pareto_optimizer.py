import streamlit as st
import pandas as pd
from datetime import datetime
from src.components.model_selector import render_page_model_selector
from src.pareto_optimizer import run_nsga2, plot_pareto_front, is_physically_valid

st.title("Multi-Objective Pareto Optimizer (NSGA-II)")

PARETO_VERSION = 5  # increment every time the optimizer bounds or logic change

if st.session_state.get("pareto_version") != PARETO_VERSION:
    for key in ["pareto_front_df", "pareto_front_timestamp", "pareto_front_hash"]:
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
            for key in ["pareto_front_df", "pareto_front_timestamp", "pareto_front_hash"]:
                st.session_state.pop(key, None)
            st.rerun()

# Run only when the button is clicked
if run_opt_btn:
    constraints_info = {"max_ppv": max_ppv_limit, "max_airblast": max_air_limit}
    with st.spinner("Running NSGA-II Multi-Objective Optimization..."):
        front = run_nsga2(
            model=model,
            n_gen=n_gen,
            pop_size=pop_size,
            constraints_info=constraints_info,
        )

        # Filter out any design where valid is False or violating regulatory limits
        if "valid" in front.columns:
            front = front[front["valid"] == True].reset_index(drop=True)

        front = front[
            (front["airblast_dbl"] <= 120.0 + 1e-4) &
            (front["ppv_mms"] <= 0.8 * max_ppv_limit + 1e-4)
        ].reset_index(drop=True)

        if len(front) == 0:
            st.error("No designs meet regulatory limits. Check constraints.")
            st.stop()

        front_hash = str(pd.util.hash_pandas_object(front).sum())
        front.to_csv("pareto_front.csv", index=False)
        st.session_state["pareto_front_df"] = front
        st.session_state["pareto_front_timestamp"] = datetime.now().isoformat()
        st.session_state["pareto_front_hash"] = front_hash

# Load from cache for display
df = st.session_state.get("pareto_front_df", None)

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

        violations = (df["airblast_dbl"] > 120.0).sum()
        if violations == 0:
            st.success(f"✅ All {len(df)} designs comply with Botswana limits.")
        else:
            st.error(f"❌ {violations} designs exceed the 120 dB airblast limit.")

        ppv_col = "ppv_mms" if "ppv_mms" in df.columns else "vibration_ppv_mms"
        fig_pareto = plot_pareto_front(df, x_objective="d80_mm", y_objective=ppv_col)
        st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Pareto Front Design Candidates")
    st.dataframe(df, use_container_width=True)
