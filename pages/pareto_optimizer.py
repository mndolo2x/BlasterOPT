import streamlit as st
import pandas as pd
from src.components.model_selector import render_page_model_selector
from src.pareto_optimizer import run_nsga2, plot_pareto_front, is_physically_valid

st.title("Multi-Objective Pareto Optimizer (NSGA-II)")

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

    run_opt_btn = st.button("Run Pareto Optimizer 🚀", type="primary")

if run_opt_btn or st.session_state.get("pareto_front_df") is None:
    constraints_info = {"max_ppv": max_ppv_limit, "max_airblast": max_air_limit}
    with st.spinner("Running NSGA-II Multi-Objective Optimization..."):
        front = run_nsga2(model=model, n_gen=n_gen, pop_size=pop_size, constraints_info=constraints_info)

        # Filter out any design where valid is False
        if "valid" in front.columns:
            front = front[front["valid"] == True].reset_index(drop=True)

        front = front[
            (front["airblast_dbl"] <= 120.0 + 1e-4) &
            (front["ppv_mms"] <= 0.8 * max_ppv_limit + 1e-4)
        ].reset_index(drop=True)

        if len(front) == 0:
            st.error("No designs meet regulatory limits. Check constraints.")
            st.stop()

        front.to_csv("pareto_front.csv", index=False)
        st.session_state["pareto_front_df"] = front

df = st.session_state.get("pareto_front_df")

if df is not None and not df.empty:
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
