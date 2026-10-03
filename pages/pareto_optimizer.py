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

        if len(front) == 0:
            st.error("No physically valid designs found. Adjust the constraints.")
            st.stop()

        front.to_csv("pareto_front.csv", index=False)
        st.session_state["pareto_front_df"] = front

df = st.session_state.get("pareto_front_df")

if df is not None and not df.empty:
    with col2:
        st.subheader("Physical Constraint Verification")

        invalid_count = (df["spacing_m"] < df["burden_m"]).sum()
        if invalid_count == 0:
            st.success(f"✅ All {len(df)} designs are physically valid.")
        else:
            st.error(f"❌ {invalid_count} designs violate spacing >= burden. This is a bug.")

        ppv_col = "ppv_mms" if "ppv_mms" in df.columns else "vibration_ppv_mms"
        fig_pareto = plot_pareto_front(df, x_objective="d80_mm", y_objective=ppv_col)
        st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Pareto Front Design Candidates")
    st.dataframe(df, use_container_width=True)
