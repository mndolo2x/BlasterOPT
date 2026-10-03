import streamlit as st
import pandas as pd
from src.components.model_selector import render_page_model_selector
from src.pareto_optimizer import run_nsga2, plot_pareto_front, _check_constraints

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
        front.to_csv("pareto_front.csv", index=False)
        st.session_state["pareto_front_df"] = front

front_df = st.session_state.get("pareto_front_df")

if front_df is not None and not front_df.empty:
    with col2:
        st.subheader("Physical Constraint Verification")

        failed_count = 0
        total_count = len(front_df)
        ppv_col = "ppv_mms" if "ppv_mms" in front_df.columns else "vibration_ppv_mms"

        for idx, row in front_df.iterrows():
            burden = float(row["burden_m"])
            spacing = float(row["spacing_m"])
            stemming = float(row["stemming_m"])
            ppv = float(row[ppv_col])

            if (spacing < burden or
                spacing > 1.5 * burden + 1e-4 or
                stemming < 0.5 * burden - 1e-4 or
                stemming > 1.0 * burden + 1e-4 or
                ppv > 0.8 * max_ppv_limit + 1e-4 or
                burden < 2.0 or burden > 12.0):
                failed_count += 1

        if failed_count == 0:
            st.success(f"✅ All {total_count} Pareto designs are physically valid.")
        else:
            st.error(f"❌ {failed_count} designs violate physical constraints.")

        fig_pareto = plot_pareto_front(front_df, x_objective="d80_mm", y_objective=ppv_col)
        st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Pareto Front Design Candidates")
    st.dataframe(front_df, use_container_width=True)
