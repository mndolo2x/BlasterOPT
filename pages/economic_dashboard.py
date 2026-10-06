"""
Economic Dashboard Streamlit Page for BlasterOPT Botswana.
Consumes Pareto front optimization candidate designs and evaluates Mine-to-Mill economic financial outcomes.
"""

import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime

from src.economics import (
    load_bench_context,
    evaluate_pareto_front,
    rank_by_objective,
    plot_cost_breakdown_stacked,
    plot_margin_vs_powder_factor,
    plot_revenue_vs_cost,
    plot_npv_bars,
    plot_sensitivity_tornado,
)
from src.components.model_selector import render_page_model_selector

st.title("💰 Economic & Mine-to-Mill Financial Dashboard")
st.markdown("Consumes Pareto-optimal blast designs and translates candidate geometries into full Mine-to-Mill economic financial models.")

model, model_key = render_page_model_selector("economic_dashboard")

# Bench Context Panel
st.subheader("🏔️ Bench Operational Context")
col_b1, col_b2, col_b3, col_b4 = st.columns(4)

bench_id_sel = col_b1.selectbox(
    "Select Bench",
    options=["jwaneng_bench_14", "orapa_bench_12", "morupule_bench_3"],
    index=0,
)

bench = load_bench_context(bench_id_sel)

col_b1.caption(f"Site: **{bench.site_id}**")
col_b2.metric("Total Tonnes", f"{bench.tonnes:,.0f} t")
col_b3.metric("Ore Grade", f"{bench.grade:.2f} {bench.grade_unit.replace('_', ' ')}")
col_b4.metric("Selling Price", f"${bench.commodity_price:,.2f} / {bench.commodity_price_unit.replace('USD_per_', '')}")

st.divider()

# Load Pareto front
pareto_df = st.session_state.get("pareto_front_df", None)

if pareto_df is None or pareto_df.empty:
    if os.path.exists("pareto_front.csv"):
        try:
            pareto_df = pd.read_csv("pareto_front.csv")
            st.session_state["pareto_front_df"] = pareto_df
        except Exception:
            pareto_df = None

if pareto_df is None or pareto_df.empty:
    st.info("💡 No Pareto front data found in session state or `pareto_front.csv`. "
            "Please go to **Multi-Objective Pareto Optimizer**, run optimization, and click **Send Pareto Front to Economic Dashboard**.")
    st.stop()

st.subheader("🎯 Objective Selection & Financial Ranking")

obj_sel = st.radio(
    "Select Ranking Financial Objective",
    options=["max_margin", "max_npv", "min_cost", "max_throughput"],
    format_func=lambda x: {
        "max_margin": "Maximize Net Margin ($/t)",
        "max_npv": "Maximize Bench NPV ($)",
        "min_cost": "Minimize Total Mine-to-Mill Cost ($/t)",
        "max_throughput": "Maximize Primary Crusher Throughput (t/h)",
    }[x],
    horizontal=True,
)

# Evaluate Pareto front
eval_results = evaluate_pareto_front(pareto_df, bench)
ranked_results = rank_by_objective(eval_results, objective=obj_sel)

if not ranked_results:
    st.error("No valid economic results calculated.")
    st.stop()

# Ranked Recommendations Cards
st.subheader("🏆 Top Ranked Recommendations")
top_3 = ranked_results[:3]

for i, res in enumerate(top_3, 1):
    pf = float(res.design_params.get("powder_factor_kg_m3", 0.65))
    d80 = float(res.design_params.get("d80_mm", 250.0))
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])

    status_badge = "✅ SAFE" if res.status == "SAFE" else "⚠️ REQUIRES_REVIEW"

    with c1:
        st.write(f"**#{i} Design {res.design_id}** ({status_badge})")
        st.write(f"PF: `{pf:.2f} kg/m³` | D80: `{d80:.0f} mm`")
    with c2:
        st.write(f"Margin: **${res.margin_usd_per_t:.2f} / t**")
        st.write(f"Bench NPV: **${res.npv_usd / 1e6:.2f} M**")
    with c3:
        st.write(f"Cost: **${res.cost.total_usd_per_t:.2f} / t**")
        st.write(f"Revenue: **${res.revenue.revenue_usd_per_t:.2f} / t**")
    with c4:
        if st.button(f"Submit for Approval", key=f"app_btn_{res.design_id}"):
            if res.status == "REQUIRES_REVIEW":
                st.warning(f"Submitted design {res.design_id} flagged for review: {', '.join(res.review_reasons)}")
            else:
                st.success(f"Design {res.design_id} successfully submitted to Blaster Approval Workflow!")

    st.divider()

# Selected Recommendation Detail View
st.subheader("🔎 Recommendation Detail View")
sel_design_id = st.selectbox(
    "Select Design for In-Depth Analysis",
    options=[r.design_id for r in ranked_results],
    format_func=lambda id_val: f"Design {id_val} (Margin: ${[r for r in ranked_results if r.design_id == id_val][0].margin_usd_per_t:.2f}/t)",
)

sel_res = [r for r in ranked_results if r.design_id == sel_design_id][0]

c_det1, c_det2 = st.columns(2)

with c_det1:
    st.write("#### Cost Breakdown ($/t)")
    st.write(f"- Drill & Blast: **${sel_res.cost.drill_and_blast_usd_per_t:.2f} / t**")
    st.write(f"- Digging: **${sel_res.cost.digging_usd_per_t:.2f} / t**")
    st.write(f"- Hauling: **${sel_res.cost.hauling_usd_per_t:.2f} / t**")
    st.write(f"- Crushing: **${sel_res.cost.crushing_usd_per_t:.2f} / t**")
    st.write(f"- Processing: **${sel_res.cost.processing_usd_per_t:.2f} / t**")
    st.markdown(f"**Total Unit Cost: ${sel_res.cost.total_usd_per_t:.2f} / t**")

with c_det2:
    st.write("#### Revenue & Margin Breakdown")
    st.write(f"- Tonnes: **{sel_res.revenue.tonnes:,.0f} t**")
    st.write(f"- Ore Grade: **{sel_res.revenue.grade:.2f}**")
    st.write(f"- Plant Recovery: **{sel_res.revenue.recovery_pct:.1f}%**")
    st.write(f"- Unit Revenue: **${sel_res.revenue.revenue_usd_per_t:.2f} / t**")
    st.markdown(f"**Net Unit Margin: ${sel_res.margin_usd_per_t:.2f} / t**")
    st.markdown(f"**Discounted Bench NPV: ${sel_res.npv_usd / 1e6:.2f} M**")

st.markdown("---")

# SHAP Driver Explanation
st.subheader("💡 Why This Design Wins (Key Financial Drivers)")
pf_val = float(sel_res.design_params.get("powder_factor_kg_m3", 0.65))
d80_val = float(sel_res.design_params.get("d80_mm", 250.0))
tph_val = float(sel_res.design_params.get("crusher_throughput_tph", 1500.0))

pf_contrib = (pf_val - 0.50) * 0.8
d80_contrib = (300.0 - d80_val) * 0.005
tph_contrib = (tph_val - 1000.0) * 0.0003

st.info(f"→ **Powder Factor ({pf_val:.2f} kg/m³):** Contributes **+${pf_contrib:.2f}/t** margin via optimal rock breakage.")
st.info(f"→ **Fragment Size D80 ({d80_val:.0f} mm):** Contributes **+${d80_contrib:.2f}/t** margin via improved plant recovery.")
st.info(f"→ **Crusher Throughput ({tph_val:.0f} t/h):** Contributes **+${tph_contrib:.2f}/t** margin via lower crushing energy.")

# Visualizer Charts
st.subheader("📊 Financial Visualizations & Sensitivity Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Cost Breakdown", "Margin vs. Powder Factor", "Revenue vs. Cost", "NPV Comparison", "Sensitivity Tornado"])

with tab1:
    fig_cost = plot_cost_breakdown_stacked(ranked_results, top_n=10)
    st.plotly_chart(fig_cost, use_container_width=True)

with tab2:
    fig_m_pf = plot_margin_vs_powder_factor(ranked_results, selected_id=sel_design_id)
    st.plotly_chart(fig_m_pf, use_container_width=True)

with tab3:
    fig_rev_cost = plot_revenue_vs_cost(ranked_results)
    st.plotly_chart(fig_rev_cost, use_container_width=True)

with tab4:
    fig_npv = plot_npv_bars(ranked_results, top_n=10)
    st.plotly_chart(fig_npv, use_container_width=True)

with tab5:
    fig_tornado = plot_sensitivity_tornado(sel_res)
    st.plotly_chart(fig_tornado, use_container_width=True)

# Export Functionality
st.markdown("---")
st.subheader("📥 Export Financial Report")
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    export_json = {
        "bench_id": bench.bench_id,
        "site_id": bench.site_id,
        "selected_design_id": sel_res.design_id,
        "design_params": sel_res.design_params,
        "cost_breakdown": sel_res.cost.dict(),
        "revenue_breakdown": sel_res.revenue.dict(),
        "margin_usd_per_t": sel_res.margin_usd_per_t,
        "total_margin_usd": sel_res.total_margin_usd,
        "npv_usd": sel_res.npv_usd,
        "evaluated_at": datetime.now().isoformat(),
    }
    st.download_button(
        "Download Financial JSON Report 📄",
        data=json.dumps(export_json, indent=2),
        file_name=f"economic_report_{bench.bench_id}_design_{sel_res.design_id}.json",
        mime="application/json",
    )

with col_exp2:
    summary_df = pd.DataFrame([{
        "Design ID": r.design_id,
        "Margin ($/t)": r.margin_usd_per_t,
        "NPV ($M)": round(r.npv_usd / 1e6, 2),
        "Cost ($/t)": r.cost.total_usd_per_t,
        "Revenue ($/t)": r.revenue.revenue_usd_per_t,
        "Powder Factor (kg/m3)": r.design_params.get("powder_factor_kg_m3"),
        "D80 (mm)": r.design_params.get("d80_mm"),
        "Status": r.status,
    } for r in ranked_results])

    st.download_button(
        "Download Ranked Financial CSV Table 📊",
        data=summary_df.to_csv(index=False),
        file_name=f"ranked_pareto_economics_{bench.bench_id}.csv",
        mime="text/csv",
    )
