"""
Similar Blasts Recommender Streamlit Page.
"""

import streamlit as st
from src.recommender import find_similar_blasts

st.title("👥 Similar Blast Recommender & Knowledge Transfer")
st.markdown("Query historical blast logs to find nearest-neighbor designs and learn from past outcomes.")

# Step 1: Verify data source
df = st.session_state.get("df")
if df is None or len(df) == 0:
    st.error(
        "⚠️ No historical blast data loaded. "
        "Go to **Data Ingestion & Generator**, generate a dataset, then return here."
    )
    st.stop()

# Required column checks
required_cols = [
    "burden_m", "spacing_m", "powder_factor_kg_m3", "stemming_m", "rock_factor_A",
    "airblast_db", "cost_per_tonne_usd"
]

if df is not None and len(df) > 0:
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"⚠️ Dataset is missing required column(s): {', '.join(missing_cols)}")

c_rec1, c_rec2 = st.columns([1, 2])

with c_rec1:
    st.subheader("Query Parameters")
    last_in = st.session_state.get("last_predict_inputs", {})

    burden = st.number_input("Burden (m)", 2.0, 12.0, float(last_in.get("burden_m", 4.2)), step=0.1, key="sb_b")
    spacing = st.number_input("Spacing (m)", 2.0, 15.0, float(last_in.get("spacing_m", 5.1)), step=0.1, key="sb_s")
    pf = st.number_input("Powder Factor (kg/m³)", 0.2, 2.5, float(last_in.get("powder_factor_kg_m3", 0.65)), step=0.05, key="sb_pf")
    stemming = st.number_input("Stemming (m)", 1.0, 10.0, float(last_in.get("stemming_m", 3.0)), step=0.1, key="sb_stem")
    rock_factor = st.number_input("Rock Factor (A)", 4.0, 16.0, float(last_in.get("rock_factor_A", 8.0)), step=0.5, key="sb_rock")

    n_matches = st.slider("Number of Similar Blasts to Retrieve", 1, 20, 5, key="sb_topk")

# Step 3: Wire the button to the search
if st.button("🔍 Find Similar Blasts", type="primary"):
    query_design = {
        "burden_m": burden,
        "spacing_m": spacing,
        "powder_factor_kg_m3": pf,
        "stemming_m": stemming,
        "rock_factor_A": rock_factor,
    }

    with st.spinner(f"Searching for {n_matches} similar blasts..."):
        results = find_similar_blasts(
            query_design=query_design,
            top_k=n_matches,
        )

    if len(results) == 0:
        st.error(
            "⚠️ No similar blasts found. "
            "Make sure a dataset is loaded in **Data Ingestion & Generator**."
        )
    else:
        st.session_state["similar_blasts_results"] = results
        st.session_state["similar_blasts_query"] = query_design
        st.success(f"✅ Found {len(results)} similar blasts.")

# Step 4: Display the results
if "similar_blasts_results" in st.session_state:
    results = st.session_state["similar_blasts_results"]

    st.divider()
    st.subheader(f"Top {len(results)} Similar Blasts")

    st.caption(
        "**How to read these results:**  \n"
        "• **Similarity** — lower means more similar to your current design  \n"
        "• **Outcomes are historical** — these are real past blasts, not predictions  \n"
        "• **Compliant** — green check means the blast met Botswana's PPV and airblast limits  \n"
        "• **Non-compliant blasts are still shown** — they teach us what to avoid"
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "compliant": st.column_config.CheckboxColumn(
                "Compliant",
                help="Within Botswana's PPV (5.0 mm/s) and airblast (120 dB) limits",
                default=False,
            ),
            "distance": st.column_config.NumberColumn(
                "Similarity",
                format="%.4f",
                help="Lower = more similar to your query",
            ),
            "d80_mm": st.column_config.NumberColumn("D80 (mm)", format="%.1f"),
            "ppv_mms": st.column_config.NumberColumn("PPV (mm/s)", format="%.2f"),
            "airblast_db": st.column_config.NumberColumn("Airblast (dB)", format="%.1f"),
            "cost_per_tonne_usd": st.column_config.NumberColumn("Cost ($/t)", format="$%.2f"),
        },
    )

    # Show the query
    query = st.session_state["similar_blasts_query"]
    st.caption(
        "Query: "
        + " | ".join(f"{k} = {v:.2f}" if isinstance(v, float) else f"{k} = {v}"
                      for k, v in query.items())
    )

    # Operational Insights Section
    compliant_count = int(results["compliant"].sum()) if "compliant" in results else 0
    total_count = len(results)

    st.divider()
    st.subheader("Operational Insights")

    col1, col2, col3 = st.columns(3)
    col1.metric("Blasts Returned", total_count)
    col2.metric("Compliant", f"{compliant_count}/{total_count}")
    d80_mean = results['d80_mm'].mean() if 'd80_mm' in results and len(results) > 0 else 0.0
    col3.metric(
        "Avg D80 (mm)",
        f"{d80_mean:.0f}",
    )

    if len(results) > 0:
        # Best-performing blast
        best = results.loc[results["cost_per_tonne_usd"].idxmin()]
        st.write(
            f"**Lowest cost in the match set:** "
            f"`{best['blast_id']}` at **${best['cost_per_tonne_usd']:.2f}/t** "
            f"(D80 = {best['d80_mm']:.0f} mm, PPV = {best['ppv_mms']:.2f} mm/s)"
        )

        # Largest vibration risk
        worst_ppv = results.loc[results["ppv_mms"].idxmax()]
        st.write(
            f"**Highest vibration in the match set:** "
            f"`{worst_ppv['blast_id']}` at **{worst_ppv['ppv_mms']:.2f} mm/s** "
            f"(limit is 5.0 mm/s)"
        )

    # Clear button
    if st.button("Clear results"):
        del st.session_state["similar_blasts_results"]
        del st.session_state["similar_blasts_query"]
        st.rerun()
