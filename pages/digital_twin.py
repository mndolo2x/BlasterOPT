import os
import streamlit as st
import pandas as pd
import numpy as np
from src.components.model_selector import render_page_model_selector
from src.ui.bench_twin import (
    make_bench_figure,
    load_holes_from_design,
    load_geology_overlays,
)

st.title("Digital Twin of Bench")

model, model_key = render_page_model_selector("digital_twin")
if model is None:
    st.stop()

# Get active design or default
active_design = st.session_state.get("current_design", st.session_state.get("last_predict_inputs"))

st.subheader("3D Spatial Digital Twin Viewer")

# 1. Control toggles
col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    show_holes = st.toggle("Show drill holes", value=True)
with col_t2:
    show_geology = st.toggle("Show geology overlays", value=True)
with col_t3:
    show_faults = st.toggle("Show fault planes", value=True)

# 2. Render 3D Bench Scene
holes = load_holes_from_design(active_design)
bench_id = getattr(active_design, "bench_id", "B14") if active_design else "B14"
site_id = getattr(active_design, "site_id", "jwaneng") if active_design else "jwaneng"
overlays = load_geology_overlays(bench_id, site_id)

fig = make_bench_figure(
    holes=holes,
    overlays=overlays,
    show_holes=show_holes,
    show_geology=show_geology,
    show_faults=show_faults,
    title=f"Digital Twin — {bench_id}",
)

st.plotly_chart(fig, use_container_width=True, config={"renderer": "svg"})

# 3. Legend and Stats Panel
st.divider()
st.subheader("Bench Statistics & Geology Legend")

col_stat1, col_stat2 = st.columns(2)

with col_stat1:
    st.markdown("### 🏷️ Geology Overlays Legend")
    for ov in overlays:
        st.markdown(f"- <span style='color:{ov[\"color\"]}; font-weight:bold;'>■</span> **{ov[\"name\"]}** (Opacity: {ov[\"opacity\"]})", unsafe_allow_html=True)

with col_stat2:
    st.markdown("### 📊 Bench Pattern Stats")
    num_holes = len(holes)
    total_charge = sum(h["charge_kg"] for h in holes) if holes else 0.0
    avg_charge = total_charge / max(num_holes, 1)

    st.metric("Total Holes", num_holes)
    st.metric("Total Explosive Charge", f"{total_charge:,.1f} kg")
    st.metric("Average Charge per Hole", f"{avg_charge:.1f} kg")
    st.caption("Bench Dimensions: **60.0 m × 40.0 m × 12.0 m**")

# 4. Navigation to Blast Design Engine
st.divider()
if st.button("Send to Blast Design Engine →", type="primary"):
    st.session_state["bench_twin_data"] = {
        "bench_id": bench_id,
        "site_id": site_id,
        "holes": holes,
        "overlays": overlays,
    }
    st.success("Transferred bench twin data to Blast Design Engine!")

# 5. Export options
st.divider()
st.subheader("Export Options")
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    if st.button("Export as PNG"):
        try:
            img_bytes = fig.to_image(format="png")
            st.download_button(
                label="Download PNG Image",
                data=img_bytes,
                file_name=f"bench_twin_{bench_id}.png",
                mime="image/png",
            )
        except Exception as e:
            st.error(f"PNG export failed (requires kaleido): {e}")

with col_exp2:
    if st.button("Export as HTML"):
        try:
            html_bytes = fig.to_html().encode("utf-8")
            st.download_button(
                label="Download Standalone HTML",
                data=html_bytes,
                file_name=f"bench_twin_{bench_id}.html",
                mime="text/html",
            )
        except Exception as e:
            st.error(f"HTML export failed: {e}")
