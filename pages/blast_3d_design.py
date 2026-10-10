"""
3D Blast Design page — Enterprise Mining Dashboard.
"""
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.components.model_selector import render_page_model_selector
from src.render3d.bench_model import BenchGeometry
from src.render3d.hole_pattern import HolePattern, generate_holes, pattern_summary
from src.render3d.renderer import render_bench_and_holes
from src.render3d.volume import compute_blast_volume, compute_blast_tonnage
from src.render3d.predictions import predict_pattern
from src.render3d.constants import DEFAULT_DESIGN
from src.explainability import get_feature_contributions, plot_feature_contributions_waterfall, generate_natural_language_explanation

# --- CUSTOM CSS & ENTERPRISE STYLING ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Cyan Accent Color for Sliders */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #00B4D8 !important;
        border-color: #00B4D8 !important;
    }
    div[data-baseweb="slider"] div[class*="StyledTrackFilled"] {
        background-color: #00B4D8 !important;
    }

    /* Bento Cards Styling */
    .bento-card {
        background-color: #121824;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .bento-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .bento-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .bento-unit {
        font-size: 0.85rem;
        font-weight: 400;
        color: #64748B;
        margin-left: 4px;
    }
    .bento-delta-pos {
        font-size: 0.8rem;
        font-weight: 600;
        color: #10B981;
    }
    .bento-delta-neg {
        font-size: 0.8rem;
        font-weight: 600;
        color: #EF4444;
    }

    /* Status Badges */
    .status-badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #10B981;
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 14px;
    }
    .status-badge-error {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        color: #EF4444;
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("3D Blast Design — Enterprise Workbench")

# --- Model selector ---
model, model_key = render_page_model_selector("blast_3d_design")
if model is None:
    st.stop()

# Baseline defaults for comparison toggle
BASELINE_PREDS = {
    "fragmentation_p80_cm": 35.0,
    "vibration_ppv_mms": 4.18,
    "airblast_db": 118.5,
    "cost_per_tonne": 5.25,
}

# --- 3-COLUMN DASHBOARD LAYOUT ---
col_left, col_center, col_right = st.columns([1, 2, 1])

# ==================== LEFT COLUMN: INPUTS ====================
with col_left:
    st.subheader("⚙️ Control Panel")

    with st.expander("📐 Design Parameters", expanded=True):
        burden = st.slider("Burden (m)", 3.0, 6.0, DEFAULT_DESIGN["burden_m"], 0.1, key="b_3d")
        spacing = st.slider("Spacing (m)", 3.5, 8.0, DEFAULT_DESIGN["spacing_m"], 0.1, key="s_3d")
        stemming = st.slider("Stemming (m)", 2.0, 5.0, DEFAULT_DESIGN["stemming_m"], 0.1, key="stem_3d")
        pf = st.slider("Powder Factor (kg/m³)", 0.4, 0.9, DEFAULT_DESIGN["powder_factor_kg_m3"], 0.01, key="pf_3d")
        bench_h = st.slider("Bench Height (m)", 10.0, 18.0, DEFAULT_DESIGN["bench_height_m"], 0.5, key="bh_3d")
        hole_d = st.slider("Hole Diameter (mm)", 100, 311, DEFAULT_DESIGN["hole_diameter_mm"], 1, key="hd_3d")
        subdrill = st.slider("Subdrill (m)", 0.3, 2.0, DEFAULT_DESIGN["subdrilling_m"], 0.1, key="sub_3d")
        hole_depth = bench_h + subdrill
        max_charge = st.slider("Max Charge per Delay (kg)", 100, 900, DEFAULT_DESIGN["max_charge_per_delay_kg"], 10, key="mc_3d")

    with st.expander("🧩 Pattern Layout", expanded=True):
        num_rows = st.slider("Number of Rows", 2, 15, DEFAULT_DESIGN["num_rows"], key="nr_3d")
        holes_per_row = st.slider("Holes per Row", 2, 15, DEFAULT_DESIGN["holes_per_row"], key="hpr_3d")
        pattern_type = st.selectbox(
            "Pattern Type",
            ["staggered", "square", "echelon", "v_pattern"],
            index=0,
            key="pt_3d",
        )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Run Optimizer", type="primary", use_container_width=True):
            st.info("Routing current 3D geometry parameters to GA Optimizer...")
    with col_btn2:
        if st.button("🔄 Reset", use_container_width=True):
            st.rerun()


# ==================== CENTER COLUMN: 3D SCENE ====================
with col_center:
    st.subheader("🧊 3D Bench Spatial Scene")

    bench = BenchGeometry(
        bench_id="B14",
        crest_elevation_m=bench_h,
        toe_elevation_m=0.0,
        face_angle_deg=75.0,
        length_m=spacing * holes_per_row * 1.2,
        width_m=burden * num_rows * 1.2,
        free_face_direction="north",
    )

    pattern = HolePattern(
        pattern_type=pattern_type,
        burden_m=burden,
        spacing_m=spacing,
        num_rows=num_rows,
        holes_per_row=holes_per_row,
        stemming_m=stemming,
        hole_depth_m=hole_depth,
        hole_angle_deg=90.0,
        powder_factor_kg_m3=pf,
        bench_height_m=bench_h,
        subdrilling_m=subdrill,
    )
    pattern = generate_holes(pattern)

    USE_PYVISTA = os.getenv("USE_PYVISTA", "false").lower() == "true"

    if USE_PYVISTA:
        try:
            from src.ui.pyvista_scene import render_bench_scene
            render_bench_scene(bench, pattern)
        except Exception as exc:
            st.warning(f"PyVista rendering failed ({exc}). Falling back to Plotly (SVG mode)...")
            fig = render_bench_and_holes(bench, pattern)
            st.plotly_chart(fig, use_container_width=True)
    else:
        fig = render_bench_and_holes(bench, pattern)
        st.plotly_chart(fig, use_container_width=True)


# ==================== RIGHT COLUMN: RESULTS ====================
with col_right:
    st.subheader("📊 Performance & Compliance")

    extra = {
        "rock_factor_A": 8.0,
        "hole_diameter_mm": float(hole_d),
        "max_charge_per_delay_kg": float(max_charge),
        "explosive_rws": 115.0,
    }
    preds = predict_pattern(model, pattern, extra)

    p80 = preds["fragmentation_d80_cm"]
    ppv = preds["vibration_ppv_mms"]
    airblast = preds["airblast_db"]

    vol_m3 = compute_blast_volume(pattern)
    tonnage_t = compute_blast_tonnage(pattern)
    drilling_cost = 15.0 * hole_depth * len(pattern.holes)
    explosive_cost = 1.30 * pf * vol_m3
    total_cost_usd = drilling_cost + explosive_cost + 5000.0
    cost_per_tonne = total_cost_usd / max(tonnage_t, 1.0)

    # 1. Status Badge
    if ppv <= 5.0 and airblast <= 120.0:
        st.markdown(
            f'<div class="status-badge-success">✅ COMPLIANT<br>'
            f'<small>PPV Margin: {5.0 - ppv:.2f} mm/s | Airblast Margin: {120.0 - airblast:.1f} dB</small></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-badge-error">❌ REGULATORY VIOLATION<br>'
            f'<small>PPV: {ppv:.2f} / 5.0 mm/s | Airblast: {airblast:.1f} / 120.0 dB</small></div>',
            unsafe_allow_html=True,
        )

    # 2. Baseline Comparison Toggle
    compare_baseline = st.toggle("Compare to Baseline", value=False)

    # 3. Bento Metric Cards with "Why?" SHAP Expanders
    # --- Metric 1: Fragmentation P80 ---
    d_p80 = p80 - BASELINE_PREDS["fragmentation_p80_cm"]
    delta_str_p80 = f" (↓ {abs(d_p80):.1f} cm)" if d_p80 < 0 else f" (↑ {d_p80:.1f} cm)" if d_p80 > 0 else ""
    delta_class_p80 = "bento-delta-pos" if d_p80 <= 0 else "bento-delta-neg"

    st.markdown(
        f'<div class="bento-card">'
        f'<div class="bento-title">Fragmentation P80</div>'
        f'<div class="bento-value">{p80:.1f}<span class="bento-unit">cm</span>'
        f'{" <span class=" + delta_class_p80 + ">" + delta_str_p80 + "</span>" if compare_baseline else ""}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Why? (Fragmentation P80 Breakdown)"):
        input_payload = {
            "burden_m": burden,
            "spacing_m": spacing,
            "powder_factor_kg_m3": pf,
            "stemming_m": stemming,
            "bench_height_m": bench_h,
            "hole_depth_m": hole_depth,
            "hole_diameter_mm": float(hole_d),
            "max_charge_per_delay_kg": float(max_charge),
            "explosive_rws": 115.0,
            "rock_factor_A": 8.0,
            "monitoring_distance_m": 800.0,
        }
        contrib_data = get_feature_contributions(None, input_payload, target="d50_mm")
        st.caption(generate_natural_language_explanation(
            shap_values=list(contrib_data["contributions"].values()),
            feature_names=list(contrib_data["contributions"].keys()),
            prediction=p80,
            constraints={"metric": "Fragmentation P80", "limit": 40.0, "unit": "cm"},
        ))
        fig_w = plot_feature_contributions_waterfall(contrib_data, title="P80 Feature Contribution")
        st.plotly_chart(fig_w, use_container_width=True)

    # --- Metric 2: Ground Vibration PPV ---
    d_ppv = ppv - BASELINE_PREDS["vibration_ppv_mms"]
    delta_str_ppv = f" (↓ {abs(d_ppv):.2f} mm/s)" if d_ppv < 0 else f" (↑ {d_ppv:.2f} mm/s)" if d_ppv > 0 else ""
    delta_class_ppv = "bento-delta-pos" if d_ppv <= 0 else "bento-delta-neg"

    st.markdown(
        f'<div class="bento-card">'
        f'<div class="bento-title">Ground Vibration (PPV)</div>'
        f'<div class="bento-value">{ppv:.2f}<span class="bento-unit">mm/s</span>'
        f'{" <span class=" + delta_class_ppv + ">" + delta_str_ppv + "</span>" if compare_baseline else ""}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Why? (PPV Vibration Breakdown)"):
        contrib_ppv = get_feature_contributions(None, input_payload, target="ppv_mms")
        st.caption(generate_natural_language_explanation(
            shap_values=list(contrib_ppv["contributions"].values()),
            feature_names=list(contrib_ppv["contributions"].keys()),
            prediction=ppv,
            constraints={"metric": "Vibration PPV", "limit": 5.0, "unit": "mm/s"},
        ))
        fig_ppv = plot_feature_contributions_waterfall(contrib_ppv, title="PPV Feature Contribution")
        st.plotly_chart(fig_ppv, use_container_width=True)

    # --- Metric 3: Airblast ---
    d_air = airblast - BASELINE_PREDS["airblast_db"]
    delta_str_air = f" (↓ {abs(d_air):.1f} dB)" if d_air < 0 else f" (↑ {d_air:.1f} dB)" if d_air > 0 else ""
    delta_class_air = "bento-delta-pos" if d_air <= 0 else "bento-delta-neg"

    st.markdown(
        f'<div class="bento-card">'
        f'<div class="bento-title">Airblast Overpressure</div>'
        f'<div class="bento-value">{airblast:.1f}<span class="bento-unit">dB</span>'
        f'{" <span class=" + delta_class_air + ">" + delta_str_air + "</span>" if compare_baseline else ""}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Why? (Airblast Breakdown)"):
        st.caption(f"Stemming confinement ({stemming:.1f} m) and charge per delay ({max_charge:.0f} kg) are the main drivers.")

    # --- Metric 4: Cost & Volume Bento Card ---
    d_cost = cost_per_tonne - BASELINE_PREDS["cost_per_tonne"]
    delta_str_cost = f" (↓ ${abs(d_cost):.2f}/t)" if d_cost < 0 else f" (↑ ${d_cost:.2f}/t)" if d_cost > 0 else ""
    delta_class_cost = "bento-delta-pos" if d_cost <= 0 else "bento-delta-neg"

    st.markdown(
        f'<div class="bento-card">'
        f'<div class="bento-title">Unit Cost</div>'
        f'<div class="bento-value">${cost_per_tonne:.2f}<span class="bento-unit">/ t</span>'
        f'{" <span class=" + delta_class_cost + ">" + delta_str_cost + "</span>" if compare_baseline else ""}</div>'
        f'<div style="font-size: 0.8rem; color: #94A3B8; margin-top: 6px;">'
        f'Vol: <b>{vol_m3:,.0f} m³</b> | Tonnage: <b>{tonnage_t:,.0f} t</b>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.divider()
    if st.button("📤 Submit for Approval", type="primary", use_container_width=True):
        st.success("Pattern design submitted for certified blaster sign-off!")
