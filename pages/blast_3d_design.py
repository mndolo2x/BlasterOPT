"""
3D Blast Design page.
"""
import streamlit as st
from src.components.model_selector import render_page_model_selector
from src.render3d.bench_model import BenchGeometry
from src.render3d.hole_pattern import HolePattern, generate_holes, pattern_summary
from src.render3d.renderer import render_bench_and_holes
from src.render3d.volume import compute_blast_volume, compute_blast_tonnage
from src.render3d.predictions import predict_pattern
from src.render3d.constants import DEFAULT_DESIGN

st.title("3D Blast Design")
st.caption(
    "Interactive 3D drill pattern layout with timing, geology, and model predictions."
)

# --- Model selector (reused component) ---
model, model_key = render_page_model_selector("blast_3d_design")
if model is None:
    st.stop()

st.divider()

# --- Input panel ---
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Design Parameters")
    burden = st.slider("Burden (m)", 3.0, 6.0, DEFAULT_DESIGN["burden_m"], 0.1)
    spacing = st.slider("Spacing (m)", 3.5, 8.0, DEFAULT_DESIGN["spacing_m"], 0.1)
    stemming = st.slider("Stemming (m)", 2.0, 5.0, DEFAULT_DESIGN["stemming_m"], 0.1)
    pf = st.slider("Powder Factor (kg/m³)", 0.4, 0.9, DEFAULT_DESIGN["powder_factor_kg_m3"], 0.01)
    bench_h = st.slider("Bench Height (m)", 10.0, 18.0, DEFAULT_DESIGN["bench_height_m"], 0.5)
    hole_d = st.slider("Hole Diameter (mm)", 100, 311, DEFAULT_DESIGN["hole_diameter_mm"], 1)
    subdrill = st.slider("Subdrill (m)", 0.3, 2.0, DEFAULT_DESIGN["subdrilling_m"], 0.1)
    hole_depth = bench_h + subdrill
    max_charge = st.slider("Max Charge per Delay (kg)", 100, 900, DEFAULT_DESIGN["max_charge_per_delay_kg"], 10)

    st.subheader("Pattern Layout")
    num_rows = st.slider("Number of Rows", 2, 15, DEFAULT_DESIGN["num_rows"])
    holes_per_row = st.slider("Holes per Row", 2, 15, DEFAULT_DESIGN["holes_per_row"])
    pattern_type = st.selectbox(
        "Pattern Type",
        ["staggered", "square", "echelon", "v_pattern"],
        index=0,
    )

with col_right:
    st.subheader("3D Scene")

    # Build pattern
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

    fig = render_bench_and_holes(bench, pattern)
    st.plotly_chart(fig, use_container_width=True)

# --- Summary and predictions ---
st.divider()
col_a, col_b, col_c = st.columns(3)

summary = pattern_summary(pattern)
with col_a:
    st.metric("Total Holes", summary["num_holes"])
    st.metric("Total Charge (kg)", f"{summary['total_charge_kg']:.0f}")

with col_b:
    st.metric("Blast Volume (m³)", f"{compute_blast_volume(pattern):,.0f}")
    st.metric("Tonnage (t)", f"{compute_blast_tonnage(pattern):,.0f}")

with col_c:
    extra = {
        "rock_factor_A": 8.0,
        "hole_diameter_mm": float(hole_d),
        "max_charge_per_delay_kg": float(max_charge),
        "explosive_rws": 115.0,
    }
    preds = predict_pattern(model, pattern, extra)
    st.metric("Fragmentation D80 (cm)", f"{preds['fragmentation_d80_cm']:.1f}")
    st.metric("Vibration PPV (mm/s)", f"{preds['vibration_ppv_mms']:.2f}")
    st.metric("Airblast (dB)", f"{preds['airblast_db']:.1f}")

# --- Compliance check ---
st.divider()
st.subheader("Compliance Check")
ppv = preds["vibration_ppv_mms"]
airblast = preds["airblast_db"]
if ppv <= 5.0 and airblast <= 120.0:
    st.success(
        f"✅ Compliant with Botswana limits. "
        f"PPV margin: {5.0 - ppv:.2f} mm/s. "
        f"Airblast margin: {120.0 - airblast:.1f} dB."
    )
else:
    st.error(
        f"❌ Non-compliant. PPV: {ppv:.2f} (limit 5.0), "
        f"Airblast: {airblast:.1f} (limit 120)."
    )
