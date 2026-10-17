"""Digital Twin of Bench — BlastOPT.

Collects blast design parameters, calculates derived metrics, and passes
data into a 3D rendering viewport using load_and_inject_scene.
"""
import math
import streamlit as st
import streamlit.components.v1 as components
from src.ui.digital_twin.loader import load_and_inject_scene

st.title("Digital Twin of Bench")
st.caption("3D Spatial Twin & What-If Design Simulator")

col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_left:
    st.subheader("Design Parameters")

    with st.expander("Expander 1: Bench Geometry", expanded=True):
        bench_id = st.text_input("Bench ID", value="BENCH_14")
        bench_height_m = st.number_input("Bench Height (m)", min_value=1.0, max_value=50.0, value=15.0, step=0.5)
        bench_width_m = st.number_input("Bench Width (m)", min_value=5.0, max_value=200.0, value=60.0, step=1.0)
        bench_length_m = st.number_input("Bench Length (m)", min_value=5.0, max_value=200.0, value=40.0, step=1.0)

    with st.expander("Expander 2: Rock Mass Properties", expanded=False):
        rock_density_kg_m3 = st.number_input("Rock Density (kg/m³)", min_value=1000, max_value=4000, value=2500, step=50)
        ucs_mpa = st.number_input("Uniaxial Compressive Strength - UCS (MPa)", min_value=10.0, max_value=300.0, value=100.0, step=5.0)
        rqd_percent = st.slider("Rock Quality Designation - RQD (%)", min_value=0, max_value=100, value=85)
        rock_factor = st.number_input("Rock Factor / Blastability Index", min_value=1.0, max_value=20.0, value=12.4, step=0.1)

    with st.expander("Expander 3: Drill & Blast Design", expanded=False):
        hole_diameter_mm = st.number_input("Hole Diameter (mm)", min_value=50, max_value=400, value=165, step=5)
        burden_m = st.number_input("Burden (m)", min_value=0.5, max_value=10.0, value=4.20, step=0.10)
        spacing_m = st.number_input("Spacing (m)", min_value=0.5, max_value=15.0, value=5.10, step=0.10)
        hole_depth_m = st.number_input("Hole Depth (m)", min_value=1.0, max_value=40.0, value=12.0, step=0.5)
        stemming_m = st.number_input("Stemming (m)", min_value=0.1, max_value=15.0, value=3.00, step=0.1)
        powder_factor_kg_m3 = st.number_input("Powder Factor (kg/m³)", min_value=0.1, max_value=3.0, value=0.65, step=0.05)
        explosive_type = st.selectbox("Explosive Type", options=["ANFO", "Heavy ANFO", "Emulsion"], index=0)

    with st.expander("Expander 4: Constraints", expanded=False):
        max_ppv_limit_mms = st.number_input("Max PPV Limit (mm/s)", min_value=0.5, max_value=50.0, value=5.0, step=0.5)

    generate_btn = st.button("Generate Digital Twin", type="primary")

# Default inputs if not set
default_inputs = {
    "bench_id": bench_id,
    "bench_height_m": bench_height_m,
    "bench_width_m": bench_width_m,
    "bench_length_m": bench_length_m,
    "rock_density_kg_m3": rock_density_kg_m3,
    "ucs_mpa": ucs_mpa,
    "rqd_percent": rqd_percent,
    "rock_factor": rock_factor,
    "hole_diameter_mm": hole_diameter_mm,
    "burden_m": burden_m,
    "spacing_m": spacing_m,
    "hole_depth_m": hole_depth_m,
    "stemming_m": stemming_m,
    "powder_factor_kg_m3": powder_factor_kg_m3,
    "explosive_type": explosive_type,
    "max_ppv_limit_mms": max_ppv_limit_mms,
}

if generate_btn:
    # Validation
    errors = []
    if stemming_m >= hole_depth_m:
        errors.append("Stemming length must be less than hole depth.")
    if burden_m <= 0:
        errors.append("Burden must be greater than 0.")
    if spacing_m <= 0:
        errors.append("Spacing must be greater than 0.")

    if errors:
        for err in errors:
            st.error(f"❌ Validation Error: {err}")
    else:
        st.session_state["twin_inputs"] = default_inputs
        st.success("✅ Inputs validated and saved to Digital Twin session state.")

twin_data = st.session_state.get("twin_inputs", default_inputs)

# Calculate derived metrics
total_holes_w = max(1, int(twin_data["bench_width_m"] / twin_data["spacing_m"]))
total_holes_l = max(1, int(twin_data["bench_length_m"] / twin_data["burden_m"]))
total_holes = total_holes_w * total_holes_l

bench_volume_m3 = twin_data["bench_width_m"] * twin_data["bench_length_m"] * twin_data["bench_height_m"]
total_tonnage_t = bench_volume_m3 * (twin_data["rock_density_kg_m3"] / 1000.0)
hole_radius_m = (twin_data["hole_diameter_mm"] / 1000.0) / 2.0
hole_area_m2 = math.pi * (hole_radius_m ** 2)
charge_length_m = max(0.0, twin_data["hole_depth_m"] - twin_data["stemming_m"])
total_charge_kg = bench_volume_m3 * twin_data["powder_factor_kg_m3"]

# Render derived metrics in right column
with col_right:
    st.subheader("Derived Bench Metrics")
    st.metric("Total Pattern Holes", f"{total_holes} ({total_holes_w} × {total_holes_l})")
    st.metric("Total Volume", f"{bench_volume_m3:,.0f} m³")
    st.metric("Total Tonnage", f"{total_tonnage_t:,.0f} tonnes")
    st.metric("Total Explosive Charge", f"{total_charge_kg:,.1f} kg")
    st.metric("Explosive Charge Length", f"{charge_length_m:.1f} m / hole")

# Render 3D scene in middle column
with col_mid:
    st.subheader("3D Spatial Viewer")
    html_content = load_and_inject_scene(twin_data)
    if html_content is None:
        st.info("3D Viewer Not Loaded. Please upload scene_template.html to src/ui/digital_twin/.")
    else:
        components.html(html_content, height=800, scrolling=False)
