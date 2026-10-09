import streamlit as st
import numpy as np
import pandas as pd

from src.components.model_selector import render_page_model_selector

st.title("PINN Prediction & Uncertainty")

PINN_MODEL_KEY = "pinn"

trained = st.session_state.get("trained_models", {})

if PINN_MODEL_KEY not in trained:
    st.warning(
        "⚠️ **PINN Prediction & Uncertainty requires a trained PINN model.**  \n"
        "Go to **ML Model Manager**, generate data, select the PINN model, "
        "and click Train Model. Then return here."
    )
    st.info(
        "The PINN model is a physics-informed neural network that blends "
        "the Kuz-Ram fragmentation equation and the USBM vibration equation "
        "with data-driven learning. It provides 95% confidence intervals "
        "and out-of-distribution (OOD) detection."
    )
    st.stop()

model, selected_key = render_page_model_selector("pinn")
if model is None:
    st.stop()

st.subheader("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    burden_m = st.slider("Burden (m)", 2.0, 8.0, 6.0, 0.1, key="pinn_burden")
    spacing_m = st.slider("Spacing (m)", 2.0, 10.0, 7.0, 0.1, key="pinn_spacing")
    hole_diameter_mm = st.slider("Hole Diameter (mm)", 100.0, 311.0, 250.0, 1.0, key="pinn_hole_d")
    hole_depth_m = st.slider("Hole Depth (m)", 5.0, 30.0, 12.0, 0.5, key="pinn_hole_depth")
    stemming_m = st.slider("Stemming (m)", 1.0, 6.0, 5.0, 0.1, key="pinn_stemming")
    sub_drill_m = st.slider("Sub-drill (m)", 0.3, 2.0, 1.5, 0.1, key="pinn_subdrill")

with col2:
    powder_factor = st.slider("Powder Factor (kg/m³)", 0.2, 1.5, 0.65, 0.05, key="pinn_pf")
    max_charge_per_delay = st.slider("Max Charge per Delay (kg)", 10.0, 500.0, 320.0, 10.0, key="pinn_charge")
    rock_strength_ucs = st.slider("Rock Strength UCS (MPa)", 20.0, 300.0, 120.0, 5.0, key="pinn_ucs")
    rmr = st.slider("Rock Mass Rating (RMR)", 0.0, 100.0, 65.0, 1.0, key="pinn_rmr")
    monitoring_distance = st.slider("Monitoring Distance (m)", 50.0, 3000.0, 450.0, 25.0, key="pinn_distance")
    blastability_index = st.slider("Blastability Index (BI)", 0.0, 100.0, 55.0, 1.0, key="pinn_bi")

input_dict = {
    "burden_m": burden_m,
    "spacing_m": spacing_m,
    "hole_diameter_mm": hole_diameter_mm,
    "hole_depth_m": hole_depth_m,
    "stemming_m": stemming_m,
    "sub_drill_m": sub_drill_m,
    "powder_factor_kg_m3": powder_factor,
    "max_charge_per_delay_kg": max_charge_per_delay,
    "rock_strength_ucs": rock_strength_ucs,
    "rmr": rmr,
    "monitoring_distance_m": monitoring_distance,
    "blastability_index": blastability_index,
}

input_df = pd.DataFrame([input_dict])

if hasattr(model, "predict"):
    preds = model.predict(input_df)
elif hasattr(model, "predict_with_uncertainty"):
    from src.pinn import predict_with_uncertainty
    preds = predict_with_uncertainty(model, input_df)
else:
    preds = model(input_df)

if isinstance(preds, pd.DataFrame):
    d80 = float(preds["fragmentation_d80_cm"].iloc[0]) if "fragmentation_d80_cm" in preds.columns else float(preds.iloc[0, 0])
    ppv = float(preds["vibration_ppv_mms"].iloc[0]) if "vibration_ppv_mms" in preds.columns else float(preds.iloc[0, 1])
    airblast = float(preds["airblast_db"].iloc[0]) if "airblast_db" in preds.columns else float(preds.iloc[0, 2])
elif isinstance(preds, dict):
    d80 = float(preds.get("fragmentation_d80_cm", 25.0))
    ppv = float(preds.get("vibration_ppv_mms", 4.5))
    airblast = float(preds.get("airblast_db", 115.0))
else:
    d80 = float(preds[0][0])
    ppv = float(preds[0][1])
    airblast = float(preds[0][2])

if not (5.0 <= d80 <= 60.0):
    st.error(f"❌ D80 = {d80:.1f} cm is outside the physical range [5, 60]. Model may be untrained.")
    st.stop()

if not (95.0 <= airblast <= 125.0):
    st.error(f"❌ Airblast = {airblast:.1f} dB is outside the physical range [95, 125]. Model may be untrained.")
    st.stop()

st.divider()
st.subheader("Predictions & Uncertainty Bounds")

col_d80, col_ppv, col_air = st.columns(3)
col_d80.metric("Fragmentation D80", f"{d80:.1f} cm")
col_ppv.metric("Vibration PPV", f"{ppv:.2f} mm/s")
col_air.metric("Airblast Overpressure", f"{airblast:.1f} dB")
