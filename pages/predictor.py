import streamlit as st
import pandas as pd
from src.components.model_selector import render_page_model_selector

st.title("Predictor & Kuz-Ram Curve")

model, model_key = render_page_model_selector("predictor")
if model is None:
    st.stop()

st.divider()

# Existing slider UI & inputs
last_in = st.session_state.get("last_predict_inputs", {
    "rock_factor_A": 8.0,
    "bench_height_m": 12.0,
    "hole_diameter_mm": 250.0,
    "burden_m": 6.0,
    "spacing_m": 7.0,
    "stemming_m": 5.0,
    "powder_factor_kg_m3": 0.65,
    "charge_mass_per_hole_kg": 320.0,
    "max_charge_per_delay_kg": 640.0,
    "monitoring_distance_m": 450.0,
    "explosive_rws": 100.0,
})

rock_A = st.number_input("Rock Factor (A)", 4.0, 16.0, float(last_in["rock_factor_A"]), step=0.5)
bench_h = st.number_input("Bench Height (m)", 5.0, 30.0, float(last_in["bench_height_m"]), step=0.5)
hole_d = st.number_input("Hole Diameter (mm)", 80.0, 380.0, float(last_in["hole_diameter_mm"]), step=10.0)
burden = st.number_input("Burden (m)", 2.0, 12.0, float(last_in["burden_m"]), step=0.2)
spacing = st.number_input("Spacing (m)", 2.0, 15.0, float(last_in["spacing_m"]), step=0.2)
stemming = st.number_input("Stemming (m)", 1.0, 10.0, float(last_in["stemming_m"]), step=0.2)
pf = st.number_input("Powder Factor (kg/m3)", 0.2, 2.5, float(last_in["powder_factor_kg_m3"]), step=0.05)
dist = st.number_input("Distance to Structure (m)", 50.0, 3000.0, float(last_in["monitoring_distance_m"]), step=25.0)

input_dict = {
    "rock_factor_A": rock_A,
    "bench_height_m": bench_h,
    "hole_diameter_mm": hole_d,
    "burden_m": burden,
    "spacing_m": spacing,
    "stemming_m": stemming,
    "powder_factor_kg_m3": pf,
    "charge_mass_per_hole_kg": 320.0,
    "max_charge_per_delay_kg": 640.0,
    "monitoring_distance_m": dist,
    "explosive_rws": 100.0,
}

if st.button("Predict", type="primary"):
    X = pd.DataFrame([input_dict])
    predictions = model.predict(X)
    st.write(predictions)
