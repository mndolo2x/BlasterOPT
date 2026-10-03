import streamlit as st
import pandas as pd
from src.components.model_selector import render_page_model_selector

st.title("Economic Dashboard")

model, model_key = render_page_model_selector("economic_dashboard")
if model is None:
    st.stop()

input_dict = {
    "rock_factor_A": 8.0,
    "bench_height_m": 12.0,
    "hole_diameter_mm": 250.0,
    "burden_m": 6.0,
    "spacing_m": 7.0,
    "stemming_m": 5.0,
    "powder_factor_kg_m3": 0.65,
    "monitoring_distance_m": 450.0,
}

X = pd.DataFrame([input_dict])
preds = model.predict(X)
d80_cm = preds["fragmentation_d80_cm"].iloc[0] if isinstance(preds, pd.DataFrame) and "fragmentation_d80_cm" in preds.columns else 20.0

from src.predict import total_cost_per_tonne
cost_info = total_cost_per_tonne({"powder_factor_kg_m3": 0.65, "d50_mm": d80_cm * 10.0})
st.write(cost_info)
