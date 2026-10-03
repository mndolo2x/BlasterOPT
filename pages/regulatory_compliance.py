import streamlit as st
import pandas as pd
from src.components.model_selector import render_page_model_selector

st.title("Regulatory Compliance")

model, model_key = render_page_model_selector("regulatory_compliance")
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
ppv = preds["vibration_ppv_mms"].iloc[0] if isinstance(preds, pd.DataFrame) and "vibration_ppv_mms" in preds.columns else 5.0
airblast = preds["airblast_db"].iloc[0] if isinstance(preds, pd.DataFrame) and "airblast_db" in preds.columns else 115.0

from src.regulatory import check_compliance
result = check_compliance({"ppv_mms": ppv, "airblast_db": airblast})
st.write(result)
