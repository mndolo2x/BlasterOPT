import streamlit as st
import numpy as np
from src.components.model_selector import render_page_model_selector

st.title("PINN Prediction & Uncertainty")

model, model_key = render_page_model_selector("pinn")
if model is None:
    st.stop()

from src.pinn import predict_with_uncertainty
dummy_input = np.random.randn(1, 12)
result = predict_with_uncertainty(model, dummy_input)
st.write(result)
