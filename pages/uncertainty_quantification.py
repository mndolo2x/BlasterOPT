import streamlit as st
import numpy as np
from src.components.model_selector import render_page_model_selector

st.title("Uncertainty Quantification")

model, model_key = render_page_model_selector("uncertainty_quantification")
if model is None:
    st.stop()

from src.ensemble_uncertainty import predict_with_uncertainty
dummy_input = np.random.randn(1, 12)
result = predict_with_uncertainty(model, dummy_input)
st.write(result)
