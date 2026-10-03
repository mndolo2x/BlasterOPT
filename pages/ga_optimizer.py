import streamlit as st
from src.components.model_selector import render_page_model_selector

st.title("Genetic Algorithm Optimizer")

model, model_key = render_page_model_selector("ga_optimizer")
if model is None:
    st.stop()

from src.optimize import BlastOptimizer
optimizer = BlastOptimizer(model=model)
