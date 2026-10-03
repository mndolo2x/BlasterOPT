import streamlit as st
from src.components.model_selector import render_page_model_selector

st.title("Multi-Objective Pareto Optimizer")

model, model_key = render_page_model_selector("pareto_optimizer")
if model is None:
    st.stop()

from src.pareto_optimizer import run_nsga2
front = run_nsga2(model=model, n_gen=100, pop_size=100)
