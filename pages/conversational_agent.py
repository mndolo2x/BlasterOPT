import streamlit as st
from src.components.model_selector import render_page_model_selector

st.title("Conversational Agent")

model, model_key = render_page_model_selector("conversational_agent")
if model is None:
    st.stop()

from src.agent.tool_binder import bind_model_to_agent
bind_model_to_agent(model_key, model)

st.write(f"Bound trained model `{model_key}` to conversational agent.")
