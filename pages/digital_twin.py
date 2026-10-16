"""Digital Twin of Bench — BlastOPT.

Embeds the standalone HTML 3D viewer inside Streamlit.
The HTML viewer is embedded with CSS overrides to display only the 3D canvas and controls.
"""
import streamlit as st
import streamlit.components.v1 as components
from src.components.model_selector import render_page_model_selector
from src.ui.digital_twin.render import load_scene_html

st.set_page_config(
    page_title="Digital Twin of Bench — BlastOPT",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    html = load_scene_html()
    components.html(html, height=520, scrolling=False)
except FileNotFoundError as exc:
    st.error(str(exc))
