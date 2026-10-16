"""Digital Twin of Bench — BlastOPT.

Embeds the standalone HTML 3D viewer inside Streamlit.
The HTML handles all rendering, controls, and dashboard chrome.
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

# Hide Streamlit chrome so the HTML dashboard fills the frame
st.markdown(
    "<style>"
    "#MainMenu {visibility: hidden;}"
    "footer {visibility: hidden;}"
    "header {visibility: hidden;}"
    ".block-container {padding: 0 !important; max-width: 100% !important;}"
    "iframe {display: block; width: 100%; border: none;}"
    "</style>",
    unsafe_allow_html=True,
)

try:
    _ = render_page_model_selector("digital_twin")
    html = load_scene_html()
    components.html(html, height=920, scrolling=False)
except FileNotFoundError as exc:
    st.error(str(exc))
