# Confirmed session state keys in ML Model Manager:
# st.session_state["trained_models"][selected_key] = model
# st.session_state["trained_model_metadata"][selected_key] = {...}

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from src.models import MODEL_REGISTRY, FEATURE_COLS
from src.data_ingestion import clean_and_preprocess, engineer_features, dataframe_fingerprint

st.markdown("""
<style>
    /* FORCE GOLD TABS */
    [data-baseweb="tab-highlight"] {
        background-color: #F2A900 !important;
    }
    [data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    [aria-selected="true"] {
        color: #ffffff !important;
    }

    /* FORCE GOLD RADIO BUTTONS */
    div[role="radiogroup"] > label > div:first-child {
        border-color: #F2A900 !important;
    }
    div[role="radiogroup"] > label > div:first-child > div {
        background-color: #F2A900 !important;
    }

    /* FORCE NAVY BUTTONS */
    .stButton > button {
        background-color: #002B5B !important;
        color: #F2A900 !important;
        border: 1px solid #F2A900 !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        background-color: #F2A900 !important;
        color: #002B5B !important;
    }

    /* FORCE DARK TABLES */
    [data-testid="stDataFrame"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }

    /* HEADERS */
    h1, h2, h3 {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="color: #ffffff; font-weight: 700; margin-bottom: 0;">Machine Learning Model Manager</h1>
<p style="color: #94a3b8; font-size: 16px; margin-top: 4px;">Train, evaluate, and deploy AI models for blast prediction.</p>
<hr style="border-color: #1e293b; margin: 16px 0 24px 0;">
""", unsafe_allow_html=True)

if "df" not in st.session_state:
    st.warning("⚠️ Please generate or upload data in Data Ingestion first.")
    st.stop()

df = st.session_state["df"]

st.subheader("Data Preview")
st.dataframe(df, hide_index=True, use_container_width=True, height=350)

st.subheader("1. Select Algorithm")
selected_key = st.selectbox(
    "Algorithm",
    options=list(MODEL_REGISTRY.keys()),
    format_func=lambda k: MODEL_REGISTRY[k]["display_name"],
)

config = MODEL_REGISTRY[selected_key]
st.caption(f"Outputs: {', '.join(config['outputs'])}")

if st.button("Train Model", type="primary"):
    from src.models import (
        GAANNModel, ANN_RF_Ensemble, PSOANNModel,
        AirblastMinimizerModel, FlyrockPredictor, CostPredictor,
    )
    from sklearn.ensemble import RandomForestRegressor as SklearnRF
    from sklearn.linear_model import Ridge as SklearnRidge
    from xgboost import XGBRegressor as SklearnXGB

    required_features = getattr(
        GAANNModel, "INPUT_COLUMNS", FEATURE_COLS
    ) if selected_key == "ga_ann_jwaneng" else FEATURE_COLS
    required_outputs = config["outputs"]

    X = df[[c for c in required_features if c in df.columns]]
    y = df[[c for c in required_outputs if c in df.columns]]

    model_classes = {
        "ga_ann_jwaneng": lambda: GAANNModel(input_size=len(required_features)),
        "ann_rf_ensemble_jwaneng": ANN_RF_Ensemble,
        "pso_ann_orapa": lambda: PSOANNModel(input_size=7),
        "airblast_minimizer": lambda: AirblastMinimizerModel(input_size=8),
        "flyrock_predictor": FlyrockPredictor,
        "cost_predictor": CostPredictor,
        "random_forest_baseline": lambda: SklearnRF(n_estimators=100, random_state=42),
        "xgboost_baseline": lambda: SklearnXGB(n_estimators=100, random_state=42),
        "ridge_baseline": lambda: SklearnRidge(alpha=1.0),
    }

    factory = model_classes.get(selected_key, lambda: SklearnRF(n_estimators=100, random_state=42))
    model = factory()

    with st.spinner(f"Training {config['display_name']}..."):
        model.fit(X, y)

    if "trained_models" not in st.session_state:
        st.session_state["trained_models"] = {}

    st.session_state["trained_models"][selected_key] = model

    if "trained_model_metadata" not in st.session_state:
        st.session_state["trained_model_metadata"] = {}

    st.session_state["trained_model_metadata"][selected_key] = {
        "display_name": config["display_name"],
        "trained_at": datetime.now().isoformat(),
        "rows": df.shape[0],
        "fingerprint": st.session_state.get("data_fingerprint", dataframe_fingerprint(df)),
        "outputs": config["outputs"],
    }

    st.success(
        f"✅ Trained **{config['display_name']}** on {df.shape[0]} rows. "
        f"It is now available on all compatible pages."
    )
