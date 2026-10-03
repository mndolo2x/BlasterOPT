"""
Registry of pages that can be powered by trained models.

Each page declares:
- display_name: human-readable name matching the sidebar
- required_outputs: output columns a model must produce to power this page
- description: what the model will do on this page
"""

from typing import Dict, List, Any


PAGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "predictor": {
        "display_name": "Predictor & Kuz-Ram Curve",
        "required_outputs": [
            "fragmentation_d80_cm",
            "vibration_ppv_mms",
            "airblast_db",
        ],
        "description": "Predict fragmentation, vibration, and airblast from sliders",
    },
    "ga_optimizer": {
        "display_name": "Genetic Algorithm Optimizer",
        "required_outputs": ["fragmentation_d80_cm", "vibration_ppv_mms"],
        "description": "Optimize burden, spacing, powder factor",
    },
    "pareto_optimizer": {
        "display_name": "Multi-Objective Pareto Optimizer",
        "required_outputs": [
            "fragmentation_d80_cm",
            "vibration_ppv_mms",
            "airblast_db",
        ],
        "description": "Multi-objective trade-off optimization",
    },
    "economic_dashboard": {
        "display_name": "Economic Dashboard",
        "required_outputs": ["fragmentation_d80_cm"],
        "description": "Cost per tonne and crusher throughput",
    },
    "regulatory_compliance": {
        "display_name": "Regulatory Compliance",
        "required_outputs": ["vibration_ppv_mms", "airblast_db"],
        "description": "PPV and airblast check against Botswana limits",
    },
    "pinn": {
        "display_name": "PINN Prediction & Uncertainty",
        "required_outputs": [
            "fragmentation_d80_cm",
            "vibration_ppv_mms",
            "airblast_db",
        ],
        "description": "Physics-informed predictions with 95% confidence intervals",
        "requires_specific_model": "pinn",
    },
    "uncertainty_quantification": {
        "display_name": "Uncertainty Quantification",
        "required_outputs": [
            "fragmentation_d80_cm",
            "vibration_ppv_mms",
            "airblast_db",
        ],
        "description": "Aleatoric vs epistemic uncertainty decomposition",
        "requires_specific_model": "ensemble_uq",
    },
    "digital_twin": {
        "display_name": "Digital Twin of Bench",
        "required_outputs": [
            "fragmentation_d80_cm",
            "vibration_ppv_mms",
            "airblast_db",
        ],
        "description": "3D what-if scenario analysis",
    },
    "conversational_agent": {
        "display_name": "Conversational Agent",
        "required_outputs": [],  # accepts any model
        "description": "Natural language blast design",
    },
}


def get_compatible_models(page_key: str) -> List[str]:
    """
    Return the list of trained model keys that can power the given page.

    Args:
        page_key: key in PAGE_REGISTRY

    Returns:
        List of model keys from st.session_state["trained_models"] that
        produce all outputs required by the page.
    """
    import streamlit as st
    from src.models import MODEL_REGISTRY

    if page_key not in PAGE_REGISTRY:
        raise KeyError(f"Unknown page: {page_key}")

    page = PAGE_REGISTRY[page_key]
    required = set(page["required_outputs"])
    specific = page.get("requires_specific_model")

    trained = st.session_state.get("trained_models", {})
    compatible: List[str] = []

    for model_key in trained.keys():
        if specific and model_key != specific:
            continue
        model_outputs = set(MODEL_REGISTRY.get(model_key, {}).get("outputs", []))
        if not required:  # conversational agent accepts any model
            compatible.append(model_key)
        elif required.issubset(model_outputs):
            compatible.append(model_key)

    return compatible
