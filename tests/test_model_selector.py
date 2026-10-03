"""
Unit tests for PAGE_REGISTRY, get_compatible_models, and render_page_model_selector.
"""
import pytest
import subprocess


def test_page_registry_completeness():
    try:
        from src.components.page_registry import PAGE_REGISTRY
    except ImportError:
        from src.page_registry import PAGE_REGISTRY

    assert len(PAGE_REGISTRY) == 9
    for key, config in PAGE_REGISTRY.items():
        assert "display_name" in config
        assert "required_outputs" in config
        assert "description" in config


def test_get_compatible_models_filters_by_outputs():
    import streamlit as st
    try:
        from src.components.page_registry import get_compatible_models
    except ImportError:
        from src.page_registry import get_compatible_models

    from src.models import GAANNModel, AirblastMinimizerModel

    st.session_state["trained_models"] = {
        "ga_ann_jwaneng": GAANNModel(input_size=11),
        "airblast_minimizer": AirblastMinimizerModel(input_size=8),
    }

    # Predictor requires 3 outputs → only GA-ANN qualifies
    predictor_models = get_compatible_models("predictor")
    assert "ga_ann_jwaneng" in predictor_models
    assert "airblast_minimizer" not in predictor_models

    # Regulatory compliance requires ppv + airblast → only GA-ANN qualifies
    compliance_models = get_compatible_models("regulatory_compliance")
    assert "ga_ann_jwaneng" in compliance_models
    assert "airblast_minimizer" not in compliance_models


def test_get_compatible_models_respects_specific_requirement():
    import streamlit as st
    try:
        from src.components.page_registry import get_compatible_models
    except ImportError:
        from src.page_registry import get_compatible_models

    from src.models import GAANNModel

    st.session_state["trained_models"] = {
        "ga_ann_jwaneng": GAANNModel(input_size=11),
    }

    # PINN page requires the PINN model specifically
    pinn_models = get_compatible_models("pinn")
    assert "ga_ann_jwaneng" not in pinn_models
    assert len(pinn_models) == 0


def test_selector_returns_none_when_no_models():
    import streamlit as st
    from src.components.model_selector import render_page_model_selector

    st.session_state["trained_models"] = {}
    model, key = render_page_model_selector("predictor")
    assert model is None
    assert key is None


def test_all_pages_use_selector():
    """Grep to verify each page calls render_page_model_selector."""
    pages = {
        "predictor": "pages/predictor.py",
        "ga_optimizer": "pages/ga_optimizer.py",
        "pareto_optimizer": "pages/pareto_optimizer.py",
        "economic_dashboard": "pages/economic_dashboard.py",
        "regulatory_compliance": "pages/regulatory_compliance.py",
        "pinn": "pages/pinn.py",
        "uncertainty_quantification": "pages/uncertainty_quantification.py",
        "digital_twin": "pages/digital_twin.py",
        "conversational_agent": "pages/conversational_agent.py",
    }
    for page_key, path in pages.items():
        result = subprocess.run(
            ["grep", "-q", f'render_page_model_selector("{page_key}")', path],
            capture_output=True,
        )
        assert result.returncode == 0, f"{path} does not call render_page_model_selector"
