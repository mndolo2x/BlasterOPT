"""
Unit tests for src/models.py MODEL_REGISTRY and model initialization.
"""

def test_registry_contains_four_biust_models():
    from src.models import MODEL_REGISTRY
    biust = [k for k, c in MODEL_REGISTRY.items() if c.get("type") == "biust"]
    assert len(biust) == 4, f"Expected 4 BIUST models, got {len(biust)}"

def test_all_models_have_display_name():
    from src.models import MODEL_REGISTRY
    for key, config in MODEL_REGISTRY.items():
        assert "display_name" in config, f"{key} missing display_name"
        assert "type" in config, f"{key} missing type"

def test_baselines_present():
    from src.models import MODEL_REGISTRY
    baseline = [k for k, c in MODEL_REGISTRY.items() if c.get("type") == "baseline"]
    assert len(baseline) >= 3, f"Expected at least 3 baselines, got {len(baseline)}"
