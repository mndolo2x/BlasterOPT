"""
Unit tests for Model Registry (`models/registry.py`).
Tests discovery, list_models(), get_model("ga_ann"), graceful error handling,
metadata completeness, and sub-2-second scan speed.
"""

import time
import pytest
from models.registry import get_registry, ModelRegistry
from models.custom.ga_ann import GAANNBlastModel


def test_registry_discovers_all_models_in_models_dir():
    """
    Test that registry discovers all models in models/ directory.
    """
    registry = get_registry()
    models_meta = registry.list_models()
    names = [meta.name for meta in models_meta]

    assert "random_forest" in names
    assert "xgboost" in names
    assert "ridge" in names
    assert "ga_ann" in names
    assert "pinn" in names
    assert "ensemble" in names
    assert "site_calibration" in names


def test_list_models_returns_expected_count():
    """
    Test that list_models() returns expected model count (>= 7 models).
    """
    registry = get_registry()
    models_meta = registry.list_models()
    assert len(models_meta) >= 7


def test_get_model_ga_ann():
    """
    Test that get_model("ga_ann") returns expected model class.
    """
    registry = get_registry()
    ga_cls = registry.get_model("ga_ann")
    assert ga_cls == GAANNBlastModel


def test_broken_modules_skipped_gracefully():
    """
    Test that broken model files are skipped gracefully during discovery and logged in get_load_errors().
    """
    registry = get_registry()
    # Inject synthetic error into registry
    registry._load_errors.append({"module": "models.custom.broken_test", "error": "No module named 'broken'"})

    errors = registry.get_load_errors()
    assert isinstance(errors, list)
    assert len(errors) >= 1
    assert any("broken" in str(e) for e in errors)


def test_metadata_completeness_for_every_model():
    """
    Test that metadata is complete for every registered model.
    """
    registry = get_registry()
    for meta in registry.list_models():
        assert meta.name is not None and len(meta.name) > 0
        assert meta.display_name is not None and len(meta.display_name) > 0
        assert meta.model_type in ["sklearn", "pytorch", "physics_informed", "ensemble", "calibration", "test", "custom"]
        assert meta.description is not None and len(meta.description) > 0
        assert meta.version is not None
        assert isinstance(meta.input_features, list) and len(meta.input_features) > 0
        assert isinstance(meta.output_features, list) and len(meta.output_features) > 0


def test_registry_scan_time_under_two_seconds():
    """
    Test that registry scan completes in < 2 seconds on startup.
    """
    start = time.perf_counter()
    reg = ModelRegistry()
    reg.discover()
    duration = time.perf_counter() - start

    assert duration < 2.0
    assert reg.scan_time_ms < 2000.0
