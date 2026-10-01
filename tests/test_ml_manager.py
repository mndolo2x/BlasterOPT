"""
Tests for ML Model Manager configuration and registry integration.
"""
import pytest
import numpy as np
import pandas as pd
from src.models import MODEL_REGISTRY, GAANNModel, FEATURE_COLS
from models.registry import get_registry


def test_page_reads_outputs_from_registry():
    """Verify registry metadata returns outputs from MODEL_REGISTRY for ga_ann_jwaneng or ga_ann."""
    reg = get_registry()
    meta = reg.get_metadata("ga_ann")
    assert meta is not None
    assert meta.output_features == [
        "fragmentation_d80_cm",
        "vibration_ppv_mms",
        "airblast_db",
    ]


def test_cv_loop_uses_registry_outputs():
    """Verify multi-output training targets match registry outputs."""
    config = MODEL_REGISTRY["ga_ann_jwaneng"]
    outputs = config["outputs"]
    assert len(outputs) == 3
    assert outputs == [
        "fragmentation_d80_cm",
        "vibration_ppv_mms",
        "airblast_db",
    ]


def test_feature_cols_matches_model_input_size():
    """Verify FEATURE_COLS length matches GAANNModel input layer and architecture."""
    assert len(FEATURE_COLS) == 10
    model = GAANNModel(input_size=len(FEATURE_COLS))
    assert model.hidden1.in_features == 10
    assert model.output.out_features == 3
    expected_arch = f"{len(FEATURE_COLS)}-70-25-3"
    assert MODEL_REGISTRY["ga_ann_jwaneng"]["architecture"] == expected_arch
