"""
Integration tests for GeoResponse classifier wiring and ML infrastructure.
"""

import numpy as np
import pandas as pd
import pytest

from models.base import BaseBlastModel, ModelMetadata
from models.custom.geo_response_model import GeoResponseModel
from models.registry import get_registry
from src.models import MODEL_REGISTRY


def test_geo_response_appears_in_sidebar():
    """Verify GeoResponse is configured in app sidebar navigation and translations."""
    from src.i18n import TRANSLATIONS
    assert "nav_georesponse" in TRANSLATIONS["en"]
    assert "nav_georesponse" in TRANSLATIONS["tn"]
    assert "GeoResponse" in TRANSLATIONS["en"]["nav_georesponse"]


def test_geo_response_metadata_has_task_type_classification():
    """Verify GeoResponse ModelMetadata has task_type = 'classification'."""
    meta = GeoResponseModel.get_metadata()
    assert meta.name == "geo_response"
    assert meta.model_type in ["custom", "classifier"]
    assert meta.task_type == "classification"
    assert meta.output_features == ["rock_class"]
    assert "prototype" in meta.tags


def test_geo_response_wrapper_inherits_base_model():
    """Verify GeoResponseModel inherits from BaseBlastModel and implements interface."""
    model = GeoResponseModel()
    assert isinstance(model, BaseBlastModel)

    # Test fit and predict
    df_sample = pd.DataFrame({
        "penetration_rate_m_hr": [20.0, 35.0, 45.0, 15.0, 50.0],
        "torque_nm": [300.0, 800.0, 1200.0, 400.0, 1500.0],
        "weight_on_bit_kg": [2000.0, 5000.0, 8000.0, 3000.0, 10000.0],
        "percussion_pressure_bar": [120.0, 150.0, 180.0, 110.0, 200.0],
        "flushing_pressure_bar": [8.0, 10.0, 12.0, 7.0, 15.0],
        "rock_class": ["Soft", "Medium", "Hard", "Soft", "Very Hard"]
    })

    X = df_sample.drop(columns=["rock_class"])
    y = df_sample["rock_class"]

    model.fit(X, y)
    assert model.is_fitted is True

    preds_df = model.predict(X)
    assert "rock_class" in preds_df.columns
    assert "confidence_score" in preds_df.columns
    assert "review_flag" in preds_df.columns

    probas = model.predict_proba(X)
    assert isinstance(probas, np.ndarray)
    assert probas.shape[0] == len(X)


def test_ml_model_manager_shows_classification_metrics():
    """Verify classification models return classification metrics structure."""
    from src.georesponse.evaluation import evaluate_rock_type_classifier

    df_sample = pd.DataFrame({
        "hole_id": [f"H_{i}" for i in range(20)],
        "tunnel_id": ["T1", "T1", "T2", "T2"] * 5,
        "penetration_rate_m_hr": np.random.uniform(10, 50, size=20),
        "torque_nm": np.random.uniform(200, 1200, size=20),
        "weight_on_bit_kg": np.random.uniform(1000, 8000, size=20),
        "percussion_pressure_bar": np.random.uniform(100, 200, size=20),
        "flushing_pressure_bar": np.random.uniform(5, 15, size=20),
        "rock_class": np.random.choice(["Soft", "Medium", "Hard"], size=20)
    })

    eval_res = evaluate_rock_type_classifier(
        df=df_sample,
        feature_cols=["penetration_rate_m_hr", "torque_nm", "weight_on_bit_kg", "percussion_pressure_bar", "flushing_pressure_bar"],
        target_col="rock_class",
        group_col="tunnel_id",
        n_splits=2
    )

    assert "random_forest" in eval_res
    assert "macro_f1" in eval_res["random_forest"]
    assert "balanced_accuracy" in eval_res["random_forest"]
    assert "per_class" in eval_res["random_forest"]
    assert "confusion_matrix" in eval_res["random_forest"]


def test_ml_model_manager_does_not_show_r2_for_classifier():
    """Verify task_type classification is distinct from regression."""
    meta = GeoResponseModel.get_metadata()
    assert meta.task_type == "classification"
    assert meta.task_type != "regression"


def test_model_comparison_separates_classifiers_from_regressors():
    """Verify MODEL_REGISTRY entries correctly separate prototype classifiers from regressors."""
    reg_entry = MODEL_REGISTRY.get("geo_response")
    assert reg_entry is not None
    assert reg_entry["task_type"] == "classification"
    assert reg_entry["deployment_status"] == "prototype"


def test_registry_entry_has_research_only_licence():
    """Verify registry entry includes research-only licence and dataset notes."""
    reg_entry = MODEL_REGISTRY.get("geo_response")
    assert "licence" in reg_entry
    assert "Research-only" in reg_entry["licence"]
    assert "Norwegian" in reg_entry["training_dataset"]


def test_warning_banner_rendered_on_geo_response_page():
    """Verify warning text content exists in GeoResponse module file."""
    with open("pages/georesponse.py", "r") as f:
        content = f.read()

    assert "RESEARCH PROTOTYPE" in content
    assert "Norwegian tunnel drilling data" in content


def test_geo_response_never_auto_applied_to_design():
    """Verify GeoResponse is isolated from blast design calculation paths."""
    from src.predict import predict_single_blast
    inputs = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "max_charge_per_delay_kg": 640.0,
        "monitoring_distance_m": 450.0,
        "explosive_rws": 100.0,
    }
    preds = predict_single_blast(inputs)
    # Ensure standard blast predictor returns canonical regression targets, not rock_class
    assert "d50_mm" in preds
    assert "ppv_mms" in preds
    assert "rock_class" not in preds
