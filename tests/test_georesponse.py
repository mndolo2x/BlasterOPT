import numpy as np
import pandas as pd
import pytest

from src.georesponse.config import MINIMAL_REQUIRED_MWD_COLUMNS, OPTIONAL_MWD_COLUMNS, TARGET_COLUMN
from src.georesponse.evaluation import evaluate_rock_type_classifier
from src.georesponse.features import compute_mwd_features, prepare_feature_matrix
from src.georesponse.ingestion import normalize_mwd_schema, validate_mwd_dataset
from src.georesponse.models import GeoResponseClassifier, MajorityClassBaseline
from models.custom.geo_response_model import GeoResponseModel


def test_mwd_penetration_rate_m_hr_ingestion():
    df = pd.DataFrame({
        "penetration_rate_m_hr": [30.0, 45.0],
        "rock_type_label": ["Granite", "Gneiss"]
    })
    df_norm, report = normalize_mwd_schema(df)
    assert report["is_eligible_for_training"] is True
    assert "penetration_rate_m_hr" in df_norm.columns
    assert df_norm["penetration_rate_m_hr"].iloc[0] == 30.0


def test_mwd_penetration_rate_m_min_conversion():
    df = pd.DataFrame({
        "penetration_rate_m_min": [0.5, 0.75],
        "rock_type_label": ["Granite", "Gneiss"]
    })
    df_norm, report = normalize_mwd_schema(df)
    assert report["is_eligible_for_training"] is True
    assert "penetration_rate_m_hr" in df_norm.columns
    assert df_norm["penetration_rate_m_hr"].iloc[0] == 30.0
    assert df_norm["penetration_rate_m_hr"].iloc[1] == 45.0


def test_mwd_penetration_rate_both_consistent():
    df = pd.DataFrame({
        "penetration_rate_m_min": [0.5, 1.0],
        "penetration_rate_m_hr": [30.0, 60.0],
        "rock_type_label": ["Granite", "Gneiss"]
    })
    df_norm, report = normalize_mwd_schema(df)
    assert report["is_eligible_for_training"] is True
    assert len(report["errors"]) == 0


def test_mwd_penetration_rate_both_conflicting():
    df = pd.DataFrame({
        "penetration_rate_m_min": [0.5, 1.0],
        "penetration_rate_m_hr": [10.0, 20.0],  # Severe mismatch with m_min * 60
        "rock_type_label": ["Granite", "Gneiss"]
    })
    df_norm, report = normalize_mwd_schema(df)
    assert report["is_eligible_for_training"] is False
    assert len(report["errors"]) > 0
    assert "Conflicting penetration rate" in report["errors"][0]


def test_mwd_missing_weight_on_bit_kg_graceful():
    df = pd.DataFrame({
        "penetration_rate_m_hr": [30.0, 40.0],
        "torque_nm": [500.0, 600.0],
        "rock_type_label": ["Granite", "Gneiss"]
    })
    df_norm, report = normalize_mwd_schema(df)
    assert report["is_eligible_for_training"] is True
    assert "weight_on_bit_kg" in report["missing_optional_columns"]

    df_feat, feat_rep = compute_mwd_features(df_norm)
    assert "mwd_specific_energy_mj_m3" in feat_rep["skipped_features"]
    assert "rop_torque_ratio" in feat_rep["computed_features"]


def test_mwd_missing_required_columns():
    df = pd.DataFrame({
        "torque_nm": [500.0, 600.0]
    })
    df_norm, report = normalize_mwd_schema(df)
    assert report["is_eligible_for_training"] is False
    assert "penetration_rate_m_hr" in report["missing_required_columns"]


def test_mwd_optional_feature_unavailable():
    df = pd.DataFrame({
        "penetration_rate_m_hr": [20.0, 30.0],
        "rock_type_label": ["Soft", "Hard"]
    })
    df_norm, report = normalize_mwd_schema(df)
    X, feat_cols, feat_rep = prepare_feature_matrix(df_norm)

    assert "penetration_rate_m_hr" in feat_cols
    assert "mwd_specific_energy_mj_m3" not in feat_cols
    assert len(feat_rep["skipped_features"]) > 0


def test_mwd_prediction_incompatible_schema():
    df_train = pd.DataFrame({
        "penetration_rate_m_hr": [20.0, 30.0, 40.0, 50.0],
        "torque_nm": [300.0, 400.0, 500.0, 600.0],
        "rock_class": ["A", "B", "A", "B"]
    })

    model = GeoResponseModel()
    model.fit(df_train.drop(columns=["rock_class"]), df_train["rock_class"])

    # Predict with missing required trained feature
    df_pred_invalid = pd.DataFrame({
        "flushing_pressure_bar": [10.0, 12.0]
    })

    with pytest.raises(ValueError, match="Incompatible dataset schema during prediction"):
        model.predict(df_pred_invalid)


def test_mwd_missing_numerical_values():
    df = pd.DataFrame({
        "penetration_rate_m_hr": [20.0, np.nan, 40.0, 30.0],
        "torque_nm": [300.0, 400.0, np.nan, 500.0],
        "rock_type_label": ["Soft", "Soft", "Hard", "Hard"]
    })

    X, feat_cols, _ = prepare_feature_matrix(df)
    assert not X.isnull().any().any()


def test_successful_end_to_end_mwd_training_and_evaluation():
    np.random.seed(42)
    n = 60
    df_full = pd.DataFrame({
        "tunnel_id": np.random.choice(["T1", "T2", "T3"], size=n),
        "penetration_rate_m_min": np.random.uniform(0.3, 1.2, size=n),
        "torque_nm": np.random.uniform(200.0, 800.0, size=n),
        "weight_on_bit_kg": np.random.uniform(2000.0, 10000.0, size=n),
        "percussion_pressure_bar": np.random.uniform(100.0, 200.0, size=n),
        "flushing_pressure_bar": np.random.uniform(5.0, 15.0, size=n),
        "rock_type_label": np.random.choice(["Soft", "Medium", "Hard"], size=n)
    })

    df_norm, report = normalize_mwd_schema(df_full)
    assert report["is_eligible_for_training"] is True

    X, feat_cols, _ = prepare_feature_matrix(df_norm)
    df_eval = df_norm.copy()
    for c in feat_cols:
        df_eval[c] = X[c]

    eval_res = evaluate_rock_type_classifier(
        df=df_eval,
        feature_cols=feat_cols,
        target_col="rock_type_label",
        group_col="tunnel_id",
        n_splits=3
    )

    assert eval_res["random_forest"]["macro_f1"] >= 0.0
    assert eval_res["random_forest"]["balanced_accuracy"] >= 0.0

    model = GeoResponseModel()
    model.fit(df_norm.drop(columns=["rock_type_label"]), df_norm["rock_type_label"])
    preds = model.predict(df_norm.drop(columns=["rock_type_label"]))

    assert "rock_class" in preds.columns
    assert "confidence_score" in preds.columns
    assert "review_flag" in preds.columns
