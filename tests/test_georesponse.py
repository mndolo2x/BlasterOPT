import numpy as np
import pandas as pd
import pytest

from src.georesponse.config import REQUIRED_MWD_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from src.georesponse.evaluation import evaluate_rock_type_classifier
from src.georesponse.features import compute_mwd_features, prepare_feature_matrix
from src.georesponse.ingestion import validate_mwd_dataset
from src.georesponse.models import GeoResponseClassifier, MajorityClassBaseline


@pytest.fixture
def sample_mwd_data():
    np.random.seed(42)
    n = 100
    tunnels = ["T1", "T2", "T3", "T4", "T5"]
    rock_types = ["Granite", "Gneiss", "Schist"]

    df = pd.DataFrame(
        {
            "hole_id": [f"H_{i}" for i in range(n)],
            "tunnel_id": np.random.choice(tunnels, size=n),
            "blast_id": np.random.choice(["B1", "B2"], size=n),
            "penetration_rate_m_hr": np.random.uniform(10.0, 50.0, size=n),
            "torque_nm": np.random.uniform(200.0, 500.0, size=n),
            "weight_on_bit_kg": np.random.uniform(500.0, 1500.0, size=n),
            "percussion_pressure_bar": np.random.uniform(100.0, 200.0, size=n),
            "flushing_pressure_bar": np.random.uniform(5.0, 15.0, size=n),
            "rock_type_label": np.random.choice(rock_types, size=n),
        }
    )
    return df


def test_validate_mwd_dataset_success(sample_mwd_data):
    is_valid, report = validate_mwd_dataset(sample_mwd_data)
    assert is_valid is True
    assert report["num_rows"] == len(sample_mwd_data)
    assert len(report["provenance_hash"]) == 16


def test_validate_mwd_dataset_missing_columns():
    invalid_df = pd.DataFrame({"hole_id": ["H1"], "penetration_rate_m_hr": [10.0]})
    is_valid, report = validate_mwd_dataset(invalid_df)
    assert is_valid is False
    assert len(report["missing_columns"]) > 0


def test_compute_mwd_features(sample_mwd_data):
    df_feat = compute_mwd_features(sample_mwd_data)
    assert "mwd_specific_energy_mj_m3" in df_feat.columns
    assert "rop_torque_ratio" in df_feat.columns
    assert not df_feat["mwd_specific_energy_mj_m3"].isna().any()


def test_majority_baseline(sample_mwd_data):
    df_feat = compute_mwd_features(sample_mwd_data)
    baseline = MajorityClassBaseline()
    X = df_feat[FEATURE_COLUMNS]
    y = df_feat[TARGET_COLUMN]
    baseline.fit(X, y)

    preds = baseline.predict(X)
    assert len(preds) == len(X)
    assert len(np.unique(preds)) == 1


def test_georesponse_classifier(sample_mwd_data):
    df_feat = compute_mwd_features(sample_mwd_data)
    X = df_feat[FEATURE_COLUMNS]
    y = df_feat[TARGET_COLUMN]

    clf = GeoResponseClassifier(model_type="random_forest")
    clf.fit(X, y)

    results_df = clf.predict_with_review_flags(X)
    assert "predicted_rock_type" in results_df.columns
    assert "confidence_score" in results_df.columns
    assert "requires_review_flag" in results_df.columns

    low_conf_mask = results_df["confidence_score"] < 0.70
    assert (results_df.loc[low_conf_mask, "requires_review_flag"] == True).all()


def test_evaluate_rock_type_classifier(sample_mwd_data):
    df_feat = compute_mwd_features(sample_mwd_data)
    eval_res = evaluate_rock_type_classifier(
        df=df_feat,
        feature_cols=FEATURE_COLUMNS,
        target_col=TARGET_COLUMN,
        group_col="tunnel_id",
        n_splits=3,
    )

    assert "random_forest" in eval_res
    assert "majority_baseline" in eval_res
    assert eval_res["random_forest"]["macro_f1"] >= 0.0
    assert eval_res["random_forest"]["balanced_accuracy"] >= 0.0
    assert "confusion_matrix" in eval_res["random_forest"]
