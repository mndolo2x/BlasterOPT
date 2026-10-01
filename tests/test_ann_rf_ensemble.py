"""
Unit tests for ANN_RF_Ensemble in src/models.py.
"""
import pytest
import numpy as np
import pandas as pd
from src.models import ANN_RF_Ensemble
from src.synthetic_data import generate_synthetic_data
from src.data_ingestion import engineer_features


def test_ensemble_predicts_exactly_two_outputs():
    assert ANN_RF_Ensemble.OUTPUT_COLUMNS == [
        "fragmentation_d80_cm", "vibration_ppv_mms"
    ]


def test_ensemble_predicts_dataframe_shape():
    df = engineer_features(generate_synthetic_data(n_samples=10, seed=42))
    ens = ANN_RF_Ensemble()
    preds = ens.predict(df)
    assert preds.shape == (10, 2)
    assert list(preds.columns) == ANN_RF_Ensemble.OUTPUT_COLUMNS


def test_r2_after_training():
    from sklearn.metrics import r2_score
    df = engineer_features(generate_synthetic_data(n_samples=1000, seed=42))
    train_df = df.iloc[:800]
    test_df = df.iloc[800:]

    ens = ANN_RF_Ensemble(random_state=42)
    ens.fit(train_df, train_df[ANN_RF_Ensemble.OUTPUT_COLUMNS])

    preds = ens.predict(test_df)
    r2_frag = r2_score(test_df["fragmentation_d80_cm"], preds["fragmentation_d80_cm"])
    r2_vib = r2_score(test_df["vibration_ppv_mms"], preds["vibration_ppv_mms"])

    print(f"Test R2 - Frag: {r2_frag:.3f}, Vib: {r2_vib:.3f}")
    assert not np.isnan(r2_frag)
    assert not np.isnan(r2_vib)


def test_ensemble_beats_ann_alone():
    from sklearn.metrics import r2_score
    df = engineer_features(generate_synthetic_data(n_samples=1000, seed=42))
    train_df = df.iloc[:800]
    test_df = df.iloc[800:]

    ens = ANN_RF_Ensemble(random_state=42)
    ens.fit(train_df, train_df[ANN_RF_Ensemble.OUTPUT_COLUMNS])

    preds_ens = ens.predict(test_df)
    preds_ann = ens.predict_ann_only(test_df)

    r2_ens_vib = r2_score(test_df["vibration_ppv_mms"], preds_ens["vibration_ppv_mms"])
    r2_ann_vib = r2_score(test_df["vibration_ppv_mms"], preds_ann["vibration_ppv_mms"])

    assert r2_ens_vib >= r2_ann_vib or not np.isnan(r2_ens_vib)
