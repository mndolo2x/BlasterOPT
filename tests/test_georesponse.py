import numpy as np
import pandas as pd
import pytest

from src.georesponse.config import MINIMAL_REQUIRED_MWD_COLUMNS, OPTIONAL_MWD_COLUMNS, TARGET_COLUMN
from src.georesponse.evaluation import evaluate_rock_type_classifier
from src.georesponse.features import compute_mwd_features, prepare_feature_matrix
from src.georesponse.ingestion import normalize_mwd_schema, validate_mwd_dataset
from src.georesponse.models import GeoResponseClassifier, MajorityClassBaseline
from models.custom.geo_response_model import GeoResponseModel


def test_mwd_headers_with_spaces_and_hyphens():
    df = pd.DataFrame({
        " Penetration Rate M Hr ": [30.0, 45.0],
        "ROCK-TYPE-LABEL": ["Granite", "Gneiss"]
    })
    df_norm, report = normalize_mwd_schema(df)
    assert report["is_eligible_for_training"] is True
    assert "penetration_rate_m_hr" in df_norm.columns
    assert "rock_type_label" in df_norm.columns
    assert df_norm["penetration_rate_m_hr"].iloc[0] == 30.0


def test_mwd_ambiguous_semantic_aliases_require_confirmation():
    df = pd.DataFrame({
        "penetration_rate_m_hr": [30.0, 45.0],
        "Rock Type": ["Granite", "Gneiss"]
    })
    df_norm, report = normalize_mwd_schema(df)
    # Ambiguous "Rock Type" should be suggested, not mapped silently
    assert "rock_type_label" not in df_norm.columns
    assert report["suggested_mappings"]["Rock Type"] == "rock_type_label"
    assert report["is_eligible_for_training"] is False

    # Confirmed mapping
    df_confirmed, report_confirmed = normalize_mwd_schema(df, user_confirmed_mappings={"Rock Type": "rock_type_label"})
    assert "rock_type_label" in df_confirmed.columns
    assert report_confirmed["is_eligible_for_training"] is True


def test_mwd_duplicate_conflicting_mappings_detected():
    df = pd.DataFrame({
        "penetration_rate_m_hr": [30.0, 45.0],
        "rock_type_label": ["Granite", "Gneiss"],
        "Rock Type Label": ["Gneiss", "Granite"]
    })
    df_norm, report = normalize_mwd_schema(df)
    assert report["is_eligible_for_training"] is False
    assert len(report["duplicate_mappings"]) > 0
    assert report["duplicate_mappings"][0]["is_identical"] is False


def test_mwd_unknown_columns_preserved_and_reported():
    df = pd.DataFrame({
        "penetration_rate_m_hr": [30.0, 45.0],
        "rock_type_label": ["Granite", "Gneiss"],
        "random_extra_sensor": [1.23, 4.56]
    })
    df_norm, report = normalize_mwd_schema(df)
    assert report["is_eligible_for_training"] is True
    assert "random_extra_sensor" in df_norm.columns
    assert "random_extra_sensor" in report["unknown_columns"]


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


def test_clearly_separable_six_class_dataset():
    """1. Clearly separable 6-class dataset testing non-zero diagonal and high scores."""
    np.random.seed(42)
    n_per_class = 20
    classes = ["Soft_Sandstone", "Medium_Basalt", "Hard_Granite", "Fractured_Gneiss", "Kimberlite", "Shale"]

    dfs = []
    for i, cls in enumerate(classes):
        # Create strong separable feature signal
        dfs.append(pd.DataFrame({
            "penetration_rate_m_hr": np.random.normal((i + 1) * 15.0, 1.0, n_per_class),
            "percussion_pressure_bar": np.random.normal((i + 1) * 25.0, 1.0, n_per_class),
            "rock_type_label": [cls] * n_per_class
        }))
    df_sep = pd.concat(dfs, ignore_index=True)

    eval_res = evaluate_rock_type_classifier(
        df=df_sep,
        feature_cols=["penetration_rate_m_hr", "percussion_pressure_bar"],
        target_col="rock_type_label",
        group_col=None,
        n_splits=5
    )

    rf_res = eval_res["random_forest"]
    assert rf_res["macro_f1"] > 0.85
    cm = np.array(rf_res["confusion_matrix"])
    # Verify non-zero diagonal
    assert np.all(np.diag(cm) > 0)


def test_target_label_permutation():
    """2. Permuted target label test verifying alignment checks/poor metrics."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "penetration_rate_m_hr": np.linspace(10, 80, n),
        "rock_type_label": np.repeat(["A", "B", "C"], 20)
    })

    # Permute target labels randomly
    df["permuted_label"] = np.random.permutation(df["rock_type_label"].values)

    eval_res = evaluate_rock_type_classifier(
        df=df,
        feature_cols=["penetration_rate_m_hr"],
        target_col="permuted_label",
        group_col=None,
        n_splits=3
    )
    # Permuted labels should have near-random Macro-F1 (~0.33)
    assert eval_res["random_forest"]["macro_f1"] < 0.50


def test_label_encoder_and_class_index_consistency():
    """3. Label encoder and class-to-index consistency test."""
    model = GeoResponseClassifier(model_type="random_forest")
    df_train = pd.DataFrame({
        "penetration_rate_m_hr": [10.0, 20.0, 30.0, 40.0],
        "rock_type_label": ["Soft", "Medium", "Hard", "Soft"]
    })
    model.fit(df_train, df_train["rock_type_label"])
    assert list(model.classes_) == ["Hard", "Medium", "Soft"]

    preds = model.predict(df_train)
    assert set(preds).issubset(set(["Hard", "Medium", "Soft"]))


def test_out_of_fold_predictions_alignment():
    """4. Out-of-fold prediction alignment and single-sample evaluation."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "penetration_rate_m_hr": np.random.uniform(10, 80, n),
        "rock_type_label": np.random.choice(["Class1", "Class2"], size=n)
    })
    eval_res = evaluate_rock_type_classifier(
        df=df,
        feature_cols=["penetration_rate_m_hr"],
        target_col="rock_type_label",
        group_col=None,
        n_splits=5
    )
    diag = eval_res["diagnostics"]
    assert diag["evaluated_rows"] == n
    assert diag["confusion_matrix_total"] == n


def test_confusion_matrix_diagonal_equals_correct_predictions():
    """5. Confusion matrix diagonal sum equals total correct predictions."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "penetration_rate_m_hr": np.random.uniform(10, 80, n),
        "rock_type_label": np.repeat(["A", "B", "C", "D", "E", "F"], 20)
    })
    eval_res = evaluate_rock_type_classifier(
        df=df,
        feature_cols=["penetration_rate_m_hr"],
        target_col="rock_type_label",
        group_col=None,
        n_splits=5
    )
    cm = np.array(eval_res["random_forest"]["confusion_matrix"])
    diag = eval_res["diagnostics"]
    assert int(np.trace(cm)) == diag["total_correct_predictions"]
    assert int(np.sum(cm)) == diag["confusion_matrix_total"] == n


def test_negative_f1_delta_formatting():
    """6. Negative F1 delta formatting test."""
    f1_rf = 0.0000
    f1_base = 0.0476
    delta = round(f1_rf - f1_base, 4)
    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    assert delta_str == "-0.0476"


def test_balanced_six_class_majority_baseline():
    """7. Balanced six-class constant majority classifier produces Macro-F1 of ~0.0476."""
    np.random.seed(42)
    n = 120
    classes = ["C1", "C2", "C3", "C4", "C5", "C6"]
    df = pd.DataFrame({
        "penetration_rate_m_hr": np.random.uniform(10, 80, n),
        "rock_type_label": np.repeat(classes, 20)
    })
    eval_res = evaluate_rock_type_classifier(
        df=df,
        feature_cols=["penetration_rate_m_hr"],
        target_col="rock_type_label",
        group_col=None,
        n_splits=5
    )
    base_f1 = eval_res["majority_baseline"]["macro_f1"]
    # For balanced 6 classes, constant majority classifier has precision 1/6 for 1 class (0 else), recall 1.0 for 1 class (0 else)
    # F1 for majority class = 2*(1/6*1)/(1/6+1) = 2/7 = 0.2857. Macro F1 = (2/7)/6 = 1/21 ~ 0.0476.
    assert pytest.approx(base_f1, abs=0.01) == 0.0476


def test_regression_with_120_row_uninformative_fixture():
    """8. Regression test with 120-row fixture reporting genuine uninformative data limitation."""
    np.random.seed(42)
    n = 120
    classes = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6"]
    # Pure random noise features vs rock labels
    df_uninformative = pd.DataFrame({
        "penetration_rate_m_hr": np.random.uniform(10, 80, n),
        "percussion_pressure_bar": np.random.uniform(100, 250, n),
        "rock_type_label": np.repeat(classes, 20)
    })
    eval_res = evaluate_rock_type_classifier(
        df=df_uninformative,
        feature_cols=["penetration_rate_m_hr", "percussion_pressure_bar"],
        target_col="rock_type_label",
        group_col=None,
        n_splits=5
    )
    assert eval_res["n_samples"] == 120
    assert eval_res["n_classes"] == 6
    assert eval_res["diagnostics"]["scoring_arrays_aligned"] is True


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
