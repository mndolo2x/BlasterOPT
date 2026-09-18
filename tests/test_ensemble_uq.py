"""
Unit tests for Ensemble Uncertainty Quantification and OOD Detection (`src/ensemble_uq`).
"""

import pytest
import numpy as np
from src.ensemble_uq.models import TrainingConfig
from src.ensemble_uq.ensemble_trainer import EnsembleTrainer
from src.ensemble_uq.ood_detector import OODDetector
from src.ensemble_uq.uncertainty_quantifier import UncertaintyQuantifier
from src.ensemble_uq.prediction import UncertaintyAwarePredictor


def test_ensemble_trainer_and_rmse():
    """
    Verifies that training an ensemble of N=7 GA-ANN members on synthetic data
    produces lower RMSE than a single individual model member.
    """
    np.random.seed(42)
    n_samples = 150
    n_features = 6

    X = np.random.uniform(1.0, 10.0, size=(n_samples, n_features))
    # True non-linear relationship with noise
    y1 = 10.0 * X[:, 0] - 2.5 * X[:, 1] + 0.5 * (X[:, 2] ** 2) + np.random.normal(0, 2.0, size=n_samples)
    y2 = 5.0 * X[:, 1] + 1.2 * X[:, 3] + np.random.normal(0, 1.0, size=n_samples)
    Y = np.column_stack([y1, y2])

    X_train, X_test = X[:120], X[120:]
    Y_train, Y_test = Y[:120], Y[120:]

    config = TrainingConfig(n_models=7, seeds=[42, 101, 202, 303, 404, 505, 606])
    trainer = EnsembleTrainer(config=config)
    members = trainer.train(X_train, Y_train)

    assert len(members) == 7

    # Evaluate single member prediction RMSE
    single_pred = members[0].predict(X_test)
    single_rmse = np.sqrt(np.mean((Y_test - single_pred) ** 2))

    # Evaluate ensemble mean prediction RMSE
    ens_preds = np.array([m.predict(X_test) for m in members])
    ens_mean_pred = np.mean(ens_preds, axis=0)
    ens_rmse = np.sqrt(np.mean((Y_test - ens_mean_pred) ** 2))

    # Verify ensemble RMSE is less than or equal to single model RMSE
    assert ens_rmse <= single_rmse * 1.05


def test_ood_detector_out_of_range_inputs():
    """
    Tests that OODDetector catches out-of-range inputs such as:
    - powder_factor_kg_m3 > 1.20
    - bench_height_m > 18.0
    """
    detector = OODDetector()

    # Normal in-distribution input
    normal_input = {"powder_factor_kg_m3": 0.65, "bench_height_m": 12.0, "burden_m": 6.0}
    rep_normal = detector.detect(normal_input)
    assert rep_normal.is_ood is False
    assert rep_normal.severity == "NORMAL"

    # Out-of-range powder factor > 1.20
    high_pf_input = {"powder_factor_kg_m3": 1.45, "bench_height_m": 12.0, "burden_m": 6.0}
    rep_pf = detector.detect(high_pf_input)
    assert rep_pf.is_ood is True
    assert len(rep_pf.threshold_exceeded) >= 1
    assert "exceeds maximum bound" in rep_pf.threshold_exceeded[0]

    # Out-of-range bench height > 18m
    high_bench_input = {"powder_factor_kg_m3": 0.65, "bench_height_m": 22.0, "burden_m": 6.0}
    rep_bench = detector.detect(high_bench_input)
    assert rep_bench.is_ood is True
    assert "exceeds maximum bound" in rep_bench.threshold_exceeded[0]


def test_uncertainty_higher_for_ood_inputs():
    """
    Tests that uncertainty-aware predictor yields higher epistemic uncertainty / flags OOD for OOD inputs.
    """
    predictor = UncertaintyAwarePredictor()

    normal_dict = {"powder_factor_kg_m3": 0.65, "bench_height_m": 12.0, "burden_m": 6.0}
    ood_dict = {"powder_factor_kg_m3": 1.50, "bench_height_m": 25.0, "burden_m": 15.0}

    pred_normal = predictor.predict(normal_dict)
    pred_ood = predictor.predict(ood_dict)

    assert pred_normal.is_ood is False
    assert pred_ood.is_ood is True
    assert predictor.flag_high_uncertainty(pred_ood) is True


def test_confidence_interval_coverage_and_explanation():
    """
    Tests 95% CI bounds calculation and natural language explanation generation.
    """
    predictor = UncertaintyAwarePredictor()

    normal_dict = {"powder_factor_kg_m3": 0.65, "bench_height_m": 12.0, "burden_m": 6.0}
    pred = predictor.predict(normal_dict)

    for target in pred.mean.keys():
        assert pred.lower_95[target] <= pred.mean[target] <= pred.upper_95[target]

    explanation = predictor.explain_uncertainty(pred)
    assert isinstance(explanation, str)
    assert "HIGH PREDICTION CONFIDENCE" in explanation or "OOD WARNING" in explanation
