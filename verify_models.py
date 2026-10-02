"""
verify_models.py

Verify that all trained models produce physically plausible predictions.
Run from the repo root:
    python verify_models.py

Verification Run Output:
========================================================================
BLASTOPT MODEL VERIFICATION
========================================================================

────────────────────────────────────────────────────────────────────────
MODEL: GA-ANN Jwaneng Multi-Output Model
────────────────────────────────────────────────────────────────────────
  ✅ fragmentation_d80_cm
      R²   = 0.9782   (expected 0.85 – 0.99)
  ✅ RMSE = 3.1810   (expected 0.50 – 5.00)
  ✅ vibration_ppv_mms
      R²   = 0.9811   (expected 0.85 – 0.99)
  ✅ RMSE = 0.6565   (expected 0.05 – 1.00)
  ✅ airblast_db
      R²   = 0.8998   (expected 0.88 – 0.99)
  ✅ RMSE = 1.8303   (expected 0.50 – 5.00)

  Sample predictions (first 3 rows):
    ✅ fragmentation_d80_cm
        actual:    [50.86 67.52 58.93]
        predicted: [52.31  66.194 58.175]
    ✅ vibration_ppv_mms
        actual:    [6.9  1.81 1.57]
        predicted: [6.506 1.85  1.661]
    ✅ airblast_db
        actual:    [120.9 115.1 117.4]
        predicted: [122.534 115.378 116.632]

────────────────────────────────────────────────────────────────────────
MODEL: ANN-RF Ensemble Jwaneng Predictor
────────────────────────────────────────────────────────────────────────
  ✅ fragmentation_d80_cm
      R²   = 0.9610   (expected 0.85 – 0.99)
  ✅ RMSE = 4.1803   (expected 0.50 – 5.00)
  ✅ vibration_ppv_mms
      R²   = 0.9841   (expected 0.85 – 0.99)
  ✅ RMSE = 0.6008   (expected 0.05 – 1.00)

  Sample predictions (first 3 rows):
    ✅ fragmentation_d80_cm
        actual:    [50.86 67.52 58.93]
        predicted: [49.481 68.266 57.399]
    ✅ vibration_ppv_mms
        actual:    [6.9  1.81 1.57]
        predicted: [6.692 1.795 1.644]

────────────────────────────────────────────────────────────────────────
MODEL: PSO-ANN Orapa Fragmentation Model
────────────────────────────────────────────────────────────────────────
  ✅ fragmentation_d80_cm
      R²   = 0.9766   (expected 0.85 – 0.99)
  ✅ RMSE = 3.2535   (expected 0.50 – 5.00)

  Sample predictions (first 3 rows):
    ✅ fragmentation_d80_cm
        actual:    [50.86 67.52 58.93]
        predicted: [46.794 73.068 58.819]

────────────────────────────────────────────────────────────────────────
MODEL: Debswana Open-Pit Airblast Minimizer
────────────────────────────────────────────────────────────────────────
  ✅ airblast_db
      R²   = 0.8654   (expected 0.85 – 0.99)
  ✅ RMSE = 2.1303   (expected 0.50 – 5.00)

  Sample predictions (first 3 rows):
    ✅ airblast_db
        actual:    [120.9 115.1 117.4]
        predicted: [120.653 115.111 116.506]

========================================================================
✅ ALL MODELS VERIFIED — demo-ready
========================================================================
"""
import numpy as np
import pandas as pd
from src.models import MODEL_REGISTRY, FEATURE_COLS
from src.synthetic_data import generate_synthetic_data

# Define alias if generate_for_model is expected
def generate_for_model(model_key, n_samples=1000):
    """Generate synthetic dataset suitable for the specified model."""
    return generate_synthetic_data(n_samples=n_samples)

from src.models import manual_cv_score


# Expected R² ranges per model, based on BIUST papers and physics
EXPECTED_R2 = {
    "ga_ann_jwaneng": {
        "fragmentation_d80_cm": (0.85, 0.99),
        "vibration_ppv_mms":    (0.85, 0.99),
        "airblast_db":          (0.88, 0.99),
    },
    "ann_rf_ensemble_jwaneng": {
        "fragmentation_d80_cm": (0.85, 0.99),
        "vibration_ppv_mms":    (0.85, 0.99),
    },
    "pso_ann_orapa": {
        "fragmentation_d80_cm": (0.85, 0.99),
    },
    "airblast_minimizer": {
        "airblast_db": (0.85, 0.99),
    },
    "flyrock_predictor": {
        "flyrock_m": (0.90, 1.00),
    },
    "cost_predictor": {
        "cost_per_tonne_usd": (0.90, 1.00),
    },
}

# Physical bounds for each output
PHYSICAL_RANGES = {
    "fragmentation_d80_cm": (5.0, 150.0),
    "vibration_ppv_mms":    (0.1, 50.0),
    "airblast_db":          (40.0, 140.0),
    "flyrock_m":            (5.0, 500.0),
    "cost_per_tonne_usd":   (0.30, 3.00),
}

# Expected RMSE ranges
EXPECTED_RMSE = {
    "fragmentation_d80_cm": (0.5, 5.0),
    "vibration_ppv_mms":    (0.05, 1.0),
    "airblast_db":          (0.5, 5.0),
    "flyrock_m":            (1.0, 50.0),
    "cost_per_tonne_usd":   (0.01, 0.30),
}


def get_model_class(key):
    from src.models import (
        GAANNModel, ANN_RF_Ensemble, PSOANNModel,
        AirblastMinimizerModel, FlyrockPredictor, CostPredictor,
    )
    return {
        "ga_ann_jwaneng": GAANNModel,
        "ann_rf_ensemble_jwaneng": ANN_RF_Ensemble,
        "pso_ann_orapa": PSOANNModel,
        "airblast_minimizer": AirblastMinimizerModel,
        "flyrock_predictor": FlyrockPredictor,
        "cost_predictor": CostPredictor,
    }[key]


def verify_all_models(n_samples=1000, cv=5):
    print("=" * 72)
    print("BLASTOPT MODEL VERIFICATION")
    print("=" * 72)

    all_pass = True

    for model_key, config in MODEL_REGISTRY.items():
        if model_key not in EXPECTED_R2:
            continue

        print(f"\n{'─' * 72}")
        print(f"MODEL: {config['display_name']}")
        print(f"{'─' * 72}")

        outputs = config["outputs"]
        expected_r2 = EXPECTED_R2.get(model_key, {})

        # 1. Generate data for this model
        try:
            df = generate_for_model(model_key, n_samples=n_samples)
        except Exception as e:
            print(f"  ❌ Data generation failed: {e}")
            all_pass = False
            continue

        # 2. Split features and targets
        model_cls = get_model_class(model_key)
        features = getattr(model_cls, "INPUT_COLUMNS", FEATURE_COLS)
        X = df[features]
        y = df[outputs]

        # 3. Cross-validate
        try:
            results = manual_cv_score(
                model_class=get_model_class(model_key),
                X=X, y=y, cv=cv,
            )
        except Exception as e:
            print(f"  ❌ Cross-validation failed: {e}")
            all_pass = False
            continue

        # 4. Check R² and RMSE per output
        for output in outputs:
            r2 = results[output]["r2_mean"]
            rmse = results[output]["rmse_mean"]

            lo, hi = expected_r2.get(output, (0.5, 1.0))
            r2_ok = lo <= r2 <= hi
            status = "✅" if r2_ok else "❌"
            if not r2_ok:
                all_pass = False

            rmse_lo, rmse_hi = EXPECTED_RMSE.get(output, (0, 1e9))
            rmse_ok = rmse_lo <= rmse <= rmse_hi
            rmse_status = "✅" if rmse_ok else "❌"
            if not rmse_ok:
                all_pass = False

            print(f"  {status} {output}")
            print(f"      R²   = {r2:.4f}   (expected {lo:.2f} – {hi:.2f})")
            print(f"  {rmse_status} RMSE = {rmse:.4f}   (expected {rmse_lo:.2f} – {rmse_hi:.2f})")

        # 5. Sample predictions sanity check
        try:
            model = get_model_class(model_key)()
            model.fit(X, y)
            preds = model.predict(X.head(3))
        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")
            all_pass = False
            continue

        print(f"\n  Sample predictions (first 3 rows):")
        for output in outputs:
            actual = y[output].head(3).values
            predicted = preds[output].values
            bounds_lo, bounds_hi = PHYSICAL_RANGES.get(output, (0, 1e9))

            in_range = all(bounds_lo <= p <= bounds_hi for p in predicted)
            status = "✅" if in_range else "❌"
            if not in_range:
                all_pass = False

            print(f"    {status} {output}")
            print(f"        actual:    {np.round(actual, 3)}")
            print(f"        predicted: {np.round(predicted, 3)}")

    print("\n" + "=" * 72)
    if all_pass:
        print("✅ ALL MODELS VERIFIED — demo-ready")
    else:
        print("❌ SOME MODELS FAILED — review the ❌ items above")
    print("=" * 72)

    return all_pass


if __name__ == "__main__":
    verify_all_models()
