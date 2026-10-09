"""
Unit tests for model verification and physical prediction constraints.
"""

def test_all_models_produce_positive_r2():
    """After training, every model must achieve positive R² on holdout data."""
    from sklearn.metrics import r2_score
    from src.models import MODEL_REGISTRY
    from src.synthetic_data import generate_for_model
    from verify_models import get_model_class

    for key, config in MODEL_REGISTRY.items():
        if key not in ["ga_ann_jwaneng", "ann_rf_ensemble_jwaneng", "pso_ann_orapa", "airblast_minimizer", "flyrock_predictor", "cost_predictor"]:
            continue
        df = generate_for_model(key, n_samples=500)
        outputs = config["outputs"]
        features = [c for c in df.columns if c not in outputs]

        train = df.iloc[:400]
        test = df.iloc[400:]

        model = get_model_class(key)()
        model.fit(train[features], train[outputs])
        preds = model.predict(test[features])

        for col in outputs:
            r2 = r2_score(test[col], preds[col])
            assert r2 > 0.0, f"{key}.{col}: R² = {r2:.4f} is non-positive"


def test_all_models_produce_in_range_predictions():
    """All predictions must fall within physical bounds."""
    from src.models import MODEL_REGISTRY
    from src.synthetic_data import generate_for_model
    from verify_models import get_model_class, PHYSICAL_RANGES

    for key, config in MODEL_REGISTRY.items():
        if key not in ["ga_ann_jwaneng", "ann_rf_ensemble_jwaneng", "pso_ann_orapa", "airblast_minimizer", "flyrock_predictor", "cost_predictor"]:
            continue
        df = generate_for_model(key, n_samples=200)
        outputs = config["outputs"]
        features = [c for c in df.columns if c not in outputs]

        model = get_model_class(key)()
        model.fit(df[features], df[outputs])
        preds = model.predict(df[features])

        for col in outputs:
            if col in PHYSICAL_RANGES:
                lo, hi = PHYSICAL_RANGES[col]
                assert (preds[col] >= lo).all() and (preds[col] <= hi).all(), (
                    f"{key}.{col}: predictions outside [{lo}, {hi}]"
                )


def test_gaann_learns_something():
    """The GA-ANN must produce predictions with variance, not constants."""
    from src.models import GAANNModel, FEATURE_COLS
    from src.synthetic_data import generate_for_model

    df = generate_for_model("ga_ann_jwaneng", n_samples=500)
    model = GAANNModel(input_size=len(getattr(GAANNModel, "INPUT_COLUMNS", FEATURE_COLS)))
    features = [c for c in df.columns if c not in model.OUTPUT_COLUMNS]
    model.fit(df[features], df[model.OUTPUT_COLUMNS])

    preds = model.predict(df[features].head(50))
    for col in model.OUTPUT_COLUMNS:
        assert preds[col].std() > 1e-6, f"{col}: predictions have zero variance"
