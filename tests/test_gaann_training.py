"""
Unit tests for GA-ANN model training, physics baseline accuracy, and residual bounds.
"""

def test_baseline_matches_data():
    """The physics baseline must match the synthetic data with R² > 0.95."""
    from src.models import GAANNModel, FEATURE_COLS
    from src.synthetic_data import generate_synthetic_data
    import numpy as np

    df = generate_synthetic_data(n_samples=500)
    model = GAANNModel(input_size=len(FEATURE_COLS))
    baseline = model._physics_baseline(df)

    for col in model.OUTPUT_COLUMNS:
        actual = df[col].values
        predicted = baseline[col].values
        r2 = 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
        assert r2 > 0.95, f"{col}: baseline R² = {r2:.4f}"


def test_trained_model_r2_above_90():
    """After training, the model must achieve R² > 0.90 on synthetic data."""
    from sklearn.metrics import r2_score
    from src.models import GAANNModel, FEATURE_COLS
    from src.synthetic_data import generate_synthetic_data

    df = generate_synthetic_data(n_samples=500)
    train = df.iloc[:400]
    test = df.iloc[400:]

    model = GAANNModel(input_size=len(FEATURE_COLS))
    model.fit(train[FEATURE_COLS], train[model.OUTPUT_COLUMNS])

    preds = model.predict(test)
    for col in model.OUTPUT_COLUMNS:
        r2 = r2_score(test[col], preds[col])
        assert r2 > 0.90, f"{col}: R² = {r2:.4f}"


def test_predictions_stay_close_to_baseline():
    """Predictions must not deviate more than 50% from the physics baseline."""
    from src.models import GAANNModel, FEATURE_COLS
    from src.synthetic_data import generate_synthetic_data
    import numpy as np

    df = generate_synthetic_data(n_samples=200)
    model = GAANNModel(input_size=len(FEATURE_COLS))
    model.fit(df[FEATURE_COLS], df[model.OUTPUT_COLUMNS])

    baseline = model._physics_baseline(df)
    preds = model.predict(df)

    for col in model.OUTPUT_COLUMNS:
        ratio = preds[col].values / baseline[col].values
        assert (ratio >= 0.5).all() and (ratio <= 1.5).all(), \
            f"{col}: predictions outside 50-150% of baseline"
