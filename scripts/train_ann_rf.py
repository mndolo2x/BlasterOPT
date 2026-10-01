"""
Training script for ANN-RF Ensemble (`scripts/train_ann_rf.py`).
Trains ANN_RF_Ensemble on synthetic blast dataset and prints R2 and RMSE metrics for both outputs.
"""

import sys
import os

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import r2_score, root_mean_squared_error
from src.synthetic_data import generate_synthetic_data
from src.data_ingestion import engineer_features
from src.models import ANN_RF_Ensemble


def train_and_evaluate_ann_rf():
    print("Generating synthetic blast dataset (1000 samples)...")
    df = engineer_features(generate_synthetic_data(n_samples=1000, seed=42))

    train_df = df.iloc[:800]
    test_df = df.iloc[800:]

    input_cols = [
        "burden_m", "spacing_m", "powder_factor_kg_m3", "stemming_m",
        "rock_factor_A", "hole_depth_m", "hole_diameter_mm",
        "max_charge_per_delay_kg", "explosive_rws", "bench_height_m"
    ]

    print(f"Training ANN-RF Ensemble Model on {len(train_df)} samples...")
    model = ANN_RF_Ensemble(random_state=42)
    model.fit(train_df[input_cols], train_df[ANN_RF_Ensemble.OUTPUT_COLUMNS])

    print(f"Evaluating on holdout test set ({len(test_df)} samples)...")
    preds = model.predict(test_df[input_cols])

    print("\n=== ANN-RF Ensemble Evaluation Results ===")
    for col in ANN_RF_Ensemble.OUTPUT_COLUMNS:
        y_true = test_df[col].values
        y_pred = preds[col].values
        r2 = r2_score(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        print(f"  {col}: R² = {r2:.4f} | RMSE = {rmse:.4f}")

    return model, preds


if __name__ == "__main__":
    train_and_evaluate_ann_rf()
