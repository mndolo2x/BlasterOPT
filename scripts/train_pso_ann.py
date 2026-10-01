"""
Training script for PSO-ANN Model (`scripts/train_pso_ann.py`).
Trains PSOANNModel using Particle Swarm Optimization and prints R2 and RMSE metrics.
"""

import sys
import os

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import r2_score, root_mean_squared_error
from src.synthetic_data import generate_synthetic_data
from src.data_ingestion import engineer_features
from src.models import PSOANNModel


def train_and_evaluate_pso_ann():
    print("Generating synthetic blast dataset (1000 samples)...")
    df = engineer_features(generate_synthetic_data(n_samples=1000, seed=42))

    train_df = df.iloc[:800]
    test_df = df.iloc[800:]

    input_cols = PSOANNModel.INPUT_COLUMNS

    print(f"Training PSO-ANN Model via PSO (30 particles, 100 iterations) on {len(train_df)} samples...")
    model = PSOANNModel(input_size=len(input_cols))
    model.fit(
        train_df[input_cols],
        train_df[PSOANNModel.OUTPUT_COLUMNS],
        n_particles=30,
        n_iterations=100,
        c1=1.5,
        c2=1.5,
        w=0.7,
    )

    print(f"Evaluating on holdout test set ({len(test_df)} samples)...")
    preds = model.predict(test_df[input_cols])

    print("\n=== PSO-ANN Evaluation Results ===")
    for col in PSOANNModel.OUTPUT_COLUMNS:
        y_true = test_df[col].values
        y_pred = preds[col].values
        r2 = r2_score(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        print(f"  {col}: R² = {r2:.4f} | RMSE = {rmse:.4f}")

    return model, preds


if __name__ == "__main__":
    train_and_evaluate_pso_ann()
