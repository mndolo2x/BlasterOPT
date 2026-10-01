"""
Training and sensitivity analysis script for Airblast Minimizer Model (`scripts/train_airblast.py`).
Trains AirblastMinimizerModel, evaluates R2/RMSE, prints sensitivity ranking, and minimizes airblast.
"""

import sys
import os

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import r2_score, root_mean_squared_error
from src.synthetic_data import generate_synthetic_data
from src.data_ingestion import engineer_features
from src.models import AirblastMinimizerModel


def train_and_evaluate_airblast():
    print("Generating synthetic blast dataset (1000 samples)...")
    df = engineer_features(generate_synthetic_data(n_samples=1000, seed=42))

    train_df = df.iloc[:800]
    test_df = df.iloc[800:]

    input_cols = AirblastMinimizerModel.INPUT_COLUMNS

    print(f"Training Airblast Minimizer Model on {len(train_df)} samples...")
    model = AirblastMinimizerModel(input_size=len(input_cols))
    model.fit(train_df[input_cols], train_df[AirblastMinimizerModel.OUTPUT_COLUMNS])

    print(f"Evaluating on holdout test set ({len(test_df)} samples)...")
    preds = model.predict(test_df[input_cols])

    print("\n=== Airblast Minimizer Evaluation Results ===")
    for col in AirblastMinimizerModel.OUTPUT_COLUMNS:
        y_true = test_df[col].values
        y_pred = preds[col].values
        r2 = r2_score(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        print(f"  {col}: R² = {r2:.4f} | RMSE = {rmse:.4f}")

    print("\n=== Sensitivity Ranking ===")
    sensitivity_ranking = model.compute_sensitivity(test_df.head(1))
    for rank, (feature, sens_score) in enumerate(sensitivity_ranking, 1):
        print(f"  #{rank} {feature}: sensitivity score = {sens_score:.4f}")

    top_3 = [item[0] for item in sensitivity_ranking[:3]]
    bottom_3 = [item[0] for item in sensitivity_ranking[-3:]]

    print(f"\nStemming in top 3 sensitive parameters: {'stemming_m' in top_3} ({top_3})")
    print(f"Spacing in bottom 3 sensitive parameters: {'spacing_m' in bottom_3} ({bottom_3})")

    print("\n=== Input Minimization ===")
    sample_input = test_df.head(1)
    init_airblast = model.predict(sample_input)["airblast_db"].values[0]
    min_input = model.minimize_airblast(sample_input)
    min_airblast = model.predict(min_input)["airblast_db"].values[0]
    print(f"  Initial predicted airblast: {init_airblast:.2f} dB")
    print(f"  Optimized predicted airblast: {min_airblast:.2f} dB")

    return model, preds


if __name__ == "__main__":
    train_and_evaluate_airblast()
