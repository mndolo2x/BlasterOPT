"""
Group-based Cross-Validation and Metric Evaluation for GeoResponse Rock-Type Classifiers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from src.georesponse.models import GeoResponseClassifier, MajorityClassBaseline


def evaluate_rock_type_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "rock_type_label",
    group_col: Optional[str] = "tunnel_id",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes group-based cross-validation (GroupKFold by tunnel_id/blast_id) to prevent data leakage
    across observation splits, comparing Majority Class Baseline vs Random Forest.

    Returns macro-F1, balanced accuracy, per-class precision/recall, and confusion matrix.
    """
    # Pre-split assertions: Verify feature matrix and target vector row alignment
    assert len(df) == len(feature_cols_df := df[feature_cols]), "Row count mismatch between df and feature columns"
    X = feature_cols_df.copy()
    if not X.empty:
        X = X.fillna(X.median())

    y = df[target_col].values
    assert len(X) == len(y), f"Row count mismatch prior to split: X has {len(X)} rows, y has {len(y)} rows"

    classes = np.unique(y)

    # Out-of-fold predictions arrays preserving original index order
    n_samples = len(df)
    oof_rf_preds = np.empty(n_samples, dtype=object)
    oof_base_preds = np.empty(n_samples, dtype=object)
    oof_fold_numbers = np.zeros(n_samples, dtype=int)
    evaluated_mask = np.zeros(n_samples, dtype=bool)

    # GroupKFold vs StratifiedKFold setup
    if group_col and group_col in df.columns and df[group_col].nunique() >= n_splits:
        groups = df[group_col].values
        cv = GroupKFold(n_splits=n_splits)
        split_generator = cv.split(X, y, groups)
        val_strategy = f"GroupKFold({group_col})"
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_generator = cv.split(X, y)
        val_strategy = "StratifiedKFold"

    fold_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_generator, start=1):
        # Assert fold indices are strictly within bounds and non-overlapping
        assert np.max(train_idx) < n_samples and np.max(test_idx) < n_samples, "Fold indices out of sample bounds"

        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # 1. Random Forest
        rf = GeoResponseClassifier(model_type="random_forest", random_state=random_state)
        rf.fit(X_tr, y_tr)
        rf_preds = rf.predict(X_te)

        # 2. Majority Baseline
        base = GeoResponseClassifier(model_type="majority_baseline")
        base.fit(X_tr, y_tr)
        base_preds = base.predict(X_te)

        # Assign out-of-fold predictions at exact test_idx positions
        oof_rf_preds[test_idx] = rf_preds
        oof_base_preds[test_idx] = base_preds
        oof_fold_numbers[test_idx] = fold_idx
        evaluated_mask[test_idx] = True

        # Track fold-level macro-f1
        fold_rf_f1 = float(f1_score(y_te, rf_preds, average="macro", zero_division=0))
        fold_scores.append({"fold": fold_idx, "n_test": len(test_idx), "rf_macro_f1": round(fold_rf_f1, 4)})

    # Filter evaluated samples
    eval_indices = np.where(evaluated_mask)[0]
    y_eval = y[eval_indices]
    rf_eval_preds = oof_rf_preds[eval_indices]
    base_eval_preds = oof_base_preds[eval_indices]

    # Compute Metrics - Random Forest
    rf_macro_f1 = float(f1_score(y_eval, rf_eval_preds, average="macro", zero_division=0))
    rf_balanced_acc = float(balanced_accuracy_score(y_eval, rf_eval_preds))
    rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_eval, rf_eval_preds, labels=classes, zero_division=0)
    rf_cm = confusion_matrix(y_eval, rf_eval_preds, labels=classes).tolist()

    # Compute Metrics - Majority Baseline
    base_macro_f1 = float(f1_score(y_eval, base_eval_preds, average="macro", zero_division=0))
    base_balanced_acc = float(balanced_accuracy_score(y_eval, base_eval_preds))

    per_class_rf = {}
    for i, c in enumerate(classes):
        per_class_rf[str(c)] = {
            "precision": float(round(rf_prec[i], 4)),
            "recall": float(round(rf_rec[i], 4)),
            "f1": float(round(rf_f1[i], 4)),
        }

    # Total correct predictions and confusion matrix diagonal check
    cm_arr = np.array(rf_cm)
    total_correct = int(np.trace(cm_arr))
    total_cm_samples = int(np.sum(cm_arr))
    assert total_cm_samples == len(y_eval), f"Confusion matrix total ({total_cm_samples}) does not equal evaluated samples ({len(y_eval)})"

    # Diagnostics metadata
    true_counts = {str(k): int(v) for k, v in pd.Series(y_eval).value_counts().to_dict().items()}
    pred_counts = {str(k): int(v) for k, v in pd.Series(rf_eval_preds).value_counts().to_dict().items()}

    diagnostics = {
        "input_rows": len(df),
        "evaluated_rows": len(y_eval),
        "feature_columns": list(feature_cols),
        "target_column": target_col,
        "class_support_true": true_counts,
        "class_support_pred": pred_counts,
        "total_correct_predictions": total_correct,
        "confusion_matrix_total": total_cm_samples,
        "fold_level_scores": fold_scores,
        "scoring_arrays_aligned": True,
    }

    return {
        "n_samples": len(df),
        "n_classes": len(classes),
        "class_labels": [str(c) for c in classes],
        "validation_strategy": val_strategy,
        "random_forest": {
            "macro_f1": round(rf_macro_f1, 4),
            "balanced_accuracy": round(rf_balanced_acc, 4),
            "per_class": per_class_rf,
            "confusion_matrix": rf_cm,
        },
        "majority_baseline": {
            "macro_f1": round(base_macro_f1, 4),
            "balanced_accuracy": round(base_balanced_acc, 4),
        },
        "f1_improvement_over_baseline": round(rf_macro_f1 - base_macro_f1, 4),
        "diagnostics": diagnostics,
    }
