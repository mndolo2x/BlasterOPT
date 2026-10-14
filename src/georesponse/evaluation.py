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
    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    y = df[target_col].values
    classes = np.unique(y)

    # GroupKFold setup
    if group_col and group_col in df.columns and df[group_col].nunique() >= n_splits:
        groups = df[group_col].values
        cv = GroupKFold(n_splits=n_splits)
        split_generator = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_generator = cv.split(X, y)

    rf_y_true, rf_y_pred = [], []
    base_y_true, base_y_pred = [], []

    for train_idx, test_idx in split_generator:
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # 1. Random Forest
        rf = GeoResponseClassifier(model_type="random_forest", random_state=random_state)
        rf.fit(X_tr, y_tr)
        rf_preds = rf.predict(X_te)

        rf_y_true.extend(y_te)
        rf_y_pred.extend(rf_preds)

        # 2. Majority Baseline
        base = GeoResponseClassifier(model_type="majority_baseline")
        base.fit(X_tr, y_tr)
        base_preds = base.predict(X_te)

        base_y_true.extend(y_te)
        base_y_pred.extend(base_preds)

    # Compute Metrics - Random Forest
    rf_macro_f1 = float(f1_score(rf_y_true, rf_y_pred, average="macro"))
    rf_balanced_acc = float(balanced_accuracy_score(rf_y_true, rf_y_pred))
    rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(rf_y_true, rf_y_pred, labels=classes)
    rf_cm = confusion_matrix(rf_y_true, rf_y_pred, labels=classes).tolist()

    # Compute Metrics - Majority Baseline
    base_macro_f1 = float(f1_score(base_y_true, base_y_pred, average="macro"))
    base_balanced_acc = float(balanced_accuracy_score(base_y_true, base_y_pred))

    per_class_rf = {}
    for i, c in enumerate(classes):
        per_class_rf[str(c)] = {
            "precision": float(round(rf_prec[i], 4)),
            "recall": float(round(rf_rec[i], 4)),
            "f1": float(round(rf_f1[i], 4)),
        }

    return {
        "n_samples": len(df),
        "n_classes": len(classes),
        "class_labels": [str(c) for c in classes],
        "validation_strategy": f"GroupKFold({group_col})" if group_col and group_col in df.columns else "StratifiedKFold",
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
    }
