"""
Metrics Calculation & Automated Data Quality Health Warnings for Blast Performance Feedback.

Implements exact error metric definitions:
- Signed Error e_i = predicted - measured
- MAE = (1/n) * sum(|e_i|)
- RMSE = sqrt((1/n) * sum(e_i^2))
- Bias = (1/n) * sum(e_i)
- Zero-safe percentage errors
- Automated Data Quality & Model Health Warning flags (Insufficient Data, Review Required, Model Performance Degraded, No Valid Comparison Available)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from src.performance_feedback.models import ComparisonResult


def calculate_regression_metrics(
    predictions: List[float], actuals: List[float]
) -> Dict[str, Any]:
    """
    Calculates MAE, RMSE, Bias, and MAPE for paired predictions and actuals.

    Signed-Error Convention:
        e_i = predicted_i - actual_i
        Positive bias indicates consistent overprediction.
    """
    if not predictions or not actuals or len(predictions) != len(actuals):
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "mape": None,
            "warning": "No Valid Comparison Available",
        }

    preds = np.asarray(predictions, dtype=float)
    acts = np.asarray(actuals, dtype=float)
    n = len(preds)

    if n == 0:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "mape": None,
            "warning": "No Valid Comparison Available",
        }

    # Signed errors
    errors = preds - acts
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    # Zero-safe percentage error handling
    valid_pct_mask = np.abs(acts) > 1e-6
    if np.any(valid_pct_mask):
        pct_errors = (abs_errors[valid_pct_mask] / np.abs(acts[valid_pct_mask])) * 100.0
        mape = float(np.mean(pct_errors))
    else:
        mape = None

    # Data health warnings logic
    warning = None
    if n < 5:
        warning = "Insufficient Data"
    elif mae > 50.0 or abs(bias) > 20.0:
        warning = "Model Performance Degraded"
    elif abs(bias) > 5.0:
        warning = "Review Required"

    return {
        "count": n,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "mape": round(mape, 2) if mape is not None else None,
        "warning": warning,
    }


def evaluate_comparisons(comparisons: List[ComparisonResult]) -> Dict[str, Any]:
    """Evaluates list of ComparisonResult objects into summary metrics and Health status."""
    if not comparisons:
        return {
            "count": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "bias": 0.0,
            "mape": 0.0,
            "health_status": "No Valid Comparison Available",
            "status_color": "gray",
        }

    preds = [c.predicted_value for c in comparisons]
    acts = [c.measured_value for c in comparisons]

    metrics = calculate_regression_metrics(preds, acts)
    warning = metrics.get("warning")

    if warning == "Insufficient Data":
        status_color = "orange"
        health_status = "Insufficient Data (sample count < 5)"
    elif warning == "Model Performance Degraded":
        status_color = "red"
        health_status = "Model Performance Degraded"
    elif warning == "Review Required":
        status_color = "yellow"
        health_status = "Review Required (systematic bias detected)"
    else:
        status_color = "green"
        health_status = "Model Operating Within Normal Parameters"

    metrics["health_status"] = health_status
    metrics["status_color"] = status_color
    return metrics
