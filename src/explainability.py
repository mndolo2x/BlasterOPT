"""
Explainability Module for BlasterOPT / BlastOpt Botswana.

Provides model explainability, feature contribution attributions, and visual waterfall charts
to help certified blasters understand and trust ML predictions (d50, PPV, flyrock, cost).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.data_ingestion import engineer_features
from src.models import BlastMLPipeline, FEATURE_COLS


def get_feature_contributions(
    model_pipeline: Optional[BlastMLPipeline],
    input_payload: Dict[str, float],
    target: str = "d50_mm",
) -> Dict[str, Any]:
    """
    Computes feature contribution attributions for a single blast design input.

    Parameters:
    -----------
    model_pipeline : BlastMLPipeline, optional
        Trained machine learning pipeline.
    input_payload : Dict[str, float]
        Input blast parameters dictionary.
    target : str, default="d50_mm"
        Target outcome variable to explain ("d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd").

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing baseline value, predicted outcome, and feature contribution scores.
    """
    if not isinstance(input_payload, dict):
        return {"error": "Invalid input: input_payload must be a dictionary."}

    df_single = pd.DataFrame([input_payload])
    df_feat = engineer_features(df_single)

    if model_pipeline is None or target not in model_pipeline.models:
        # Fallback heuristic contribution estimate based on feature weights
        pf = float(input_payload.get("powder_factor_kg_m3", 0.65))
        burden = float(input_payload.get("burden_m", 6.0))
        rock_a = float(input_payload.get("rock_factor_A", 8.0))
        dist = float(input_payload.get("monitoring_distance_m", 450.0))

        baseline = 220.0 if target == "d50_mm" else (10.0 if target == "ppv_mms" else 100.0)

        contributions = {
            "powder_factor_kg_m3": round(-45.0 * (pf - 0.6), 2),
            "rock_factor_A": round(25.0 * (rock_a - 8.0), 2),
            "burden_m": round(15.0 * (burden - 6.0), 2),
            "monitoring_distance_m": round(-20.0 * ((dist - 400.0) / 100.0), 2),
        }
        pred_val = baseline + sum(contributions.values())

        return {
            "target": target,
            "baseline_value": baseline,
            "predicted_value": round(pred_val, 2),
            "contributions": contributions,
            "explanation": "Heuristic physics-guided feature contribution attribution.",
        }

    model = model_pipeline.models[target]
    feature_names = model_pipeline.feature_names if model_pipeline.feature_names else FEATURE_COLS

    X = df_feat[[c for c in feature_names if c in df_feat.columns]].copy()
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_names]

    pred_val = float(model.predict(X)[0])
    baseline = 200.0 if target == "d50_mm" else 10.0

    # Extract feature importances or coefficients
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Scale importances to represent net deviation from baseline
    diff = pred_val - baseline
    total_imp = float(np.sum(importances)) if np.sum(importances) > 0 else 1.0

    contributions = {}
    for name, imp in zip(feature_names, importances):
        contrib = float(np.round((imp / total_imp) * diff, 2))
        if abs(contrib) > 0.01:
            contributions[name] = contrib

    return {
        "target": target,
        "baseline_value": round(baseline, 2),
        "predicted_value": round(pred_val, 2),
        "contributions": contributions,
        "explanation": "Model feature contribution breakdown relative to domain baseline.",
    }


def plot_feature_contributions_waterfall(
    contribution_data: Dict[str, Any], title: str = "Feature Contribution Breakdown for Certified Blasters"
) -> go.Figure:
    """
    Generates a Plotly Waterfall chart depicting feature contributions to the predicted outcome.

    Parameters:
    -----------
    contribution_data : Dict[str, Any]
        Dictionary returned by get_feature_contributions.
    title : str
        Chart title.

    Returns:
    --------
    go.Figure
        Plotly Figure displaying the waterfall contribution breakdown.
    """
    if "error" in contribution_data:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {contribution_data['error']}", showarrow=False)
        return fig

    baseline = contribution_data.get("baseline_value", 200.0)
    contributions = contribution_data.get("contributions", {})
    predicted = contribution_data.get("predicted_value", baseline)

    measures = ["relative"] * len(contributions) + ["total"]
    x_labels = list(contributions.keys()) + ["Predicted Outcome"]
    y_values = list(contributions.values()) + [predicted]

    fig = go.Figure(
        go.Waterfall(
            name="Contribution",
            orientation="v",
            measure=["absolute"] + measures,
            x=["Baseline"] + x_labels,
            y=[baseline] + y_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#D50000"}},
            increasing={"marker": {"color": "#00C853"}},
            totals={"marker": {"color": "#2962FF"}},
        )
    )

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Blast Parameter / Feature",
        yaxis_title=f"Impact on {contribution_data.get('target', 'Target')}",
        template="plotly_white",
        height=450,
    )

    return fig


def explain_prediction(model, input_data, feature_names=None, background_data=None):
    """
    Generate SHAP explanation for a single blast prediction.

    Key drivers from Jwaneng research:
    - Fragmentation: powder factor, burden
    - Vibration: burden, charge per delay, distance
    """
    if not HAS_SHAP:
        raise ImportError("shap package is required for explain_prediction function.")

    if background_data is None:
        if isinstance(input_data, pd.DataFrame):
            background_data = input_data
        elif hasattr(input_data, "numpy"):
            import torch
            background_data = torch.zeros_like(input_data)
        else:
            background_data = np.zeros_like(input_data)

    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(input_data)
    return shap_values
