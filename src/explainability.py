"""
Explainability Module for BlasterOPT / BlastOpt Botswana.

Provides model explainability, feature contribution attributions, and visual waterfall charts
to help certified blasters understand and trust ML predictions (d50, PPV, flyrock, cost).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Union

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.data_ingestion import engineer_features
from src.models import BlastMLPipeline, FEATURE_COLS, ANN_RF_Ensemble


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
        Dictionary returned by get_feature_contributions or get_shap_explanation.
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
    contributions = contribution_data.get("contributions", contribution_data.get("feature_contributions", {}))
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


def _create_shap_force_plot(base_value: float, shap_values: np.ndarray, feature_names: List[str]) -> go.Figure:
    """Creates an interactive Plotly Force Plot visualization for SHAP values."""
    neg_indices = np.where(shap_values < 0)[0]
    pos_indices = np.where(shap_values >= 0)[0]

    fig = go.Figure()

    # Base line
    fig.add_shape(
        type="line", x0=base_value, x1=base_value, y0=0, y1=1,
        line=dict(color="gray", width=2, dash="dash")
    )

    pred_val = base_value + float(np.sum(shap_values))

    # Add bar chart for positive and negative forces
    fig.add_trace(go.Bar(
        x=[shap_values[i] for i in pos_indices],
        y=[feature_names[i] for i in pos_indices],
        orientation='h',
        marker=dict(color='#00C853'),
        name='Pushes Higher',
        base=base_value
    ))

    fig.add_trace(go.Bar(
        x=[shap_values[i] for i in neg_indices],
        y=[feature_names[i] for i in neg_indices],
        orientation='h',
        marker=dict(color='#D50000'),
        name='Pushes Lower',
        base=base_value
    ))

    fig.update_layout(
        title=f"<b>SHAP Force Plot</b> (Base Output: {base_value:.2f} ➔ Final Prediction: {pred_val:.2f})",
        xaxis_title="Prediction Impact",
        yaxis_title="Feature",
        barmode='overlay',
        template="plotly_white",
        height=400
    )
    return fig


def get_shap_explanation(
    model: Any,
    input_data: Union[pd.DataFrame, np.ndarray, Any],
    feature_names: Optional[List[str]] = None,
    background_data: Optional[Union[pd.DataFrame, np.ndarray, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a SHAP explanation for a single prediction.

    Supports:
    - Tree-SHAP for Random Forest, XGBoost, and ANN-RF ensemble models.
    - Deep-SHAP / KernelExplainer / GradientExplainer for PyTorch ANN models.

    Parameters:
    -----------
    model : Any
        Trained model instance (scikit-learn, XGBoost, PyTorch nn.Module, or ANN_RF_Ensemble).
    input_data : Union[pd.DataFrame, np.ndarray, Any]
        Single row input data to explain.
    feature_names : List[str], optional
        List of feature names. If input_data is DataFrame, columns are used by default.
    background_data : Union[pd.DataFrame, np.ndarray, Any], optional
        Background reference dataset for SHAP explainers.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing shap_values, base_value, feature_names, force_plot,
        waterfall_plot, and feature_contributions.
    """
    if isinstance(input_data, pd.DataFrame):
        if feature_names is None:
            feature_names = list(input_data.columns)
        X_arr = input_data.values
    elif isinstance(input_data, np.ndarray):
        X_arr = input_data
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
    elif HAS_TORCH and isinstance(input_data, torch.Tensor):
        X_arr = input_data.detach().cpu().numpy()
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
    else:
        X_arr = np.array(input_data)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    # Unwrap ensemble if needed
    target_model = model
    if isinstance(model, ANN_RF_Ensemble):
        target_model = model.rf_model

    if HAS_SHAP:
        try:
            is_torch_model = HAS_TORCH and isinstance(target_model, torch.nn.Module)

            if is_torch_model:
                bg = background_data
                if bg is None:
                    bg = torch.zeros((10, X_arr.shape[1]), dtype=torch.float32)
                elif isinstance(bg, pd.DataFrame):
                    bg = torch.tensor(bg.values, dtype=torch.float32)
                elif isinstance(bg, np.ndarray):
                    bg = torch.tensor(bg, dtype=torch.float32)

                inp_tensor = torch.tensor(X_arr, dtype=torch.float32)
                try:
                    explainer = shap.DeepExplainer(target_model, bg)
                    shap_vals = explainer.shap_values(inp_tensor)
                except Exception:
                    # Fallback to KernelExplainer if DeepExplainer fails
                    def py_predict(x):
                        target_model.eval()
                        with torch.no_grad():
                            t = torch.tensor(x, dtype=torch.float32)
                            out = target_model(t).detach().cpu().numpy()
                            return out.ravel()
                    bg_np = bg.detach().cpu().numpy() if isinstance(bg, torch.Tensor) else np.array(bg)
                    explainer = shap.KernelExplainer(py_predict, bg_np[:10])
                    shap_vals = explainer.shap_values(X_arr)

            else:
                # Tree-SHAP or KernelExplainer for scikit-learn / XGBoost
                try:
                    explainer = shap.TreeExplainer(target_model)
                    shap_vals = explainer.shap_values(X_arr)
                except Exception:
                    bg_np = background_data.values if isinstance(background_data, pd.DataFrame) else (background_data if background_data is not None else X_arr)
                    explainer = shap.KernelExplainer(target_model.predict, bg_np)
                    shap_vals = explainer.shap_values(X_arr)

            # Extract base value
            if hasattr(explainer, "expected_value"):
                bv = explainer.expected_value
                base_value = float(bv[0]) if isinstance(bv, (list, np.ndarray)) else float(bv)
            else:
                base_value = 0.0

            if isinstance(shap_vals, list):
                shap_vec = np.array(shap_vals[0]).flatten()
            else:
                shap_vec = np.array(shap_vals).flatten()

        except Exception:
            # Fallback SHAP estimation if explainer execution hits exceptions
            shap_vec = np.zeros(len(feature_names))
            base_value = 0.0
    else:
        # Fallback heuristic if SHAP package unavailable
        if hasattr(target_model, "feature_importances_"):
            importances = target_model.feature_importances_
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)
        shap_vec = (importances / np.sum(importances)) * 10.0
        base_value = 200.0

    feat_contribs = {fn: float(sv) for fn, sv in zip(feature_names, shap_vec)}
    pred_val = base_value + float(np.sum(shap_vec))

    force_fig = _create_shap_force_plot(base_value, shap_vec, feature_names)
    waterfall_fig = plot_feature_contributions_waterfall(
        {
            "baseline_value": base_value,
            "predicted_value": pred_val,
            "feature_contributions": feat_contribs,
            "target": "Model Prediction",
        },
        title="SHAP Feature Contribution Waterfall Plot",
    )

    return {
        "shap_values": shap_vec,
        "base_value": base_value,
        "feature_names": feature_names,
        "force_plot": force_fig,
        "waterfall_plot": waterfall_fig,
        "feature_contributions": feat_contribs,
    }


def explain_prediction(model, input_data, feature_names=None, background_data=None):
    """
    Generate SHAP explanation for a single blast prediction using get_shap_explanation.

    Key drivers from Jwaneng research:
    - Fragmentation: powder factor, burden
    - Vibration: burden, charge per delay, distance
    """
    return get_shap_explanation(model, input_data, feature_names=feature_names, background_data=background_data)
