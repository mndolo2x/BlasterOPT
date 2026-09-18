"""
Plotly Visualizer Module for Ensemble Uncertainty Quantification & OOD Distribution.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from src.ensemble_uq.models import EnsemblePrediction


def plot_prediction_intervals(
    actuals: np.ndarray,
    predictions: List[EnsemblePrediction],
    target_name: str = "d50_mm"
) -> go.Figure:
    """
    Plots predicted vs. actual values with 95% error bars.
    """
    means = [p.mean.get(target_name, 0.0) for p in predictions]
    lowers = [p.lower_95.get(target_name, 0.0) for p in predictions]
    uppers = [p.upper_95.get(target_name, 0.0) for p in predictions]
    err_upper = [u - m for u, m in zip(uppers, means)]
    err_lower = [m - l for l, m in zip(lowers, means)]

    indices = list(range(1, len(actuals) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices,
        y=actuals,
        mode="markers",
        name="Actual Ground Truth",
        marker=dict(size=9, color="#00C853", symbol="circle"),
    ))

    fig.add_trace(go.Scatter(
        x=indices,
        y=means,
        mode="markers",
        name="Ensemble Predicted Mean",
        marker=dict(size=9, color="#2962FF", symbol="diamond"),
        error_y=dict(type="data", symmetric=False, array=err_upper, arrayminus=err_lower, color="#2962FF"),
    ))

    fig.update_layout(
        title=f"<b>Ensemble Prediction 95% Confidence Intervals [{target_name}]</b>",
        xaxis_title="Sample Index",
        yaxis_title=f"Value [{target_name}]",
        template="plotly_white",
        height=420,
    )
    return fig


def plot_uncertainty_decomposition(pred: EnsemblePrediction) -> go.Figure:
    """
    Plots stacked bar chart decomposing aleatoric (data noise) vs epistemic (model gap) variance.
    """
    targets = list(pred.mean.keys())
    aleatoric_vals = [pred.aleatoric.get(t, 0.0) for t in targets]
    epistemic_vals = [pred.epistemic.get(t, 0.0) for t in targets]

    fig = go.Figure(data=[
        go.Bar(name="Aleatoric Variance (Data Noise)", x=targets, y=aleatoric_vals, marker_color="#FF6D00"),
        go.Bar(name="Epistemic Variance (Model Knowledge Gap)", x=targets, y=epistemic_vals, marker_color="#D50000"),
    ])

    fig.update_layout(
        barmode="stack",
        title="<b>Uncertainty Decomposition: Aleatoric vs Epistemic Variance</b>",
        xaxis_title="Target Metric",
        yaxis_title="Variance Magnitude",
        template="plotly_white",
        height=380,
    )
    return fig


def plot_ood_distribution(
    in_dist_scores: np.ndarray,
    ood_scores: np.ndarray,
    test_score: float
) -> go.Figure:
    """
    Plots histogram distribution comparing In-Distribution vs Out-Of-Distribution Mahalanobis scores.
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=in_dist_scores,
        name="In-Distribution Training Data",
        opacity=0.6,
        marker_color="#2962FF",
    ))

    fig.add_trace(go.Histogram(
        x=ood_scores,
        name="Out-Of-Distribution (OOD) Data",
        opacity=0.6,
        marker_color="#D50000",
    ))

    fig.add_vline(
        x=test_score,
        line_dash="dash",
        line_color="black",
        line_width=3,
        annotation_text=f"Current Sample Distance ({test_score:.2f})",
        annotation_position="top right",
    )

    fig.update_layout(
        barmode="overlay",
        title="<b>Mahalanobis Distance Distribution: In-Distribution vs OOD</b>",
        xaxis_title="Mahalanobis Distance",
        yaxis_title="Sample Count",
        template="plotly_white",
        height=380,
    )
    return fig


def plot_confidence_calibration(
    actual_coverage: float = 0.95,
    target_coverage: float = 0.95
) -> go.Figure:
    """
    Plots calibration comparison of target vs actual 95% CI coverage.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Target Coverage Goal", "Actual Observed Test Coverage"],
        y=[target_coverage * 100.0, actual_coverage * 100.0],
        marker_color=["#2962FF", "#00C853" if 0.93 <= actual_coverage <= 0.97 else "#FFD600"],
        text=[f"{target_coverage*100:.1f}%", f"{actual_coverage*100:.1f}%"],
        textposition="auto",
    ))

    fig.update_layout(
        title="<b>95% Confidence Interval Test Coverage Calibration</b>",
        yaxis=dict(title="Coverage Percentage (%)", range=[0, 100]),
        template="plotly_white",
        height=320,
    )
    return fig
