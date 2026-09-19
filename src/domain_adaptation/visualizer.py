"""
Plotly Visualizer Submodule for Domain Adaptation Package.
Provides PCA/t-SNE feature distribution alignment plots, fine-tuning loss curves,
and pre- vs. post-adaptation R2 comparative bar charts.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from sklearn.decomposition import PCA


def plot_domain_feature_distribution(
    X_source: np.ndarray,
    X_target: np.ndarray,
    source_label: str = "Kimberlite (Source)",
    target_label: str = "Granite (Target)"
) -> go.Figure:
    """
    Generates a 2D PCA feature distribution plot comparing Kimberlite vs Granite domain features.
    """
    pca = PCA(n_components=2)
    X_all = np.vstack([X_source, X_target])
    pca_all = pca.fit_transform(X_all)

    n_src = len(X_source)
    pca_src = pca_all[:n_src]
    pca_tgt = pca_all[n_src:]

    fig = go.Figure()

    # Source domain scatter
    fig.add_trace(go.Scatter(
        x=pca_src[:, 0],
        y=pca_src[:, 1],
        mode="markers",
        name=source_label,
        marker=dict(color="#2962FF", size=8, opacity=0.7),
    ))

    # Target domain scatter
    fig.add_trace(go.Scatter(
        x=pca_tgt[:, 0],
        y=pca_tgt[:, 1],
        mode="markers",
        name=target_label,
        marker=dict(color="#FF6D00", size=9, symbol="diamond", opacity=0.85),
    ))

    fig.update_layout(
        title="<b>Geology Domain Feature Distribution Alignment (PCA 2D Projection)</b>",
        xaxis_title=f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)",
        yaxis_title=f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)",
        template="plotly_white",
        height=420,
    )
    return fig


def plot_adaptation_r2_comparison(
    pre_adaptation_r2: float,
    post_adaptation_r2: float,
    target_geology: str = "Granite"
) -> go.Figure:
    """
    Generates a comparative bar chart showing model R2 before and after transfer adaptation.
    """
    fig = go.Figure()

    categories = ["Pre-Adaptation (Zero-Shot)", "Post-Adaptation (Transfer Learning)"]
    r2_values = [max(0.0, pre_adaptation_r2), max(0.0, post_adaptation_r2)]
    colors = ["#D50000", "#00C853"]

    fig.add_trace(go.Bar(
        x=categories,
        y=r2_values,
        marker_color=colors,
        text=[f"R² = {val:.2f}" for val in r2_values],
        textposition="auto",
    ))

    fig.update_layout(
        title=f"<b>Target Domain ({target_geology}) Model Accuracy Improvement</b>",
        yaxis_title="Target Domain Mean R² Score",
        yaxis=dict(range=[0, 1.0]),
        template="plotly_white",
        height=380,
    )
    return fig
