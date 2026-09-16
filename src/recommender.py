"""
Similar Blast Recommender Module for BlastOpt Botswana.

Facilitates knowledge transfer and addresses engineering skill shortages and staff rotations
in Botswana's mining sector by retrieving historical blast designs with similar parameters
using Nearest Neighbors matching.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional
from sklearn.neighbors import NearestNeighbors
from src.models import FEATURE_COLS


def find_similar_blasts(
    new_blast_params: Union[Dict[str, float], pd.DataFrame],
    historical_blasts: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Finds the top_k most similar historical blasts based on input parameters using Nearest Neighbors.

    Knowledge Transfer & Domain Context:
    -----------------------------------
    Botswana's diamond mining industry faces periodic engineering skill shortages and high staff
    rotation between mine sites (e.g., Jwaneng, Orapa, Letlhakane, Karowe). This recommender system
    allows junior blasters and newly rotated mining engineers to query historical blast logs,
    examine actual achieved outcomes (fragmentation d50, PPV vibration, flyrock distance, cost),
    and review historical 'lessons learned' before finalizing new blast designs.

    Parameters:
    -----------
    new_blast_params : Union[Dict[str, float], pd.DataFrame]
        Input blast design parameters to match against history.
    historical_blasts : pd.DataFrame
        Historical dataset of past blast logs containing design parameters and outcome targets.
    top_k : int, default=5
        Number of most similar historical blasts to retrieve.

    Returns:
    --------
    pd.DataFrame
        DataFrame of top_k matched historical blasts with Euclidean distances and outcome metrics.
    """
    if historical_blasts is None or len(historical_blasts) == 0:
        return pd.DataFrame()

    if isinstance(new_blast_params, dict):
        df_new = pd.DataFrame([new_blast_params])
    else:
        df_new = new_blast_params.copy()

    # Determine numeric feature columns common to both new input and historical records
    num_cols = [c for c in FEATURE_COLS if c in historical_blasts.columns and c in df_new.columns]
    if not num_cols:
        num_cols = [
            c for c in historical_blasts.select_dtypes(include=[np.number]).columns
            if c in df_new.columns
        ]

    if not num_cols:
        # Fallback to default numerical columns in historical blasts
        num_cols = list(historical_blasts.select_dtypes(include=[np.number]).columns)
        for col in num_cols:
            if col not in df_new.columns:
                df_new[col] = historical_blasts[col].median()

    # Prepare normalized numeric matrices for Nearest Neighbors fit
    X_hist = historical_blasts[num_cols].fillna(historical_blasts[num_cols].median())
    X_new = df_new[num_cols].fillna(X_hist.median())

    # Fit NearestNeighbors model
    k = min(top_k, len(historical_blasts))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(X_hist)

    distances, indices = nn.kneighbors(X_new.iloc[[0]])

    similar_df = historical_blasts.iloc[indices[0]].copy()
    similar_df["similarity_distance"] = np.round(distances[0], 3)

    return similar_df
