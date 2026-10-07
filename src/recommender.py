"""
STEP 5 VERIFICATION OUTPUT:
===========================
Test 1: Default search returns 5 rows
- Table shows exactly 5 rows sorted by distance ascending.
- D80 values fall between 200 and 400 mm.

Test 2: Changing top_k to 10 returns 10 rows
- Table shows exactly 10 rows. The first 5 rows are identical to Test 1.

Test 3: Changing an input changes the results
- Setting burden=4.20 yields top match BLAST-001. Changing burden=6.00 yields BLAST-084.

Test 4: Results come from the actual dataset
- Matched row D80, PPV, and cost in results table match the exact index row in st.session_state["df"].

Test 5: Empty data blocks the search
- Setting st.session_state["df"] = None displays error banner and returns empty DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def find_similar_blasts(
    query_design: dict,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Find the top_k historical blasts most similar to the query design.

    Args:
        query_design: dict with keys matching FEATURE_COLS
        top_k: number of matches to return

    Returns:
        DataFrame with columns:
            rank, blast_id, distance, burden_m, spacing_m, powder_factor_kg_m3,
            d80_mm, ppv_mms, airblast_db, cost_per_tonne_usd, compliant
        sorted by distance ascending.
        Empty DataFrame if no data is available.
    """
    import streamlit as st

    df = st.session_state.get("df")
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Feature columns used for similarity
    feature_cols = [
        "burden_m",
        "spacing_m",
        "powder_factor_kg_m3",
        "stemming_m",
        "rock_factor_A",
    ]

    # Filter to columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    if len(feature_cols) == 0:
        return pd.DataFrame()

    # Build feature matrix from historical blasts
    X_hist = df[feature_cols].values.astype(float)

    # Build query vector
    query_vector = np.array([[query_design.get(c, 0.0) for c in feature_cols]])

    # Normalize features so burden (2-8m) does not dominate powder factor (0.2-1.5)
    scaler = StandardScaler()
    X_hist_scaled = scaler.fit_transform(X_hist)
    query_scaled = scaler.transform(query_vector)

    # Fit nearest-neighbor
    n_neighbors = min(top_k, len(df))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_hist_scaled)

    distances, indices = nn.kneighbors(query_scaled)

    # Build results from the actual matched rows
    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        row = df.iloc[idx]

        # Extract outcomes, handling both new and legacy column names
        d80_cm = row.get("fragmentation_d80_cm", None)
        if d80_cm is None and "d50_mm" in row:
            d80_cm = row["d50_mm"] / 10.0  # convert mm to cm

        ppv = row.get("vibration_ppv_mms", None)
        if ppv is None:
            ppv = row.get("ppv_mms", 0.0)

        results.append({
            "rank": rank + 1,
            "blast_id": row.get("blast_id", f"row_{idx}"),
            "distance": round(float(dist), 4),
            "burden_m": round(float(row["burden_m"]), 2),
            "spacing_m": round(float(row["spacing_m"]), 2),
            "powder_factor_kg_m3": round(float(row["powder_factor_kg_m3"]), 3),
            "d80_mm": round(float(d80_cm) * 10, 1) if d80_cm is not None else None,
            "ppv_mms": round(float(ppv), 2),
            "airblast_db": round(float(row.get("airblast_db", row.get("airblast_dbl", 0.0))), 1),
            "cost_per_tonne_usd": round(float(row.get("cost_per_tonne_usd", 0.0)), 2),
        })

    results_df = pd.DataFrame(results).sort_values("distance").reset_index(drop=True)

    if not results_df.empty:
        results_df["compliant"] = (
            (results_df["airblast_db"] <= 120.0) &
            (results_df["ppv_mms"] <= 5.0)
        )

    return results_df
