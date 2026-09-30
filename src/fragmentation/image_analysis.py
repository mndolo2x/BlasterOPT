"""
WipFrag / Split-Desktop image analysis integration and distribution fitting for fragmentation sizing.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.fragmentation.swebrec import SwebrecModel
from src.fragmentation.distributions import calculate_goodness_of_fit, rosin_rammler_curve, lognormal_curve


class ImageAnalysisImporter:
    """
    Imports fragmentation results from WipFrag or Split-Desktop.

    Both tools export CSV files with columns:
        'size_mm', 'percent_passing', 'percent_retained'

    Reference: Sanchidrian, J.A. et al. (2009). "On the accuracy of fragment
    size measurement by image analysis." Rock Mechanics and Rock Engineering,
    42, 95-110.
    """

    def import_wipfrag_csv(self, filepath: str) -> pd.DataFrame:
        """
        Parses a WipFrag CSV export.
        Returns a DataFrame with 'size_mm' and 'percent_passing'.
        """
        if not os.path.exists(filepath):
            raise ValueError(f"File not found at filepath '{filepath}'")

        df = pd.read_csv(filepath)
        req_cols = ["size_mm", "percent_passing"]
        for col in req_cols:
            if col not in df.columns:
                raise ValueError(f"WipFrag CSV missing required column '{col}'")

        return df[["size_mm", "percent_passing"]].sort_values("size_mm").reset_index(drop=True)

    def import_split_desktop_csv(self, filepath: str) -> pd.DataFrame:
        """
        Parses a Split-Desktop CSV export.
        Returns a DataFrame with 'size_mm' and 'percent_passing'.
        """
        if not os.path.exists(filepath):
            raise ValueError(f"File not found at filepath '{filepath}'")

        df = pd.read_csv(filepath)
        req_cols = ["size_mm", "percent_passing"]
        for col in req_cols:
            if col not in df.columns:
                raise ValueError(f"Split-Desktop CSV missing required column '{col}'")

        return df[["size_mm", "percent_passing"]].sort_values("size_mm").reset_index(drop=True)

    def calculate_d80_from_image(self, measured_df: pd.DataFrame) -> float:
        """Interpolates the D80 from the measured distribution."""
        if measured_df.empty:
            raise ValueError("measured_df DataFrame cannot be empty")

        sizes = measured_df["size_mm"].values
        passing = measured_df["percent_passing"].values

        # Linear interpolation for size at 80% passing
        d80 = float(np.interp(80.0, passing, sizes))
        return float(round(d80, 1))

    def fit_distributions_to_measured(self, measured_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fits Rosin-Rammler, Swebrec, and log-normal to measured data.

        Returns:
            {
                "rosin_rammler": {"params": {...}, "r_squared": float},
                "swebrec": {"params": {...}, "r_squared": float},
                "lognormal": {"params": {...}, "r_squared": float},
                "best_fit": str,  # which distribution fit best
            }
        """
        if measured_df.empty:
            raise ValueError("measured_df DataFrame cannot be empty")

        sizes = measured_df["size_mm"].values
        passing = measured_df["percent_passing"].values

        d50 = float(np.interp(50.0, passing, sizes))
        x_max = float(np.max(sizes))

        # 1. Fit Swebrec
        swe_mod = SwebrecModel(x_max=max(x_max, d50 * 1.5), x_50=d50, b=1.0)
        fit_sw = swe_mod.fit_from_data(sizes, passing)
        r2_sw = fit_sw["r_squared"]

        # 2. Fit Rosin-Rammler (via curve comparison)
        rr_pred = rosin_rammler_curve(x_50=d50, n=1.25, x_min=min(sizes), x_max=max(sizes), n_points=len(sizes))
        fit_rr = calculate_goodness_of_fit(measured_df, rr_pred)
        r2_rr = fit_rr["r_squared"]

        # 3. Fit Log-Normal
        ln_pred = lognormal_curve(x_50=d50, sigma=0.6, x_min=min(sizes), x_max=max(sizes), n_points=len(sizes))
        fit_ln = calculate_goodness_of_fit(measured_df, ln_pred)
        r2_ln = fit_ln["r_squared"]

        r2_dict = {
            "swebrec": r2_sw,
            "rosin_rammler": r2_rr,
            "lognormal": r2_ln,
        }
        best_fit = max(r2_dict, key=r2_dict.get)

        return {
            "rosin_rammler": {"params": {"x_50": d50, "n": 1.25}, "r_squared": r2_rr},
            "swebrec": {"params": fit_sw, "r_squared": r2_sw},
            "lognormal": {"params": {"x_50": d50, "sigma": 0.6}, "r_squared": r2_ln},
            "best_fit": best_fit,
        }


def parse_wipfrag_data(image_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy helper wrapper for image metrics."""
    scale = image_metrics.get("scale_px_per_cm", 10.0)
    areas = np.array(image_metrics.get("particle_areas_px2", [100, 400, 900, 1600, 2500]))

    diameters_cm = np.sqrt(4.0 * areas / np.pi) / scale
    diameters_mm = diameters_cm * 10.0
    sorted_diams = np.sort(diameters_mm)
    weights = sorted_diams ** 3.0
    cum_pct = (np.cumsum(weights) / np.sum(weights)) * 100.0

    df_dist = pd.DataFrame({"size_mm": sorted_diams, "percent_passing": cum_pct})
    importer = ImageAnalysisImporter()
    d80 = importer.calculate_d80_from_image(df_dist)

    return {
        "d10_mm": round(float(np.percentile(sorted_diams, 10)), 1),
        "d50_mm": round(float(np.percentile(sorted_diams, 50)), 1),
        "d80_mm": d80,
        "d95_mm": round(float(np.percentile(sorted_diams, 95)), 1),
        "size_distribution_df": df_dist,
    }
