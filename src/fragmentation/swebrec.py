"""
Swebrec fragmentation distribution model (Ouchterlony 2005, 2009).
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional
from scipy.optimize import curve_fit


class SwebrecModel:
    """
    Swebrec fragmentation distribution (Ouchterlony 2005, 2009).

    The cumulative form is:
        P(x) = 100 / (1 + [ln(x_max / x) / ln(x_max / x_50)]^b)

    Reference: Ouchterlony, F. (2005). "The Swebrec function: linking
    fragmentation by blasting and crushing." Mining Technology, 114(1), 29-46.
    """

    def __init__(self, x_max: float, x_50: float, b: float = 1.0) -> None:
        if x_max <= x_50:
            raise ValueError(f"x_max ({x_max}) must be strictly greater than x_50 ({x_50})")
        if x_50 <= 0:
            raise ValueError(f"x_50 ({x_50}) must be positive")
        if b <= 0:
            raise ValueError(f"b ({b}) must be positive")

        self.x_max = x_max
        self.x_50 = x_50
        self.b = b

    def percent_passing(self, x: float) -> float:
        """Returns P(x) for a given fragment size x."""
        if x <= 0:
            return 0.0
        if x >= self.x_max:
            return 100.0

        num_base = math.log(self.x_max / x)
        denom_base = math.log(self.x_max / self.x_50)

        # Formula: P(x) = 100 / (1 + [ln(x_max / x) / ln(x_max / x_50)]^b)
        p_val = 100.0 / (1.0 + (num_base / denom_base) ** self.b)
        return float(round(p_val, 2))

    def compute_curve(self, x_min: float = 1.0, x_max: Optional[float] = None, n_points: int = 200) -> pd.DataFrame:
        """Returns a DataFrame with columns 'size_mm' and 'percent_passing'."""
        max_limit = x_max if x_max is not None else self.x_max
        sizes = np.logspace(np.log10(max(0.1, x_min)), np.log10(max_limit), n_points)
        passings = [self.percent_passing(s) for s in sizes]

        return pd.DataFrame({"size_mm": sizes, "percent_passing": passings})

    def fit_from_data(self, sizes: np.ndarray, passing: np.ndarray) -> Dict[str, float]:
        """
        Fits Swebrec parameters (x_max, x_50, b) to measured data.
        Uses scipy.optimize.curve_fit.
        Returns {'x_max': float, 'x_50': float, 'b': float, 'r_squared': float}.
        """
        if len(sizes) == 0 or len(passing) == 0:
            raise ValueError("sizes and passing arrays cannot be empty")

        def swebrec_func(x, x_max_fit, x_50_fit, b_fit):
            num = np.log(np.maximum(x_max_fit, x + 1e-5) / np.maximum(x, 1e-5))
            denom = np.log(np.maximum(x_max_fit, x_50_fit + 1e-5) / np.maximum(x_50_fit, 1e-5))
            return 100.0 / (1.0 + (num / np.maximum(denom, 1e-5)) ** b_fit)

        p0 = [max(sizes) * 1.1, np.median(sizes), 1.0]
        bounds = ([max(sizes), 0.1, 0.1], [max(sizes) * 10.0, max(sizes), 5.0])

        popt, _ = curve_fit(swebrec_func, sizes, passing, p0=p0, bounds=bounds, maxfev=5000)

        x_max_fit, x_50_fit, b_fit = popt

        residuals = passing - swebrec_func(sizes, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((passing - np.mean(passing))**2)
        r_squared = 1.0 - (ss_res / max(1e-5, ss_tot))

        self.x_max = float(x_max_fit)
        self.x_50 = float(x_50_fit)
        self.b = float(b_fit)

        return {
            "x_max": float(round(x_max_fit, 1)),
            "x_50": float(round(x_50_fit, 1)),
            "b": float(round(b_fit, 3)),
            "r_squared": float(round(r_squared, 4)),
        }

    def calculate_b_from_uniformity(self, n_uniformity: float) -> float:
        """
        Estimates b from the Kuz-Ram uniformity index n.

        Relation (Ouchterlony 2005):
            b ≈ 0.5 * n  (approximate)

        Reference: Ouchterlony, F. (2005), Eq. 15.
        """
        if n_uniformity <= 0:
            raise ValueError(f"n_uniformity ({n_uniformity}) must be positive")

        # Formula: b ≈ 0.5 * n
        b_est = 0.5 * n_uniformity
        return float(round(b_est, 3))

    def calculate_x_max_from_burden(self, burden_m: float, spacing_m: float) -> float:
        """
        Estimates x_max from blast geometry.

        Relation (Ouchterlony 2005):
            x_max = min(burden_m, spacing_m) in mm
        """
        if burden_m <= 0 or spacing_m <= 0:
            raise ValueError("burden_m and spacing_m must be positive")

        # Formula: x_max = min(burden_m, spacing_m) in meters -> mm
        x_max_m = min(burden_m, spacing_m)
        x_max_mm = x_max_m * 1000.0
        return float(round(x_max_mm, 1))


def swebrec_cumulative_passing(
    x_size_mm: Union[float, np.ndarray],
    x_max_mm: float,
    x_50_mm: float,
    b_curve_factor: float = 1.0,
) -> Union[float, np.ndarray]:
    """Legacy wrapper for SwebrecModel."""
    model = SwebrecModel(x_max=x_max_mm, x_50=x_50_mm, b=b_curve_factor)
    if isinstance(x_size_mm, (int, float)):
        return model.percent_passing(float(x_size_mm))

    return np.array([model.percent_passing(float(x)) for x in x_size_mm])
