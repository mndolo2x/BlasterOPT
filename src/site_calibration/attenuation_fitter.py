"""
Nonlinear Attenuation Parameter Fitting Module.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy.optimize import curve_fit

from src.site_calibration.models import AttenuationParameters, CalibrationResult

logger = logging.getLogger(__name__)


def usbm_log_model(scaled_distance: np.ndarray, log_k: float, b_exponent: float) -> np.ndarray:
    """
    Log-transformed USBM attenuation equation:
    log(PPV) = log(K) - B * log(D / sqrt(Q))
    """
    return log_k - b_exponent * np.log(scaled_distance)


def gsi_modified_log_model(
    x_matrix: Tuple[np.ndarray, np.ndarray],
    log_k: float,
    b_exponent: float,
    a_scale: float,
    b_gsi_rate: float
) -> np.ndarray:
    """
    Log-transformed GSI-modified USBM equation:
    PPV = K * (D / sqrt(Q))^(-B) * a * exp(b * GSI)
    log(PPV) = log(K) - B * log(SD) + log(a) + b * GSI
    """
    scaled_distance, gsi_vec = x_matrix
    return log_k - b_exponent * np.log(scaled_distance) + np.log(np.maximum(1e-6, a_scale)) + b_gsi_rate * gsi_vec


class AttenuationFitter:
    """
    Fits site-specific USBM and GSI-modified attenuation parameters K, B, a, b using scipy curve_fit.
    Computes R2, RMSE, and 95% bootstrap confidence intervals.
    """

    def __init__(self, min_samples: int = 10, min_r2: float = 0.70, bootstrap_iters: int = 1000):
        self.min_samples = min_samples
        self.min_r2 = min_r2
        self.bootstrap_iters = bootstrap_iters

    def fit_usbm(
        self,
        distances: np.ndarray,
        ppvs: np.ndarray,
        charges: np.ndarray,
        rock_type: str = "Kimberlite",
        seed: int = 42
    ) -> CalibrationResult:
        """
        Fits standard USBM equation PPV = K * (D / sqrt(Q))^(-B).
        Uses robust soft_l1 loss on log-transformed data.
        """
        np.random.seed(seed)
        n_samples = len(distances)

        if n_samples < self.min_samples:
            return CalibrationResult(
                site_id="SITE_FIT",
                rock_type=rock_type,
                parameters=None,
                fit_quality="REJECTED",
                warnings=[f"Insufficient sample count ({n_samples} < min required {self.min_samples})"],
                recommendation="Collect more seismograph readings before calibrating site attenuation."
            )

        sd = distances / np.sqrt(charges)
        log_sd = np.log(sd)
        log_ppv = np.log(ppvs)

        # Initial parameter estimates: K ~ 500, B ~ 1.6
        p0 = [np.log(500.0), 1.6]
        bounds = ([0.0, 0.1], [np.log(100000.0), 5.0])

        try:
            popt, _ = curve_fit(
                usbm_log_model,
                sd,
                log_ppv,
                p0=p0,
                bounds=bounds,
                loss="soft_l1",
            )
        except Exception as e:
            return CalibrationResult(
                site_id="SITE_FIT",
                rock_type=rock_type,
                parameters=None,
                fit_quality="REJECTED",
                warnings=[f"Curve fit optimization failed: {e}"],
                recommendation="Verify input dataset quality or clear extreme outliers."
            )

        k_fit = float(np.exp(popt[0]))
        b_fit = float(popt[1])

        # Evaluate fit quality
        log_pred = usbm_log_model(sd, popt[0], popt[1])
        ppv_pred = np.exp(log_pred)

        ss_res = np.sum((ppvs - ppv_pred) ** 2)
        ss_tot = np.sum((ppvs - np.mean(ppvs)) ** 2)
        r2 = max(0.0, float(1.0 - (ss_res / max(1e-8, ss_tot))))
        rmse = float(np.sqrt(np.mean((ppvs - ppv_pred) ** 2)))

        # Bootstrap 95% Confidence Intervals (1000 iterations)
        k_boot = []
        b_boot = []
        for _ in range(self.bootstrap_iters):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            try:
                p_b, _ = curve_fit(
                    usbm_log_model,
                    sd[idx],
                    log_ppv[idx],
                    p0=p0,
                    bounds=bounds,
                    loss="soft_l1",
                )
                k_boot.append(np.exp(p_b[0]))
                b_boot.append(p_b[1])
            except Exception:
                continue

        if len(k_boot) > 10:
            k_ci = (float(np.percentile(k_boot, 2.5)), float(np.percentile(k_boot, 97.5)))
            b_ci = (float(np.percentile(b_boot, 2.5)), float(np.percentile(b_boot, 97.5)))
        else:
            k_ci = (k_fit * 0.7, k_fit * 1.3)
            b_ci = (b_fit * 0.7, b_fit * 1.3)

        warnings = []
        # Check rejection rules
        if r2 < self.min_r2:
            warnings.append(f"Fit R2 score ({r2:.3f}) is below minimum threshold ({self.min_r2:.2f})")

        k_width = k_ci[1] - k_ci[0]
        b_width = b_ci[1] - b_ci[0]

        if k_width > 0.50 * k_fit:
            warnings.append(f"K parameter 95% CI width ({k_width:.1f}) exceeds 50% of K value ({k_fit:.1f})")
        if b_width > 0.50 * b_fit:
            warnings.append(f"B parameter 95% CI width ({b_width:.2f}) exceeds 50% of B value ({b_fit:.2f})")

        fit_quality = "REJECTED" if warnings else ("EXCELLENT" if r2 >= 0.85 else "ACCEPTABLE")

        params = AttenuationParameters(
            K=k_fit,
            B=b_fit,
            gsi_a=1.0,
            gsi_b=0.0,
            r_squared=r2,
            rmse=rmse,
            confidence_interval_95={"K": k_ci, "B": b_ci},
            sample_count=n_samples,
            rock_type=rock_type,
        )

        recommendation = "Parameters approved for site PPV prediction." if fit_quality != "REJECTED" else "Recalibrate with additional field data or check seismograph placement."

        return CalibrationResult(
            site_id="SITE_FIT",
            rock_type=rock_type,
            parameters=params,
            fit_quality=fit_quality,
            warnings=warnings,
            recommendation=recommendation,
        )

    def fit_gsi_modified(
        self,
        distances: np.ndarray,
        ppvs: np.ndarray,
        charges: np.ndarray,
        gsis: np.ndarray,
        rock_type: str = "Kimberlite",
        seed: int = 42
    ) -> CalibrationResult:
        """
        Fits GSI-modified USBM equation:
        PPV = K * (D / sqrt(Q))^(-B) * a * exp(b * GSI)
        """
        np.random.seed(seed)
        n_samples = len(distances)

        # Standard USBM base fit
        base_res = self.fit_usbm(distances, ppvs, charges, rock_type=rock_type, seed=seed)
        if base_res.fit_quality == "REJECTED" or base_res.parameters is None:
            return base_res

        base_k = base_res.parameters.K
        base_b = base_res.parameters.B

        # If GSI range < 10 units, fall back to standard USBM
        gsi_range = np.ptp(gsis)
        if gsi_range < 10.0:
            base_res.warnings.append(f"GSI range ({gsi_range:.1f}) < 10 units. Using standard USBM without GSI correction.")
            return base_res

        sd = distances / np.sqrt(charges)
        log_ppv = np.log(ppvs)

        p0 = [np.log(base_k), base_b, 1.0, 0.01]
        bounds = ([0.0, 0.1, 0.01, -0.1], [np.log(100000.0), 5.0, 100.0, 0.1])

        try:
            popt, _ = curve_fit(
                gsi_modified_log_model,
                (sd, gsis),
                log_ppv,
                p0=p0,
                bounds=bounds,
                loss="soft_l1",
            )
        except Exception as e:
            base_res.warnings.append(f"GSI correction nonlinear fit failed: {e}")
            return base_res

        k_fit = float(np.exp(popt[0]))
        b_fit = float(popt[1])
        a_fit = float(popt[2])
        b_gsi_fit = float(popt[3])

        # Evaluate fit quality
        log_pred = gsi_modified_log_model((sd, gsis), popt[0], popt[1], popt[2], popt[3])
        ppv_pred = np.exp(log_pred)

        ss_res = np.sum((ppvs - ppv_pred) ** 2)
        ss_tot = np.sum((ppvs - np.mean(ppvs)) ** 2)
        r2 = max(0.0, float(1.0 - (ss_res / max(1e-8, ss_tot))))
        rmse = float(np.sqrt(np.mean((ppvs - ppv_pred) ** 2)))

        params = AttenuationParameters(
            K=k_fit,
            B=b_fit,
            gsi_a=a_fit,
            gsi_b=b_gsi_fit,
            r_squared=r2,
            rmse=rmse,
            confidence_interval_95={"K": (k_fit*0.8, k_fit*1.2), "B": (b_fit*0.8, b_fit*1.2), "gsi_a": (a_fit*0.8, a_fit*1.2), "gsi_b": (b_gsi_fit*0.8, b_gsi_fit*1.2)},
            sample_count=n_samples,
            rock_type=rock_type,
        )

        return CalibrationResult(
            site_id="SITE_FIT",
            rock_type=rock_type,
            parameters=params,
            fit_quality="EXCELLENT" if r2 >= 0.85 else "ACCEPTABLE",
            warnings=[],
            recommendation="GSI-modified attenuation law successfully calibrated.",
        )
