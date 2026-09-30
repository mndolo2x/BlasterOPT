"""
KCO (Kuznetsov-Cunningham-Ouchterlony) fragmentation model (Ouchterlony 2005).
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional
from src.fragmentation.kuz_ram import predict_kuz_ram
from src.fragmentation.swebrec import SwebrecModel


class KCOModel:
    """
    Kuznetsov-Cunningham-Ouchterlony (KCO) fragmentation model.

    Reference: Ouchterlony, F. (2005). "The Swebrec function: linking
    fragmentation by blasting and crushing." Mining Technology, 114(1), 29-46.

    The KCO model:
        1. Calculates x_50 using the Kuznetsov equation (same as Kuz-Ram)
        2. Calculates n using the Cunningham uniformity index
        3. Derives b from n
        4. Calculates x_max from blast geometry
        5. Builds the Swebrec distribution
    """

    def __init__(self, kuz_ram_model=None, swebrec_model=None) -> None:
        self.kuz_ram_model = kuz_ram_model
        self.swebrec_model = swebrec_model

    def predict(self, blast_params: dict) -> Dict[str, Any]:
        """
        Predicts the full fragment size distribution using KCO.

        Input blast_params must include:
            burden_m, spacing_m, bench_height_m, hole_diameter_mm,
            powder_factor_kg_m3, explosive_rws, rock_factor_a,
            blastability_index, in_situ_block_size_m

        Returns:
            {
                "x_50_cm": float,
                "x_max_cm": float,
                "b": float,
                "n_uniformity": float,
                "distribution": pd.DataFrame,
                "model": "KCO",
            }
        """
        required = [
            "burden_m",
            "spacing_m",
            "bench_height_m",
            "hole_diameter_mm",
            "powder_factor_kg_m3",
            "explosive_rws",
            "rock_factor_a",
        ]
        for req in required:
            if req not in blast_params:
                raise ValueError(f"Missing required parameter '{req}' in blast_params")

        burden_m = float(blast_params["burden_m"])
        spacing_m = float(blast_params["spacing_m"])
        bench_h_m = float(blast_params["bench_height_m"])
        pf_kg_m3 = float(blast_params["powder_factor_kg_m3"])
        rws = float(blast_params["explosive_rws"])
        rock_a = float(blast_params["rock_factor_a"])

        if burden_m <= 0 or spacing_m <= 0 or bench_h_m <= 0 or pf_kg_m3 <= 0 or rws <= 0:
            raise ValueError("All physical inputs must be strictly positive")

        # 1. Calculate x_50 using Kuznetsov equation
        vol_per_hole = burden_m * spacing_m * bench_h_m
        q_mass_kg = vol_per_hole * pf_kg_m3

        kr_res = predict_kuz_ram(
            powder_factor_kg_m3=pf_kg_m3,
            charge_mass_per_hole_kg=q_mass_kg,
            rock_factor_a=rock_a,
            explosive_relative_weight_strength=rws,
        )

        x_50_cm = kr_res["d50_cm"]
        x_50_mm = kr_res["d50_mm"]

        # 2. Cunningham uniformity index n
        n_uniformity = kr_res["uniformity_index_n"]

        # 3. Derive b from n (b ≈ 0.5 * n)
        swe_temp = SwebrecModel(x_max=2000.0, x_50=x_50_mm, b=1.0)
        b = swe_temp.calculate_b_from_uniformity(n_uniformity)

        # 4. Calculate x_max from blast geometry (min(burden, spacing))
        x_max_mm = swe_temp.calculate_x_max_from_burden(burden_m, spacing_m)
        if x_max_mm <= x_50_mm:
            x_max_mm = x_50_mm * 2.5

        x_max_cm = x_max_mm / 10.0

        # 5. Build Swebrec distribution
        swe_final = SwebrecModel(x_max=x_max_mm, x_50=x_50_mm, b=b)
        dist_df = swe_final.compute_curve(x_min=1.0, x_max=x_max_mm, n_points=100)

        return {
            "x_50_cm": round(x_50_cm, 2),
            "x_max_cm": round(x_max_cm, 2),
            "b": round(b, 3),
            "n_uniformity": round(n_uniformity, 2),
            "distribution": dist_df,
            "model": "KCO",
        }

    def compare_with_kuz_ram(self, blast_params: dict) -> pd.DataFrame:
        """
        Returns a DataFrame comparing the Kuz-Ram (Rosin-Rammler) and
        KCO (Swebrec) predictions at the same x_50.
        Columns: 'size_mm', 'kuz_ram_passing', 'kco_passing'.
        """
        kco_res = self.predict(blast_params)
        dist_kco = kco_res["distribution"]

        sizes_mm = dist_kco["size_mm"].values
        kco_passing = dist_kco["percent_passing"].values

        x_50_mm = kco_res["x_50_cm"] * 10.0
        n_uniformity = kco_res["n_uniformity"]

        # Rosin-Rammler
        x_c = x_50_mm / ((math.log(2.0)) ** (1.0 / n_uniformity))
        kr_passing = 100.0 * (1.0 - np.exp(-((sizes_mm / x_c) ** n_uniformity)))

        return pd.DataFrame({
            "size_mm": sizes_mm,
            "kuz_ram_passing": np.clip(kr_passing, 0, 100),
            "kco_passing": np.clip(kco_passing, 0, 100),
        })


def predict_kco(
    powder_factor_kg_m3: float,
    charge_mass_per_hole_kg: float,
    rock_factor_a: float = 7.0,
    explosive_relative_weight_strength: float = 100.0,
    x_max_mm: float = 1000.0,
    b_curve_factor: float = 1.25,
) -> Dict[str, Any]:
    """Legacy helper wrapper for KCOModel."""
    params = {
        "burden_m": 4.0,
        "spacing_m": 5.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "powder_factor_kg_m3": powder_factor_kg_m3,
        "explosive_rws": explosive_relative_weight_strength,
        "rock_factor_a": rock_factor_a,
    }
    kco = KCOModel()
    res = kco.predict(params)
    dist = res["distribution"]

    idx_80 = np.argmin(np.abs(dist["percent_passing"].values - 80.0))
    d80_mm = float(dist["size_mm"].values[idx_80])

    swe_mod = SwebrecModel(x_max=res["x_max_cm"] * 10.0, x_50=res["x_50_cm"] * 10.0, b=res["b"])

    fines = float(swe_mod.percent_passing(10.0))
    oversize = float(100.0 - swe_mod.percent_passing(300.0))

    return {
        "d50_mm": res["x_50_cm"] * 10.0,
        "d80_mm": round(d80_mm, 1),
        "fines_pct_minus_10mm": round(fines, 2),
        "oversize_pct_plus_300mm": round(max(0.0, oversize), 2),
        "distribution_curve": {
            "sizes_mm": dist["size_mm"].tolist(),
            "passing_pct": dist["percent_passing"].tolist(),
        },
    }
