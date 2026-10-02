"""
Synthetic Blast Data Generator for BlastOpt Botswana.

Generates realistic open-pit mining blast datasets based on domain physics models:
- Kuz-Ram Fragmentation Model (Mean fragment size d50, uniformity index n)
- USBM Scale Distance Ground Vibration Model (PPV mm/s)
- Scaled Charge Empirical Flyrock Model (Flyrock distance meters)
- Mining Operational Cost Model (USD per ton)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def generate_synthetic_data(
    n_samples: int = 100,
    seed: Optional[int] = 42,
    rock_factor_range: Tuple[float, float] = (6.0, 12.0),
    bench_height_range: Tuple[float, float] = (10.0, 15.0),
    hole_diameter_range: Tuple[float, float] = (150.0, 311.0), # mm
) -> pd.DataFrame:
    """
    Wrapper alias for generating synthetic blast datasets with n_samples parameter.
    """
    return generate_synthetic_blast_data(
        num_samples=n_samples,
        seed=seed,
        rock_factor_range=rock_factor_range,
        bench_height_range=bench_height_range,
        hole_diameter_range=hole_diameter_range,
    )


def generate_synthetic_blast_data(
    num_samples: int = 500,
    seed: Optional[int] = 42,
    rock_factor_range: Tuple[float, float] = (6.0, 12.0),
    bench_height_range: Tuple[float, float] = (10.0, 15.0),
    hole_diameter_range: Tuple[float, float] = (150.0, 311.0), # mm
) -> pd.DataFrame:
    """
    Generates realistic synthetic blast logs for Botswana mining operations.

    Parameters:
    -----------
    num_samples : int
        Number of blast events to simulate.
    seed : int, optional
        Random seed for reproducibility.
    rock_factor_range : tuple
        Min and max blastability rock factor A (6 = soft, 12 = hard rock like kimberlite/basalt).
    bench_height_range : tuple
        Bench height in meters.
    hole_diameter_range : tuple
        Hole diameter in millimeters.

    Returns:
    --------
    pd.DataFrame: Synthetic blasting dataset with parameters, outputs, and metadata.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Mine Site & Pit Locations in Botswana
    mine_sites = ["Jwaneng Mine", "Orapa Mine", "Karowe Mine", "Letlhakane Mine", "Damtshaa Mine"]
    rock_types = ["Kimberlite", "Basalt", "Granite", "Sandstone", "Shale"]
    explosive_types = ["ANFO", "Heavy ANFO (70/30)", "Emulsion (Matrix)", "Slurry"]

    # Randomly assign mine and rock types
    sites = np.random.choice(mine_sites, size=num_samples)
    rocks = np.random.choice(rock_types, size=num_samples)
    explosives = np.random.choice(explosive_types, size=num_samples)

    # Density based on rock type (t/m3)
    rock_density_map = {
        "Kimberlite": 2.65,
        "Basalt": 2.85,
        "Granite": 2.70,
        "Sandstone": 2.40,
        "Shale": 2.30
    }
    rock_density = np.array([rock_density_map[r] + np.random.normal(0, 0.05) for r in rocks])
    rock_density = np.clip(rock_density, 2.0, 3.2)

    # Rock blastability factor A (Kuz-Ram A factor)
    rock_factor = np.random.uniform(rock_factor_range[0], rock_factor_range[1], size=num_samples)

    # 2. Blast Geometry Input Variables
    bench_height = np.random.uniform(bench_height_range[0], bench_height_range[1], size=num_samples) # meters
    hole_diameter = np.random.uniform(hole_diameter_range[0], hole_diameter_range[1], size=num_samples) # mm

    # Burden (m): typically 20 - 25 x hole diameter in meters
    burden = (hole_diameter / 1000.0) * np.random.uniform(20.0, 25.0, size=num_samples)
    # Spacing (m): typically 1.1 - 1.4 x Burden
    spacing = burden * np.random.uniform(1.1, 1.35, size=num_samples)
    # Stemming length (m): typically 0.85 - 1.05 x Burden
    stemming = burden * np.random.uniform(0.85, 1.05, size=num_samples)
    # Subdrilling (m): typically 0.15 - 0.22 x Burden
    subdrilling = burden * np.random.uniform(0.15, 0.22, size=num_samples)

    # Total hole depth (m)
    hole_depth = bench_height + subdrilling
    # Charge length per hole (m)
    charge_length = np.maximum(0.5, hole_depth - stemming)

    # Explosive relative weight strength (RWS) relative to ANFO (ANFO = 100)
    rws_map = {
        "ANFO": 100.0,
        "Heavy ANFO (70/30)": 115.0,
        "Emulsion (Matrix)": 125.0,
        "Slurry": 105.0
    }
    rws = np.array([rws_map[e] + np.random.normal(0, 2.0) for e in explosives])

    # Explosive density (g/cm3 or kg/l)
    exp_density_map = {
        "ANFO": 0.82,
        "Heavy ANFO (70/30)": 1.15,
        "Emulsion (Matrix)": 1.25,
        "Slurry": 1.10
    }
    exp_density = np.array([exp_density_map[e] + np.random.normal(0, 0.02) for e in explosives])

    # Hole volume & explosive mass per hole (kg)
    hole_radius_m = (hole_diameter / 1000.0) / 2.0
    hole_cross_area = np.pi * (hole_radius_m ** 2)
    charge_mass_per_hole = hole_cross_area * charge_length * (exp_density * 1000.0) # kg

    # Volume & tonnage broken per hole
    rock_volume_per_hole = burden * spacing * bench_height # m3
    rock_mass_per_hole = rock_volume_per_hole * rock_density # tonnes

    # Powder factor (kg / m3 and kg / tonne)
    powder_factor_kg_m3 = charge_mass_per_hole / rock_volume_per_hole
    powder_factor_kg_t = charge_mass_per_hole / rock_mass_per_hole

    # Maximum Charge Weight Per Delay (Q in kg) - assuming 1 to 3 holes per delay
    holes_per_delay = np.random.randint(1, 4, size=num_samples)
    max_charge_per_delay = charge_mass_per_hole * holes_per_delay

    # Distance to nearest critical structure / monitor point (m)
    monitoring_distance = np.random.uniform(200.0, 1200.0, size=num_samples)

    # 3. Physics Models for Targets

    # --- 3. Physics Models for Targets (aligned with src/physics_core.py) ---
    from src.physics_core import (
        kuznetsov_x50, cunningham_uniformity, rosin_rammler_d80,
        usbm_ppv, siskind_airblast, lundborg_flyrock, total_cost_per_tonne
    )

    x50_list, n_list, d80_list, ppv_list, ab_list, fly_list, cost_list = [], [], [], [], [], [], []

    for i in range(num_samples):
        b = burden[i]
        s = spacing[i]
        rf = rock_factor[i]
        d_mm = hole_diameter[i]
        h = hole_depth[i]
        bh = bench_height[i]
        stem = stemming[i]
        q_delay = max_charge_per_delay[i]
        q_hole = charge_mass_per_hole[i]
        dist = monitoring_distance[i]
        rws_val = rws[i]

        x50_cm = kuznetsov_x50(rf, b, s, h, q_hole, rws_val)
        n_val = cunningham_uniformity(b, s, d_mm, bh, max(h - stem, 1.0))
        d80_val = rosin_rammler_d80(x50_cm, n_val)
        ppv_val = usbm_ppv(q_delay, dist)
        ab_val = siskind_airblast(q_delay, dist)
        fly_val = lundborg_flyrock(q_hole, stem, b)
        cost_val = total_cost_per_tonne(15.0, h, 50, 1.30, q_hole, 5000.0, 50000.0)

        x50_list.append(x50_cm)
        n_list.append(n_val)
        d80_list.append(d80_val)
        ppv_list.append(ppv_val)
        ab_list.append(ab_val)
        fly_list.append(fly_val)
        cost_list.append(cost_val)

    noise_d80 = np.random.normal(1.0, 0.012, num_samples)
    noise_ppv = np.random.normal(1.0, 0.08, num_samples)
    noise_ab = np.random.normal(1.0, 0.012, num_samples)

    d80_cm = np.clip(np.array(d80_list) * noise_d80, 5.0, 150.0)
    d50_mm = np.clip(np.array(x50_list) * 10.0 * noise_d80, 20.0, 800.0)
    n_uniformity = np.clip(np.array(n_list), 0.7, 2.2)
    ppv_mms = np.clip(np.array(ppv_list) * noise_ppv, 0.1, 50.0)
    airblast_db = np.clip(np.array(ab_list) * noise_ab, 40.0, 140.0)
    flyrock_dist_m = np.clip(np.array(fly_list) * noise_ppv, 5.0, 500.0)
    cost_per_tonne = np.clip(np.array(cost_list) * noise_ppv, 0.30, 3.00)

    # Assemble DataFrame
    data = pd.DataFrame({
        "blast_id": [f"BLAST-{1000+i}" for i in range(num_samples)],
        "mine_site": sites,
        "rock_type": rocks,
        "rock_density_t_m3": np.round(rock_density, 2),
        "rock_factor_A": np.round(rock_factor, 2),
        "explosive_type": explosives,
        "explosive_rws": np.round(rws, 1),
        "explosive_density_g_cm3": np.round(exp_density, 2),
        "bench_height_m": np.round(bench_height, 2),
        "hole_diameter_mm": np.round(hole_diameter, 1),
        "hole_depth_m": np.round(hole_depth, 2),
        "burden_m": np.round(burden, 2),
        "spacing_m": np.round(spacing, 2),
        "stemming_m": np.round(stemming, 2),
        "subdrilling_m": np.round(subdrilling, 2),
        "charge_mass_per_hole_kg": np.round(charge_mass_per_hole, 2),
        "powder_factor_kg_m3": np.round(powder_factor_kg_m3, 3),
        "powder_factor_kg_t": np.round(powder_factor_kg_t, 3),
        "max_charge_per_delay_kg": np.round(max_charge_per_delay, 2),
        "monitoring_distance_m": np.round(monitoring_distance, 1),
        # Target variables
        "d50_mm": np.round(d50_mm, 2),
        "fragmentation_d80_cm": np.round(d80_cm, 2),
        "uniformity_index_n": np.round(n_uniformity, 2),
        "ppv_mms": np.round(ppv_mms, 2),
        "vibration_ppv_mms": np.round(ppv_mms, 2),
        "airblast_db": np.round(airblast_db, 1),
        "flyrock_m": np.round(flyrock_dist_m, 2),
        "cost_per_tonne_usd": np.round(cost_per_tonne, 2),
    })

    assert data.shape[0] == num_samples, f"Expected {num_samples} rows, got {data.shape[0]}"
    return data


def validate_synthetic_data(df: pd.DataFrame) -> bool:
    """
    Validate that synthetic dataset DataFrame is non-empty and contains required columns.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns:
    --------
    bool
        True if valid.

    Raises:
    -------
    ValueError
        If DataFrame is empty or missing expected target columns.
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty or None.")

    required_targets = [
        "fragmentation_d80_cm",
        "vibration_ppv_mms",
        "airblast_db",
    ]
    missing = [col for col in required_targets if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required target columns: {missing}")

    return True


def generate_for_model(model_key: str, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic blast dataset suitable for the specified model key."""
    return generate_synthetic_blast_data(num_samples=n_samples, seed=seed)


if __name__ == "__main__":
    df = generate_synthetic_blast_data(num_samples=100)
    print("Generated synthetic dataset sample shape:", df.shape)
    print(df.head())
