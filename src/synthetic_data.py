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

    # Burden (m): typically 25 - 40 x hole diameter in meters
    burden = (hole_diameter / 1000.0) * np.random.uniform(28, 38, size=num_samples)
    # Spacing (m): typically 1.1 - 1.4 x Burden
    spacing = burden * np.random.uniform(1.1, 1.35, size=num_samples)
    # Stemming length (m): typically 0.7 - 1.2 x Burden
    stemming = burden * np.random.uniform(0.7, 1.1, size=num_samples)
    # Subdrilling (m): typically 0.2 - 0.3 x Burden
    subdrilling = burden * np.random.uniform(0.2, 0.3, size=num_samples)

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

    # Maximum Charge Weight Per Delay (Q in kg) - assuming 2 to 4 holes per delay
    holes_per_delay = np.random.randint(1, 4, size=num_samples)
    max_charge_per_delay = charge_mass_per_hole * holes_per_delay

    # Distance to nearest critical structure / monitor point (m)
    monitoring_distance = np.random.uniform(150.0, 1200.0, size=num_samples)

    # 3. Physics Models for Targets

    # --- Target A: Mean Fragment Size d50 (cm) using Kuz-Ram Equation ---
    # d50 (cm) = A * (K)^(-0.8) * Q_hole^(1/6) * (115 / RWS)^(19/30)
    # K = Powder factor in kg/m3
    kuz_ram_d50_cm = (
        rock_factor
        * (powder_factor_kg_m3 ** (-0.8))
        * (charge_mass_per_hole ** (1/6))
        * ((115.0 / rws) ** (19/30))
    )
    # Convert cm to mm
    d50_mm = kuz_ram_d50_cm * 10.0
    # Add random operational noise (±10%)
    d50_mm = d50_mm * np.random.normal(1.0, 0.08, size=num_samples)
    d50_mm = np.clip(d50_mm, 20.0, 800.0)

    # Kuz-Ram Uniformity Index (n)
    # n = (2.2 - 14 * B / d) * (1 - S / B)^0.5 * (1 + (abs(S/B - 1)) / 2) * (L / H) ...
    spacing_burden_ratio = spacing / burden
    n_uniformity = (2.2 - 14 * (burden / (hole_diameter / 1000.0))) * (
        1 + (spacing_burden_ratio - 1) / 2
    ) * (charge_length / bench_height)
    n_uniformity = np.clip(np.abs(n_uniformity) + 0.8, 0.7, 2.2)

    # --- Target B: Peak Particle Velocity PPV (mm/s) using USBM Scaled Distance Law ---
    # PPV = K_vib * ( ScaledDistance )^(-beta)
    # Scaled Distance SD = Distance / sqrt(Q_delay)
    scaled_distance = monitoring_distance / np.sqrt(max_charge_per_delay)

    # Ground attenuation constants K_vib (typically 500 - 2000) and beta (typically 1.4 - 1.8)
    k_vib = np.random.uniform(800, 1600, size=num_samples)
    beta = np.random.uniform(1.4, 1.7, size=num_samples)

    ppv_mms = k_vib * (scaled_distance ** (-beta))
    # Operational noise
    ppv_mms = ppv_mms * np.random.normal(1.0, 0.12, size=num_samples)
    ppv_mms = np.clip(ppv_mms, 0.5, 120.0)

    # --- Target C: Flyrock Distance (m) ---
    # Scaled Charge empirical formula: L_fly = K_fly * (Q_hole^(2/3) / Burden)
    k_fly = np.random.uniform(15, 25, size=num_samples)
    flyrock_dist_m = k_fly * ((charge_mass_per_hole ** (2/3)) / burden) * (stemming / burden) ** (-0.5)
    flyrock_dist_m = flyrock_dist_m * np.random.normal(1.0, 0.10, size=num_samples)
    flyrock_dist_m = np.clip(flyrock_dist_m, 10.0, 450.0)

    # --- Target D: Drilling & Blasting Cost ($ / tonne) ---
    # Cost = (Drilling Cost + Explosive Cost + Accessories Cost) / Tonnes per hole
    drilling_cost_per_m = 12.0 + (hole_diameter / 100.0) * 8.0 # $12 - $35 / m
    explosive_cost_per_kg = 1.2 + (rws / 100.0) * 0.8 # $1.5 - $2.5 / kg
    accessories_cost_per_hole = 15.0 # Detonators, boosters, surface delays

    total_hole_cost = (
        (hole_depth * drilling_cost_per_m)
        + (charge_mass_per_hole * explosive_cost_per_kg)
        + accessories_cost_per_hole
    )
    cost_per_tonne = total_hole_cost / rock_mass_per_hole
    cost_per_tonne = cost_per_tonne * np.random.normal(1.0, 0.05, size=num_samples)
    cost_per_tonne = np.clip(cost_per_tonne, 0.5, 15.0)

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
        "uniformity_index_n": np.round(n_uniformity, 2),
        "ppv_mms": np.round(ppv_mms, 2),
        "flyrock_m": np.round(flyrock_dist_m, 2),
        "cost_per_tonne_usd": np.round(cost_per_tonne, 2),
    })

    return data


if __name__ == "__main__":
    df = generate_synthetic_blast_data(num_samples=100)
    print("Generated synthetic dataset sample shape:", df.shape)
    print(df.head())
