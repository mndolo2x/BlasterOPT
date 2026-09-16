"""
3D Digital Twin of the Bench Module for BlastOpt Botswana.

Constructs 3D spatial models of open-pit bench blocks, integrating geological block models,
as-drilled geometry, and structural joint data. Links simulated fragmentation size distributions
to downstream Mine-to-Mill economic outcomes (digger productivity, truck payload, crusher throughput).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from src.predict import total_cost_per_tonne, predict_crusher_throughput


def build_digital_twin(
    bench_id: str = "BENCH_JWA_15S",
    geological_data: Optional[Dict[str, Any]] = None,
    as_drilled_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Constructs a 3D Digital Twin of the mining bench block incorporating geological models,
    as-drilled drillhole geometry, and structural joint orientations.

    Digital Twin Domain Context:
    ----------------------------
    A 3D Digital Twin bridges primary drill-and-blast design with downstream mining value.
    By mapping 3D spatial variations in rock hardness (Kimberlite vs Waste Granite), jointing density,
    and actual as-drilled hole locations (accounting for collar deviation and dip/azimuth errors),
    the digital twin enables precision blast simulations and closes the loop with downstream processing plant performance.

    Parameters:
    -----------
    bench_id : str, default="BENCH_JWA_15S"
        Bench identifier code (e.g. Jwaneng Cut 8 Bench 15 South).
    geological_data : Dict[str, Any], optional
        Geological block model parameters (rock_type, rock_factor_A, density_t_m3, joint_spacing_m).
    as_drilled_data : pd.DataFrame, optional
        DataFrame containing as-drilled hole coordinates (hole_id, x_m, y_m, z_m, depth_m, diameter_mm).

    Returns:
    --------
    Dict[str, Any]
        Structured 3D Digital Twin object containing mesh points, drillhole geometry, block model, and metadata.
    """
    if geological_data is None:
        geological_data = {
            "rock_type": "Kimberlite_Hard",
            "rock_factor_A": 8.5,
            "density_t_m3": 2.65,
            "joint_spacing_m": 0.8,
            "hardness_index": 12.0,  # Bond Work Index kWh/t
        }

    if as_drilled_data is None or as_drilled_data.empty:
        # Generate default grid of 24 as-drilled holes
        holes = []
        for r in range(4):
            for h in range(6):
                holes.append({
                    "hole_id": f"R{r+1}-H{h+1}",
                    "x_m": h * 7.0 + (r % 2) * 3.5,
                    "y_m": r * 6.0,
                    "z_m": 150.0,
                    "depth_m": 15.0 + (np.random.rand() - 0.5) * 0.8,
                    "diameter_mm": 250.0,
                    "dip_deg": 90.0,
                })
        as_drilled_data = pd.DataFrame(holes)

    # Construct 3D bench block geometry coordinates (x, y, z)
    bench_length = float(as_drilled_data["x_m"].max() + 7.0) if not as_drilled_data.empty else 45.0
    bench_width = float(as_drilled_data["y_m"].max() + 6.0) if not as_drilled_data.empty else 25.0
    bench_height = 15.0
    base_z = 135.0

    # 3D Bounding box vertices
    vertices = np.array([
        [0.0, 0.0, base_z],
        [bench_length, 0.0, base_z],
        [bench_length, bench_width, base_z],
        [0.0, bench_width, base_z],
        [0.0, 0.0, base_z + bench_height],
        [bench_length, 0.0, base_z + bench_height],
        [bench_length, bench_width, base_z + bench_height],
        [0.0, bench_width, base_z + bench_height],
    ])

    # 3D Trimesh / PyVista mesh reference if libraries available
    trimesh_obj = None
    if HAS_TRIMESH:
        try:
            faces = np.array([
                [0, 1, 2], [0, 2, 3],  # Bottom
                [4, 5, 6], [4, 6, 7],  # Top
                [0, 1, 5], [0, 5, 4],  # Front
                [2, 3, 7], [2, 7, 6],  # Back
                [1, 2, 6], [1, 6, 5],  # Right
                [0, 3, 7], [0, 7, 4],  # Left
            ])
            trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        except Exception:
            trimesh_obj = None

    digital_twin = {
        "bench_id": bench_id,
        "geological_data": geological_data,
        "as_drilled_data": as_drilled_data,
        "dimensions": {
            "length_m": bench_length,
            "width_m": bench_width,
            "height_m": bench_height,
            "volume_m3": round(bench_length * bench_width * bench_height, 2),
            "tonnage_t": round(bench_length * bench_width * bench_height * geological_data.get("density_t_m3", 2.65), 2),
        },
        "vertices_3d": vertices,
        "trimesh_mesh": trimesh_obj,
    }

    return digital_twin


def simulate_fragmentation(
    digital_twin: Dict[str, Any],
    blast_params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Simulates rock fragmentation size distribution (d10, d50, d80, d90, Rosin-Rammler parameters)
    for the digital twin bench block.

    Parameters:
    -----------
    digital_twin : Dict[str, Any]
        3D Digital Twin bench dictionary built by build_digital_twin.
    blast_params : Dict[str, float], optional
        Blast geometry parameters (powder_factor_kg_m3, burden_m, spacing_m, stemming_m).

    Returns:
    --------
    Dict[str, Any]
        Dictionary of simulated fragmentation percent passing sizes (mm) and Rosin-Rammler parameters.
    """
    if blast_params is None:
        blast_params = {
            "powder_factor_kg_m3": 0.65,
            "burden_m": 6.0,
            "spacing_m": 7.0,
            "stemming_m": 5.0,
            "charge_mass_per_hole_kg": 320.0,
        }

    geo = digital_twin.get("geological_data", {})
    rock_A = float(geo.get("rock_factor_A", 8.5))
    pf = float(blast_params.get("powder_factor_kg_m3", 0.65))
    q = float(blast_params.get("charge_mass_per_hole_kg", 320.0))

    # Extended Kuz-Ram d50 (cm) calculation
    d50_cm = rock_A * (max(pf, 0.05) ** (-0.8)) * (max(q, 1.0) ** (1.0 / 6.0)) * ((115.0 / 100.0) ** (19.0 / 30.0))
    d50_mm = float(np.clip(d50_cm * 10.0, 30.0, 1000.0))

    # Rosin-Rammler uniformity index n
    n_val = 1.25

    # Rosin-Rammler characteristic size x_c = d50 / (ln 2)^(1/n)
    x_c = d50_mm / (np.log(2.0) ** (1.0 / n_val))

    # Derivation of d10, d80, d90 percent passing sizes
    # P(x) = 1 - exp(-(x/x_c)^n)  ==> x = x_c * (-ln(1 - P))^(1/n)
    d10_mm = x_c * ((-np.log(1.0 - 0.10)) ** (1.0 / n_val))
    d80_mm = x_c * ((-np.log(1.0 - 0.80)) ** (1.0 / n_val))
    d90_mm = x_c * ((-np.log(1.0 - 0.90)) ** (1.0 / n_val))

    # Boulder percentage (> 500 mm / 50 cm)
    boulder_passing_pct = (1.0 - np.exp(-1.0 * ((500.0 / x_c) ** n_val))) * 100.0
    boulder_pct = max(0.0, 100.0 - boulder_passing_pct)

    # Fines percentage (< 10 mm / 1 cm)
    fines_pct = (1.0 - np.exp(-1.0 * ((10.0 / x_c) ** n_val))) * 100.0

    return {
        "d10_mm": round(float(d10_mm), 1),
        "d50_mm": round(float(d50_mm), 1),
        "d80_mm": round(float(d80_mm), 1),
        "d90_mm": round(float(d90_mm), 1),
        "characteristic_size_xc_mm": round(float(x_c), 1),
        "uniformity_index_n": round(float(n_val), 2),
        "boulder_percentage": round(float(boulder_pct), 2),
        "fines_percentage": round(float(fines_pct), 2),
    }


def link_to_downstream(
    digital_twin: Dict[str, Any],
    fragmentation: Dict[str, Any],
) -> Dict[str, float]:
    """
    Feeds simulated bench fragmentation parameters into a downstream Mine-to-Mill value model predicting
    excavator productivity, truck payload fill factor, primary crusher throughput, and total unit cost.

    Parameters:
    -----------
    digital_twin : Dict[str, Any]
        3D Digital Twin bench dictionary.
    fragmentation : Dict[str, Any]
        Fragmentation size distribution dictionary from simulate_fragmentation.

    Returns:
    --------
    Dict[str, float]
        Dictionary of predicted downstream operating KPIs (digger_productivity_tph, truck_fill_factor,
        crusher_throughput_tph, specific_energy_kwh_t, total_cost_per_tonne_usd).
    """
    d50 = float(fragmentation.get("d50_mm", 220.0))
    d80_cm = float(fragmentation.get("d80_mm", 350.0)) / 10.0
    boulder_pct = float(fragmentation.get("boulder_percentage", 5.0))

    geo = digital_twin.get("geological_data", {})
    wi = float(geo.get("hardness_index", 12.0))

    # 1. Shovel / Digger Productivity (t/h)
    # Optimum diggability at d50 = 150-250mm; coarse boulders degrade bucket fill factor & cycle time
    base_digger_tph = 2500.0
    bucket_fill_factor = float(np.clip(0.95 - (boulder_pct / 100.0) * 0.40 - (max(d50 - 250.0, 0.0) / 1000.0), 0.50, 0.98))
    cycle_time_sec = float(np.clip(28.0 + (boulder_pct * 0.8), 25.0, 60.0))
    digger_productivity_tph = base_digger_tph * bucket_fill_factor * (30.0 / cycle_time_sec)

    # 2. Haul Truck Payload Fill Factor (%)
    truck_fill_factor = float(np.clip(0.98 - (boulder_pct / 100.0) * 0.35, 0.60, 0.99))

    # 3. Crusher Throughput & Specific Energy Consumption
    crusher_res = predict_crusher_throughput(d80_cm=d80_cm, ore_hardness=wi)

    # 4. Total Mine-to-Mill Cost ($/t)
    cost_res = total_cost_per_tonne({
        "powder_factor_kg_m3": 0.65,
        "bench_height_m": 15.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "d50_mm": d50,
    })

    return {
        "digger_productivity_tph": round(float(digger_productivity_tph), 1),
        "bucket_fill_factor_pct": round(float(bucket_fill_factor * 100.0), 1),
        "truck_fill_factor_pct": round(float(truck_fill_factor * 100.0), 1),
        "crusher_throughput_tph": crusher_res.get("throughput_tph", 2200.0),
        "specific_energy_kwh_t": crusher_res.get("specific_energy_kwh_t", 4.2),
        "total_cost_per_tonne_usd": cost_res.get("total_cost_usd_t", 5.20),
    }
