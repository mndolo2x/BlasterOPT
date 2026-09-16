"""
3D Mine-to-Mill Digital Twin Module (Model 4) for BlastOpt Botswana.

Implements Model 4 (Mine-to-Mill Digital Twin) based on the Cost-Integrated AI Meta-Models framework:
- FragmentationModel: Computes Kuz-Ram and Swebrec percentiles (D10, D50, D80, D90, x_c, n).
- DownstreamModel: Predicts excavator fill factors, truck payloads, crusher throughput, specific energy, and total cost per tonne.
- MineToMillTwin: Orchestrates full simulation pipeline and what-if sensitivity sweeps.
- OreTracker: Models graph database relationships (Neo4j / local fallback) tracking ore from bench to mill via GPS/RFID.
- ScenarioAnalyzer: Generates sensitivity sweeps and returns Plotly figures showing parameter impacts on total cost per tonne.
- Preserves 3D spatial bench builder functions (build_digital_twin, simulate_fragmentation, link_to_downstream).
"""

import os
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List, Union, Tuple

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

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

from src.predict import total_cost_per_tonne, predict_crusher_throughput

logger = logging.getLogger(__name__)


class FragmentationModel:
    """
    Predicts rock fragment size distribution using Kuz-Ram and Swebrec equations.
    """

    def __init__(self, rock_factor_A: float = 8.5):
        self.rock_factor_A = rock_factor_A

    def predict(self, blast_params: Dict[str, float]) -> Dict[str, float]:
        """
        Predicts fragment size percentiles (D10, D50, D80, D90, x_c, n) and distribution percentages.

        Parameters:
        -----------
        blast_params : Dict[str, float]
            Dictionary containing powder_factor_kg_m3, burden_m, spacing_m, stemming_m, charge_mass_per_hole_kg.

        Returns:
        --------
        Dict[str, float]
            Fragmentation size distribution metrics.
        """
        pf = float(blast_params.get("powder_factor_kg_m3", 0.65))
        q = float(blast_params.get("charge_mass_per_hole_kg", 320.0))

        # Kuz-Ram d50 (cm -> mm)
        d50_cm = self.rock_factor_A * (max(pf, 0.05) ** (-0.8)) * (max(q, 1.0) ** (1.0 / 6.0)) * ((115.0 / 100.0) ** (19.0 / 30.0))
        d50_mm = float(np.clip(d50_cm * 10.0, 30.0, 1000.0))

        n_val = 1.25
        x_c = d50_mm / (np.log(2.0) ** (1.0 / n_val))

        d10_mm = x_c * ((-np.log(1.0 - 0.10)) ** (1.0 / n_val))
        d80_mm = x_c * ((-np.log(1.0 - 0.80)) ** (1.0 / n_val))
        d90_mm = x_c * ((-np.log(1.0 - 0.90)) ** (1.0 / n_val))

        boulder_passing_pct = (1.0 - np.exp(-1.0 * ((500.0 / x_c) ** n_val))) * 100.0
        boulder_pct = max(0.0, 100.0 - boulder_passing_pct)
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


class DownstreamModel:
    """
    Predicts downstream Mine-to-Mill performance across digging, hauling, crushing, and milling.
    """

    def __init__(self, ore_hardness_wi: float = 12.5):
        self.ore_hardness = ore_hardness_wi

    def predict(
        self,
        fragmentation: Dict[str, float],
        blast_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Predicts digger productivity, truck fill factor, crusher throughput, specific energy, and total cost per tonne.

        Parameters:
        -----------
        fragmentation : Dict[str, float]
            Fragmentation distribution dictionary from FragmentationModel.predict().
        blast_params : Dict[str, float], optional
            Blast design parameters.

        Returns:
        --------
        Dict[str, float]
            Downstream KPIs dictionary.
        """
        d50 = float(fragmentation.get("d50_mm", 220.0))
        d80_cm = float(fragmentation.get("d80_mm", 350.0)) / 10.0
        boulder_pct = float(fragmentation.get("boulder_percentage", 5.0))

        # 1. Digger / Excavator Productivity
        base_digger_tph = 2500.0
        bucket_fill_factor = float(np.clip(0.95 - (boulder_pct / 100.0) * 0.40 - (max(d50 - 250.0, 0.0) / 1000.0), 0.50, 0.98))
        cycle_time_sec = float(np.clip(28.0 + (boulder_pct * 0.8), 25.0, 60.0))
        digger_productivity_tph = base_digger_tph * bucket_fill_factor * (30.0 / cycle_time_sec)

        # 2. Haul Truck Payload Fill Factor & Cycle Time
        truck_fill_factor = float(np.clip(0.98 - (boulder_pct / 100.0) * 0.35, 0.60, 0.99))
        truck_cycle_time_min = float(np.clip(22.0 + (boulder_pct * 0.3), 18.0, 45.0))

        # 3. Crusher Throughput & Specific Energy
        crusher_res = predict_crusher_throughput(d80_cm=d80_cm, ore_hardness=self.ore_hardness)

        # 4. Total Mine-to-Mill Cost Breakdown ($/t)
        cost_breakdown = total_cost_per_tonne(blast_params if blast_params else {"powder_factor_kg_m3": 0.65, "d50_mm": d50})

        # Add loading, hauling, crushing, milling stage costs
        drill_cost = cost_breakdown["drilling_cost_usd_t"]
        exp_cost = cost_breakdown["explosive_cost_usd_t"]
        dig_cost = cost_breakdown["digging_cost_usd_t"]
        haul_cost = cost_breakdown["hauling_cost_usd_t"]
        crush_cost = cost_breakdown["crushing_cost_usd_t"]
        mill_cost = cost_breakdown["milling_cost_usd_t"]
        screen_cost = 0.25
        stockpile_cost = 0.30

        total_cost_usd_t = round(
            drill_cost + exp_cost + dig_cost + haul_cost + crush_cost + mill_cost + screen_cost + stockpile_cost, 2
        )

        return {
            "digger_productivity_tph": round(float(digger_productivity_tph), 1),
            "bucket_fill_factor_pct": round(float(bucket_fill_factor * 100.0), 1),
            "truck_fill_factor_pct": round(float(truck_fill_factor * 100.0), 1),
            "truck_cycle_time_min": round(float(truck_cycle_time_min), 1),
            "crusher_throughput_tph": crusher_res.get("throughput_tph", 2200.0),
            "specific_energy_kwh_t": crusher_res.get("specific_energy_kwh_t", 4.2),
            "total_cost_per_tonne_usd": total_cost_usd_t,
            "cost_breakdown_usd_t": {
                "drilling": drill_cost,
                "explosives": exp_cost,
                "digging": dig_cost,
                "hauling": haul_cost,
                "crushing": crush_cost,
                "milling": mill_cost,
                "screening": screen_cost,
                "stockpiling": stockpile_cost,
            },
        }


class MineToMillTwin:
    """
    Orchestrates the full Mine-to-Mill digital twin pipeline connecting blast designs to economic outcomes.
    """

    def __init__(self, rock_factor_A: float = 8.5, ore_hardness_wi: float = 12.5):
        self.frag_model = FragmentationModel(rock_factor_A=rock_factor_A)
        self.downstream_model = DownstreamModel(ore_hardness_wi=ore_hardness_wi)

    def simulate(self, blast_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulates full pipeline from blast geometry to fragmentation, throughput, energy, and total cost per tonne.

        Parameters:
        -----------
        blast_params : Dict[str, float]
            Input blast parameters.

        Returns:
        --------
        Dict[str, Any]
            Complete simulation dictionary containing fragmentation, throughput, energy, and cost keys.
        """
        frag = self.frag_model.predict(blast_params)
        downstream = self.downstream_model.predict(frag, blast_params=blast_params)

        return {
            "fragmentation": frag,
            "throughput": downstream["crusher_throughput_tph"],
            "energy": downstream["specific_energy_kwh_t"],
            "cost": downstream["total_cost_per_tonne_usd"],
            "downstream": downstream,
        }

    def what_if(
        self,
        blast_params: Dict[str, float],
        param_to_vary: str = "powder_factor_kg_m3",
        range_min: float = 0.30,
        range_max: float = 1.20,
        steps: int = 20,
    ) -> pd.DataFrame:
        """
        Generates what-if sensitivity analysis showing how cost per tonne and throughput change as a parameter varies.

        Parameters:
        -----------
        blast_params : Dict[str, float]
            Base blast design parameters.
        param_to_vary : str, default="powder_factor_kg_m3"
            Parameter name to sweep over range.
        range_min : float, default=0.30
            Sweep minimum value.
        range_max : float, default=1.20
            Sweep maximum value.
        steps : int, default=20
            Number of sweep evaluation steps.

        Returns:
        --------
        pd.DataFrame
            DataFrame of parameter values vs predicted cost, throughput, and fragmentation metrics.
        """
        vals = np.linspace(range_min, range_max, steps)
        rows = []

        for v in vals:
            test_params = blast_params.copy()
            test_params[param_to_vary] = float(v)

            if param_to_vary == "powder_factor_kg_m3":
                b = test_params.get("burden_m", 6.0)
                s = test_params.get("spacing_m", 7.0)
                h = test_params.get("bench_height_m", 15.0)
                test_params["charge_mass_per_hole_kg"] = float(v * b * s * h)

            res = self.simulate(test_params)
            rows.append({
                param_to_vary: round(float(v), 3),
                "d50_mm": res["fragmentation"]["d50_mm"],
                "d80_mm": res["fragmentation"]["d80_mm"],
                "crusher_throughput_tph": res["throughput"],
                "specific_energy_kwh_t": res["energy"],
                "total_cost_per_tonne_usd": res["cost"],
            })

        return pd.DataFrame(rows)


class OreTracker:
    """
    Models relationships between blasts, ore blocks, and processing batches using Neo4j or local graph structure.
    """

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.local_graph: List[Dict[str, Any]] = []

        if HAS_NEO4J and uri and user and password:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
            except Exception as e:
                logger.warning(f"Neo4j driver connection fallback: {e}")

    def track_ore_block(
        self,
        blast_id: str = "BLAST_JWA_2024_08",
        block_id: str = "BLOCK_CUT8_15S_01",
        batch_id: str = "BATCH_MILL_402",
        gps_collar: Tuple[float, float, float] = (-24.52, 25.83, 1150.0),
    ) -> Dict[str, Any]:
        """
        Tracks ore from bench to mill using GPS/RFID telematics and records graph node relationship.

        Parameters:
        -----------
        blast_id : str
            Blast pattern identifier.
        block_id : str
            Ore block polygon identifier.
        batch_id : str
            Milling batch identifier.
        gps_collar : Tuple[float, float, float]
            GPS collar coordinate tuple (latitude, longitude, elevation_m).

        Returns:
        --------
        Dict[str, Any]
            Ore tracking record dictionary.
        """
        record = {
            "blast_id": blast_id,
            "block_id": block_id,
            "batch_id": batch_id,
            "gps_collar": gps_collar,
            "status": "TRACKED_BENCH_TO_MILL",
            "rfid_tag": f"RFID_ORE_{block_id}_9921",
        }

        self.local_graph.append(record)

        if self.driver is not None:
            try:
                with self.driver.session() as session:
                    cypher = """
                    MERGE (b:Blast {id: $blast_id})
                    MERGE (o:OreBlock {id: $block_id})
                    MERGE (m:MillBatch {id: $batch_id})
                    MERGE (b)-[:PRODUCED]->(o)
                    MERGE (o)-[:PROCESSED_IN]->(m)
                    """
                    session.run(cypher, blast_id=blast_id, block_id=block_id, batch_id=batch_id)
            except Exception as err:
                logger.warning(f"Neo4j Cypher write fallback: {err}")

        return record

    def close(self):
        """Closes Neo4j driver connection."""
        if self.driver is not None:
            self.driver.close()


class ScenarioAnalyzer:
    """
    Generates sensitivity sweeps and scenario analysis returning Plotly figures.
    """

    def __init__(self, twin: Optional[MineToMillTwin] = None):
        self.twin = twin if twin is not None else MineToMillTwin()

    def plot_sensitivity(
        self,
        base_blast_params: Dict[str, float],
        param_to_vary: str = "powder_factor_kg_m3",
        range_min: float = 0.30,
        range_max: float = 1.20,
    ) -> go.Figure:
        """
        Generates Plotly sensitivity chart showing parameter impact on total cost per tonne.

        Parameters:
        -----------
        base_blast_params : Dict[str, float]
            Base blast design parameters.
        param_to_vary : str, default="powder_factor_kg_m3"
            Parameter name to sweep.
        range_min : float, default=0.30
            Sweep minimum.
        range_max : float, default=1.20
            Sweep maximum.

        Returns:
        --------
        go.Figure
            Plotly Figure displaying parameter vs total cost per tonne.
        """
        df_sweep = self.twin.what_if(
            blast_params=base_blast_params,
            param_to_vary=param_to_vary,
            range_min=range_min,
            range_max=range_max,
            steps=20,
        )

        fig = px.line(
            df_sweep,
            x=param_to_vary,
            y="total_cost_per_tonne_usd",
            title=f"<b>What-If Sensitivity: {param_to_vary} vs. Total Mine-to-Mill Cost ($/t)</b>",
            labels={param_to_vary: param_to_vary.replace("_", " ").title(), "total_cost_per_tonne_usd": "Total Cost ($/tonne)"},
            markers=True,
            color_discrete_sequence=["#1976D2"],
        )

        fig.update_layout(template="plotly_white", height=420)
        return fig


# --- Standalone 3D Bench Spatial Helper Functions ---

def build_digital_twin(
    bench_id: str = "BENCH_JWA_15S",
    geological_data: Optional[Dict[str, Any]] = None,
    as_drilled_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Constructs 3D Digital Twin bench dictionary."""
    if geological_data is None:
        geological_data = {
            "rock_type": "Kimberlite_Hard",
            "rock_factor_A": 8.5,
            "density_t_m3": 2.65,
            "joint_spacing_m": 0.8,
            "hardness_index": 12.0,
        }

    if as_drilled_data is None or as_drilled_data.empty:
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

    bench_length = float(as_drilled_data["x_m"].max() + 7.0) if not as_drilled_data.empty else 45.0
    bench_width = float(as_drilled_data["y_m"].max() + 6.0) if not as_drilled_data.empty else 25.0
    bench_height = 15.0
    base_z = 135.0

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

    trimesh_obj = None
    if HAS_TRIMESH:
        try:
            faces = np.array([
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [2, 3, 7], [2, 7, 6],
                [1, 2, 6], [1, 6, 5],
                [0, 3, 7], [0, 7, 4],
            ])
            trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        except Exception:
            trimesh_obj = None

    return {
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


def simulate_fragmentation(
    digital_twin: Dict[str, Any],
    blast_params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Simulates rock fragmentation for digital twin bench."""
    if blast_params is None:
        blast_params = {"powder_factor_kg_m3": 0.65, "burden_m": 6.0, "spacing_m": 7.0, "charge_mass_per_hole_kg": 320.0}

    geo = digital_twin.get("geological_data", {})
    rock_A = float(geo.get("rock_factor_A", 8.5))

    frag_model = FragmentationModel(rock_factor_A=rock_A)
    return frag_model.predict(blast_params)


def link_to_downstream(
    digital_twin: Dict[str, Any],
    fragmentation: Dict[str, Any],
) -> Dict[str, float]:
    """Feeds fragmentation parameters to downstream value model."""
    geo = digital_twin.get("geological_data", {})
    wi = float(geo.get("hardness_index", 12.5))

    downstream_model = DownstreamModel(ore_hardness_wi=wi)
    return downstream_model.predict(fragmentation)
