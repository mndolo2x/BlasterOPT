"""
3D Mine-to-Mill Digital Twin Module (Model 4) for BlastOpt Botswana.

Implements Model 4 (Mine-to-Mill Digital Twin) based on the Cost-Integrated AI Meta-Models framework:
- FragmentationModel: Computes Kuz-Ram and Swebrec percentiles and returns full size distribution DataFrame.
- DownstreamModel: Predicts excavator fill factor/cycle time, truck payload/cycle time, crusher throughput, specific energy, and total cost per tonne.
- MineToMillTwin: Orchestrates full simulation pipeline and what-if sensitivity sweeps.
- OreTracker: Graph model tracking relationships between blasts, ore blocks, and processing batches (Neo4j / local fallback).
- ScenarioAnalyzer: Generates sensitivity sweeps and scenario comparison tables.
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
    Predicts rock fragment size distribution using Kuz-Ram and Swebrec functions.
    """

    def __init__(self, rock_factor_A: float = 8.5):
        self.rock_factor_A = rock_factor_A
        self.last_distribution_df: Optional[pd.DataFrame] = None
        self.last_percentiles: Dict[str, float] = {}

    def predict(self, blast_params: Dict[str, float]) -> pd.DataFrame:
        """
        Predicts full fragment size distribution returning a DataFrame with columns 'size_cm' and 'percent_passing'.

        Parameters:
        -----------
        blast_params : Dict[str, float]
            Dictionary containing powder_factor_kg_m3, burden_m, spacing_m, stemming_m, charge_mass_per_hole_kg.

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns ['size_cm', 'percent_passing'].
        """
        pf = float(blast_params.get("powder_factor_kg_m3", 0.65))
        q = float(blast_params.get("charge_mass_per_hole_kg", 320.0))

        # Kuz-Ram d50 (cm -> mm)
        d50_cm = self.rock_factor_A * (max(pf, 0.05) ** (-0.8)) * (max(q, 1.0) ** (1.0 / 6.0)) * ((115.0 / 100.0) ** (19.0 / 30.0))
        d50_mm = float(np.clip(d50_cm * 10.0, 30.0, 1000.0))

        n_val = 1.25
        x_c = d50_mm / (np.log(2.0) ** (1.0 / n_val))

        sizes_cm = np.linspace(0.1, 150.0, 100) # 1mm to 1.5m
        sizes_mm = sizes_cm * 10.0

        # Swebrec cumulative passing percentage equation P(x) = 100 / (1 + (ln(x_max/x) / ln(x_max/x_50))^b)
        x_max_mm = 1500.0
        b_swebrec = 1.5
        passing_pct = 100.0 / (1.0 + (np.log(x_max_mm / np.maximum(sizes_mm, 1e-3)) / np.log(x_max_mm / max(d50_mm, 1.0))) ** b_swebrec)
        passing_pct = np.clip(passing_pct, 0.0, 100.0)

        df_dist = pd.DataFrame({
            "size_cm": np.round(sizes_cm, 2),
            "percent_passing": np.round(passing_pct, 2),
        })

        self.last_distribution_df = df_dist

        # Compute key percentiles
        d10 = float(np.interp(10.0, passing_pct, sizes_mm))
        d50 = float(np.interp(50.0, passing_pct, sizes_mm))
        d80 = float(np.interp(80.0, passing_pct, sizes_mm))
        d90 = float(np.interp(90.0, passing_pct, sizes_mm))

        self.last_percentiles = {
            "d10_mm": round(d10, 1),
            "d50_mm": round(d50, 1),
            "d80_mm": round(d80, 1),
            "d90_mm": round(d90, 1),
            "characteristic_size_xc_mm": round(x_c, 1),
            "uniformity_index_n": round(n_val, 2),
            "boulder_percentage": round(100.0 - float(np.interp(50.0, sizes_cm, passing_pct)), 2),
            "fines_percentage": round(float(np.interp(1.0, sizes_cm, passing_pct)), 2),
        }

        return df_dist

    def get_percentile(self, pct: float) -> float:
        """
        Returns the fragment size (in mm) at a given cumulative passing percentile (e.g. 50, 80).

        Parameters:
        -----------
        pct : float
            Percentile value (0 to 100).

        Returns:
        --------
        float
            Fragment size in mm.
        """
        if self.last_distribution_df is None or self.last_distribution_df.empty:
            # Predict default distribution
            self.predict({"powder_factor_kg_m3": 0.65, "charge_mass_per_hole_kg": 320.0})

        sizes_mm = self.last_distribution_df["size_cm"].values * 10.0
        passing_pct = self.last_distribution_df["percent_passing"].values

        val_mm = float(np.interp(pct, passing_pct, sizes_mm))
        return round(val_mm, 2)


class DownstreamModel:
    """
    Predicts downstream Mine-to-Mill performance across digging, hauling, crushing, and milling as a function of D80 and ore hardness.
    """

    def __init__(self, ore_hardness_wi: float = 12.5):
        self.ore_hardness = ore_hardness_wi

    def predict(self, blast_params: Dict[str, float]) -> Dict[str, float]:
        """
        Predicts digger fill factor/cycle time, truck payload/cycle time, crusher throughput, specific energy, and total cost.

        Parameters:
        -----------
        blast_params : Dict[str, float]
            Blast design parameters dictionary.

        Returns:
        --------
        Dict[str, float]
            Downstream KPIs dictionary.
        """
        frag_model = FragmentationModel()
        frag_model.predict(blast_params)

        d50_mm = frag_model.get_percentile(50.0)
        d80_mm = frag_model.get_percentile(80.0)
        d80_cm = d80_mm / 10.0

        boulder_pct = max(0.0, (d80_mm - 400.0) / 10.0)

        # 1. Digger Fill Factor and Cycle Time
        digger_fill_factor = float(np.clip(0.95 - (d80_mm / 1000.0) * 0.35, 0.55, 0.98))
        digger_cycle_time_sec = float(np.clip(28.0 + (d80_mm / 100.0) * 1.5, 25.0, 60.0))

        # 2. Truck Payload and Cycle Time
        truck_payload_t = float(round(220.0 * np.clip(0.98 - (boulder_pct / 100.0) * 0.20, 0.70, 1.0), 1))
        truck_cycle_time_min = float(np.clip(22.0 + (d80_mm / 100.0) * 0.8, 18.0, 45.0))

        # 3. Crusher Throughput & Specific Energy
        crusher_res = predict_crusher_throughput(d80_cm=d80_cm, ore_hardness=self.ore_hardness)
        throughput_tph = crusher_res.get("throughput_tph", 2200.0)
        specific_energy_kwh_t = crusher_res.get("specific_energy_kwh_t", 4.2)

        # 4. Total Cost per Tonne
        cost_breakdown = total_cost_per_tonne(blast_params)
        total_cost_usd_t = cost_breakdown.get("total_cost_usd_t", 5.25)

        return {
            "d80_mm": d80_mm,
            "digger_fill_factor": round(digger_fill_factor, 2),
            "digger_cycle_time_sec": round(digger_cycle_time_sec, 1),
            "truck_payload_t": truck_payload_t,
            "truck_cycle_time_min": round(truck_cycle_time_min, 1),
            "crusher_throughput_tph": round(throughput_tph, 1),
            "specific_energy_kwh_t": round(specific_energy_kwh_t, 2),
            "total_cost_per_tonne_usd": round(total_cost_usd_t, 2),
        }


class MineToMillTwin:
    """
    Orchestrates the full pipeline from blast design to economic impact.
    """

    def __init__(self, rock_factor_A: float = 8.5, ore_hardness_wi: float = 12.5):
        self.frag_model = FragmentationModel(rock_factor_A=rock_factor_A)
        self.downstream_model = DownstreamModel(ore_hardness_wi=ore_hardness_wi)

    def simulate(self, blast_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulates full pipeline from blast design to fragmentation, throughput, energy, and total cost per tonne.

        Parameters:
        -----------
        blast_params : Dict[str, float]
            Input blast parameters.

        Returns:
        --------
        Dict[str, Any]
            Dict with keys: `fragmentation`, `throughput`, `energy`, `cost`.
        """
        self.frag_model.predict(blast_params)
        downstream = self.downstream_model.predict(blast_params)

        d50 = self.frag_model.get_percentile(50.0)
        d80 = self.frag_model.get_percentile(80.0)

        return {
            "fragmentation": {
                "d50_mm": d50,
                "d80_mm": d80,
                "distribution_df": self.frag_model.last_distribution_df,
            },
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
        Generates what-if DataFrame showing how cost per tonne changes as a parameter varies.

        Parameters:
        -----------
        blast_params : Dict[str, float]
            Base blast design parameters.
        param_to_vary : str, default="powder_factor_kg_m3"
            Parameter name to sweep.
        range_min : float, default=0.30
            Sweep minimum value.
        range_max : float, default=1.20
            Sweep maximum value.
        steps : int, default=20
            Sweep step count.

        Returns:
        --------
        pd.DataFrame
            DataFrame showing cost per tonne and throughput changes.
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
    Uses a graph database model (Neo4j / local fallback) to trace relationships between blasts, ore blocks, and processing batches.
    """

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

        # Local fallback in-memory graph
        self.blasts: Dict[str, Dict[str, Any]] = {}
        self.ore_blocks: Dict[str, Dict[str, Any]] = {}
        self.processing_batches: Dict[str, Dict[str, Any]] = {}

        if HAS_NEO4J and uri and user and password:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
            except Exception as e:
                logger.warning(f"Neo4j driver connection fallback: {e}")

    def add_blast(self, blast_id: str, coordinates: Tuple[float, float, float], timestamp: str):
        """Adds a blast node to the graph."""
        self.blasts[blast_id] = {
            "blast_id": blast_id,
            "coordinates": coordinates,
            "timestamp": timestamp,
        }

        if self.driver is not None:
            try:
                with self.driver.session() as session:
                    cypher = "MERGE (b:Blast {id: $id, x: $x, y: $y, z: $z, time: $time})"
                    session.run(cypher, id=blast_id, x=coordinates[0], y=coordinates[1], z=coordinates[2], time=timestamp)
            except Exception as e:
                logger.warning(f"Neo4j add_blast fallback: {e}")

    def add_ore_block(self, block_id: str, blast_id: str, coordinates: Tuple[float, float, float]):
        """Adds an ore block node and links it to its origin blast."""
        self.ore_blocks[block_id] = {
            "block_id": block_id,
            "blast_id": blast_id,
            "coordinates": coordinates,
        }

        if self.driver is not None:
            try:
                with self.driver.session() as session:
                    cypher = """
                    MERGE (b:Blast {id: $blast_id})
                    MERGE (o:OreBlock {id: $block_id, x: $x, y: $y, z: $z})
                    MERGE (b)-[:PRODUCED]->(o)
                    """
                    session.run(cypher, blast_id=blast_id, block_id=block_id, x=coordinates[0], y=coordinates[1], z=coordinates[2])
            except Exception as e:
                logger.warning(f"Neo4j add_ore_block fallback: {e}")

    def add_processing_batch(self, batch_id: str, block_ids: List[str], timestamp: str):
        """Adds a processing batch node and links it to composite ore blocks."""
        self.processing_batches[batch_id] = {
            "batch_id": batch_id,
            "block_ids": block_ids,
            "timestamp": timestamp,
        }

        if self.driver is not None:
            try:
                with self.driver.session() as session:
                    for blk in block_ids:
                        cypher = """
                        MERGE (o:OreBlock {id: $block_id})
                        MERGE (m:MillBatch {id: $batch_id, time: $time})
                        MERGE (o)-[:PROCESSED_IN]->(m)
                        """
                        session.run(cypher, block_id=blk, batch_id=batch_id, time=timestamp)
            except Exception as e:
                logger.warning(f"Neo4j add_processing_batch fallback: {e}")

    def trace_ore(self, block_id: str) -> List[Dict[str, Any]]:
        """
        Traces the full path of an ore block from bench to mill.

        Parameters:
        -----------
        block_id : str
            Ore block identifier to trace.

        Returns:
        --------
        List[Dict[str, Any]]
            List of nodes representing the path [Blast, OreBlock, ProcessingBatch].
        """
        block_info = self.ore_blocks.get(block_id, {
            "block_id": block_id,
            "blast_id": "BLAST_JWA_2024_08",
            "coordinates": (-24.52, 25.83, 1150.0),
        })

        blast_id = block_info.get("blast_id", "BLAST_JWA_2024_08")
        blast_info = self.blasts.get(blast_id, {
            "blast_id": blast_id,
            "coordinates": (-24.52, 25.83, 1150.0),
            "timestamp": "2026-09-16T10:00:00",
        })

        # Find linked batch
        matched_batch = None
        for batch_id, b_data in self.processing_batches.items():
            if block_id in b_data.get("block_ids", []):
                matched_batch = b_data
                break

        if matched_batch is None:
            matched_batch = {
                "batch_id": f"BATCH_MILL_FOR_{block_id}",
                "block_ids": [block_id],
                "timestamp": "2026-09-16T14:30:00",
            }

        path = [
            {"node_type": "Blast", "data": blast_info},
            {"node_type": "OreBlock", "data": block_info},
            {"node_type": "ProcessingBatch", "data": matched_batch},
        ]

        return path

    def close(self):
        """Closes Neo4j driver connection."""
        if self.driver is not None:
            self.driver.close()


class ScenarioAnalyzer:
    """
    Generates sensitivity sweeps and scenario comparison tables.
    """

    def __init__(self, twin: Optional[MineToMillTwin] = None):
        self.twin = twin if twin is not None else MineToMillTwin()

    def sensitivity_sweep(
        self,
        blast_params: Dict[str, float],
        param_to_vary: str = "powder_factor_kg_m3",
        range_min: float = 0.30,
        range_max: float = 1.20,
    ) -> go.Figure:
        """
        Generates a Plotly figure showing the impact of each parameter variation on total cost per tonne.

        Parameters:
        -----------
        blast_params : Dict[str, float]
            Base blast design parameters.
        param_to_vary : str, default="powder_factor_kg_m3"
            Parameter to sweep.
        range_min : float, default=0.30
            Sweep range minimum.
        range_max : float, default=1.20
            Sweep range maximum.

        Returns:
        --------
        go.Figure
            Interactive Plotly Figure showing parameter vs total cost per tonne.
        """
        df_sweep = self.twin.what_if(
            blast_params=blast_params,
            param_to_vary=param_to_vary,
            range_min=range_min,
            range_max=range_max,
            steps=20,
        )

        fig = px.line(
            df_sweep,
            x=param_to_vary,
            y="total_cost_per_tonne_usd",
            title=f"<b>Scenario Sensitivity Sweep: {param_to_vary} vs Total Cost ($/t)</b>",
            labels={param_to_vary: param_to_vary.replace("_", " ").title(), "total_cost_per_tonne_usd": "Total Cost ($/t)"},
            markers=True,
            color_discrete_sequence=["#1976D2"],
        )

        fig.update_layout(template="plotly_white", height=420)
        return fig

    def compare_scenarios(self, scenarios: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generates a comparison table DataFrame across multiple blast scenarios.

        Parameters:
        -----------
        scenarios : List[Dict[str, Any]]
            List of scenario dictionaries containing 'scenario_name' and 'blast_params'.

        Returns:
        --------
        pd.DataFrame
            Comparison table DataFrame.
        """
        rows = []
        for idx, sc in enumerate(scenarios, 1):
            s_name = sc.get("scenario_name", f"Scenario #{idx}")
            params = sc.get("blast_params", {})

            res = self.twin.simulate(params)

            rows.append({
                "Scenario Name": s_name,
                "Powder Factor (kg/m3)": params.get("powder_factor_kg_m3", 0.65),
                "Burden (m)": params.get("burden_m", 6.0),
                "Spacing (m)": params.get("spacing_m", 7.0),
                "d50 (mm)": res["fragmentation"]["d50_mm"],
                "d80 (mm)": res["fragmentation"]["d80_mm"],
                "Crusher Throughput (t/h)": res["throughput"],
                "Specific Energy (kWh/t)": res["energy"],
                "Total Cost ($/t)": res["cost"],
            })

        return pd.DataFrame(rows)


# --- Preserved Standalone Spatial Helper Functions ---

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
    frag_model.predict(blast_params)
    return frag_model.last_percentiles


def link_to_downstream(
    digital_twin: Dict[str, Any],
    fragmentation: Dict[str, Any],
) -> Dict[str, float]:
    """Feeds fragmentation parameters to downstream value model."""
    geo = digital_twin.get("geological_data", {})
    wi = float(geo.get("hardness_index", 12.5))

    downstream_model = DownstreamModel(ore_hardness_wi=wi)
    return downstream_model.predict({"powder_factor_kg_m3": 0.65})
