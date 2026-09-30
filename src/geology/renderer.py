"""
3D visualization of geological data on the bench for BlasterOPT.
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
try:
    from src.render3d.bench_model import BenchModel
except Exception:
    class BenchModel:
        def __init__(self, bench_id: str = "B1", crest_elevation_m: float = 100.0, toe_elevation_m: float = 85.0, face_angle_deg: float = 75.0) -> None:
            self.bench_id = bench_id
            self.crest_elevation_m = crest_elevation_m
            self.toe_elevation_m = toe_elevation_m
            self.face_angle_deg = face_angle_deg
        def get_face_surface(self, resolution: int = 20) -> np.ndarray:
            return np.zeros((resolution, 3))
from src.geology.structures import StructuralModel
from src.geology.ore_body import OreBodyModel


class GeologyRenderer3D:
    """3D Renderer for geological structures, joint sets, and domains."""

    def __init__(self, joint_sets: Optional[List[dict]] = None, structures: Optional[StructuralModel] = None) -> None:
        self.joint_sets = joint_sets or []
        self.structures = structures

    def render_plotly(self) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(title="3D Geological Structural Discontinuities")
        return fig


class GeologyRenderer:
    """
    3D visualization of geological data on the bench.
    """

    def __init__(self, plotter: Any = None) -> None:
        self.plotter = plotter

    def _init_pyvista_plotter(self):
        if self.plotter is None:
            try:
                import pyvista as pv
                self.plotter = pv.Plotter(off_screen=True)
            except Exception:
                self.plotter = None
        return self.plotter

    def render_bench(self, bench: BenchModel) -> None:
        plotter = self._init_pyvista_plotter()
        if plotter is not None:
            import pyvista as pv
            face_pts = bench.get_face_surface(resolution=30)
            mesh = pv.PolyData(face_pts)
            plotter.add_mesh(mesh, color="saddlebrown", opacity=0.5, label="Bench Surface")

    def render_joint_sets(self, joint_sets: List[dict]) -> None:
        plotter = self._init_pyvista_plotter()
        if plotter is not None and joint_sets:
            import pyvista as pv
            for js in joint_sets:
                dip = js.get("mean_dip_deg", 60.0)
                dd = js.get("mean_dip_direction_deg", 180.0)
                rad_dip = np.radians(dip)
                rad_dd = np.radians(dd)
                normal = np.array([np.sin(rad_dip) * np.sin(rad_dd), np.sin(rad_dip) * np.cos(rad_dd), -np.cos(rad_dip)])
                line = pv.Line((0, 0, 100), (normal[0] * 5, normal[1] * 5, 100 + normal[2] * 5))
                plotter.add_mesh(line, color="cyan", line_width=4)

    def render_structures(self, structures: StructuralModel) -> None:
        plotter = self._init_pyvista_plotter()
        if plotter is not None and structures:
            import pyvista as pv
            for f in structures.faults:
                poly = f["polyline"]
                if len(poly) > 0:
                    mesh = pv.PolyData(poly)
                    plotter.add_mesh(mesh, color="red", line_width=5, label=f"Fault {f['fault_id']}")
            for d in structures.dykes:
                poly = d["polyline"]
                if len(poly) > 0:
                    mesh = pv.PolyData(poly)
                    plotter.add_mesh(mesh, color="black", line_width=6, label=f"Dyke {d['dyke_id']}")

    def render_ore_boundary(self, ore_body: OreBodyModel) -> None:
        plotter = self._init_pyvista_plotter()
        if plotter is not None and ore_body is not None:
            import pyvista as pv
            df = ore_body.classify_ore_waste()
            ore_pts = df[df["material_type"] == "ore"][["x", "y", "z"]].values
            if len(ore_pts) > 0:
                cloud = pv.PolyData(ore_pts)
                plotter.add_mesh(cloud, color="gold", point_size=10, render_points_as_spheres=True, label="Ore Body")

    def render_rmr_zones(self, rmr_map: dict) -> None:
        plotter = self._init_pyvista_plotter()
        if plotter is not None and rmr_map:
            import pyvista as pv
            coords = np.array(rmr_map.get("coordinates", [[0, 0, 100]]))
            scores = np.array(rmr_map.get("rmr_scores", [50]))
            cloud = pv.PolyData(coords)
            cloud["rmr"] = scores
            plotter.add_mesh(cloud, scalars="rmr", cmap="RdYlGn", point_size=12, render_points_as_spheres=True)

    def render_dilution_zone(self, dilution: dict) -> None:
        plotter = self._init_pyvista_plotter()
        if plotter is not None and dilution.get("total_dilution_pct", 0) > 0:
            import pyvista as pv
            box = pv.Box(bounds=(-5, 5, -5, 5, 85, 100))
            plotter.add_mesh(box, color="magenta", opacity=0.3, label="Dilution Zone")

    def render_all(self, bench: Optional[BenchModel] = None) -> Any:
        plotter = self._init_pyvista_plotter()
        if plotter is not None:
            if bench:
                self.render_bench(bench)
            return plotter

        fig = go.Figure()
        if bench:
            face_pts = bench.get_face_surface(resolution=20)
            fig.add_trace(
                go.Scatter3d(
                    x=face_pts[:, 0], y=face_pts[:, 1], z=face_pts[:, 2],
                    mode="markers", marker=dict(size=3, color="saddlebrown", opacity=0.5),
                    name="Bench Face",
                )
            )
        fig.update_layout(title="3D Geological Visualization Engine")
        return fig
