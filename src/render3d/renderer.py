"""
Main PyVista / Plotly 3D blast renderer for BlasterOPT 3D engine.
Combines bench geometry, drill holes, free faces, backbreak zone, toe burden, and timing overlays.
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from src.render3d.bench_model import BenchModel
from src.render3d.hole_pattern import HolePattern, Hole
from src.render3d.free_face import identify_free_faces


class BlastRenderer3D:
    """
    3D renderer unifying Plotly rendering for BlasterOPT 3D engine.
    """

    def __init__(self, bench: BenchModel, pattern: Optional[HolePattern] = None) -> None:
        self.bench = bench
        self.pattern = pattern

    def render_plotly(self) -> go.Figure:
        """
        Render interactive 3D Plotly figure.
        """
        fig = go.Figure()

        # Bench face surface
        face_pts = self.bench.get_face_surface(resolution=30)
        fig.add_trace(
            go.Scatter3d(
                x=face_pts[:, 0],
                y=face_pts[:, 1],
                z=face_pts[:, 2],
                mode="markers",
                marker=dict(size=3, color="saddlebrown", opacity=0.5),
                name="Bench Face Mesh",
            )
        )

        # Crest and Toe lines
        crest = self.bench.get_crest_line()
        toe = self.bench.get_toe_line()

        fig.add_trace(
            go.Scatter3d(
                x=crest[:, 0], y=crest[:, 1], z=crest[:, 2],
                mode="lines",
                line=dict(color="red", width=5),
                name="Crest Edge",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=toe[:, 0], y=toe[:, 1], z=toe[:, 2],
                mode="lines",
                line=dict(color="blue", width=5),
                name="Toe Edge",
            )
        )

        # Holes
        if self.pattern is not None:
            holes = self.pattern.generate_holes(n_rows=3, n_per_row=4)
            collars = np.array([[h.x, h.y, h.z] for h in holes])
            toes = np.array([h.get_toe_coordinate() for h in holes])

            fig.add_trace(
                go.Scatter3d(
                    x=collars[:, 0], y=collars[:, 1], z=collars[:, 2],
                    mode="markers",
                    marker=dict(size=8, color="green"),
                    name="Hole Collars",
                )
            )

            for c, t in zip(collars, toes):
                fig.add_trace(
                    go.Scatter3d(
                        x=[c[0], t[0]], y=[c[1], t[1]], z=[c[2], t[2]],
                        mode="lines",
                        line=dict(color="black", width=3),
                        showlegend=False,
                    )
                )

        fig.update_layout(
            title=f"3D Bench Model - {self.bench.bench_id}",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Elevation Z (m)",
            ),
        )

        return fig


class Blast3DRenderer:
    """
    Main renderer that combines bench, holes, timing, and analysis overlays.
    Uses PyVista for 3D rendering with Plotly fallback.
    """

    def __init__(self, bench: BenchModel, holes: List[Hole], timing: dict) -> None:
        self.bench = bench
        self.holes = holes
        self.timing = timing
        self.plotter = None

    def _init_pyvista_plotter(self):
        """Lazy initialize PyVista plotter if pyvista is available."""
        if self.plotter is None:
            try:
                import pyvista as pv
                self.plotter = pv.Plotter(off_screen=True)
            except Exception:
                self.plotter = None
        return self.plotter

    def render_bench(self) -> None:
        """Render bench surface, crest, and toe lines."""
        plotter = self._init_pyvista_plotter()
        if plotter is not None:
            import pyvista as pv
            face_pts = self.bench.get_face_surface(resolution=30)
            mesh = pv.PolyData(face_pts)
            plotter.add_mesh(mesh, color="saddlebrown", opacity=0.6, label="Bench Face")

            crest_pts = self.bench.get_crest_line()
            crest_line = pv.PolyData(crest_pts)
            plotter.add_mesh(crest_line, color="red", line_width=4, label="Crest")

            toe_pts = self.bench.get_toe_line()
            toe_line = pv.PolyData(toe_pts)
            plotter.add_mesh(toe_line, color="blue", line_width=4, label="Toe")

    def render_holes(self) -> None:
        """Render drill holes as cylinders or lines."""
        plotter = self._init_pyvista_plotter()
        if plotter is not None and self.holes:
            import pyvista as pv
            for h in self.holes:
                toe = h.get_toe_coordinate()
                line = pv.Line((h.x, h.y, h.z), toe)
                plotter.add_mesh(line, color="green", line_width=5)

    def render_free_faces(self) -> None:
        """Render detected free faces."""
        plotter = self._init_pyvista_plotter()
        faces = identify_free_faces(self.bench)
        if plotter is not None and faces:
            import pyvista as pv
            for f in faces:
                centroid = f["centroid"]
                point = pv.PolyData([centroid])
                plotter.add_mesh(point, color="yellow", point_size=15, render_points_as_spheres=True)

    def render_backbreak_zone(self, backbreak_m: float) -> None:
        """Render predicted backbreak zone surface behind back row."""
        plotter = self._init_pyvista_plotter()
        if plotter is not None and backbreak_m > 0:
            import pyvista as pv
            crest = self.bench.get_crest_line()
            backbreak_pts = crest.copy()
            backbreak_pts[:, 1] += backbreak_m  # Offset behind back row
            bb_mesh = pv.PolyData(backbreak_pts)
            plotter.add_mesh(bb_mesh, color="darkred", opacity=0.3, label="Backbreak Zone")

    def render_toe_burden(self, toe_burden_m: float) -> None:
        """Render toe burden volume at floor level."""
        plotter = self._init_pyvista_plotter()
        if plotter is not None and toe_burden_m > 0:
            import pyvista as pv
            toe = self.bench.get_toe_line()
            toe_mesh = pv.PolyData(toe)
            plotter.add_mesh(toe_mesh, color="purple", opacity=0.4, label="Toe Burden")

    def render_timing(self, time_ms: float) -> None:
        """Render timing state at specific timestamp time_ms."""
        plotter = self._init_pyvista_plotter()
        if plotter is not None and self.holes:
            import pyvista as pv
            delays_map = self.timing.get("delays_ms", {})
            for h in self.holes:
                delay = delays_map.get(h.hole_id, 0.0)
                color = "gray" if time_ms >= delay else ("red" if abs(time_ms - delay) <= 20 else "blue")
                point = pv.PolyData([(h.x, h.y, h.z)])
                plotter.add_mesh(point, color=color, point_size=12, render_points_as_spheres=True)

    def render_all(self) -> Any:
        """
        Build and return complete PyVista plotter or Plotly figure fallback.
        """
        plotter = self._init_pyvista_plotter()
        if plotter is not None:
            self.render_bench()
            self.render_holes()
            self.render_free_faces()
            self.render_backbreak_zone(backbreak_m=2.0)
            self.render_toe_burden(toe_burden_m=6.0)
            return plotter

        # Fallback Plotly figure
        fig = go.Figure()
        face_pts = self.bench.get_face_surface(resolution=20)
        fig.add_trace(
            go.Scatter3d(
                x=face_pts[:, 0], y=face_pts[:, 1], z=face_pts[:, 2],
                mode="markers", marker=dict(size=3, color="saddlebrown", opacity=0.5),
                name="Bench Surface",
            )
        )
        if self.holes:
            fig.add_trace(
                go.Scatter3d(
                    x=[h.x for h in self.holes],
                    y=[h.y for h in self.holes],
                    z=[h.z for h in self.holes],
                    mode="markers", marker=dict(size=8, color="green"),
                    name="Drill Holes",
                )
            )
        fig.update_layout(title=f"3D Blast Renderer - {self.bench.bench_id}")
        return fig
