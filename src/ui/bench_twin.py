"""
Improved 3D bench view for Streamlit + Plotly.

This version keeps the existing API surface used by the app while upgrading the
visual quality substantially: higher-res terrain, better lighting, more realistic
rock relief, and clearer hole coloring. It remains dependency-free and works in
Streamlit as-is.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.colors import sample_colorscale

BG = "#0b1220"
CSCALE = "Turbo"


def terrain(nx: int = 160, ny: int = 100, width: float = 60.0, length: float = 40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bench top (z=0) with stepped faces and fine noise for rock texture."""
    x = np.linspace(0, width, nx)
    y = np.linspace(0, length, ny)
    X, Y = np.meshgrid(x, y)
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, X.shape)
    for _ in range(3):
        noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
                 + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5
    Z = 0.25 * noise
    for k in range(3):
        Z -= 4.0 / (1 + np.exp(-(X - (44 + 4 * k)) * 3.0))
    return X, Y, Z, noise


def terrain_from_survey(survey_data: np.ndarray, nx: int = 100, ny: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate terrain grid from an array of (x, y, z) survey points."""
    try:
        from scipy.interpolate import griddata
    except Exception:
        return terrain(nx=nx, ny=ny)

    x_min, x_max = survey_data[:, 0].min(), survey_data[:, 0].max()
    y_min, y_max = survey_data[:, 1].min(), survey_data[:, 1].max()

    grid_x, grid_y = np.mgrid[x_min:x_max:complex(0, nx), y_min:y_max:complex(0, ny)]
    grid_z = griddata(survey_data[:, :2], survey_data[:, 2], (grid_x, grid_y), method="cubic", fill_value=0.0)
    noise = np.zeros_like(grid_z)
    return grid_x, grid_y, grid_z, noise


def demo_holes(rows: int = 8, per_row: int = 6, depth: float = 12.0) -> List[Dict[str, Any]]:
    """Generate explicit synthetic blastholes for demonstration fallback."""
    rng = np.random.default_rng(7)
    recs = []
    for r in range(rows):
        for h in range(per_row):
            x0, y0 = 10 + 5 * h, 4 + 4 * r
            dx, dy = rng.normal(0, 0.25, 2)
            charge = float(np.clip(50 + 200 * (h / max(per_row - 1, 1)) * rng.uniform(0.6, 1.1), 50, 250))
            recs.append({
                "hole_id": f"R{r + 1}-H{h + 1}",
                "x": float(x0),
                "y": float(y0),
                "z": 0.0,
                "x_toe": float(x0 + dx),
                "y_toe": float(y0 + dy),
                "z_toe": float(-depth),
                "charge_kg": charge,
            })
    return recs


# Backward-compatible alias used elsewhere in the app.
synthetic_holes_for_demo = demo_holes


def mesh_quad(pts: Any, color: str, name: str, opacity: float) -> go.Mesh3d:
    """Build a 3D Mesh3d trace from 4 corner vertices."""
    pts = np.asarray(pts, float)
    return go.Mesh3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        i=[0, 1],
        j=[1, 3],
        k=[2, 2],
        color=color,
        opacity=opacity,
        name=name,
        showlegend=True,
        hoverinfo="name",
    )


def load_holes_from_design(design: Any) -> List[Dict[str, Any]]:
    """Convert a design object or dict into a list of hole dictionaries."""
    if design is None:
        return demo_holes()

    if isinstance(design, dict):
        holes = design.get("holes", [])
        num_rows = design.get("num_rows", 8)
        holes_per_row = design.get("holes_per_row", 6)
        burden = design.get("burden_m", 4.2)
        spacing = design.get("spacing_m", 5.1)
        depth = design.get("hole_depth_m", 12.0)
        pf = design.get("powder_factor_kg_m3", 0.65)
        bench_h = design.get("bench_height_m", 12.0)
    elif hasattr(design, "holes"):
        holes = getattr(design, "holes", [])
        num_rows = getattr(design, "num_rows", 8)
        holes_per_row = getattr(design, "holes_per_row", 6)
        burden = getattr(design, "burden_m", 4.2)
        spacing = getattr(design, "spacing_m", 5.1)
        depth = getattr(design, "hole_depth_m", 12.0)
        pf = getattr(design, "powder_factor_kg_m3", 0.65)
        bench_h = getattr(design, "bench_height_m", 12.0)
    else:
        return demo_holes()

    if holes:
        recs = []
        for h in holes:
            if isinstance(h, dict):
                h_id = h.get("hole_id", "H-01")
                x = float(h.get("x_m", h.get("x", 0.0)))
                y = float(h.get("y_m", h.get("y", 0.0)))
                z = float(h.get("z_collar_m", h.get("z", 0.0)))
                x_toe = float(h.get("x_toe_m", h.get("x_toe", x)))
                y_toe = float(h.get("y_toe_m", h.get("y_toe", y)))
                z_toe = float(h.get("z_toe_m", h.get("z_toe", -depth)))
                charge = float(h.get("charge_kg", h.get("charge_mass_kg", 150.0)))
            else:
                h_id = getattr(h, "hole_id", "H-01")
                x = float(getattr(h, "x_m", 0.0))
                y = float(getattr(h, "y_m", 0.0))
                z = float(getattr(h, "z_collar_m", 0.0))
                x_toe = float(getattr(h, "x_toe", x))
                y_toe = float(getattr(h, "y_toe", y))
                z_toe = float(getattr(h, "z_toe_m", -depth))
                charge = float(getattr(h, "charge_kg", 150.0))

            recs.append({
                "hole_id": h_id,
                "x": x,
                "y": y,
                "z": z,
                "x_toe": x_toe,
                "y_toe": y_toe,
                "z_toe": z_toe,
                "charge_kg": charge,
            })
        return recs

    # Generate from pattern geometry if holes list is empty
    recs = []
    charge = pf * burden * spacing * bench_h
    for r in range(num_rows):
        for c in range(holes_per_row):
            x = c * spacing + 10.0
            y = r * burden + 4.0
            recs.append({
                "hole_id": f"R{r + 1}-H{c + 1}",
                "x": float(x),
                "y": float(y),
                "z": 0.0,
                "x_toe": float(x),
                "y_toe": float(y),
                "z_toe": float(-depth),
                "charge_kg": float(charge),
            })
    return recs


def load_geology_overlays(bench_id: str = "B14", site_id: str = "jwaneng") -> List[Dict[str, Any]]:
    """Read geological block model overlays from config/benches YAML or fallback."""
    yaml_path = f"config/benches/{site_id}_bench_{bench_id.replace('BENCH_', '').replace('B', '')}.yaml"

    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                overlays = cfg.get("geology_overlays", [])
                if overlays:
                    return overlays
        except Exception:
            pass

    return [
        {
            "name": "Kimberlite zone",
            "color": "#8e44ad",
            "opacity": 0.40,
            "type": "geology",
            "vertices": [(0, 0, 0.1), (22, 0, 0.1), (0, 40, 0.1), (22, 40, 0.1)],
        },
        {
            "name": "Fault plane",
            "color": "#d93a3a",
            "opacity": 0.35,
            "type": "fault",
            "vertices": [(26, 0, 0.1), (40, 40, 0.1), (32, 0, -14), (46, 40, -14)],
        },
    ]


def make_bench_figure(
    holes: Optional[List[Dict[str, Any]]] = None,
    overlays: Optional[List[Dict[str, Any]]] = None,
    show_holes: bool = True,
    show_geology: bool = True,
    show_faults: bool = True,
    title: str = "Digital Twin of Bench",
) -> go.Figure:
    """Return an interactive Plotly Figure for the 3D bench digital twin."""
    holes = holes if holes is not None else demo_holes()
    overlays = overlays if overlays is not None else load_geology_overlays()

    X, Y, Z, noise = terrain(nx=160, ny=100, width=60.0, length=40.0)
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        surfacecolor=Z + 0.5 * noise,
        showscale=False,
        hoverinfo="skip",
        colorscale=[[0, "#3b2c20"], [0.45, "#7a5a3a"], [0.8, "#b48a58"], [1, "#d2ab74"]],
        lighting=dict(ambient=0.55, diffuse=0.9, specular=0.15, roughness=0.9, fresnel=0.1),
        lightposition=dict(x=-200, y=-300, z=400),
    ))

    for ov in overlays:
        if ov is None:
            continue
        is_fault = "fault" in str(ov.get("name", "")).lower() or ov.get("type") == "fault"
        if is_fault and not show_faults:
            continue
        if not is_fault and not show_geology:
            continue

        vertices = ov.get("vertices")
        if not vertices:
            continue
        fig.add_trace(mesh_quad(vertices, ov["color"], ov["name"], ov["opacity"]))

    if show_holes and holes:
        charges = [h["charge_kg"] for h in holes]
        cmin, cmax = min(charges), max(charges)
        if cmax == cmin:
            cmax = cmin + 1.0

        for h in holes:
            c = sample_colorscale(CSCALE, [float((h["charge_kg"] - cmin) / (cmax - cmin))])[0]
            fig.add_trace(go.Scatter3d(
                x=[h["x"], h["x_toe"]],
                y=[h["y"], h["y_toe"]],
                z=[h["z"], h["z_toe"]],
                mode="lines",
                line=dict(color=c, width=7),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter3d(
            x=[h["x"] for h in holes],
            y=[h["y"] for h in holes],
            z=[h["z"] + 0.3 for h in holes],
            mode="markers",
            showlegend=False,
            marker=dict(
                size=5,
                color=[h["charge_kg"] for h in holes],
                colorscale=CSCALE,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title=dict(text="Charge Weight (kg)", font=dict(color="white")),
                    tickfont=dict(color="white"),
                    orientation="h",
                    len=0.3,
                    x=0.02,
                    xanchor="left",
                    y=0.04,
                    thickness=12,
                ),
            ),
            text=[h["hole_id"] for h in holes],
            customdata=[h["charge_kg"] for h in holes],
            hovertemplate="%{text}<br>Charge: %{customdata:.0f} kg<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"{title} - {len(holes)} holes" if show_holes else title,
            x=0.5,
            font=dict(color="white", size=18),
        ),
        paper_bgcolor=BG,
        margin=dict(l=0, r=0, t=50, b=0),
        height=640,
        legend=dict(x=0.01, y=0.97, bgcolor="rgba(11,18,32,0.8)", font=dict(color="white")),
        scene=dict(
            bgcolor=BG,
            aspectmode="data",
            camera=dict(eye=dict(x=1.1, y=-1.5, z=0.8)),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )

    return fig
