"""
HIGH-RESOLUTION 3D BENCH DIGITAL TWIN
Renders mine bench with:
  - 160×100 hi-res terrain mesh with realistic rock texture
  - Professional 3D lighting (ambient + diffuse + specular)
  - Drill holes colored by charge weight
  - Geology overlays (Kimberlite, faults)
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

try:
    import yaml
except ImportError:
    yaml = None

# ============================================================================
# CONSTANTS
# ============================================================================
BG = "#0b1220"  # Dark background
CSCALE = "Turbo"  # Color scale for charge weight
TERRAIN_NX = 160  # Terrain grid resolution (x)
TERRAIN_NY = 100  # Terrain grid resolution (y)
BENCH_WIDTH = 60.0  # Bench width (m)
BENCH_LENGTH = 40.0  # Bench length (m)
BENCH_DEPTH = 12.0  # Standard hole depth (m)

# ============================================================================
# TERRAIN GENERATION
# ============================================================================


def terrain(
    nx: int = TERRAIN_NX,
    ny: int = TERRAIN_NY,
    width: float = BENCH_WIDTH,
    length: float = BENCH_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a realistic bench terrain surface.
    
    Features:
    - Bench top at z=0
    - Three 4-meter bench faces (stepped cut)
    - Fine noise texture for rock relief
    
    Returns:
      (X, Y, Z, noise) - 2D arrays of coordinates and elevation
    """
    x = np.linspace(0, width, nx)
    y = np.linspace(0, length, ny)
    X, Y = np.meshgrid(x, y)

    # Rock texture: smooth Gaussian noise
    rng = np.random.default_rng(seed=3)
    noise = rng.normal(0, 1, X.shape)
    for _ in range(3):
        noise = (
            noise
            + np.roll(noise, 1, 0)
            + np.roll(noise, -1, 0)
            + np.roll(noise, 1, 1)
            + np.roll(noise, -1, 1)
        ) / 5.0

    # Base surface + texture
    Z = 0.25 * noise

    # Bench faces: three 4m drops with smooth sigmoid transitions
    for k in range(3):
        face_pos = 44 + 4 * k
        Z -= 4.0 / (1 + np.exp(-(X - face_pos) * 3.0))

    return X, Y, Z, noise


# ============================================================================
# DEMO DATA
# ============================================================================


def demo_holes(
    rows: int = 8, per_row: int = 6, depth: float = BENCH_DEPTH
) -> List[Dict[str, Any]]:
    """Generate synthetic drill holes for demo/fallback."""
    rng = np.random.default_rng(seed=7)
    holes = []

    for r in range(rows):
        for h in range(per_row):
            x0, y0 = 10 + 5 * h, 4 + 4 * r
            dx, dy = rng.normal(0, 0.25, 2)  # As-drilled deviation
            charge = float(
                np.clip(
                    50
                    + 200
                    * (h / max(per_row - 1, 1))
                    * rng.uniform(0.6, 1.1),
                    50,
                    250,
                )
            )
            holes.append(
                {
                    "hole_id": f"R{r + 1}-H{h + 1}",
                    "x": float(x0),
                    "y": float(y0),
                    "z": 0.0,
                    "x_toe": float(x0 + dx),
                    "y_toe": float(y0 + dy),
                    "z_toe": float(-depth),
                    "charge_kg": charge,
                }
            )

    return holes


# Backward compat alias
synthetic_holes_for_demo = demo_holes


# ============================================================================
# LOADER FUNCTIONS
# ============================================================================


def load_holes_from_design(design: Any) -> List[Dict[str, Any]]:
    """Load drill holes from a design object (dict or class)."""
    if design is None:
        return demo_holes()

    # Extract parameters
    if isinstance(design, dict):
        holes = design.get("holes", [])
        num_rows = design.get("num_rows", 8)
        holes_per_row = design.get("holes_per_row", 6)
        burden = design.get("burden_m", 4.2)
        spacing = design.get("spacing_m", 5.1)
        depth = design.get("hole_depth_m", BENCH_DEPTH)
        pf = design.get("powder_factor_kg_m3", 0.65)
        bench_h = design.get("bench_height_m", BENCH_DEPTH)
    elif hasattr(design, "holes"):
        holes = getattr(design, "holes", [])
        num_rows = getattr(design, "num_rows", 8)
        holes_per_row = getattr(design, "holes_per_row", 6)
        burden = getattr(design, "burden_m", 4.2)
        spacing = getattr(design, "spacing_m", 5.1)
        depth = getattr(design, "hole_depth_m", BENCH_DEPTH)
        pf = getattr(design, "powder_factor_kg_m3", 0.65)
        bench_h = getattr(design, "bench_height_m", BENCH_DEPTH)
    else:
        return demo_holes()

    # Convert hole list
    if holes:
        result = []
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

            result.append(
                {
                    "hole_id": h_id,
                    "x": x,
                    "y": y,
                    "z": z,
                    "x_toe": x_toe,
                    "y_toe": y_toe,
                    "z_toe": z_toe,
                    "charge_kg": charge,
                }
            )
        return result

    # Generate from pattern
    result = []
    charge = pf * burden * spacing * bench_h
    for r in range(num_rows):
        for c in range(holes_per_row):
            x = c * spacing + 10.0
            y = r * burden + 4.0
            result.append(
                {
                    "hole_id": f"R{r + 1}-H{c + 1}",
                    "x": float(x),
                    "y": float(y),
                    "z": 0.0,
                    "x_toe": float(x),
                    "y_toe": float(y),
                    "z_toe": float(-depth),
                    "charge_kg": float(charge),
                }
            )
    return result


def load_geology_overlays(
    bench_id: str = "B14", site_id: str = "jwaneng"
) -> List[Dict[str, Any]]:
    """Load geology overlays from YAML or return defaults."""
    if yaml is None:
        return _default_overlays()

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

    return _default_overlays()


def _default_overlays() -> List[Dict[str, Any]]:
    """Default geology overlays."""
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


# ============================================================================
# 3D MESH BUILDER
# ============================================================================


def mesh_quad(pts: Any, color: str, name: str, opacity: float) -> go.Mesh3d:
    """Build a Mesh3d trace from 4 vertices."""
    pts = np.asarray(pts, dtype=float)
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


# ============================================================================
# MAIN FIGURE BUILDER
# ============================================================================


def make_bench_figure(
    holes: Optional[List[Dict[str, Any]]] = None,
    overlays: Optional[List[Dict[str, Any]]] = None,
    show_holes: bool = True,
    show_geology: bool = True,
    show_faults: bool = True,
    title: str = "Digital Twin of Bench",
) -> go.Figure:
    """
    Build the interactive 3D bench figure.
    
    Args:
      holes: List of hole dicts with keys: hole_id, x, y, z, x_toe, y_toe, z_toe, charge_kg
      overlays: List of geology overlay dicts with: name, color, opacity, type, vertices
      show_holes: Toggle drill holes
      show_geology: Toggle geology overlays
      show_faults: Toggle fault planes
      title: Figure title
    
    Returns:
      go.Figure with 3D scene
    """
    holes = holes if holes is not None else demo_holes()
    overlays = overlays if overlays is not None else load_geology_overlays()

    # Generate terrain
    X, Y, Z, noise = terrain(nx=TERRAIN_NX, ny=TERRAIN_NY)
    fig = go.Figure()

    # ========================================================================
    # 1. TERRAIN SURFACE
    # ========================================================================
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=Z + 0.5 * noise,  # Elevation + texture
            showscale=False,
            hoverinfo="skip",
            colorscale=[
                [0, "#3b2c20"],  # Dark brown (deep)
                [0.45, "#7a5a3a"],  # Mid brown
                [0.8, "#b48a58"],  # Lighter brown
                [1, "#d2ab74"],  # Tan (high)
            ],
            lighting=dict(
                ambient=0.55,
                diffuse=0.9,
                specular=0.15,
                roughness=0.9,
                fresnel=0.1,
            ),
            lightposition=dict(x=-200, y=-300, z=400),
        )
    )

    # ========================================================================
    # 2. GEOLOGY OVERLAYS
    # ========================================================================
    for ov in overlays:
        if ov is None:
            continue

        is_fault = (
            "fault" in str(ov.get("name", "")).lower()
            or ov.get("type") == "fault"
        )
        if is_fault and not show_faults:
            continue
        if not is_fault and not show_geology:
            continue

        vertices = ov.get("vertices")
        if not vertices:
            continue

        fig.add_trace(
            mesh_quad(
                vertices,
                ov.get("color", "#888888"),
                ov.get("name", "Overlay"),
                ov.get("opacity", 0.3),
            )
        )

    # ========================================================================
    # 3. DRILL HOLES
    # ========================================================================
    if show_holes and holes:
        charges = [h["charge_kg"] for h in holes]
        cmin, cmax = min(charges), max(charges)
        if cmax == cmin:
            cmax = cmin + 1.0

        # Draw collar-to-toe lines
        for h in holes:
            norm_charge = (h["charge_kg"] - cmin) / (cmax - cmin)
            c = sample_colorscale(CSCALE, [norm_charge])[0]
            fig.add_trace(
                go.Scatter3d(
                    x=[h["x"], h["x_toe"]],
                    y=[h["y"], h["y_toe"]],
                    z=[h["z"], h["z_toe"]],
                    mode="lines",
                    line=dict(color=c, width=7),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Add collar markers with colorbar
        fig.add_trace(
            go.Scatter3d(
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
                        title=dict(
                            text="Charge Weight (kg)",
                            font=dict(color="white"),
                        ),
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
            )
        )

    # ========================================================================
    # 4. LAYOUT & CAMERA
    # ========================================================================
    title_text = f"{title} - {len(holes)} holes" if show_holes else title

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(color="white", size=18)),
        paper_bgcolor=BG,
        margin=dict(l=0, r=0, t=50, b=0),
        height=640,
        legend=dict(
            x=0.01,
            y=0.97,
            bgcolor="rgba(11,18,32,0.8)",
            font=dict(color="white"),
        ),
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
