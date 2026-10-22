"""
3D bench view for Streamlit + Plotly (professional mining-engineering digital twin renderer).
"""
import os
import yaml
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from typing import List, Dict, Any, Optional, Tuple

BG = "#0b1220"
CSCALE = "Turbo"


def terrain(
    nx: int = 120,
    ny: int = 80,
    width: float = 40.0,
    length: float = 60.0,
    height: float = 15.0,
    vertical_exaggeration: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate parametric 3D bench surface with:
    - Uneven, irregular top surface (procedural noise).
    - Three clearly visible stepped drops (approx 3.5m - 5m each depending on bench height).
    - Thick, near-vertical exposed rock faces with horizontal geological strata.
    - Visible crest, toe, and bench floor at z=0.
    """
    x = np.linspace(0, length, nx)
    y = np.linspace(0, width, ny)
    X, Y = np.meshgrid(x, y)

    # Procedural terrain noise for top surface roughness
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, X.shape)
    for _ in range(3):
        noise = (
            noise
            + np.roll(noise, 1, 0)
            + np.roll(noise, -1, 0)
            + np.roll(noise, 1, 1)
            + np.roll(noise, -1, 1)
        ) / 5.0

    # Base elevation starts at full bench height plus top surface noise
    Z = height + 0.20 * noise

    # Three stepped drops along the high-X side
    step_height = height / 3.0
    x_step1 = 0.52 * length
    x_step2 = 0.68 * length
    x_step3 = 0.84 * length

    # Sigmoid smooth step drops for steep exposed rock faces
    drop1 = step_height / (1.0 + np.exp(-0.8 * (X - x_step1)))
    drop2 = step_height / (1.0 + np.exp(-0.8 * (X - x_step2)))
    drop3 = step_height / (1.0 + np.exp(-0.8 * (X - x_step3)))

    Z = Z - drop1 - drop2 - drop3
    Z = np.clip(Z, 0.0, None)

    # Apply vertical exaggeration scaling to Z coordinates
    Z_scaled = Z * vertical_exaggeration

    return X, Y, Z_scaled, noise


def generate_3d_holes(
    rows: int = 8,
    per_row: int = 6,
    burden: float = 4.2,
    spacing: float = 5.1,
    diameter_mm: float = 165.0,
    depth: float = 12.0,
    bench_length: float = 60.0,
    bench_width: float = 40.0,
    bench_height: float = 15.0,
    vertical_exaggeration: float = 1.0,
    powder_factor: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    Generate 3D drill hole records anchored to the terrain surface.
    """
    recs = []
    charge_per_hole = powder_factor * burden * spacing * depth
    rng = np.random.default_rng(7)

    start_x = 8.0
    start_y = 5.0

    for r in range(rows):
        for c in range(per_row):
            x = start_x + r * burden
            y = start_y + c * spacing
            if x >= bench_length * 0.50:
                # Keep holes on the top bench surface before stepped faces
                x = 8.0 + (r % 4) * burden

            z_collar = 0.0
            z_toe = -depth * vertical_exaggeration

            # Slight random tilt for realistic drilling deviation
            dx = float(rng.normal(0, 0.08))
            dy = float(rng.normal(0, 0.08))

            charge_kg = float(
                np.clip(charge_per_hole * rng.uniform(0.85, 1.15), 50.0, 350.0)
            )

            recs.append(
                {
                    "hole_id": f"R{r + 1}-H{c + 1}",
                    "x": float(x),
                    "y": float(y),
                    "z": float(z_collar),
                    "x_toe": float(x + dx),
                    "y_toe": float(y + dy),
                    "z_toe": float(z_toe),
                    "diameter_mm": float(diameter_mm),
                    "depth_m": float(depth),
                    "charge_kg": charge_kg,
                }
            )

    return recs


def demo_holes(
    rows: int = 8, per_row: int = 6, depth: float = 12.0
) -> List[Dict[str, Any]]:
    """Generate synthetic drill holes for demo mode."""
    return generate_3d_holes(
        rows=rows,
        per_row=per_row,
        depth=depth,
    )


def synthetic_holes_for_demo(
    rows: int = 8, per_row: int = 6, depth: float = 12.0
) -> List[Dict[str, Any]]:
    """Alias for demo_holes."""
    return demo_holes(rows=rows, per_row=per_row, depth=depth)


def load_holes_from_design(
    design: Any,
    bench_length: float = 60.0,
    bench_width: float = 40.0,
    bench_height: float = 15.0,
    vertical_exaggeration: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Convert design object or dict into a list of 3D hole dictionaries.
    """
    if design is None:
        return generate_3d_holes(
            bench_length=bench_length,
            bench_width=bench_width,
            bench_height=bench_height,
            vertical_exaggeration=vertical_exaggeration,
        )

    if isinstance(design, dict):
        holes = design.get("holes", [])
        rows = design.get("num_rows", design.get("rows", 8))
        per_row = design.get("holes_per_row", design.get("cols", 6))
        burden = design.get("burden_m", 4.2)
        spacing = design.get("spacing_m", 5.1)
        diameter = design.get("hole_diameter_mm", 165.0)
        depth = design.get("hole_depth_m", 12.0)
        pf = design.get("powder_factor_kg_m3", 0.65)
    elif hasattr(design, "holes"):
        holes = getattr(design, "holes", [])
        rows = getattr(design, "num_rows", getattr(design, "rows", 8))
        per_row = getattr(design, "holes_per_row", getattr(design, "cols", 6))
        burden = getattr(design, "burden_m", 4.2)
        spacing = getattr(design, "spacing_m", 5.1)
        diameter = getattr(design, "hole_diameter_mm", 165.0)
        depth = getattr(design, "hole_depth_m", 12.0)
        pf = getattr(design, "powder_factor_kg_m3", 0.65)
    else:
        return generate_3d_holes(
            bench_length=bench_length,
            bench_width=bench_width,
            bench_height=bench_height,
            vertical_exaggeration=vertical_exaggeration,
        )

    if holes:
        recs = []
        for h in holes:
            if isinstance(h, dict):
                h_id = h.get("hole_id", "H-01")
                x = float(h.get("x_m", h.get("x", 10.0)))
                y = float(h.get("y_m", h.get("y", 5.0)))
                z = float(h.get("z_collar_m", h.get("z", 0.0)))
                x_toe = float(h.get("x_toe_m", h.get("x_toe", x)))
                y_toe = float(h.get("y_toe_m", h.get("y_toe", y)))
                z_toe = float(h.get("z_toe_m", h.get("z_toe", z - depth))) * vertical_exaggeration
                charge = float(h.get("charge_kg", h.get("charge_mass_kg", 150.0)))
                diam = float(h.get("diameter_mm", diameter))
            else:
                h_id = getattr(h, "hole_id", "H-01")
                x = float(getattr(h, "x_m", 10.0))
                y = float(getattr(h, "y_m", 5.0))
                z = float(getattr(h, "z_collar_m", 0.0))
                x_toe = float(getattr(h, "x_toe", x))
                y_toe = float(getattr(h, "y_toe", y))
                z_toe = float(getattr(h, "z_toe_m", z - depth)) * vertical_exaggeration
                charge = float(getattr(h, "charge_kg", 150.0))
                diam = float(getattr(h, "diameter_mm", diameter))

            recs.append(
                {
                    "hole_id": h_id,
                    "x": x,
                    "y": y,
                    "z": z,
                    "x_toe": x_toe,
                    "y_toe": y_toe,
                    "z_toe": z_toe,
                    "diameter_mm": diam,
                    "depth_m": depth,
                    "charge_kg": charge,
                }
            )
        return recs

    return generate_3d_holes(
        rows=rows,
        per_row=per_row,
        burden=burden,
        spacing=spacing,
        diameter_mm=diameter,
        depth=depth,
        bench_length=bench_length,
        bench_width=bench_width,
        bench_height=bench_height,
        vertical_exaggeration=vertical_exaggeration,
        powder_factor=pf,
    )


def load_geology_overlays(
    bench_id: str = "B14",
    site_id: str = "jwaneng",
    bench_length: float = 60.0,
    bench_width: float = 40.0,
    bench_height: float = 15.0,
    vertical_exaggeration: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Get 3D geological zone overlays with exact color and opacity specifications:
    - Kimberlite: Purple (#8e44ad), default opacity 0.60
    - Dolomite waste: Grey (#7f8c8d), default opacity 0.45
    - Granite contact: Orange (#e67e22), default opacity 0.55
    """
    ve = vertical_exaggeration
    h = bench_height

    return [
        {
            "name": "Kimberlite zone",
            "color": "#8e44ad",
            "opacity": 0.60,
            "type": "geology",
            "vertices": [
                (0.0, 0.0, 0.1 * ve),
                (0.40 * bench_length, 0.0, 0.1 * ve),
                (0.0, bench_width, (h * 0.85) * ve),
                (0.40 * bench_length, bench_width, (h * 0.85) * ve),
            ],
        },
        {
            "name": "Dolomite waste",
            "color": "#7f8c8d",
            "opacity": 0.45,
            "type": "geology",
            "vertices": [
                (0.40 * bench_length, 0.0, (h * 0.20) * ve),
                (0.70 * bench_length, 0.0, (h * 0.20) * ve),
                (0.40 * bench_length, bench_width, (h * 0.95) * ve),
                (0.70 * bench_length, bench_width, (h * 0.95) * ve),
            ],
        },
        {
            "name": "Granite contact",
            "color": "#e67e22",
            "opacity": 0.55,
            "type": "geology",
            "vertices": [
                (0.70 * bench_length, 0.0, (h * 0.05) * ve),
                (1.0 * bench_length, 0.0, (h * 0.05) * ve),
                (0.70 * bench_length, bench_width, (h * 0.75) * ve),
                (1.0 * bench_length, bench_width, (h * 0.75) * ve),
            ],
        },
        {
            "name": "FAULT_JWA_01",
            "color": "#d93a3a",
            "opacity": 0.35,
            "type": "fault",
            "vertices": [
                (0.48 * bench_length, 0.0, (h * 1.02) * ve),
                (0.58 * bench_length, bench_width, (h * 1.02) * ve),
                (0.55 * bench_length, 0.0, 0.0),
                (0.65 * bench_length, bench_width, 0.0),
            ],
        },
    ]


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


def make_bench_figure(
    holes: Optional[List[Dict[str, Any]]] = None,
    overlays: Optional[List[Dict[str, Any]]] = None,
    show_holes: bool = True,
    show_geology: bool = True,
    show_faults: bool = True,
    bench_length: float = 60.0,
    bench_width: float = 40.0,
    bench_height: float = 15.0,
    vertical_exaggeration: float = 1.0,
    title: str = "Digital Twin of Bench",
) -> go.Figure:
    """
    Return interactive 3D Plotly Figure for the Mining Digital Twin with:
    - Stepped 3D bench terrain surface.
    - 3D drill hole shafts with circular collar rims and charge weight colormaps.
    - 3D geological zone overlays (Kimberlite purple 0.60, Dolomite grey 0.45, Granite orange 0.55).
    - Translucent red 3D fault planes intersecting terrain without floating gaps.
    - Initial isometric perspective (~28° elevation).
    """
    ve = vertical_exaggeration
    X, Y, Z, noise = terrain(
        nx=120,
        ny=80,
        width=bench_width,
        length=bench_length,
        height=bench_height,
        vertical_exaggeration=ve,
    )

    if holes is None:
        holes = load_holes_from_design(
            None,
            bench_length=bench_length,
            bench_width=bench_width,
            bench_height=bench_height,
            vertical_exaggeration=ve,
        )

    if overlays is None:
        overlays = load_geology_overlays(
            bench_length=bench_length,
            bench_width=bench_width,
            bench_height=bench_height,
            vertical_exaggeration=ve,
        )

    fig = go.Figure()

    # ----- 1. Stepped Bench Terrain Surface -----
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=Z + 0.25 * noise,
            showscale=False,
            hoverinfo="skip",
            colorscale=[
                [0.0, "#2c221a"],
                [0.3, "#5a3e22"],
                [0.6, "#7a5a3a"],
                [0.85, "#b48a58"],
                [1.0, "#d2ab74"],
            ],
            lighting=dict(
                ambient=0.6,
                diffuse=0.9,
                specular=0.15,
                roughness=0.9,
                fresnel=0.1,
            ),
            lightposition=dict(x=-200, y=-300, z=400),
            name="Bench Surface",
        )
    )

    # ----- 2. Geology & Fault Overlays -----
    if overlays:
        for ov in overlays:
            is_fault = (
                "fault" in ov.get("name", "").lower() or ov.get("type") == "fault"
            )
            if is_fault and not show_faults:
                continue
            if not is_fault and not show_geology:
                continue

            fig.add_trace(
                mesh_quad(
                    ov["vertices"],
                    ov["color"],
                    ov["name"],
                    ov["opacity"],
                )
            )

    # ----- 3. Drill Holes 3D Shafts & Collar Rim Geometry -----
    if show_holes and holes:
        charges = [h["charge_kg"] for h in holes]
        cmin, cmax = 50.0, 350.0
        if charges:
            if max(charges) > cmax or min(charges) < cmin:
                cmin, cmax = min(charges), max(charges)
                if cmax == cmin:
                    cmax = cmin + 1.0

        for h in holes:
            c_val = float(
                np.clip((h["charge_kg"] - cmin) / max(cmax - cmin, 1.0), 0.0, 1.0)
            )
            c = sample_colorscale(CSCALE, [c_val])[0]

            # Adjust Z positions to sit relative to top bench surface height
            z_top = h["z"] if h["z"] > 0.0 else bench_height * ve
            z_bottom = z_top + h["z_toe"] if h["z_toe"] < 0 else h["z_toe"]

            # 3D Cylindrical shaft line
            fig.add_trace(
                go.Scatter3d(
                    x=[h["x"], h["x_toe"]],
                    y=[h["y"], h["y_toe"]],
                    z=[z_top, z_bottom],
                    mode="lines",
                    line=dict(color=c, width=8),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Dark borehole bottom indicator
            fig.add_trace(
                go.Scatter3d(
                    x=[h["x_toe"]],
                    y=[h["y_toe"]],
                    z=[z_bottom],
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=4, color="#111111"),
                    hoverinfo="skip",
                )
            )

        # Raised/bevelled circular collar rim markers
        fig.add_trace(
            go.Scatter3d(
                x=[h["x"] for h in holes],
                y=[h["y"] for h in holes],
                z=[(h["z"] if h["z"] > 0.0 else bench_height * ve) + 0.08 for h in holes],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=7,
                    color=[h["charge_kg"] for h in holes],
                    colorscale=CSCALE,
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(
                        title=dict(
                            text="Charge Weight (kg)",
                            font=dict(color="white", size=13),
                            side="right",
                        ),
                        tickfont=dict(color="white", size=11),
                        orientation="v",
                        len=0.6,
                        x=1.02,
                        xanchor="left",
                        thickness=16,
                    ),
                    line=dict(color="#ffffff", width=1.0),
                ),
                text=[h["hole_id"] for h in holes],
                customdata=[(h["charge_kg"], h["depth_m"]) for h in holes],
                hovertemplate="<b>%{text}</b><br>Charge: %{customdata[0]:.0f} kg<br>Depth: %{customdata[1]:.1f} m<extra></extra>",
            )
        )

    # ----- 4. Isometric perspective & layout -----
    fig.update_layout(
        title=dict(
            text=f"{title} — {len(holes)} Blastholes" if (show_holes and holes) else title,
            x=0.5,
            font=dict(color="white", size=18),
        ),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=0, r=100, t=50, b=0),
        height=640,
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.97,
            bgcolor="rgba(11,18,32,0.85)",
            font=dict(color="white"),
        ),
        scene=dict(
            bgcolor=BG,
            aspectmode="manual",
            aspectratio=dict(x=1.6, y=1.0, z=0.6),
            camera=dict(
                eye=dict(x=1.4, y=-1.4, z=1.1),
                center=dict(x=0, y=0, z=-0.1),
                up=dict(x=0, y=0, z=1),
            ),
            xaxis=dict(
                title=dict(text="Length (m)", font=dict(color="white")),
                showgrid=True,
                gridcolor="#1f2d42",
                tickfont=dict(color="white"),
            ),
            yaxis=dict(
                title=dict(text="Width (m)", font=dict(color="white")),
                showgrid=True,
                gridcolor="#1f2d42",
                tickfont=dict(color="white"),
            ),
            zaxis=dict(
                title=dict(text="Height (m)", font=dict(color="white")),
                showgrid=True,
                gridcolor="#1f2d42",
                tickfont=dict(color="white"),
            ),
        ),
    )

    return fig


if __name__ == "__main__":
    fig = make_bench_figure()
    fig.write_html("bench_preview.html")
    print("wrote bench_preview.html")
