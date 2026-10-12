"""
3D bench view for Streamlit + Plotly (corrected and improved).

This version addresses visualization issues:
- Holes are embedded in the bench surface using bilinear interpolation of the terrain grid
- Holes are rendered as solid cylinder Mesh3d objects (thicker, color-coded by charge)
- Collars use a marker scatter with an enlarged, readable colorbar
- Geology overlays that sit on the bench are nudged to the surface so they are visible
- Scene camera/aspect adjusted so bench fills more of the view

Backwards-compatible API: make_bench_figure(...), load_holes_from_design(...), load_geology_overlays(...)
"""
import os
import yaml
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from typing import List, Dict, Any, Optional, Tuple

BG = "#0b1220"
CSCALE = "Turbo"


def terrain(nx: int = 120, ny: int = 80, width: float = 60.0, length: float = 40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bench top (z=0) with three stepped faces on the high-x side."""
    x = np.linspace(0, width, nx)
    y = np.linspace(0, length, ny)
    X, Y = np.meshgrid(x, y)
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, X.shape)
    for _ in range(3):
        noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
                 + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5
    Z = 0.15 * noise
    for k in range(3):
        Z -= 4.0 / (1 + np.exp(-(X - (44 + 4 * k)) * 3.0))
    return X, Y, Z, noise


def terrain_from_survey(survey_data: np.ndarray, nx: int = 120, ny: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    """
    Generate synthetic drill holes positioned within the bench area.
    Holes are spread across x=10..40 and y=5..35, sitting on top of
    the bench surface.
    """
    rng = np.random.default_rng(7)
    recs = []
    for r in range(rows):
        for h in range(per_row):
            x0 = 10 + 5.0 * h
            y0 = 5 + 3.5 * r
            dx, dy = rng.normal(0, 0.15, 2)
            charge = float(np.clip(
                50 + 200 * (h / max(per_row - 1, 1)) * rng.uniform(0.7, 1.1),
                50, 250
            ))
            recs.append(dict(
                hole_id=f"R{r + 1}-H{h + 1}",
                x=float(x0), y=float(y0), z=0.0,
                x_toe=float(x0 + dx), y_toe=float(y0 + dy), z_toe=float(-depth),
                charge_kg=charge,
            ))
    return recs


def synthetic_holes_for_demo(rows: int = 8, per_row: int = 6, depth: float = 12.0) -> List[Dict[str, Any]]:
    """Alias for demo_holes to maintain backward compatibility."""
    return demo_holes(rows=rows, per_row=per_row, depth=depth)


def load_holes_from_design(design: Any) -> List[Dict[str, Any]]:
    """
    Convert a design object or dict into a list of hole dictionaries.
    """
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
                x = float(h.get("x_m", h.get("x", 10.0)))
                y = float(h.get("y_m", h.get("y", 5.0)))
                z = float(h.get("z_collar_m", h.get("z", 0.0)))
                x_toe = float(h.get("x_toe_m", h.get("x_toe", x)))
                y_toe = float(h.get("y_toe_m", h.get("y_toe", y)))
                z_toe = float(h.get("z_toe_m", h.get("z_toe", -depth)))
                charge = float(h.get("charge_kg", h.get("charge_mass_kg", 150.0)))
            else:
                h_id = getattr(h, "hole_id", "H-01")
                x = float(getattr(h, "x_m", 10.0))
                y = float(getattr(h, "y_m", 5.0))
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
            x = c * 5.0 + 10.0
            y = r * 3.5 + 5.0
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
    """
    Read geological block model overlays from config/benches YAML or fallback.
    """
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

    # Default fallback overlays
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


def mesh_quad(pts: Any, color: str, name: str, opacity: float) -> go.Mesh3d:
    """Build a 3D Mesh3d trace from 4 corner vertices."""
    pts = np.asarray(pts, float)
    return go.Mesh3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], i=[0, 1], j=[1, 3], k=[2, 2],
                     color=color, opacity=opacity, name=name, showlegend=True, hoverinfo="name")


# ----- Utility helpers -----

def _surface_elevation_at(xq: float, yq: float, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    """Bilinear interpolation of the grid Z at query point (xq, yq)."""
    # X is shape (ny, nx) with X[0, :] increasing; Y is (ny, nx) with Y[:, 0] increasing
    xs = X[0, :]
    ys = Y[:, 0]
    if xq <= xs[0]:
        ix = 0
        xw = 0.0
    elif xq >= xs[-1]:
        ix = len(xs) - 2
        xw = 1.0
    else:
        ix = np.searchsorted(xs, xq) - 1
        x0, x1 = xs[ix], xs[ix + 1]
        xw = (xq - x0) / (x1 - x0) if (x1 - x0) != 0 else 0.0

    if yq <= ys[0]:
        iy = 0
        yw = 0.0
    elif yq >= ys[-1]:
        iy = len(ys) - 2
        yw = 1.0
    else:
        iy = np.searchsorted(ys, yq) - 1
        y0, y1 = ys[iy], ys[iy + 1]
        yw = (yq - y0) / (y1 - y0) if (y1 - y0) != 0 else 0.0

    z00 = Z[iy, ix]
    z10 = Z[iy, ix + 1]
    z01 = Z[iy + 1, ix]
    z11 = Z[iy + 1, ix + 1]

    # bilinear
    z0 = z00 * (1 - xw) + z10 * xw
    z1 = z01 * (1 - xw) + z11 * xw
    zq = z0 * (1 - yw) + z1 * yw
    return float(zq)


def _make_cylinder_mesh(xc: float, yc: float, z_top: float, z_bottom: float, radius: float = 0.1, n_seg: int = 16, color: str = "#FF6D00", name: Optional[str] = None, opacity: float = 1.0) -> go.Mesh3d:
    """Create a vertical cylinder mesh (approximated by n_seg sides) centered at (xc, yc) running z_top->z_bottom."""
    # build circle points
    theta = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    xt = xc + radius * np.cos(theta)
    yt = yc + radius * np.sin(theta)
    zt = np.full_like(xt, z_top)
    xb = xc + radius * np.cos(theta)
    yb = yc + radius * np.sin(theta)
    zb = np.full_like(xb, z_bottom)

    # vertices: top ring then bottom ring
    xs = np.concatenate([xt, xb])
    ys = np.concatenate([yt, yb])
    zs = np.concatenate([zt, zb])

    # faces (triangles) for sides
    i = []
    j = []
    k = []
    for t in range(n_seg):
        t_next = (t + 1) % n_seg
        top_t = t
        top_next = t_next
        bot_t = t + n_seg
        bot_next = t_next + n_seg
        # two triangles per quad
        i.extend([top_t, top_t])
        j.extend([top_next, bot_t])
        k.extend([bot_t, bot_next])

    # optional caps (top and bottom) - create fan triangles to center
    # add center vertices
    xs = np.concatenate([xs, [xc, xc]])
    ys = np.concatenate([ys, [yc, yc]])
    zs = np.concatenate([zs, [(z_top), (z_bottom)]])
    center_top_idx = len(xs) - 2
    center_bot_idx = len(xs) - 1
    for t in range(n_seg):
        t_next = (t + 1) % n_seg
        # top cap
        i.append(center_top_idx)
        j.append(t)
        k.append(t_next)
        # bottom cap
        i.append(center_bot_idx)
        j.append(t + n_seg)
        k.append(t_next + n_seg)

    return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color, opacity=opacity, name=name or "hole", showlegend=False)


def make_bench_figure(
    holes: Optional[List[Dict[str, Any]]] = None,
    overlays: Optional[List[Dict[str, Any]]] = None,
    show_holes: bool = True,
    show_geology: bool = True,
    show_faults: bool = True,
    title: str = "Digital Twin of Bench",
) -> go.Figure:
    """
    Return interactive Plotly Figure for the 3D Bench Digital Twin (corrected version).
    """
    holes = holes if holes is not None else demo_holes()
    overlays = overlays if overlays is not None else load_geology_overlays()

    X, Y, Z, noise = terrain()
    fig = go.Figure()

    # ----- 1. Bench terrain surface -----
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=Z + 0.3 * noise,
        showscale=False,
        hoverinfo="skip",
        colorscale=[
            [0.0, "#3b2c20"],
            [0.45, "#7a5a3a"],
            [0.8, "#b48a58"],
            [1.0, "#d2ab74"],
        ],
        lighting=dict(ambient=0.6, diffuse=0.9, specular=0.15, roughness=0.9, fresnel=0.1),
        lightposition=dict(x=-200, y=-300, z=400),
    ))

    # ----- 2. Geology & Fault Overlays -----
    if overlays:
        for ov in overlays:
            if ov is None:
                continue
            is_fault = "fault" in str(ov.get("name", "")).lower() or ov.get("type") == "fault"
            if is_fault and not show_faults:
                continue
            if not is_fault and not show_geology:
                continue

            # adjust overlay vertices that are at 'surface' heights to be placed on the terrain
            verts = []
            for vx, vy, vz in ov.get("vertices", []):
                if vz is None:
                    vz = 0.0
                if vz >= 0.0:
                    surf_z = _surface_elevation_at(vx, vy, X, Y, Z)
                    verts.append((vx, vy, surf_z + 0.02))
                else:
                    verts.append((vx, vy, vz))

            fig.add_trace(mesh_quad(verts, ov["color"], ov["name"], ov["opacity"]))

    # ----- 3. Holes: rendered as cylinders colored by charge weight -----
    if show_holes and holes:
        charges = [h["charge_kg"] for h in holes]
        cmin, cmax = (min(charges), max(charges)) if charges else (50, 250)
        if cmax == cmin:
            cmax = cmin + 1.0

        # compute surface elevations at each collar and adjust holes so collars sit on the surface
        adjusted_holes = []
        for h in holes:
            surf_z = _surface_elevation_at(h["x"], h["y"], X, Y, Z)
            # depth (positive) from collar to toe
            depth = abs(h.get("z_toe", surf_z - 12.0) - h.get("z", surf_z))
            z_top = surf_z
            z_bottom = z_top - depth
            adjusted_holes.append({**h, "z": z_top, "z_toe": z_bottom})

        # draw cylinders and collar markers
        for h in adjusted_holes:
            norm = float(np.clip((h["charge_kg"] - cmin) / max((cmax - cmin), 1.0), 0.0, 1.0))
            color_hex = sample_colorscale(CSCALE, [norm])[0]
            # radius scaled with charge but clamped
            radius = max(0.05, min(0.35, 0.06 + 0.0005 * (h["charge_kg"] - cmin)))
            cyl = _make_cylinder_mesh(h["x"], h["y"], h["z"], h["z_toe"], radius=radius, n_seg=20, color=color_hex, name=h.get("hole_id"), opacity=1.0)
            fig.add_trace(cyl)

        # collar markers: keep for hover and for colorbar
        fig.add_trace(go.Scatter3d(
            x=[h["x"] for h in adjusted_holes],
            y=[h["y"] for h in adjusted_holes],
            z=[h["z"] + 0.02 for h in adjusted_holes],
            mode="markers",
            showlegend=False,
            marker=dict(
                size=8,
                color=[h["charge_kg"] for h in adjusted_holes],
                colorscale=CSCALE,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title=dict(text="Charge Weight (kg)", font=dict(color="white", size=14)),
                    tickfont=dict(color="white", size=12),
                    orientation="v",
                    len=0.8,
                    x=1.12,
                    xanchor="left",
                    thickness=22,
                ),
                line=dict(color="black", width=0.5),
            ),
            text=[h["hole_id"] for h in adjusted_holes],
            customdata=[h["charge_kg"] for h in adjusted_holes],
            hovertemplate="%{text}<br>Charge: %{customdata:.0f} kg<extra></extra>",
        ))

    # ----- 4. Scene layout: tighter camera, bench fills more view -----
    fig.update_layout(
        title=dict(text=f"{title} — {len(holes)} holes" if show_holes else title, x=0.5, font=dict(color="white", size=18)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=0, r=140, t=50, b=0),
        height=720,
        showlegend=True,
        legend=dict(x=0.01, y=0.97, bgcolor="rgba(11,18,32,0.8)", font=dict(color="white")),
        scene=dict(
            bgcolor=BG,
            aspectmode="manual",
            aspectratio=dict(x=2.0, y=1.0, z=0.45),
            camera=dict(eye=dict(x=1.2, y=-1.0, z=0.9)),
            xaxis=dict(visible=False, showbackground=False, showgrid=False),
            yaxis=dict(visible=False, showbackground=False, showgrid=False),
            zaxis=dict(visible=False, showbackground=False, showgrid=False),
        ),
    )

    return fig


if __name__ == "__main__":
    fig = make_bench_figure()
    fig.write_html("bench_preview.html")
    print("wrote bench_preview.html")
