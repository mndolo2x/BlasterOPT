from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import os
import numpy as np

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

import matplotlib.pyplot as plt


@dataclass
class BenchConfig:
    bench_length_m: float = 60.0
    bench_width_m: float = 40.0
    bench_height_m: float = 15.0
    rock_type: str = "Kimberlite"
    burden_m: float = 4.2
    spacing_m: float = 5.1
    hole_diameter_mm: float = 165.0
    hole_depth_m: float = 12.0
    rows: int = 8
    holes_per_row: int = 6
    mesh_resolution_m: float = 0.25
    ai_rock: bool = False


@dataclass
class Hole:
    x: float
    y: float
    z: float
    depth: float
    diameter_m: float
    charge_kg: float


def build_bench_scene(cfg: BenchConfig) -> Tuple[Any, List[Hole], Dict[str, Any]]:
    """Builds 3D scene elements for the bench geometry and blastholes."""
    holes: List[Hole] = []

    # Calculate explosive charge per hole based on ANFO density (~850 kg/m^3) and charge length (~70% of hole depth)
    radius_m = (cfg.hole_diameter_mm / 1000.0) / 2.0
    hole_vol_m3 = np.pi * (radius_m ** 2) * cfg.hole_depth_m
    charge_kg = hole_vol_m3 * 0.70 * 850.0

    start_x = cfg.burden_m
    start_y = cfg.spacing_m

    for r in range(cfg.rows):
        for c in range(cfg.holes_per_row):
            x = start_x + r * cfg.burden_m
            y = start_y + c * cfg.spacing_m
            z = cfg.bench_height_m
            holes.append(
                Hole(
                    x=x,
                    y=y,
                    z=z,
                    depth=cfg.hole_depth_m,
                    diameter_m=cfg.hole_diameter_mm / 1000.0,
                    charge_kg=charge_kg,
                )
            )

    scene = {
        "config": cfg,
        "holes": holes,
    }

    if PYVISTA_AVAILABLE:
        try:
            blocks = pv.MultiBlock()
            # Bench box
            box = pv.Box(
                bounds=(
                    0,
                    cfg.bench_length_m,
                    0,
                    cfg.bench_width_m,
                    0,
                    cfg.bench_height_m,
                )
            )
            blocks.append(box, name="Bench")

            # Add hole cylinders
            for idx, h in enumerate(holes):
                cyl = pv.Cylinder(
                    center=(h.x, h.y, h.z - h.depth / 2.0),
                    direction=(0, 0, -1),
                    radius=h.diameter_m / 2.0,
                    height=h.depth,
                )
                blocks.append(cyl, name=f"Hole_{idx}")

            scene["pv_blocks"] = blocks
        except Exception:
            pass

    metadata = {
        "geometry_count": 1 + len(holes),
        "total_holes": len(holes),
        "rock_type": cfg.rock_type,
    }

    return scene, holes, metadata


def render_preview(scene: Any, output_path: str) -> str:
    """Renders 3D scene preview to an image file."""
    rendered = False

    if PYVISTA_AVAILABLE and isinstance(scene, dict) and "pv_blocks" in scene:
        try:
            pv.start_xvfb()
        except Exception:
            pass
        try:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_composite(scene["pv_blocks"], color="tan")
            plotter.set_background("white")
            plotter.show(screenshot=output_path)
            plotter.close()
            rendered = True
        except Exception:
            rendered = False

    if not rendered:
        # Fallback to matplotlib rendering if offscreen VTK fails
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        cfg = scene["config"] if isinstance(scene, dict) and "config" in scene else BenchConfig()
        holes = scene["holes"] if isinstance(scene, dict) and "holes" in scene else []

        # Draw wireframe box for bench
        l, w, h = cfg.bench_length_m, cfg.bench_width_m, cfg.bench_height_m
        ax.plot([0, l, l, 0, 0], [0, 0, w, w, 0], [0, 0, 0, 0, 0], color="brown")
        ax.plot([0, l, l, 0, 0], [0, 0, w, w, 0], [h, h, h, h, h], color="brown")
        for x_c, y_c in [(0, 0), (l, 0), (l, w), (0, w)]:
            ax.plot([x_c, x_c], [y_c, y_c], [0, h], color="brown")

        # Draw holes
        for hole in holes:
            ax.plot([hole.x, hole.x], [hole.y, hole.y], [hole.z, hole.z - hole.depth], color="red", linewidth=2)

        ax.set_title(f"3D Bench Twin ({cfg.rock_type})")
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Width (m)")
        ax.set_zlabel("Height (m)")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    return output_path


def main() -> None:
    """Main entrypoint for standalone test generation."""
    cfg = BenchConfig()
    scene, holes, metadata = build_bench_scene(cfg)
    render_preview(scene, "demo_bench.png")


if __name__ == "__main__":
    main()
