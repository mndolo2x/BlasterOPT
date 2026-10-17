"""
Script to generate the ultimate, high-resolution 3D Bench model image for BlastOPT Botswana.
Integrates stepped bench topography, geological strata, fault structures, drillholes with charge columns,
collar rings, and professional lighting & camera angles.
"""
import os
import sys
import numpy as np

# Ensure repo root is on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pyvista as pv
from src.render3d.bench_model import BenchGeometry, build_bench_mesh
from src.render3d.hole_pattern import HolePattern, generate_holes


def generate_best_3d_bench_image(output_path: str = "best_3d_bench.png") -> str:
    print("Building high-resolution 3D bench geometry...")

    # 1. Define Bench Parameters
    bench = BenchGeometry(
        bench_id="B14",
        crest_elevation_m=18.0,
        toe_elevation_m=0.0,
        face_angle_deg=75.0,
        length_m=70.0,
        width_m=45.0,
    )

    # 2. Define Hole Pattern (8 rows, 6 holes per row = 48 blastholes, staggered pattern)
    pattern_config = HolePattern(
        pattern_type="staggered",
        burden_m=4.2,
        spacing_m=5.0,
        num_rows=8,
        holes_per_row=6,
        stemming_m=3.5,
        hole_depth_m=13.5,
        hole_angle_deg=90.0,
        powder_factor_kg_m3=0.65,
        bench_height_m=18.0,
        subdrilling_m=1.5,
    )
    pattern = generate_holes(pattern_config)

    # Enable PyVista XVFB for headless rendering
    try:
        pv.start_xvfb()
    except Exception as e:
        print(f"XVFB notice: {e}")

    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    plotter.set_background("#0B132B")  # Deep slate/navy dark theme background

    # 3. Build Bench Block Mesh
    mesh_data = build_bench_mesh(bench, subdrill_m=1.5)

    # Create top crest surface mesh
    x_crest = mesh_data["crest"]["x"]
    y_crest = mesh_data["crest"]["y"]
    z_crest = mesh_data["crest"]["z"]

    grid_crest = pv.StructuredGrid(x_crest, y_crest, z_crest)
    grid_crest.point_data["Elevation"] = z_crest.flatten()
    plotter.add_mesh(
        grid_crest,
        cmap="copper",
        opacity=0.92,
        show_edges=False,
        smooth_shading=True,
        ambient=0.4,
        diffuse=0.8,
        specular=0.2,
        label="Kimberlite Bench Top",
    )

    # 4. Geological Strata Horizon Planes (Kimberlite / Dolomite / Granite)
    # Dolomite middle layer (Z = 6m to 12m)
    x_strata = np.linspace(0, bench.length_m, 50)
    y_strata = np.linspace(0, bench.width_m, 50)
    xx_s, yy_s = np.meshgrid(x_strata, y_strata)
    zz_dolomite = np.full_like(xx_s, 10.0) - 0.05 * xx_s  # Dipping layer
    grid_dolomite = pv.StructuredGrid(xx_s, yy_s, zz_dolomite)
    plotter.add_mesh(
        grid_dolomite,
        color="#705D4F",
        opacity=0.40,
        show_edges=False,
        label="Dolomite Contact Layer",
    )

    # 5. Fault Plane (FAULT_JWA_01)
    # Planes dipping across the bench
    zz_fault = 0.6 * xx_s + 0.3 * yy_s - 5.0
    grid_fault = pv.StructuredGrid(xx_s, yy_s, zz_fault)
    plotter.add_mesh(
        grid_fault,
        color="#EF4444",
        opacity=0.45,
        show_edges=True,
        edge_color="#DC2626",
        label="FAULT_JWA_01 Plane",
    )

    # 6. Build Blastholes (ANFO/Emulsion charge columns + Stemming top caps)
    charges = [h.charge_kg for h in pattern.holes]
    min_c, max_c = min(charges), max(charges)
    if max_c == min_c:
        max_c = min_c + 1.0

    get_collar = mesh_data.get("get_collar_z", lambda x, y: bench.crest_elevation_m)

    for idx, h in enumerate(pattern.holes):
        collar_z = get_collar(h.x_m, h.y_m)
        total_depth = h.depth_m
        stemming_length = 3.5
        charge_length = max(1.0, total_depth - stemming_length)

        # Charge column cylinder (lower section)
        charge_center_z = collar_z - stemming_length - (charge_length / 2.0)
        charge_cyl = pv.Cylinder(
            center=(h.x_m, h.y_m, charge_center_z),
            direction=(0, 0, -1),
            radius=0.22,  # Enlarged diameter for visual clarity
            height=charge_length,
            resolution=24,
        )

        # Color based on charge weight
        norm_charge = (h.charge_kg - min_c) / (max_c - min_c)
        r = norm_charge
        g = 0.3 + 0.6 * (1.0 - abs(norm_charge - 0.5) * 2)
        b = 1.0 - norm_charge
        color_rgb = (r, g, b)

        plotter.add_mesh(
            charge_cyl,
            color=color_rgb,
            smooth_shading=True,
            ambient=0.5,
            diffuse=0.9,
            specular=0.5,
        )

        # Stemming cap cylinder (top section)
        stem_center_z = collar_z - (stemming_length / 2.0)
        stem_cyl = pv.Cylinder(
            center=(h.x_m, h.y_m, stem_center_z),
            direction=(0, 0, -1),
            radius=0.22,
            height=stemming_length,
            resolution=24,
        )
        plotter.add_mesh(
            stem_cyl,
            color="#F59E0B",  # Amber/Yellow stemming color
            opacity=0.85,
            smooth_shading=True,
        )

        # Collar ring on bench top surface
        collar_ring = pv.Disc(
            center=(h.x_m, h.y_m, collar_z + 0.05),
            inner=0.22,
            outer=0.45,
            normal=(0, 0, 1),
            r_res=1,
            c_res=24,
        )
        plotter.add_mesh(
            collar_ring,
            color="#00B4D8",  # Cyan collar accent
            ambient=0.8,
        )

    # 7. Lighting and Camera Setup
    light1 = pv.Light(position=(100, 100, 80), color="white", intensity=0.9)
    light2 = pv.Light(position=(-50, -50, 40), color="#38BDF8", intensity=0.4)
    plotter.add_light(light1)
    plotter.add_light(light2)

    plotter.camera_position = [
        (110.0, -60.0, 65.0),  # Eye position (Isometric perspective)
        (35.0, 22.5, 8.0),     # Target center position
        (0.0, 0.0, 1.0),       # Up vector
    ]

    # Add Scale Bar & Legend Text
    plotter.add_legend(
        labels=[
            ("Kimberlite Top Bench", "#B8977E"),
            ("Stemming Column (3.5m)", "#F59E0B"),
            ("ANFO Charge Column", "#EC4899"),
            ("Collar Ring Indicator", "#00B4D8"),
            ("Dolomite Contact Layer", "#705D4F"),
            ("FAULT_JWA_01 Structure", "#EF4444"),
        ],
        bcolor="#0F172A",
        border=True,
        size=(0.28, 0.28),
        loc="upper left",
    )

    plotter.add_text(
        "BlastOpt Botswana — 3D Digital Twin Bench Model\nLocation: Jwaneng Mine Bench 14 | Geology: Kimberlite / Dolomite",
        position="lower_left",
        font_size=12,
        color="#F8FAFC",
        shadow=True,
    )

    plotter.add_axes(
        line_width=3,
        color="#00B4D8",
        x_color="#EF4444",
        y_color="#10B981",
        z_color="#3B82F6",
    )

    # Save high-res screenshot
    plotter.show(screenshot=output_path)
    plotter.close()
    print(f"High-resolution 3D bench image generated and saved to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_best_3d_bench_image("best_3d_bench.png")
