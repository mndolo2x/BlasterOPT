import os
import streamlit as st
from pathlib import Path
from src.engine.bench_model import BenchConfig, build_bench_scene, render_preview

st.title("3D Digital Twin of Bench")
st.caption("Parametric blast bench generation. Server-rendered PyVista geometry.")

# Use a 2-column layout
col1, col2 = st.columns([1, 2.5])

with col1:
    st.header("Bench Parameters")
    length = st.number_input("Bench Length (m)", min_value=10.0, max_value=200.0, value=60.0, step=5.0)
    width = st.number_input("Bench Width (m)", min_value=10.0, max_value=200.0, value=40.0, step=5.0)
    height = st.number_input("Bench Height (m)", min_value=5.0, max_value=50.0, value=15.0, step=1.0)

    rock = st.selectbox("Rock Type", ["Kimberlite", "Granite", "Limestone", "Dolomite", "Sandstone"])

    st.header("Blast Pattern")
    burden = st.number_input("Burden (m)", min_value=1.0, max_value=15.0, value=4.2, step=0.1)
    spacing = st.number_input("Spacing (m)", min_value=1.0, max_value=15.0, value=5.1, step=0.1)
    diameter = st.number_input("Hole Diameter (mm)", min_value=50, max_value=400, value=165, step=5)
    depth = st.number_input("Hole Depth (m)", min_value=1.0, max_value=30.0, value=12.0, step=0.5)
    rows = st.number_input("Rows", min_value=1, max_value=20, value=8)
    cols = st.number_input("Holes per Row", min_value=1, max_value=20, value=6)

    generate_btn = st.button("Generate 3D Bench", type="primary", use_container_width=True)

with col2:
    st.header("3D Rendering Space")

    if generate_btn:
        with st.spinner("Generating 3D mesh and rendering preview..."):
            try:
                cfg = BenchConfig(
                    bench_length_m=length,
                    bench_width_m=width,
                    bench_height_m=height,
                    rock_type=rock,
                    burden_m=burden,
                    spacing_m=spacing,
                    hole_diameter_mm=diameter,
                    hole_depth_m=depth,
                    rows=rows,
                    holes_per_row=cols,
                    mesh_resolution_m=0.25,
                    ai_rock=False
                )

                # Build the scene
                scene, holes, metadata = build_bench_scene(cfg)

                # Render preview to a temporary file
                preview_path = "temp_bench_preview.png"
                render_preview(scene, preview_path)

                # Display the image
                st.image(preview_path, use_column_width=True, caption="Parametric 3D Bench (Server Rendered)")

                # Show metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Holes", len(holes))
                total_charge = sum(h.charge_kg for h in holes)
                m2.metric("Total Charge", f"{total_charge:.0f} kg")
                m3.metric("Geometry Parts", metadata["geometry_count"])

                # Clean up temp file
                if os.path.exists(preview_path):
                    os.remove(preview_path)

            except Exception as e:
                st.error(f"An error occurred during 3D generation: {e}")
                st.exception(e)
    else:
        st.info("Configure the parameters on the left and click 'Generate 3D Bench' to render the model.")
