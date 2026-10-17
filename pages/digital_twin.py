import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from src.components.model_selector import render_page_model_selector
from src.ui.bench_twin import (
    make_bench_figure,
    load_holes_from_design,
    load_geology_overlays,
)
from src.ui.digital_twin.loader import load_and_inject_scene
from src.engine.bench_model import BenchConfig, build_bench_scene, render_preview

st.title("3D Digital Twin of Bench")
st.caption("Mining-Engineering 3D Digital Twin — Interactive 3D bench model with stepped faces, blastholes, geology overlays, and fault planes.")

model, model_key = render_page_model_selector("digital_twin")

# Get active design or default
active_design = st.session_state.get("current_design", st.session_state.get("last_predict_inputs"))

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
    pf = st.number_input("Powder Factor (kg/m³)", min_value=0.1, max_value=2.5, value=0.65, step=0.05)

    st.header("Display Options")
    show_holes = st.toggle("Show drill holes", value=True)
    show_geology = st.toggle("Show geology overlays", value=True)
    show_faults = st.toggle("Show fault planes", value=True)
    vertical_exaggeration = st.slider("Vertical Exaggeration", min_value=1.0, max_value=3.0, value=1.0, step=0.1)

    render_mode = st.radio(
        "3D View Engine Mode",
        ["Three.js Interactive Viewer 🎮", "Plotly 3D Surface View 📊", "Server PyVista Render 🖼️"],
        index=0,
    )
    generate_btn = st.button("Generate 3D Bench", type="primary", use_container_width=True)

with col2:
    st.header("3D Rendering Space")

    try:
        design_dict = {
            "num_rows": rows,
            "holes_per_row": cols,
            "burden_m": burden,
            "spacing_m": spacing,
            "hole_diameter_mm": diameter,
            "hole_depth_m": depth,
            "bench_height_m": height,
            "powder_factor_kg_m3": pf,
        }

        holes = load_holes_from_design(
            design_dict,
            bench_length=length,
            bench_width=width,
            bench_height=height,
            vertical_exaggeration=vertical_exaggeration,
        )

        overlays = load_geology_overlays(
            bench_length=length,
            bench_width=width,
            bench_height=height,
            vertical_exaggeration=vertical_exaggeration,
        )

        # Prepare scene payload for HTML/Three.js viewer
        scene_payload = {
            "bench_length": length,
            "bench_width": width,
            "bench_height": height,
            "holes": [
                {
                    "id": h["hole_id"],
                    "x": h["x"],
                    "y": h["y"],
                    "charge": h["charge_kg"],
                    "ppv_risk": float(np.clip((h["charge_kg"] - 50.0) / 300.0, 0.1, 0.95)),
                }
                for h in holes
            ] if show_holes else [],
            "faults": [
                {
                    "id": ov["name"],
                    "is_high_risk": True if "JWA_01" in ov["name"] else False,
                    "start_x": ov["vertices"][0][0],
                    "start_y": ov["vertices"][0][1],
                    "end_x": ov["vertices"][1][0],
                    "end_y": ov["vertices"][1][1],
                }
                for ov in overlays
                if "fault" in ov.get("name", "").lower() or ov.get("type") == "fault"
            ] if show_faults else [],
        }

        if render_mode == "Three.js Interactive Viewer 🎮":
            html_content = load_and_inject_scene(scene_payload)
            if html_content:
                components.html(html_content, height=720, scrolling=False)
            else:
                st.warning("Scene template not found. Falling back to Plotly 3D view.")
                fig = make_bench_figure(
                    holes=holes if show_holes else [],
                    overlays=overlays,
                    show_holes=show_holes,
                    show_geology=show_geology,
                    show_faults=show_faults,
                    bench_length=length,
                    bench_width=width,
                    bench_height=height,
                    vertical_exaggeration=vertical_exaggeration,
                    title=f"3D Bench Twin ({rock})",
                )
                st.plotly_chart(fig, use_container_width=True, config={"renderer": "svg"})

        elif render_mode == "Server PyVista Render 🖼️" or generate_btn:
            with st.spinner("Generating 3D mesh and rendering preview..."):
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
                scene, pv_holes, metadata = build_bench_scene(cfg)
                preview_path = "temp_bench_preview.png"
                render_preview(scene, preview_path)
                st.image(preview_path, use_column_width=True, caption="Parametric 3D Bench (Server Rendered PyVista)")
                if os.path.exists(preview_path):
                    os.remove(preview_path)
        else:
            # Plotly 3D Surface View Mode
            fig = make_bench_figure(
                holes=holes if show_holes else [],
                overlays=overlays,
                show_holes=show_holes,
                show_geology=show_geology,
                show_faults=show_faults,
                bench_length=length,
                bench_width=width,
                bench_height=height,
                vertical_exaggeration=vertical_exaggeration,
                title=f"3D Bench Twin ({rock})",
            )
            st.plotly_chart(fig, use_container_width=True, config={"renderer": "svg"})

        # Metrics Panel
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        num_holes = len(holes) if show_holes else 0
        total_charge = sum(h["charge_kg"] for h in holes) if show_holes else 0.0
        m1.metric("Total Holes", num_holes)
        m2.metric("Total Explosive Charge", f"{total_charge:,.0f} kg")
        m3.metric("Geology Overlays", len(overlays) if show_geology else 0)
        m4.metric("Dimensions", f"{length:.0f}m x {width:.0f}m x {height:.0f}m")

        # Geology Legend
        st.divider()
        st.markdown("### 🏷️ Geology Overlays & Fault Planes Legend")
        col_lg1, col_lg2 = st.columns(2)
        with col_lg1:
            st.markdown("- <span style='color:#8e44ad; font-weight:bold;'>■</span> **Kimberlite zone** (Opacity: 0.60)", unsafe_allow_html=True)
            st.markdown("- <span style='color:#7f8c8d; font-weight:bold;'>■</span> **Dolomite waste** (Opacity: 0.45)", unsafe_allow_html=True)
        with col_lg2:
            st.markdown("- <span style='color:#e67e22; font-weight:bold;'>■</span> **Granite contact** (Opacity: 0.55)", unsafe_allow_html=True)
            st.markdown("- <span style='color:#d93a3a; font-weight:bold;'>■</span> **FAULT_JWA_01** (Opacity: 0.35)", unsafe_allow_html=True)

        # Actions & Transfers
        st.divider()
        col_act1, col_act2, col_act3 = st.columns(3)
        with col_act1:
            if st.button("Send to Blast Design Engine →", type="primary", use_container_width=True):
                st.session_state["bench_twin_data"] = {
                    "bench_length": length,
                    "bench_width": width,
                    "bench_height": height,
                    "holes": holes,
                    "overlays": overlays,
                }
                st.success("Transferred bench twin data to Blast Design Engine!")

        with col_act2:
            try:
                fig_exp = make_bench_figure(
                    holes=holes if show_holes else [],
                    overlays=overlays,
                    bench_length=length,
                    bench_width=width,
                    bench_height=height,
                    vertical_exaggeration=vertical_exaggeration,
                )
                img_bytes = fig_exp.to_image(format="png")
                st.download_button(
                    label="Download PNG Image",
                    data=img_bytes,
                    file_name=f"bench_twin_{rock}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            except Exception:
                pass

        with col_act3:
            try:
                fig_exp = make_bench_figure(
                    holes=holes if show_holes else [],
                    overlays=overlays,
                    bench_length=length,
                    bench_width=width,
                    bench_height=height,
                    vertical_exaggeration=vertical_exaggeration,
                )
                html_bytes = fig_exp.to_html().encode("utf-8")
                st.download_button(
                    label="Download Standalone HTML",
                    data=html_bytes,
                    file_name=f"bench_twin_{rock}.html",
                    mime="text/html",
                    use_container_width=True,
                )
            except Exception:
                pass

    except Exception as e:
        st.error(f"An error occurred during 3D bench generation: {e}")
        st.exception(e)
