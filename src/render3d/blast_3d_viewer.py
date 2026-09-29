"""
3D Blast Viewer tabbed layout page implementation for Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.render3d.bench_model import BenchModel
from src.render3d.hole_pattern import HolePattern
from src.render3d.subdrill_analysis import (
    calculate_optimal_subdrill,
    analyze_subdrill_existing,
)
from src.render3d.toe_burden import (
    calculate_toe_burden,
    calculate_relief_holes,
)
from src.render3d.backbreak import (
    predict_backbreak,
    recommend_backbreak_prevention,
)
from src.render3d.free_face import (
    identify_free_faces,
    recommend_blast_direction,
)
from src.render3d.collision import (
    check_hole_collisions,
)
from src.render3d.timing_viz import (
    create_timing_animation,
)
from src.render3d.renderer import (
    BlastRenderer3D,
    Blast3DRenderer,
)


def render_blast_3d_viewer():
    """
    Renders the 8-tab 3D Blast Viewer interface.
    """
    st.header("🧊 3D Blast Engineering & Analysis Viewer")
    st.caption(
        "Interactive 3D mine bench engineering suite with subdrill optimization, toe burden analysis, "
        "backbreak prediction, free face identification, collision checks, and electronic detonator firing animation."
    )

    # Sidebar / Controls
    st.sidebar.subheader("3D Bench Parameters")
    crest_elev = st.sidebar.number_input("Crest Elevation (m)", 50.0, 200.0, 100.0, key="v3d_crest")
    toe_elev = st.sidebar.number_input("Toe Elevation (m)", 0.0, 180.0, 85.0, key="v3d_toe")
    face_angle = st.sidebar.number_input("Face Angle (deg)", 45.0, 90.0, 75.0, key="v3d_angle")
    rock_a = st.sidebar.slider("Rock Blastability Factor (A)", 4.0, 16.0, 8.0, key="v3d_rock_a")

    st.sidebar.subheader("Pattern Design")
    pattern_type = st.sidebar.selectbox("Pattern Type", ["staggered", "square", "echelon", "v_pattern"], index=0, key="v3d_ptype")
    burden_m = st.sidebar.number_input("Burden (m)", 2.0, 10.0, 4.0, key="v3d_b")
    spacing_m = st.sidebar.number_input("Spacing (m)", 2.0, 12.0, 5.0, key="v3d_s")
    subdrill_m = st.sidebar.number_input("Current Subdrill (m)", 0.0, 5.0, 1.5, key="v3d_sub")
    pf_kg_m3 = st.sidebar.number_input("Powder Factor (kg/m3)", 0.2, 2.0, 0.7, key="v3d_pf")
    stemming_m = st.sidebar.number_input("Stemming Length (m)", 1.0, 8.0, 3.5, key="v3d_stem")

    # Instantiate bench & pattern
    bench = BenchModel("BENCH_01", crest_elev, toe_elev, face_angle)
    pattern = HolePattern(burden_m, spacing_m, bench, pattern_type=pattern_type, subdrill_m=subdrill_m)
    holes = pattern.generate_holes(n_rows=3, n_per_row=4)

    # Define 8 Tabs
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        "1. Bench Geometry",
        "2. Hole Pattern",
        "3. Subdrill Analysis",
        "4. Toe Burden",
        "5. Backbreak",
        "6. Free Faces",
        "7. Collision Check",
        "8. Timing Animation",
    ])

    # Tab 1: Bench Geometry
    with t1:
        st.subheader("Bench Geometry & Face Mesh")
        c1, c2 = st.columns([2, 1])
        with c1:
            renderer = BlastRenderer3D(bench=bench, pattern=pattern)
            st.plotly_chart(renderer.render_plotly(), use_container_width=True)
        with c2:
            st.metric("Bench Height", f"{bench.get_bench_height():.2f} m")
            st.metric("Crest Elevation", f"{bench.crest_elevation_m:.2f} m")
            st.metric("Toe Elevation", f"{bench.toe_elevation_m:.2f} m")
            st.metric("Face Angle", f"{bench.face_angle_deg:.1f}°")
            st.info("Bench face generated using horizontal offset formula: height / tan(face_angle).")

    # Tab 2: Hole Pattern
    with t2:
        st.subheader("Hole Pattern Placement")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure()
            collars = np.array([[h.x, h.y, h.z] for h in holes])
            toes = np.array([h.get_toe_coordinate() for h in holes])
            fig.add_trace(go.Scatter3d(x=collars[:, 0], y=collars[:, 1], z=collars[:, 2], mode="markers+text", marker=dict(size=8, color="green"), text=[h.hole_id for h in holes]))
            for c, t in zip(collars, toes):
                fig.add_trace(go.Scatter3d(x=[c[0], t[0]], y=[c[1], t[1]], z=[c[2], t[2]], mode="lines", line=dict(color="orange", width=4), showlegend=False))
            fig.update_layout(title=f"3D Hole Pattern ({pattern_type.upper()})", scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("Total Holes", len(holes))
            st.metric("Burden x Spacing", f"{burden_m:.1f} m x {spacing_m:.1f} m")
            st.metric("Pattern Layout", pattern_type.upper())
            hole_table = [{"Hole": h.hole_id, "X": h.x, "Y": h.y, "Z": h.z, "Depth": h.depth_m} for h in holes]
            st.dataframe(pd.DataFrame(hole_table), use_container_width=True)

    # Tab 3: Subdrill Analysis
    with t3:
        st.subheader("Subdrilling Optimization (Konya, 1995)")
        opt_sub = calculate_optimal_subdrill(burden_m, bench.get_bench_height(), rock_a, hole_diameter_mm=250.0)
        sub_analysis = analyze_subdrill_existing(subdrill_m, opt_sub["optimal_subdrill_m"])

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Current Subdrill", f"{subdrill_m:.2f} m")
            st.metric("Optimal Subdrill", f"{opt_sub['optimal_subdrill_m']:.2f} m")
            st.metric("Recommended Range", f"{opt_sub['recommended_range_m'][0]} - {opt_sub['recommended_range_m'][1]} m")
            if sub_analysis["classification"] == "excessive":
                st.error(sub_analysis["status_message"])
            elif sub_analysis["classification"] == "insufficient":
                st.error(sub_analysis["status_message"])
            else:
                st.success(sub_analysis["status_message"])
        with c2:
            st.markdown(f"**Reasoning:** {opt_sub['reasoning']}")
            if opt_sub.get("excessive_risk"):
                st.warning(opt_sub["excessive_risk"])
            if opt_sub.get("insufficient_risk"):
                st.warning(opt_sub["insufficient_risk"])

    # Tab 4: Toe Burden
    with t4:
        st.subheader("Toe Burden Analysis & Relief Holes")
        tb = calculate_toe_burden(burden_m, bench.get_bench_height(), face_angle)
        relief = calculate_relief_holes(tb["toe_burden_m"], burden_m, spacing_m)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Crest Burden", f"{tb['crest_burden_m']:.2f} m")
            st.metric("Toe Burden", f"{tb['toe_burden_m']:.2f} m")
            st.metric("Difference", f"{tb['difference_m']:.2f} m")
            if tb["requires_relief"]:
                st.error("🚨 Toe burden exceeds 1.4x crest burden!")
            else:
                st.success("Toe burden is within safe limits.")
        with c2:
            st.markdown(f"**Recommendation:** {tb['recommendation']}")
            st.markdown(f"**Relief Holes Required:** {relief['relief_holes_required']}")
            st.markdown(f"**Relief Hole Positions (m):** {relief['positions_m']}")

    # Tab 5: Backbreak
    with t5:
        st.subheader("Backbreak Prediction (Scoble et al., 1997)")
        bb = predict_backbreak(pf_kg_m3, stemming_m, burden_m, rock_a, joint_spacing_m=0.8)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Predicted Backbreak", f"{bb['predicted_backbreak_m']:.2f} m")
            if bb["risk_level"] == "high":
                st.error(f"Backbreak Risk Level: {bb['risk_level'].upper()}")
            elif bb["risk_level"] == "medium":
                st.warning(f"Backbreak Risk Level: {bb['risk_level'].upper()}")
            else:
                st.success(f"Backbreak Risk Level: {bb['risk_level'].upper()}")
        with c2:
            st.markdown("**Contributing Factors:**")
            for factor in bb["contributing_factors"]:
                st.write(f"- {factor}")
            st.markdown("**Prevention Measures:**")
            for measure in bb["prevention_measures"]:
                st.write(f"- {measure}")

    # Tab 6: Free Faces
    with t6:
        st.subheader("Free Face Identification & Blast Direction")
        faces = identify_free_faces(bench)
        direction = recommend_blast_direction(faces)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Free Faces Detected", len(faces))
            st.metric("Recommended Azimuth", f"{direction['recommended_azimuth_deg']:.1f}°")
        with c2:
            st.markdown(f"**Reasoning:** {direction['reasoning']}")
            for f in faces:
                st.info(f"Face '{f['face_id']}': Height={f['height_m']}m, Width={f['width_m']}m, Quality={f['quality'].upper()}")

    # Tab 7: Collision Check
    with t7:
        st.subheader("3D Hole-to-Hole Collision Detection")
        col_res = check_hole_collisions(holes, min_separation_m=0.5, max_deviation_deg=3.0)

        if col_res["has_collisions"]:
            st.error("🚨 CRITICAL: Hole collisions detected!")
        else:
            st.success("No critical hole collisions detected.")

        st.markdown("**Recommendations:**")
        for rec in col_res["recommendations"]:
            st.write(f"- {rec}")

        if col_res["collisions"]:
            st.dataframe(pd.DataFrame(col_res["collisions"]), use_container_width=True)

    # Tab 8: Timing Animation
    with t8:
        st.subheader("Electronic Detonator Timing Animation")
        timing_seq = {"delays_ms": {h.hole_id: i * 25.0 for i, h in enumerate(holes)}}
        anim_fig = create_timing_animation(holes, timing_seq, duration_ms=500.0)
        st.plotly_chart(anim_fig, use_container_width=True)
