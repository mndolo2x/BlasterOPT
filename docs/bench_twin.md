# 3D Bench Digital Twin Documentation

## Overview
The `src/ui/bench_twin.py` module renders a 3D Bench Digital Twin scene using Plotly and Streamlit. It visualizes continuous topographic bench surface geometry, geological block model overlays (e.g. Kimberlite zones, Granite contacts), structural fault planes, and blastholes color-coded by explosive charge weight.

## Expected Data Shapes

### Blastholes (`load_holes_from_design`)
Returns a list of dictionaries with the following schema:

```python
{
    "hole_id": "R1-H1",        # Unique string identifier
    "x": 10.0,                 # Collar X position (m)
    "y": 4.0,                  # Collar Y position (m)
    "z": 0.0,                  # Collar Z elevation (m)
    "x_toe": 10.0,             # Toe X position (m)
    "y_toe": 4.0,              # Toe Y position (m)
    "z_toe": -12.0,            # Toe Z elevation (m)
    "charge_kg": 150.0,        # Total explosive charge weight (kg)
}
```

### Geological Overlays (`load_geology_overlays`)
Returns a list of dictionaries with quadrilateral vertices:

```python
{
    "name": "Kimberlite zone", # Overlay name
    "color": "#8e44ad",        # Hex color string
    "opacity": 0.40,           # Opacity float (0.0 to 1.0)
    "type": "geology",         # Overlay category ("geology" or "fault")
    "vertices": [              # 4 corner coordinates (X, Y, Z)
        (0.0, 0.0, 0.1),
        (22.0, 0.0, 0.1),
        (0.0, 40.0, 0.1),
        (22.0, 40.0, 0.1),
    ],
}
```

## Replacing Synthetic Data with Real Survey Data
To substitute synthetic terrain with a real Digital Terrain Model (DTM) or survey point cloud:

1. Ingest survey point array `survey_pts` of shape `(N, 3)` containing `[X, Y, Z]` coordinates.
2. Invoke `terrain_from_survey(survey_pts, nx=100, ny=60)`.
3. Pass the resulting grid mesh arrays `(grid_x, grid_y, grid_z)` into `make_bench_figure()`.

## Adding New Geology Overlays
Add new entries under `geology_overlays` in the bench configuration YAML files (`config/benches/[site]_bench_[id].yaml`). `load_geology_overlays` will automatically parse and render them as 3D `Mesh3d` quadrilaterals.

## Performance Guidelines
- Default grid resolution is `100 x 60` (`nx` and `ny`).
- Plotly rendering mode is forced to `config={"renderer": "svg"}` in Streamlit to guarantee non-WebGL compatibility across headless servers and remote clients.
- Renders in under 2 seconds on standard desktop hardware.
