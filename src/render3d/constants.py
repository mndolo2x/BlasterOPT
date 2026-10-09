"""
Physical constants and bounds for the 3D blast design module.
"""

# Search-space bounds for input parameters
BURDEN_RANGE_M = (3.0, 6.0)
SPACING_RANGE_M = (3.5, 8.0)
STEMMING_RANGE_M = (2.0, 5.0)
POWDER_FACTOR_RANGE = (0.40, 0.90)
HOLE_DIAMETER_RANGE_MM = (100, 311)
HOLE_DEPTH_RANGE_M = (5.0, 30.0)
BENCH_HEIGHT_RANGE_M = (10.0, 18.0)
SUBDRILL_RANGE_M = (0.3, 2.0)
EXPLOSIVE_RWS_RANGE = (80, 130)
ROCK_FACTOR_RANGE = (6.0, 12.0)

# Default test scenario (Jwaneng-like)
DEFAULT_DESIGN = {
    "burden_m": 4.2,
    "spacing_m": 5.1,
    "stemming_m": 3.0,
    "powder_factor_kg_m3": 0.65,
    "hole_diameter_mm": 165,
    "hole_depth_m": 16.5,
    "bench_height_m": 15.0,
    "subdrilling_m": 1.5,
    "hole_angle_deg": 90.0,
    "rock_factor_A": 8.0,
    "explosive_rws": 115,
    "max_charge_per_delay_kg": 640,
    "monitoring_distance_m": 800,
    "num_rows": 8,
    "holes_per_row": 6,
    "pattern_type": "staggered",
}

# Regulatory limits (Botswana)
PPV_LIMIT_MM_S = 5.0
AIRBLAST_LIMIT_DB = 120.0

# Rock density assumption for tonnage calculation
ROCK_DENSITY_T_M3 = 2.7
