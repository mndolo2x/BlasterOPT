"""
Configuration constants, schemas, and licensing notices for BlastOPT GeoResponse v0.1.
"""

# Core Telemetry Ingestion Column Aliases
COLUMN_ALIASES = {
    "penetration_rate_m_hr": ["penetration_rate_m_hr", "rop_m_hr", "penetration_rate_m_h"],
    "penetration_rate_m_min": ["penetration_rate_m_min", "rop_m_min"],
    "torque_nm": ["torque_nm", "rotation_torque_nm", "torque"],
    "weight_on_bit_kg": ["weight_on_bit_kg", "wob_kg", "weight_on_bit"],
    "percussion_pressure_bar": ["percussion_pressure_bar", "hammer_pressure_bar", "percussion_bar"],
    "feed_pressure_bar": ["feed_pressure_bar", "thrust_pressure_bar"],
    "rotation_pressure_bar": ["rotation_pressure_bar"],
    "flushing_pressure_bar": ["flushing_pressure_bar", "water_pressure_bar"],
    "rotation_speed_rpm": ["rotation_speed_rpm", "rpm"],
    "depth_m": ["depth_m", "measured_depth_m", "depth"],
    "rock_type_label": ["rock_type_label", "rock_class", "rock_type", "strata_label"],
}

# Base required sensors for minimal MWD feature engineering and model training
MINIMAL_REQUIRED_MWD_COLUMNS = [
    "penetration_rate_m_hr",
    "rock_type_label",
]

# Optional sensors used for extended feature engineering when available
OPTIONAL_MWD_COLUMNS = [
    "torque_nm",
    "weight_on_bit_kg",
    "percussion_pressure_bar",
    "feed_pressure_bar",
    "rotation_pressure_bar",
    "flushing_pressure_bar",
    "rotation_speed_rpm",
    "depth_m",
]

# Grouping Identifiers to Prevent Data Leakage Across Splits
GROUP_IDENTIFIER_COLUMNS = ["tunnel_id", "blast_id", "hole_id", "bench_id"]

# All possible feature candidates
ALL_FEATURE_COLUMNS = [
    "penetration_rate_m_hr",
    "torque_nm",
    "weight_on_bit_kg",
    "percussion_pressure_bar",
    "feed_pressure_bar",
    "rotation_pressure_bar",
    "flushing_pressure_bar",
    "rotation_speed_rpm",
    "depth_m",
    "mwd_specific_energy_mj_m3",
    "rop_torque_ratio",
    "feed_percussion_ratio",
]

TARGET_COLUMN = "rock_type_label"

# Licensing and Research Terms
DATASET_LICENSE_NOTICE = (
    "RESEARCH-ONLY BENCHMARK DATASET NOTICE:\n"
    "The Norwegian MWD dataset benchmark is used exclusively for research and evaluation purposes.\n"
    "It must not be used for commercial deployment without express written permission.\n"
    "No GeoResponse prediction automatically modifies operational blast designs."
)

CONFIDENCE_THRESHOLD = 0.70  # Predictions below 70% probability trigger a review warning
