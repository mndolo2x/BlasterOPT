"""
Configuration constants, schemas, and licensing notices for BlastOPT GeoResponse v0.1.
"""

# Mandatory MWD Ingestion Columns
REQUIRED_MWD_COLUMNS = [
    "penetration_rate_m_hr",
    "torque_nm",
    "weight_on_bit_kg",
    "percussion_pressure_bar",
    "flushing_pressure_bar",
    "rock_type_label",
]

# Optional Grouping Identifiers to Prevent Data Leakage Across Splits
GROUP_IDENTIFIER_COLUMNS = ["tunnel_id", "blast_id", "hole_id"]

# Feature Columns used for Model Training
FEATURE_COLUMNS = [
    "penetration_rate_m_hr",
    "torque_nm",
    "weight_on_bit_kg",
    "percussion_pressure_bar",
    "flushing_pressure_bar",
    "mwd_specific_energy_mj_m3",
    "rop_torque_ratio",
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
