"""
Centralized, Auditable Site Configuration Service for BlastOpt Botswana.

Parses site-specific YAML limits (PPV, Airblast, Flyrock), validates configuration status,
flags unverified examples (`is_verified: False`), and provides strict limit getters that raise
errors on missing parameters instead of silently defaulting.
"""

import os
import yaml
import logging
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

DEFAULT_YAML_PATH = "config/site_limits.yaml"


def validate_site_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates if a parsed site configuration dictionary contains all required fields and valid statuses.

    Returns:
        (is_valid, errors)
    """
    errors = []

    if not isinstance(config, dict):
        return False, ["Configuration payload must be a dictionary."]

    required_top_level = ["site_id", "config_version", "status", "limits"]
    for req in required_top_level:
        if req not in config or config[req] is None:
            errors.append(f"Missing required top-level key: '{req}'")

    limits = config.get("limits", {})
    if not isinstance(limits, dict):
        errors.append("'limits' section must be a dictionary.")
        return False, errors

    required_limits = ["max_ppv_mm_s", "max_airblast", "max_flyrock_distance_m"]
    for lim in required_limits:
        if lim not in limits or limits[lim] is None:
            errors.append(f"Missing required limit entry: '{lim}'")
        else:
            entry = limits[lim]
            if not isinstance(entry, dict) or "value" not in entry or entry["value"] is None:
                errors.append(f"Limit entry '{lim}' must be a dictionary containing a non-null 'value'.")

    is_valid = len(errors) == 0
    return is_valid, errors


def load_site_config(site_id: str = "jwaneng", config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads and parses site configuration from YAML file.
    Evaluates verification status: If any limit is missing, invalid, or marked status: "unverified_example",
    returns `is_verified: False`.
    """
    path_to_load = config_path or DEFAULT_YAML_PATH

    if not os.path.exists(path_to_load):
        raise FileNotFoundError(f"Site configuration file not found at path: {path_to_load}")

    try:
        with open(path_to_load, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Failed to parse YAML site configuration at {path_to_load}: {e}")

    is_valid, errors = validate_site_config(raw_config)

    # Determine if configuration is verified
    is_verified = True
    if not is_valid:
        is_verified = False

    top_status = str(raw_config.get("status", "")).lower()
    if top_status in ["draft", "unverified_example", "unverified"]:
        is_verified = False

    limits = raw_config.get("limits", {})
    for lim_key, lim_data in limits.items():
        if isinstance(lim_data, dict):
            st = str(lim_data.get("status", "")).lower()
            if st in ["unverified_example", "unverified", "draft"]:
                is_verified = False

    parsed_res = dict(raw_config)
    parsed_res["is_verified"] = is_verified
    parsed_res["validation_errors"] = errors
    parsed_res["unverified_warning"] = (
        "⚠️ WARNING: This design was evaluated against unverified example limits. "
        "Do not use for operational decisions."
    ) if not is_verified else ""

    return parsed_res


def get_limit(limit_name: str, site_id: str = "jwaneng", config_path: Optional[str] = None) -> float:
    """
    Retrieves a specific numeric limit from site configuration.

    Parameters:
        limit_name: Key in limits section (e.g., 'max_ppv_mm_s', 'max_airblast', 'max_flyrock_distance_m')

    Raises:
        ValueError: If the limit is missing or invalid. DOES NOT SILENTLY DEFAULT TO 5.0.
    """
    config = load_site_config(site_id=site_id, config_path=config_path)
    limits = config.get("limits", {})

    # Support canonical alias mapping
    alias_map = {
        "max_ppv": "max_ppv_mm_s",
        "max_ppv_mms": "max_ppv_mm_s",
        "ppv_limit": "max_ppv_mm_s",
        "max_airblast_db": "max_airblast",
        "max_airblast_dbl": "max_airblast",
        "airblast_limit": "max_airblast",
        "max_flyrock_m": "max_flyrock_distance_m",
        "flyrock_limit": "max_flyrock_distance_m",
    }

    target_key = alias_map.get(limit_name, limit_name)

    if target_key not in limits:
        raise ValueError(
            f"Limit '{limit_name}' (target: '{target_key}') is missing from site configuration '{site_id}'. "
            f"Refusing to execute with silent defaults."
        )

    entry = limits[target_key]
    if not isinstance(entry, dict) or "value" not in entry or entry["value"] is None:
        raise ValueError(
            f"Limit '{limit_name}' in site configuration '{site_id}' has no valid numeric 'value'."
        )

    try:
        return float(entry["value"])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Limit '{limit_name}' value '{entry.get('value')}' cannot be converted to float: {e}")
