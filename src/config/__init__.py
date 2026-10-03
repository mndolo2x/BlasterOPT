"""
Central Configuration & DEMO_MODE Flags for BlastOpt Botswana.

Controls application-wide data provenance, DEMO_MODE behavior, and strict error handling
when real hardware / API streams are unavailable.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Single environment variable controlling DEMO_MODE application behavior
DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() == "true"


def get_demo_mode() -> bool:
    """Returns True if DEMO_MODE is active."""
    return os.getenv("DEMO_MODE", "true").lower() == "true"


def require_real_data(source_name: str) -> None:
    """
    Fails loudly by raising RuntimeError if DEMO_MODE is False and real data/hardware streams are unavailable.
    Never silently substitutes mock data for missing real data.
    """
    if not get_demo_mode():
        err_msg = (
            f"REAL DATA REQUIRED — DEMO_MODE is False, but real data endpoint or stream "
            f"for '{source_name}' is unavailable. Automatic fallback to simulated/mock data is disabled."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)


def dump_model(obj: Any) -> Dict[str, Any]:
    """Safely serializes a Pydantic v1 or v2 model to a dictionary."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        return obj.dict()
    elif isinstance(obj, dict):
        return obj
    return dict(obj)


def get_provenance_badge(
    source: str = "synthetic",
    timestamp: Optional[str] = None,
    location: str = "Jwaneng_Cut8_Bench12",
    quality: str = "measured",
    model_version: Optional[str] = None
) -> Dict[str, str]:
    """
    Constructs a standardized data provenance dictionary / metadata badge.

    Fields:
    -------
    source : str
        Data origin ("synthetic", "uploaded", "API", "field_device")
    timestamp : str
        ISO UTC capture timestamp
    location : str
        Site or bench origin
    quality : str
        Data quality category ("measured", "predicted", "imputed")
    model_version : str, optional
        Model version if predicted
    """
    from datetime import datetime, timezone
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    badge = {
        "Source": source,
        "Timestamp": ts,
        "Site/Bench": location,
        "Data Quality": quality,
    }
    if model_version:
        badge["Model Version"] = model_version
    return badge
