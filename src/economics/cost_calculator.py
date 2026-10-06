"""
Cost Calculator Module with Sanity Checks for BlasterOPT.
Calculates drill-and-blast unit costs and enforces hard reasonable range checks.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from src.predict import total_cost_per_tonne
from src.services.audit_service import AuditService

MIN_REASONABLE_COST = 1.00   # USD per tonne
MAX_REASONABLE_COST = 10.00  # USD per tonne


class CostEstimate(BaseModel):
    """Pydantic model representing cost estimate and sanity status."""

    total_cost_usd: float = Field(..., description="Total cost in USD")
    cost_per_tonne_usd: float = Field(..., description="Unit cost per tonne USD/t")
    status: Literal["OK", "REQUIRES_REVIEW", "INVALID"] = Field("OK", description="Cost estimate status")
    warnings: List[str] = Field(default_factory=list, description="Sanity warnings")


def calculate_cost(design: Dict[str, Any], site_id: str = "Jwaneng Mine") -> CostEstimate:
    """
    Calculates cost per tonne and applies hard sanity bounds ($1.00 - $10.00 / t).

    Parameters:
    -----------
    design : Dict[str, Any]
        Blast design parameter dictionary.
    site_id : str, default='Jwaneng Mine'
        Mine site identifier.

    Returns:
    --------
    CostEstimate
        CostEstimate model instance.
    """
    if "cost_per_tonne_usd" in design and isinstance(design["cost_per_tonne_usd"], (int, float)):
        raw_cost = float(design["cost_per_tonne_usd"])
    else:
        cost_dict = total_cost_per_tonne(design)
        raw_cost = float(cost_dict.get("total_cost_usd_t", cost_dict.get("cost_per_tonne_usd", 4.80)))

    warnings = []

    if raw_cost < MIN_REASONABLE_COST or raw_cost > MAX_REASONABLE_COST:
        status = "INVALID"
        msg = (
            f"Cost prediction {raw_cost:.2f} USD/tonne is outside "
            f"the reasonable range [{MIN_REASONABLE_COST:.2f}, {MAX_REASONABLE_COST:.2f}]. "
            f"The design is flagged as INVALID."
        )
        warnings.append(msg)

        # Log for audit
        audit = AuditService()
        audit.log_event(
            event_type="cost_prediction_out_of_range",
            user_id="COST_CALCULATOR",
            payload={
                "event_type": "cost_prediction_out_of_range",
                "raw_value": raw_cost,
                "design": design,
                "site_id": site_id,
            },
        )

        return CostEstimate(
            total_cost_usd=round(raw_cost, 2),
            cost_per_tonne_usd=round(raw_cost, 2),
            status="INVALID",
            warnings=warnings,
        )

    return CostEstimate(
        total_cost_usd=round(raw_cost, 2),
        cost_per_tonne_usd=round(raw_cost, 2),
        status="OK",
        warnings=[],
    )
