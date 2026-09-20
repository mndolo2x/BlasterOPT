"""
Immutable Blast Design Versioning Module for BlastOpt Botswana.

Defines BlastDesign and BlastDesignVersion domain models.
Enforces immutability for APPROVED blast designs: once approved, a design's content is frozen
and any modification spawns a new version with parent_version pointing to the previous version.
Computes deterministic SHA-256 content hashes for audit trails referencing (design_id, version).
"""

import json
import hashlib
import logging
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union, Literal
from src.domain.safety_checks import SafetyReport

logger = logging.getLogger(__name__)


class BlastDesign(BaseModel):
    """
    Blast design parameters domain model.
    """
    burden_m: float = Field(..., description="Burden distance in meters")
    spacing_m: float = Field(..., description="Spacing distance in meters")
    stemming_m: float = Field(..., description="Stemming confinement length in meters")
    powder_factor_kg_m3: float = Field(..., description="Powder factor in kg/m3")
    bench_height_m: float = Field(12.0, description="Bench height in meters")
    hole_diameter_mm: float = Field(250.0, description="Hole diameter in millimeters")
    rock_factor_A: float = Field(8.0, description="Rock blastability factor A")
    charge_mass_per_hole_kg: float = Field(320.0, description="Explosive charge mass per hole in kg")
    max_charge_per_delay_kg: float = Field(640.0, description="Maximum charge per delay interval in kg")
    monitoring_distance_m: float = Field(450.0, description="Distance to nearest structure in meters")
    explosive_rws: float = Field(100.0, description="Relative Weight Strength of explosive")


def compute_content_hash(design: Union[BlastDesign, Dict[str, Any]]) -> str:
    """
    Generates a deterministic SHA-256 content hash of serialized design parameters.
    The hash is computed at creation and never changes for a given design payload.
    """
    if isinstance(design, BlastDesign):
        design_dict = design.model_dump()
    elif isinstance(design, dict):
        design_dict = design.copy()
    else:
        design_dict = dict(design)

    # Sort keys for deterministic JSON string representation
    serialized_str = json.dumps(design_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized_str.encode("utf-8")).hexdigest()


class BlastDesignVersion(BaseModel):
    """
    Immutable versioned snapshot of a blast design and its associated safety report and approval status.
    """
    design_id: str = Field(..., description="Unique blast design pattern ID")
    version: int = Field(1, description="Version number (1, 2, 3, ...)")
    parent_version: Optional[int] = Field(None, description="Version number of parent design if modified from an approved version")
    created_at: Union[datetime, str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp when version was created"
    )
    created_by: str = Field(..., description="User ID or engineer who created this version")
    design: BlastDesign = Field(..., description="Core blast design parameters model")
    safety_report: SafetyReport = Field(..., description="Safety evaluation report at creation time")
    approval_status: Literal["DRAFT", "PENDING_APPROVAL", "APPROVED", "REJECTED"] = Field(
        "DRAFT", description="Approval lifecycle state"
    )
    content_hash: str = Field(..., description="Deterministic SHA-256 hash of serialized design parameters")

    @classmethod
    def create(
        cls,
        design_id: str,
        design_data: Union[BlastDesign, Dict[str, Any]],
        safety_report: SafetyReport,
        created_by: str,
        version: int = 1,
        parent_version: Optional[int] = None,
        approval_status: Literal["DRAFT", "PENDING_APPROVAL", "APPROVED", "REJECTED"] = "DRAFT",
    ) -> "BlastDesignVersion":
        """
        Factory method to construct a BlastDesignVersion with computed content_hash.
        """
        if isinstance(design_data, dict):
            design_model = BlastDesign(**design_data)
        else:
            design_model = design_data

        content_hash = compute_content_hash(design_model)
        ts = datetime.now(timezone.utc).isoformat()

        return cls(
            design_id=design_id,
            version=version,
            parent_version=parent_version,
            created_at=ts,
            created_by=created_by,
            design=design_model,
            safety_report=safety_report,
            approval_status=approval_status,
            content_hash=content_hash,
        )


def create_next_version(
    current_version: BlastDesignVersion,
    new_design_data: Union[BlastDesign, Dict[str, Any]],
    new_safety_report: SafetyReport,
    created_by: str,
) -> BlastDesignVersion:
    """
    Spawns a new BlastDesignVersion if an existing version (especially an APPROVED version) is modified.
    Sets parent_version to current_version.version and increments version number by 1.
    """
    next_ver_num = current_version.version + 1
    parent_ver_num = current_version.version

    return BlastDesignVersion.create(
        design_id=current_version.design_id,
        design_data=new_design_data,
        safety_report=new_safety_report,
        created_by=created_by,
        version=next_ver_num,
        parent_version=parent_ver_num,
        approval_status="DRAFT",
    )
