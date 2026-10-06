"""
Bench Context Loader Module.
Loads bench parameters (tonnes, grade, commodity price) from YAML configuration files.
"""

import os
import yaml
from typing import Optional
from pydantic import BaseModel, Field


class BenchContext(BaseModel):
    """Pydantic model representing site bench operational context."""

    bench_id: str = Field(..., description="Unique bench identifier")
    site_id: str = Field(..., description="Mine site name")
    tonnes: float = Field(..., gt=0, description="Total bench ore tonnage")
    grade: float = Field(..., ge=0, description="Ore grade (carats/t, g/t, etc.)")
    grade_unit: str = Field("carats_per_tonne", description="Unit of grade measurement")
    commodity_price: float = Field(..., gt=0, description="Commodity unit selling price USD")
    commodity_price_unit: str = Field("USD_per_carat", description="Commodity price unit")
    discount_rate: float = Field(0.10, ge=0.0, le=0.50, description="Annual discount rate for NPV")
    bench_life_years: float = Field(1.0, gt=0.0, description="Estimated bench life in years")


def load_bench_context(bench_id: str, site_id: Optional[str] = None, config_dir: str = "config/benches") -> BenchContext:
    """
    Loads bench context from YAML configuration file.

    Parameters:
    -----------
    bench_id : str
        ID of bench (e.g. 'jwaneng_bench_14').
    site_id : str, optional
        Site name filter override.
    config_dir : str, default='config/benches'
        Directory containing bench context YAML files.

    Returns:
    --------
    BenchContext
        Validated Pydantic BenchContext model instance.
    """
    file_name = f"{bench_id}.yaml" if not bench_id.endswith(".yaml") else bench_id
    file_path = os.path.join(config_dir, file_name)

    if not os.path.exists(file_path):
        # Fallback default context if YAML file does not exist
        return BenchContext(
            bench_id=bench_id,
            site_id=site_id or "Jwaneng Mine",
            tonnes=250000.0,
            grade=0.42,
            grade_unit="carats_per_tonne",
            commodity_price=180.0,
            commodity_price_unit="USD_per_carat",
            discount_rate=0.10,
            bench_life_years=1.0,
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if site_id:
        data["site_id"] = site_id

    return BenchContext(**data)
