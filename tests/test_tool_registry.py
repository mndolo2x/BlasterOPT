"""
Unit tests for Conversational AI Agent Tool Registry module (src/agent/tool_registry.py).
"""

import pytest
from pydantic import BaseModel, Field, ValidationError
from src.agent.tool_registry import (
    BlastDesignInput,
    FragmentationResult,
    VibrationResult,
    AirblastResult,
    DownstreamResult,
    BlastDesign,
    RegulationLimits,
    MWDData,
    BlastRecord,
    ToolRegistry,
    TOOL_REGISTRY,
)


def test_blast_design_input_pydantic_validation():
    """Test BlastDesignInput Pydantic model validation and default values."""
    inp = BlastDesignInput(
        bench_id="BENCH_JWA_15S",
        production_target_tonnes=25000.0,
    )

    assert inp.bench_id == "BENCH_JWA_15S"
    assert inp.production_target_tonnes == 25000.0
    assert inp.max_vibration_mm_s == 5.0
    assert inp.max_airblast_db == 120.0

    # Invalid missing required field should raise ValidationError
    with pytest.raises(ValidationError):
        BlastDesignInput(bench_id="BENCH_01")


def test_tool_registry_registration_and_execution():
    """Test registering custom tools in ToolRegistry and executing handlers with Pydantic validation."""
    registry = ToolRegistry()

    class SampleInput(BaseModel):
        x: float = Field(..., description="First value")
        y: float = Field(..., description="Second value")

    def sample_handler(x: float, y: float) -> float:
        return x + y

    registry.register_tool(
        name="add_numbers",
        description="Adds two floating point numbers together.",
        input_schema=SampleInput,
        handler=sample_handler,
    )

    assert registry.get_tool("add_numbers") is not None

    # Execute tool with dict arguments
    res = registry.execute_tool("add_numbers", {"x": 10.5, "y": 4.5})
    assert res == 15.0


def test_tool_registry_openai_specs_generation():
    """Test get_openai_tools_specs generates valid OpenAI function calling JSON schema."""
    specs = TOOL_REGISTRY.get_openai_tools_specs()

    assert isinstance(specs, list)
    assert len(specs) >= 2

    for s in specs:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "description" in s["function"]
        assert "parameters" in s["function"]


def test_default_tool_execution():
    """Test executing default tools registered in global TOOL_REGISTRY."""
    res_predict = TOOL_REGISTRY.execute_tool(
        "predict_blast_outcomes",
        {"bench_id": "BENCH_ORA_12N", "production_target_tonnes": 15000.0},
    )

    assert isinstance(res_predict, dict)
    assert res_predict["bench_id"] == "BENCH_ORA_12N"
    assert "predicted_fragmentation" in res_predict
    assert "predicted_vibration" in res_predict

    res_compliance = TOOL_REGISTRY.execute_tool(
        "check_regulatory_compliance",
        {"max_ppv_mm_s": 5.0, "max_airblast_db": 120.0},
    )

    assert isinstance(res_compliance, dict)
    assert "is_compliant" in res_compliance
