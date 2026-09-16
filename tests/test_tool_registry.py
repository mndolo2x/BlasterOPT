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
        BlastDesignInput()


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
    assert len(specs) == 16

    for s in specs:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "description" in s["function"]
        assert "parameters" in s["function"]


def test_all_16_registered_tools_execution():
    """Test executing all 16 registered tools in global TOOL_REGISTRY."""
    expected_tools = [
        "predict_fragmentation",
        "predict_vibration",
        "predict_airblast",
        "predict_downstream",
        "design_blast",
        "optimize_blast",
        "explain_prediction",
        "get_mwd_data",
        "get_geology",
        "get_regulations",
        "search_past_blasts",
        "find_similar_blasts",
        "generate_report",
        "route_for_approval",
        "log_decision",
        "query_knowledge_graph",
    ]

    for tool_name in expected_tools:
        assert tool_name in TOOL_REGISTRY, f"Tool '{tool_name}' missing from TOOL_REGISTRY"
        spec = TOOL_REGISTRY.get_tool(tool_name)
        assert spec is not None
        assert spec.input_schema is not None
        assert spec.description != ""

    # Test executing representative tools
    res_frag = TOOL_REGISTRY.execute_tool("predict_fragmentation", {"bench_id": "BENCH_01"})
    assert "d80_cm" in res_frag
    assert "d50_cm" in res_frag

    res_vib = TOOL_REGISTRY.execute_tool("predict_vibration", {"bench_id": "BENCH_01"})
    assert "ppv_mm_s" in res_vib

    res_air = TOOL_REGISTRY.execute_tool("predict_airblast", {"bench_id": "BENCH_01"})
    assert "airblast_db" in res_air

    res_design = TOOL_REGISTRY.execute_tool("design_blast", {"bench_id": "BENCH_01"})
    assert res_design["bench_id"] == "BENCH_01"

    res_mwd = TOOL_REGISTRY.execute_tool(
        "get_mwd_data",
        {
            "hole_id": "HOLE_01",
            "depth_m": 15.0,
            "penetration_rate_m_min": 0.6,
            "torque_nm": 1200.0,
            "vibration_mm_s": 3.2,
            "timestamp": "2026-09-16T12:00:00",
        },
    )
    assert res_mwd["hole_id"] == "HOLE_01"

    res_kg = TOOL_REGISTRY.execute_tool("query_knowledge_graph", {"question": "Optimal powder factor?"})
    assert "Knowledge graph query answer" in res_kg
