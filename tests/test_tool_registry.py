"""
Unit tests for Conversational AI Agent Tool Registry module (src/agent/tool_registry.py and src/agent/tool_binder.py).
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
    SiteQuery,
    SearchDict,
    QuestionInput,
    ToolRegistry,
    TOOL_REGISTRY,
)
from src.agent.tool_binder import bind_tools


def test_pydantic_schemas_valid_inputs():
    """Verify that all Pydantic schemas validate correct inputs."""
    # BlastDesignInput
    bdi = BlastDesignInput(bench_id="BENCH_01", production_target_tonnes=10000.0)
    assert bdi.bench_id == "BENCH_01"

    # FragmentationResult
    frag = FragmentationResult(d80_cm=35.0, d50_cm=20.0)
    assert frag.d80_cm == 35.0

    # VibrationResult
    vib = VibrationResult(ppv_mm_s=4.5)
    assert vib.ppv_mm_s == 4.5

    # AirblastResult
    air = AirblastResult(airblast_db=115.0)
    assert air.airblast_db == 115.0

    # DownstreamResult
    down = DownstreamResult(crusher_throughput_tph=2500.0, specific_energy_kwh_t=4.1, dig_rate_tph=2100.0, total_cost_per_tonne=4.80)
    assert down.crusher_throughput_tph == 2500.0

    # BlastDesign
    design = BlastDesign(
        design_id="DES_01",
        bench_id="BENCH_01",
        powder_factor=0.65,
        burden_m=6.0,
        spacing_m=7.0,
        stemming_m=5.0,
        predicted_fragmentation=frag,
        predicted_vibration=vib,
        predicted_airblast=air,
        predicted_downstream=down,
    )
    assert design.design_id == "DES_01"

    # RegulationLimits
    reg = RegulationLimits(max_ppv_mm_s=5.0, max_airblast_db=120.0, site_id="DEBSWANA_JWANENG")
    assert reg.max_ppv_mm_s == 5.0

    # MWDData
    mwd = MWDData(hole_id="H1", depth_m=15.0, penetration_rate_m_min=0.5, torque_nm=1200.0, vibration_mm_s=2.1, timestamp="2026-09-16T12:00:00")
    assert mwd.hole_id == "H1"

    # BlastRecord
    rec = BlastRecord(blast_id="BL_01", bench_id="BENCH_01", date="2026-09-16", design=design)
    assert rec.blast_id == "BL_01"

    # SiteQuery, SearchDict, QuestionInput
    sq = SiteQuery(site_id="SITE_01")
    assert sq.site_id == "SITE_01"

    sd = SearchDict(query_params={"key": "val"})
    assert sd.query_params["key"] == "val"

    qi = QuestionInput(question="What is the optimal powder factor?")
    assert qi.question == "What is the optimal powder factor?"


def test_pydantic_schemas_invalid_inputs():
    """Verify that invalid inputs raise ValidationError."""
    with pytest.raises(ValidationError):
        BlastDesignInput()  # Missing bench_id

    with pytest.raises(ValidationError):
        FragmentationResult(d80_cm="invalid_float")  # Invalid float type

    with pytest.raises(ValidationError):
        MWDData(hole_id="H1")  # Missing required numerical fields


def test_bind_tools_returns_registry_with_bound_functions():
    """Verify bind_tools() returns a registry with all functions bound."""
    registry = bind_tools()

    assert isinstance(registry, ToolRegistry)
    assert len(registry.keys()) == 20

    for tool_name in registry.keys():
        spec = registry.get_tool(tool_name)
        assert spec is not None
        assert spec.handler is not None, f"Tool '{tool_name}' missing handler function"
        assert callable(spec.handler), f"Tool '{tool_name}' handler is not callable"


def test_tool_execution_via_registry():
    """Test executing registered tools via registry execute_tool."""
    res_frag = TOOL_REGISTRY.execute_tool("predict_fragmentation", {"bench_id": "BENCH_01"})
    assert "d80_cm" in res_frag

    res_vib = TOOL_REGISTRY.execute_tool("predict_vibration", {"bench_id": "BENCH_01"})
    assert "ppv_mm_s" in res_vib

    res_kg = TOOL_REGISTRY.execute_tool("query_knowledge_graph", {"question": "What is ANFO?"})
    assert "ISEE Handbook" in res_kg or "Knowledge Graph Result" in res_kg
    assert "ammonium nitrate" in res_kg.lower()

    res_dep = TOOL_REGISTRY.execute_tool("query_knowledge_graph", {"question": "What is cube root scaled distance?"})
    assert "PA DEP § 211.101" in res_dep or "Knowledge Graph Result" in res_dep
    assert "airblast" in res_dep.lower()
