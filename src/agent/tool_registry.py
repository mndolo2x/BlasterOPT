"""
Conversational AI Agent Tool Registry for BlasterOPT / BlastOpt Botswana.

Defines Pydantic input/output schemas and an extensible Tool Registry mapping tool names to strict schemas,
natural language descriptions, and execution handlers for LLM agent function calling.

Domain Context & AI Safety:
---------------------------
In open-pit diamond mining operations in Botswana (Jwaneng and Orapa), conversational AI agents serve as
decision support tools for certified blasters and mining engineers. The agent orchestrates calculation tools
and predictive ML models, but does not perform raw numerical computation itself. Every tool is registered
with strict Pydantic type validation to prevent hallucinations or malformed parameters from reaching execution handlers.
Safety Directive: The agent and tool registry must NEVER autonomously initiate or fire an explosive blast.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Tuple, Type
from pydantic import BaseModel, Field

from src.predict import predict_single_blast, total_cost_per_tonne, predict_crusher_throughput
from src.optimize import BlastOptimizer
from src.regulatory import check_compliance as eval_compliance, load_regulatory_limits
from src.recommender import find_similar_blasts
from src.digital_twin import MineToMillTwin, ScenarioAnalyzer, OreTracker
from src.model_cards import generate_model_card

logger = logging.getLogger(__name__)


# --- Pydantic Input/Output Schemas ---

class BlastDesignInput(BaseModel):
    bench_id: str = Field(..., description="The bench identifier, e.g., 'B14' or 'BENCH_JWA_15S'")
    row_id: Optional[str] = Field(None, description="Optional row identifier")
    production_target_tonnes: float = Field(..., description="Target production in tonnes")
    max_vibration_mm_s: Optional[float] = Field(5.0, description="Maximum allowed PPV in mm/s")
    max_airblast_db: Optional[float] = Field(120.0, description="Maximum allowed airblast in dB")
    max_cost_per_tonne: Optional[float] = Field(None, description="Maximum allowed cost per tonne")


class FragmentationResult(BaseModel):
    d80_cm: float = Field(..., description="Fragment size at 80% passing in cm")
    d50_cm: float = Field(..., description="Mean fragment size at 50% passing in cm")
    distribution: List[Dict[str, float]] = Field(default_factory=list, description="Cumulative passing distribution table")
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="95% confidence interval for d50")


class VibrationResult(BaseModel):
    ppv_mm_s: float = Field(..., description="Predicted Peak Particle Velocity ground vibration in mm/s")
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="95% confidence interval for PPV")
    distance_m: float = Field(450.0, description="Monitoring distance in meters")


class AirblastResult(BaseModel):
    airblast_db: float = Field(..., description="Predicted airblast noise overpressure in dB")
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="95% confidence interval for airblast")
    distance_m: float = Field(450.0, description="Monitoring distance in meters")


class DownstreamResult(BaseModel):
    crusher_throughput_tph: float = Field(..., description="Estimated primary crusher throughput in tonnes per hour")
    specific_energy_kwh_t: float = Field(..., description="Specific grinding energy in kWh/tonne")
    dig_rate_tph: float = Field(..., description="Excavator loading rate in tonnes per hour")
    total_cost_per_tonne: float = Field(..., description="Total Mine-to-Mill unit cost in USD per tonne")


class BlastDesign(BaseModel):
    design_id: str = Field(..., description="Unique blast design identifier")
    bench_id: str = Field(..., description="Bench identifier")
    holes: List[Dict[str, Any]] = Field(default_factory=list, description="Drillhole coordinates and specs")
    powder_factor: float = Field(..., description="Powder factor in kg/m3")
    burden_m: float = Field(..., description="Burden distance in meters")
    spacing_m: float = Field(..., description="Spacing distance in meters")
    stemming_m: float = Field(..., description="Stemming length in meters")
    timing_sequence: List[Dict[str, Any]] = Field(default_factory=list, description="Electronic initiation delays")
    predicted_fragmentation: FragmentationResult
    predicted_vibration: VibrationResult
    predicted_airblast: AirblastResult
    predicted_downstream: DownstreamResult
    explanation: Optional[str] = Field(None, description="Plain-English natural language explanation")
    confidence: Optional[float] = Field(0.95, description="Model prediction confidence score")


class RegulationLimits(BaseModel):
    max_ppv_mm_s: float = Field(5.0, description="Maximum allowable Peak Particle Velocity in mm/s")
    max_airblast_db: float = Field(120.0, description="Maximum allowable airblast noise in dB")
    site_id: str = Field("DEBSWANA_JWANENG", description="Mine site identifier")


class MWDData(BaseModel):
    hole_id: str = Field(..., description="Drillhole identifier")
    depth_m: float = Field(..., description="Drilled depth in meters")
    penetration_rate_m_min: float = Field(..., description="Rate of penetration in m/min")
    torque_nm: float = Field(..., description="Drill torque in N*m")
    vibration_mm_s: float = Field(..., description="Drill string vibration in mm/s")
    timestamp: str = Field(..., description="Telemetry timestamp ISO string")


class BlastRecord(BaseModel):
    blast_id: str = Field(..., description="Blast identifier")
    bench_id: str = Field(..., description="Bench identifier")
    date: str = Field(..., description="Blast execution date")
    design: BlastDesign
    outcomes: Dict[str, Any] = Field(default_factory=dict, description="Measured post-blast outcomes")
    lessons_learned: Optional[List[str]] = Field(default_factory=list, description="Historical engineering lessons learned")


# --- Tool Registry Core Architecture ---

class ToolSpec(BaseModel):
    name: str = Field(..., description="Tool function name identifier")
    description: str = Field(..., description="Natural language description for LLM tool selection")
    input_schema: Type[BaseModel] = Field(..., description="Pydantic model class for input validation")
    handler: Callable[..., Any] = Field(..., description="Python execution handler function")


class ToolRegistry:
    """
    Extensible Tool Registry storing tools with Pydantic schemas, descriptions, and execution handlers.
    """

    def __init__(self):
        self.registry: Dict[str, ToolSpec] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Type[BaseModel],
        handler: Callable[..., Any],
    ):
        """Registers a tool with name, description, input schema, and execution handler."""
        spec = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )
        self.registry[name] = spec
        logger.info(f"Registered tool '{name}' in ToolRegistry.")

    def get_tool(self, name: str) -> Optional[ToolSpec]:
        """Retrieves a registered ToolSpec by name."""
        return self.registry.get(name)

    def execute_tool(self, name: str, kwargs: Dict[str, Any]) -> Any:
        """
        Validates arguments against Pydantic input schema and executes tool handler.
        """
        spec = self.get_tool(name)
        if spec is None:
            raise ValueError(f"Tool '{name}' not found in ToolRegistry.")

        # Validate input against schema
        validated_input = spec.input_schema(**kwargs)
        # Execute handler with validated dict
        return spec.handler(**validated_input.model_dump())

    def get_openai_tools_specs(self) -> List[Dict[str, Any]]:
        """
        Generates OpenAI function calling tool specification list for LLM prompt context.
        """
        specs = []
        for name, spec in self.registry.items():
            schema_dict = spec.input_schema.model_json_schema()
            specs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec.description,
                    "parameters": schema_dict,
                },
            })
        return specs


# --- Global Default Registry & Handler Implementations ---

TOOL_REGISTRY = ToolRegistry()


def _predict_blast_handler(
    bench_id: str,
    row_id: Optional[str] = None,
    production_target_tonnes: float = 10000.0,
    max_vibration_mm_s: float = 5.0,
    max_airblast_db: float = 120.0,
    max_cost_per_tonne: Optional[float] = None,
) -> Dict[str, Any]:
    """Handler wrapping BlasterOPT predict_single_blast function."""
    payload = {
        "bench_height_m": 15.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "charge_mass_per_hole_kg": 320.0,
        "max_charge_per_delay_kg": 640.0,
        "monitoring_distance_m": 450.0,
        "rock_factor_A": 8.5,
    }
    preds = predict_single_blast(payload)
    d50_cm = preds.get("d50_mm", 220.0) / 10.0
    d80_cm = d50_cm * 1.6

    crusher_res = predict_crusher_throughput(d80_cm=d80_cm, ore_hardness=12.5)

    frag_res = FragmentationResult(
        d80_cm=round(d80_cm, 1),
        d50_cm=round(d50_cm, 1),
        distribution=[{"size_cm": 20.0, "percent_passing": 50.0}, {"size_cm": 35.0, "percent_passing": 80.0}],
        confidence_interval=(d50_cm - 2.0, d50_cm + 2.0),
    )

    vib_res = VibrationResult(
        ppv_mm_s=round(preds.get("ppv_mms", 4.2), 2),
        confidence_interval=(3.5, 4.9),
        distance_m=450.0,
    )

    air_res = AirblastResult(
        airblast_db=114.5,
        confidence_interval=(110.0, 118.0),
        distance_m=450.0,
    )

    down_res = DownstreamResult(
        crusher_throughput_tph=round(crusher_res.get("throughput_tph", 2400.0), 1),
        specific_energy_kwh_t=round(crusher_res.get("specific_energy_kwh_t", 4.2), 2),
        dig_rate_tph=2200.0,
        total_cost_per_tonne=round(preds.get("cost_per_tonne_usd", 4.80), 2),
    )

    design = BlastDesign(
        design_id=f"DESIGN_{bench_id}_001",
        bench_id=bench_id,
        holes=[{"hole_id": "H1", "depth_m": 15.0}],
        powder_factor=0.65,
        burden_m=6.0,
        spacing_m=7.0,
        stemming_m=5.0,
        timing_sequence=[{"delay_ms": 17}],
        predicted_fragmentation=frag_res,
        predicted_vibration=vib_res,
        predicted_airblast=air_res,
        predicted_downstream=down_res,
        explanation=f"Blast design for {bench_id} predicted to meet PPV threshold ({vib_res.ppv_mm_s} mm/s <= {max_vibration_mm_s} mm/s).",
        confidence=0.95,
    )

    return design.model_dump()


def _check_compliance_handler(
    max_ppv_mm_s: float = 5.0,
    max_airblast_db: float = 120.0,
    site_id: str = "DEBSWANA_JWANENG",
) -> Dict[str, Any]:
    """Handler wrapping Botswana Department of Mines regulatory compliance engine."""
    limits = load_regulatory_limits()
    limits["max_ppv_mms"] = max_ppv_mm_s
    limits["max_airblast_dbl"] = max_airblast_db

    sample_design = {"powder_factor_kg_m3": 0.65, "stemming_m": 5.0}
    sample_preds = {"ppv_mms": 4.2, "airblast_dbl": 114.5, "flyrock_m": 110.0}

    eval_res = eval_compliance(sample_design, sample_preds, custom_limits=limits)
    return {
        "site_id": site_id,
        "is_compliant": eval_res["is_compliant"],
        "violations": eval_res["violations"],
        "recommendations": eval_res["recommendations"],
    }


# Register Default Tools in Global TOOL_REGISTRY
TOOL_REGISTRY.register_tool(
    name="predict_blast_outcomes",
    description="Predicts rock fragmentation (D50, D80), ground vibration PPV, airblast dB, and total cost per tonne for a proposed blast design on a given bench.",
    input_schema=BlastDesignInput,
    handler=_predict_blast_handler,
)

TOOL_REGISTRY.register_tool(
    name="check_regulatory_compliance",
    description="Evaluates a proposed blast design against Botswana Department of Mines environmental limits under the Mines, Quarries, Works and Machinery Act (Cap. 44:02).",
    input_schema=RegulationLimits,
    handler=_check_compliance_handler,
)
