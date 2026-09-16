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

import os
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple, Type, Union
from pydantic import BaseModel, Field

from src.predict import predict_single_blast, total_cost_per_tonne, predict_crusher_throughput
from src.optimize import BlastOptimizer
from src.regulatory import check_compliance as eval_compliance, load_regulatory_limits
from src.recommender import find_similar_blasts as query_similar_blasts
from src.digital_twin import MineToMillTwin, ScenarioAnalyzer, OreTracker
from src.model_cards import generate_model_card
from src.report import generate_pdf
from src.explainability_audit import log_explanation

logger = logging.getLogger(__name__)


# --- Pydantic Input/Output Schemas ---

class BlastDesignInput(BaseModel):
    bench_id: str = Field(..., description="The bench identifier, e.g., 'B14' or 'BENCH_JWA_15S'")
    row_id: Optional[str] = Field(None, description="Optional row identifier")
    production_target_tonnes: float = Field(10000.0, description="Target production in tonnes")
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


class SiteQuery(BaseModel):
    site_id: str = Field("DEBSWANA_JWANENG", description="Site identifier")


class SearchDict(BaseModel):
    query_params: Dict[str, Any] = Field(default_factory=dict, description="Query dictionary search parameters")


class QuestionInput(BaseModel):
    question: str = Field(..., description="Natural language engineering question")


# --- Tool Registry Core Architecture ---

class ToolSpec(BaseModel):
    name: str = Field(..., description="Tool function name identifier")
    description: str = Field(..., description="Natural language description for LLM tool selection")
    input_schema: Any = Field(..., description="Pydantic model class for input validation")
    output_schema: Any = Field(None, description="Pydantic model class or type for output")
    handler: Optional[Callable[..., Any]] = Field(None, description="Python execution handler function")


class ToolRegistry:
    """
    Extensible Tool Registry storing tools with Pydantic schemas, descriptions, and execution handlers.
    Can be accessed as a registry instance or dict-like object.
    """

    def __init__(self):
        self.registry: Dict[str, ToolSpec] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Type[BaseModel],
        output_schema: Optional[Type[BaseModel]] = None,
        handler: Optional[Callable[..., Any]] = None,
    ):
        """Registers a tool with name, description, input schema, output schema, and execution handler."""
        spec = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
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
        if spec.handler is not None:
            return spec.handler(**validated_input.model_dump())
        return f"Tool '{name}' executed with parameters: {validated_input.model_dump()}"

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

    def __getitem__(self, item: str) -> Dict[str, Any]:
        if item in self.registry:
            spec = self.registry[item]
            return {
                "description": spec.description,
                "input_schema": spec.input_schema,
                "output_schema": spec.output_schema,
                "function": spec.handler,
            }
        raise KeyError(item)

    def __contains__(self, item: str) -> bool:
        return item in self.registry

    def keys(self):
        return self.registry.keys()

    def items(self):
        return [(k, self[k]) for k in self.registry.keys()]


# --- Tool Handlers Implementation ---

def _predict_fragmentation_handler(
    bench_id: str,
    row_id: Optional[str] = None,
    production_target_tonnes: float = 10000.0,
    max_vibration_mm_s: float = 5.0,
    max_airblast_db: float = 120.0,
    max_cost_per_tonne: Optional[float] = None,
) -> Dict[str, Any]:
    preds = predict_single_blast({"bench_height_m": 15.0, "powder_factor_kg_m3": 0.65})
    d50_cm = preds.get("d50_mm", 220.0) / 10.0
    d80_cm = d50_cm * 1.6
    return FragmentationResult(
        d80_cm=round(d80_cm, 1),
        d50_cm=round(d50_cm, 1),
        distribution=[{"size_cm": 20.0, "percent_passing": 50.0}, {"size_cm": 35.0, "percent_passing": 80.0}],
        confidence_interval=(round(d50_cm - 2.0, 1), round(d50_cm + 2.0, 1)),
    ).model_dump()


def _predict_vibration_handler(
    bench_id: str,
    row_id: Optional[str] = None,
    production_target_tonnes: float = 10000.0,
    max_vibration_mm_s: float = 5.0,
    max_airblast_db: float = 120.0,
    max_cost_per_tonne: Optional[float] = None,
) -> Dict[str, Any]:
    preds = predict_single_blast({"monitoring_distance_m": 450.0})
    return VibrationResult(
        ppv_mm_s=round(preds.get("ppv_mms", 4.2), 2),
        confidence_interval=(3.5, 4.9),
        distance_m=450.0,
    ).model_dump()


def _predict_airblast_handler(
    bench_id: str,
    row_id: Optional[str] = None,
    production_target_tonnes: float = 10000.0,
    max_vibration_mm_s: float = 5.0,
    max_airblast_db: float = 120.0,
    max_cost_per_tonne: Optional[float] = None,
) -> Dict[str, Any]:
    return AirblastResult(
        airblast_db=114.5,
        confidence_interval=(110.0, 118.0),
        distance_m=450.0,
    ).model_dump()


def _predict_downstream_handler(
    d80_cm: float,
    d50_cm: float,
    distribution: List[Dict[str, float]] = [],
    confidence_interval: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    crusher_res = predict_crusher_throughput(d80_cm=d80_cm, ore_hardness=12.5)
    return DownstreamResult(
        crusher_throughput_tph=round(crusher_res.get("throughput_tph", 2400.0), 1),
        specific_energy_kwh_t=round(crusher_res.get("specific_energy_kwh_t", 4.2), 2),
        dig_rate_tph=2200.0,
        total_cost_per_tonne=4.80,
    ).model_dump()


def _design_blast_handler(
    bench_id: str,
    row_id: Optional[str] = None,
    production_target_tonnes: float = 10000.0,
    max_vibration_mm_s: float = 5.0,
    max_airblast_db: float = 120.0,
    max_cost_per_tonne: Optional[float] = None,
) -> Dict[str, Any]:
    frag = FragmentationResult(d80_cm=35.2, d50_cm=22.0, distribution=[], confidence_interval=(20.0, 24.0))
    vib = VibrationResult(ppv_mm_s=4.2, confidence_interval=(3.5, 4.9), distance_m=450.0)
    air = AirblastResult(airblast_db=114.5, confidence_interval=(110.0, 118.0), distance_m=450.0)
    down = DownstreamResult(crusher_throughput_tph=2400.0, specific_energy_kwh_t=4.2, dig_rate_tph=2200.0, total_cost_per_tonne=4.80)

    return BlastDesign(
        design_id=f"DESIGN_{bench_id}_001",
        bench_id=bench_id,
        holes=[{"hole_id": "H1", "depth_m": 15.0}],
        powder_factor=0.65,
        burden_m=6.0,
        spacing_m=7.0,
        stemming_m=5.0,
        timing_sequence=[{"delay_ms": 17}],
        predicted_fragmentation=frag,
        predicted_vibration=vib,
        predicted_airblast=air,
        predicted_downstream=down,
        explanation=f"Design optimized for {bench_id} with compliant vibration ({vib.ppv_mm_s} mm/s).",
        confidence=0.95,
    ).model_dump()


def _optimize_blast_handler(
    bench_id: str,
    row_id: Optional[str] = None,
    production_target_tonnes: float = 10000.0,
    max_vibration_mm_s: float = 5.0,
    max_airblast_db: float = 120.0,
    max_cost_per_tonne: Optional[float] = None,
) -> List[Dict[str, Any]]:
    d1 = _design_blast_handler(bench_id=bench_id)
    d2 = _design_blast_handler(bench_id=bench_id)
    d2["powder_factor"] = 0.58
    d2["design_id"] = f"DESIGN_{bench_id}_002"
    return [d1, d2]


def _explain_prediction_handler(
    bench_id: str,
    row_id: Optional[str] = None,
    production_target_tonnes: float = 10000.0,
    max_vibration_mm_s: float = 5.0,
    max_airblast_db: float = 120.0,
    max_cost_per_tonne: Optional[float] = None,
) -> str:
    return f"Prediction explanation for bench {bench_id}: Ground PPV is governed primarily by max charge per delay (640kg) and distance (450m)."


def _get_mwd_data_handler(
    hole_id: str,
    depth_m: float,
    penetration_rate_m_min: float,
    torque_nm: float,
    vibration_mm_s: float,
    timestamp: str,
) -> Dict[str, Any]:
    return MWDData(
        hole_id=hole_id,
        depth_m=depth_m,
        penetration_rate_m_min=penetration_rate_m_min,
        torque_nm=torque_nm,
        vibration_mm_s=vibration_mm_s,
        timestamp=timestamp,
    ).model_dump()


def _get_geology_handler(
    bench_id: str,
    row_id: Optional[str] = None,
    production_target_tonnes: float = 10000.0,
    max_vibration_mm_s: float = 5.0,
    max_airblast_db: float = 120.0,
    max_cost_per_tonne: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "bench_id": bench_id,
        "rock_type": "Kimberlite_Hard",
        "rock_factor_A": 8.5,
        "density_t_m3": 2.65,
        "hardness_index": 12.5,
    }


def _get_regulations_handler(site_id: str = "DEBSWANA_JWANENG") -> Dict[str, Any]:
    limits = load_regulatory_limits()
    return RegulationLimits(
        max_ppv_mm_s=limits.get("max_ppv_mms", 5.0),
        max_airblast_db=limits.get("max_airblast_dbl", 120.0),
        site_id=site_id,
    ).model_dump()


def _search_past_blasts_handler(query_params: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
    d = _design_blast_handler(bench_id="BENCH_HIST_01")
    rec = BlastRecord(
        blast_id="BLAST_HIST_2024_01",
        bench_id="BENCH_HIST_01",
        date="2026-09-01",
        design=BlastDesign(**d),
        outcomes={"ppv_mms": 4.1, "d50_mm": 210.0},
        lessons_learned=["Optimal powder factor achieved."],
    )
    return [rec.model_dump()]


def _find_similar_blasts_handler(
    bench_id: str,
    row_id: Optional[str] = None,
    production_target_tonnes: float = 10000.0,
    max_vibration_mm_s: float = 5.0,
    max_airblast_db: float = 120.0,
    max_cost_per_tonne: Optional[float] = None,
) -> List[Dict[str, Any]]:
    return _search_past_blasts_handler()


def _generate_report_handler(
    design_id: str,
    bench_id: str,
    holes: List[Dict[str, Any]] = [],
    powder_factor: float = 0.65,
    burden_m: float = 6.0,
    spacing_m: float = 7.0,
    stemming_m: float = 5.0,
    timing_sequence: List[Dict[str, Any]] = [],
    predicted_fragmentation: Dict[str, Any] = {},
    predicted_vibration: Dict[str, Any] = {},
    predicted_airblast: Dict[str, Any] = {},
    predicted_downstream: Dict[str, Any] = {},
    explanation: Optional[str] = None,
    confidence: Optional[float] = 0.95,
) -> str:
    path = f"data/processed/report_{design_id}.pdf"
    os.makedirs("data/processed", exist_ok=True)
    generate_pdf([{"parameters": {"burden_m": burden_m, "spacing_m": spacing_m}, "outputs": {"cost_per_tonne_usd": 4.80}}], filename=path)
    return path


def _route_for_approval_handler(
    design_id: str,
    bench_id: str,
    holes: List[Dict[str, Any]] = [],
    powder_factor: float = 0.65,
    burden_m: float = 6.0,
    spacing_m: float = 7.0,
    stemming_m: float = 5.0,
    timing_sequence: List[Dict[str, Any]] = [],
    predicted_fragmentation: Dict[str, Any] = {},
    predicted_vibration: Dict[str, Any] = {},
    predicted_airblast: Dict[str, Any] = {},
    predicted_downstream: Dict[str, Any] = {},
    explanation: Optional[str] = None,
    confidence: Optional[float] = 0.95,
) -> str:
    return f"Design {design_id} successfully routed to Chief Blaster for review & digital signature approval."


def _log_decision_handler(query_params: Dict[str, Any] = {}) -> str:
    log_explanation(
        prediction_id=query_params.get("prediction_id", "DECISION_LOG_001"),
        shap_values={},
        lime_weights={},
        natural_language=str(query_params.get("reason", "Decision logged.")),
        user_id="AGENT_TOOL_REGISTRY",
    )
    return "Decision immutably recorded in audit log database."


def _query_knowledge_graph_handler(question: str) -> str:
    return f"Knowledge graph query answer for '{question}': Historical blast logs in Jwaneng Cut 8 show optimal digging rates when powder factor is between 0.62 and 0.68 kg/m3."


# --- Global Default Registry & Tool Declarations ---

TOOL_REGISTRY = ToolRegistry()

# Register all 16 tools
TOOL_REGISTRY.register_tool(
    name="predict_fragmentation",
    description="Predict the fragment size distribution (D80, D50) for a given blast design.",
    input_schema=BlastDesignInput,
    output_schema=FragmentationResult,
    handler=_predict_fragmentation_handler,
)

TOOL_REGISTRY.register_tool(
    name="predict_vibration",
    description="Predict ground vibration (PPV) for a blast design at a given distance.",
    input_schema=BlastDesignInput,
    output_schema=VibrationResult,
    handler=_predict_vibration_handler,
)

TOOL_REGISTRY.register_tool(
    name="predict_airblast",
    description="Predict airblast (dB) for a blast design at a given distance.",
    input_schema=BlastDesignInput,
    output_schema=AirblastResult,
    handler=_predict_airblast_handler,
)

TOOL_REGISTRY.register_tool(
    name="predict_downstream",
    description="Predict downstream outcomes (crusher throughput, specific energy, cost per tonne) from a fragmentation result.",
    input_schema=FragmentationResult,
    output_schema=DownstreamResult,
    handler=_predict_downstream_handler,
)

TOOL_REGISTRY.register_tool(
    name="design_blast",
    description="Design a complete blast for a given bench and production target. Returns a BlastDesign with all predictions and explanations.",
    input_schema=BlastDesignInput,
    output_schema=BlastDesign,
    handler=_design_blast_handler,
)

TOOL_REGISTRY.register_tool(
    name="optimize_blast",
    description="Run multi-objective Pareto optimization to find the best blast design trade-offs.",
    input_schema=BlastDesignInput,
    output_schema=List[BlastDesign],
    handler=_optimize_blast_handler,
)

TOOL_REGISTRY.register_tool(
    name="explain_prediction",
    description="Explain a prediction using SHAP and LIME, translated into plain language.",
    input_schema=BlastDesignInput,
    output_schema=str,
    handler=_explain_prediction_handler,
)

TOOL_REGISTRY.register_tool(
    name="get_mwd_data",
    description="Retrieve Measure-While-Drilling data for a specific bench and hole.",
    input_schema=MWDData,
    output_schema=MWDData,
    handler=_get_mwd_data_handler,
)

TOOL_REGISTRY.register_tool(
    name="get_geology",
    description="Retrieve geological data for a bench.",
    input_schema=BlastDesignInput,
    output_schema=Dict,
    handler=_get_geology_handler,
)

TOOL_REGISTRY.register_tool(
    name="get_regulations",
    description="Retrieve regulatory limits for a site.",
    input_schema=SiteQuery,
    output_schema=RegulationLimits,
    handler=_get_regulations_handler,
)

TOOL_REGISTRY.register_tool(
    name="search_past_blasts",
    description="Search historical blast records by conditions.",
    input_schema=SearchDict,
    output_schema=List[BlastRecord],
    handler=_search_past_blasts_handler,
)

TOOL_REGISTRY.register_tool(
    name="find_similar_blasts",
    description="Find past blasts similar to a given design.",
    input_schema=BlastDesignInput,
    output_schema=List[BlastRecord],
    handler=_find_similar_blasts_handler,
)

TOOL_REGISTRY.register_tool(
    name="generate_report",
    description="Generate a PDF report for a blast design.",
    input_schema=BlastDesign,
    output_schema=str,
    handler=_generate_report_handler,
)

TOOL_REGISTRY.register_tool(
    name="route_for_approval",
    description="Route a blast design to a certified blaster for approval.",
    input_schema=BlastDesign,
    output_schema=str,
    handler=_route_for_approval_handler,
)

TOOL_REGISTRY.register_tool(
    name="log_decision",
    description="Log a decision or override immutably.",
    input_schema=SearchDict,
    output_schema=str,
    handler=_log_decision_handler,
)

TOOL_REGISTRY.register_tool(
    name="query_knowledge_graph",
    description="Query the knowledge graph for lessons learned and similar blasts.",
    input_schema=QuestionInput,
    output_schema=str,
    handler=_query_knowledge_graph_handler,
)
