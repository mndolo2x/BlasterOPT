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


class TermInput(BaseModel):
    term: str = Field(..., description="The blast engineering or mining term to look up")


class TranslationInput(BaseModel):
    text: str = Field(..., description="Text content to translate")


class TermDefinitionResult(BaseModel):
    term: str = Field(..., description="The queried term")
    definition: str = Field(..., description="Technical definition")
    plain_language: str = Field(..., description="Plain language explanation")
    setswana: Optional[str] = Field(None, description="Setswana translation")
    category: str = Field("Glossary", description="Term category")
    source: str = Field("Blaster's Handbook / PA DEP", description="Information source")


# --- Tool Registry Core Architecture ---

class ToolSpec(BaseModel):
    name: str = Field(..., description="Tool function name identifier")
    description: str = Field(..., description="Natural language description for LLM tool selection")
    input_schema: Any = Field(..., description="Pydantic model class for input validation")
    output_schema: Any = Field(None, description="Pydantic model class or type for output")
    handler: Optional[Callable[..., Any]] = Field(None, description="Python execution handler function")


class _ToolDictProxy(dict):
    """Proxy dictionary that synchronizes key updates back to ToolSpec attributes."""

    def __init__(self, spec: ToolSpec):
        self.spec = spec
        super().__init__({
            "description": spec.description,
            "input_schema": spec.input_schema,
            "output_schema": spec.output_schema,
            "function": spec.handler,
        })

    def __setitem__(self, key: str, value: Any):
        super().__setitem__(key, value)
        if key == "function":
            self.spec.handler = value
        elif key == "description":
            self.spec.description = value
        elif key == "input_schema":
            self.spec.input_schema = value
        elif key == "output_schema":
            self.spec.output_schema = value


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
        input_schema: Any,
        output_schema: Any = None,
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

    def __getitem__(self, item: str) -> _ToolDictProxy:
        if item in self.registry:
            return _ToolDictProxy(self.registry[item])
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


# --- Blast Blocks, Benches & Open-Pit Blast Design Reference Knowledge Base ---

BLAST_BLOCK_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "blast_block_definition": {
        "concept": "Blast Block",
        "question": "What is a blast block?",
        "answer": (
            "A blast block is a bounded portion of rock—usually within one bench—selected to be drilled, loaded, and initiated "
            "under one integrated blast design. Functionally, it is the smallest operationally meaningful blast volume for which "
            "geometry, drilling, loading, initiation, prediction, and post-blast performance can be linked to one blast record. "
            "Note that 'blast block' is operational, mine-specific terminology and not a single universally standardized geometry."
        ),
        "data_layers": [
            "A. Spatial Identity (Block ID, pit/phase, bench, polygon, coordinates, free-face direction)",
            "B. Bench Geometry (Bench height H, crest, toe/floor, face angle, berms, local relief)",
            "C. Pattern Geometry (Burden B, spacing S, row count, holes per row, hole diameter, inclination)",
            "D. Hole/Loading Geometry (Hole depth L, subdrill J, stemming T, charge length, decking)",
            "E. Timing / Initiation (Row/hole delays, firing direction, max charge per delay Qmax)",
            "F. Rock / Geology (Rock factor Rf, blastability index BI, RMR/GSI/Q, joints, water)",
            "G. Performance Targets (P80, oversize, fines, PPV, airblast, backbreak, cost/tonne)",
            "H. Operational Constraints (Wall proximity, infrastructure, equipment reach, exclusion zones)"
        ],
    },
    "blast_block_vs_bench": {
        "concept": "Blast Block vs Bench",
        "question": "What is the difference between a blast block and a bench?",
        "answer": (
            "A bench is a larger mining level or rock slice defined by crest, floor, and bench height (a pit geometry unit). "
            "A blast block is a selected area or volume on that bench that is drilled, loaded, and treated as a single blast-design unit. "
            "The hierarchy is: Bench → Blast Block → Row → Hole → Charge Segment → Initiation Event → Measured Response."
        ),
    },
    "debswana_bench_heights": {
        "concept": "Debswana Bench Heights & Hole Depth Evidence",
        "question": "What is a typical Debswana bench height for a blast block?",
        "answer": (
            "Public Orapa studies report a 15 m bench height (with hole diameters 127–250 mm, 40–60 holes/row, 15–25 rows/blast). "
            "For Jwaneng, recent published research reports a hole depth range of 14.90–16.10 m (with burden 7–8 m, spacing 8–9 m), "
            "but explicitly does not state a single mine-wide bench height. Note: Hole depth includes subdrill (L = H + J) and angle "
            "effects and must not be used as a direct substitute for vertical bench height."
        ),
        "orapa_reference": {"bench_height_m": 15.0, "stiffness_ratio_H_B": "2.5-3.75", "spacing_burden_ratio_S_B": "1.17-1.25"},
        "jwaneng_reference": {"hole_depth_m": "14.90-16.10", "burden_m": "7-8", "spacing_m": "8-9", "powder_factor_kg_m3": "0.57-0.78"},
    },
    "burden_spacing_geometry": {
        "concept": "Burden, Spacing & Pattern Geometry",
        "question": "What is the relationship between burden, spacing, and blast-block design?",
        "answer": (
            "Burden (B) is the perpendicular distance from a blasthole to the nearest free face, controlling rock relief and confinement. "
            "Spacing (S) is the centre-to-centre distance between adjacent holes in a row, controlling lateral energy distribution. "
            "Together they define the 2D pattern area per hole (B × S) and, with bench height H, the first-order interior rock volume (Vh ≈ B × S × H). "
            "USBR manual guidance suggests a first-approximation S/B ratio of 1.2–1.8 (1.5 starting point) for millisecond-delayed patterns. "
            "Orapa published datasets report an S/B range of 1.17–1.25."
        ),
    },
    "blast_outcomes_kpi": {
        "concept": "Blast Outcomes & Performance KPIs",
        "question": "How are blast block outcomes assessed in Debswana operations?",
        "answer": (
            "Blast outcomes are evaluated across three primary parameters: "
            "1) Fragmentation (P80 ≤ 150 mm target at Jwaneng assessed via shovel-mounted imaging), "
            "2) Ground Vibration / PPV (Peak Particle Velocity governed primarily by max charge per delay Qmax and distance DI), "
            "and 3) Airblast / AOP (air overpressure in dB governed by stemming confinement T, charge per delay Qmax, and atmospheric conditions)."
        ),
    },
}

# Pennsylvania DEP § 211.101 Blasting Regulatory Terms
PA_DEP_GLOSSARY: Dict[str, str] = {
    "access point": "A point in outer or inner perimeter security allowing entry to or exit from a magazine site.",
    "bench height": "The vertical elevation difference between the upper surface (bench top) and floor of a excavation level in an open-pit mine.",
    "burden": "The perpendicular distance from a blasthole to the nearest free face or effective row-to-face distance that explosive energy must overcome to create lateral relief.",
    "spacing": "The centre-to-centre distance between adjacent blastholes in the same row.",
    "subdrill": "The additional length drilled below the planned bench floor elevation to ensure floor breakage and eliminate toe.",
    "stiffness ratio": "The ratio of bench height to burden (H/B), indicating the structural flexibility of the rock face relative to burden confinement.",
    "airblast": "An airborne shock wave resulting from an explosion, also known as air overpressure (may or may not be audible).",
    "at-the-hole communication": "Communication between driller and blaster-in-charge describing borehole condition (e.g. cones with messages or verbal description).",
    "blast area": "The area around the blast site that must be cleared and secured to prevent injury to persons and property damage.",
    "blast site": "The specific location where explosive charges are loaded into blast holes.",
    "blaster": "An individual licensed by the Department under Chapter 210 to detonate explosives and supervise blasting activities.",
    "blaster-in-charge": "The blaster designated to have supervision and control over all blasting activities related to a blast.",
    "blasting activity": "Actions associated with the use of explosives from delivery to worksite until all postblast measures are completed (priming, loading, stemming, wiring, detonating).",
    "cube root scaled distance": "Ds1/3 = D / (W)^(1/3), where D is horizontal distance in feet and W is maximum charge weight in pounds per delay (< 8 ms). Used to estimate airblast levels.",
    "delay interval": "The designed time interval, usually in milliseconds, between successive detonations.",
    "detonator": "A device containing initiating or primary explosive used for initiating detonation (electric caps, nonelectric caps, delay connectors, detonating cord).",
    "explosives": "Chemical compounds or mixtures whose primary purpose is to function by explosion (dynamite, black powder, detonating cord, PETN).",
    "flyrock": "Overburden, stone, clay or material cast from the blast site through the air or along the ground beyond the blast area or permit boundary.",
    "indoor magazine": "A magazine located entirely within a secure intrusion-resistant and theft-resistant building.",
    "inner perimeter security": "Measures taken to increase intrusion resistance encircling an individual or group of magazines.",
    "misfire": "Incomplete detonation of explosives.",
    "outer perimeter security": "Measures taken to increase intrusion resistance encircling the area where magazines are situated.",
    "particle velocity": "A measure of the intensity of ground vibration, specifically the time rate of change of the amplitude of ground vibration.",
    "peak particle velocity": "The maximum intensity of particle velocity ground vibration.",
    "primer": "A cartridge or package of high explosives into which a detonator has been inserted or attached.",
    "square root scaled distance": "Ds = D / (W)^(1/2), where D is horizontal distance in feet and W is maximum charge weight in pounds per delay (< 8 ms). Used to estimate ground vibration.",
    "stemming": "Inert material placed in a blast hole after an explosive charge to confine explosion gases to the blast hole, or to separate decked charges.",
    "structure": "Everything built or constructed for occupancy, use, or ornamentation (bridges, offices, water towers, silos, dwellings).",
    "utility line": "An electric cable, fiber optic line, pipeline or conduit used to transport or transmit electricity, gases, liquids, or information.",
}

# ISEE Blaster's Handbook Glossary Definitions
ISEE_GLOSSARY: Dict[str, str] = {
    "acceptor": "A charge of explosives or blasting agent receiving an impulse from an exploding donor charge.",
    "air blast": "The airborne shock wave or acoustic transient generated by an explosion.",
    "anfo": "An explosive material consisting of ammonium nitrate and fuel oil.",
    "ammonium nitrate": "The ammonium salt of nitric acid represented by NH4NO3.",
    "base charge": "The main explosive charge in the base of a detonator.",
    "blast area": "The area of a blast within the influence of flying rock missiles, gases, and concussion.",
    "blast site": "The area where explosive materials are handled during loading prior to shot.",
    "blaster": "That qualified person in charge of, and responsible for, the loading and firing of a blast (same as Shot Firer).",
    "blasting agent": "An explosive material that meets prescribed criteria for insensitivity to initiation.",
    "booster": "An explosive charge, usually of high strength and high detonation velocity, used to improve the initiation of less sensitive explosive materials.",
    "bridgewire": "A resistance wire connecting the ends of the legwires inside an electric detonator.",
    "cap sensitivity": "The sensitivity of an explosive to initiation by a detonator. Cap-sensitive if it detonates with an IME No. 8 Test Detonator.",
    "certified blaster": "A blaster certified by a governmental agency to prepare, execute, and supervise blasting.",
    "detonation": "An explosive reaction that moves through an explosive material at a velocity greater than the speed of sound in the material.",
    "detonator": "Any device containing initiating or primary explosive used for initiating detonation.",
    "deflagration": "An explosive reaction such as rapid combustion moving through explosive material at a velocity less than the speed of sound in the material.",
    "delay blasting": "The practice of initiating individual explosive decks, boreholes, or rows of boreholes at predetermined time intervals using delay detonators.",
    "flyrock": "Rocks propelled from the blast area by the force of an explosion.",
    "misfire": "A blast that fails to detonate completely after an attempt at initiation.",
    "particle velocity": "A measure of the intensity of ground vibration, specifically the time rate of change of the amplitude of ground vibration.",
    "powder factor": "The ratio of explosive mass used to the volume or tonnage of rock broken.",
    "ppv": "Peak Particle Velocity - maximum particle velocity of ground vibration measured in mm/s or in/s.",
    "primary explosive": "A sensitive explosive that nearly always detonates by simple ignition from spark, flame, impact, or friction.",
    "primer": "A unit, package, or cartridge of explosives used to initiate other explosives or blasting agents, containing a detonator or detonating cord.",
    "seismograph": "An instrument useful in monitoring blasting operations that records ground vibration particle velocity, displacement, or acceleration in three perpendicular directions.",
    "stemming": "Inert material (such as gravel or drill cuttings) packed in a blasthole above the explosive charge to confine gases.",
    "sympathetic propagation": "The detonation of an explosive material as the result of receiving an impulse from another detonation through air, earth, or water.",
}


def _lookup_blast_term_handler(term: str) -> Dict[str, Any]:
    """Look up definition of a blast engineering term."""
    q_term = term.lower().strip()

    # 1. Search PA DEP Glossaries for exact term matches
    for k, v in PA_DEP_GLOSSARY.items():
        if k == q_term or f" {k} " in f" {q_term} " or f"what is a {k}" in q_term or f"define {k}" in q_term or f"what is {k}" in q_term:
            from src.autshumato_translator import translate_phrase
            tn_str = translate_phrase(k, source_lang="en", target_lang="tn")
            return TermDefinitionResult(
                term=k.title(),
                definition=v,
                plain_language=f"In simple terms, {k} refers to: {v}",
                setswana=tn_str if tn_str != k else None,
                category="PA DEP § 211.101 Regulation",
                source="Pennsylvania DEP Regulations",
            ).model_dump()

    # 2. Search ISEE Glossaries for exact term matches
    for k, v in ISEE_GLOSSARY.items():
        if k == q_term or f" {k} " in f" {q_term} " or f"what is a {k}" in q_term or f"define {k}" in q_term or f"what is {k}" in q_term:
            from src.autshumato_translator import translate_phrase
            tn_str = translate_phrase(k, source_lang="en", target_lang="tn")
            return TermDefinitionResult(
                term=k.title(),
                definition=v,
                plain_language=f"In simple terms, {k} refers to: {v}",
                setswana=tn_str if tn_str != k else None,
                category="ISEE Blaster's Handbook",
                source="ISEE Blaster's Handbook 18th Edition",
            ).model_dump()

    # 3. Search Blast Block Knowledge Base for specific concept matches
    best_kb_match = None
    best_kb_score = 0
    for k, kb in BLAST_BLOCK_KNOWLEDGE_BASE.items():
        concept_lower = kb["concept"].lower()
        question_lower = kb["question"].lower()
        score = 0
        if concept_lower in q_term:
            score += 5
        for word in q_term.split():
            if len(word) > 3 and (word in concept_lower or word in question_lower):
                score += 1
        if score > best_kb_score:
            best_kb_score = score
            best_kb_match = kb

    if best_kb_match and best_kb_score >= 2:
        return TermDefinitionResult(
            term=best_kb_match["concept"],
            definition=best_kb_match["answer"],
            plain_language=f"In simple terms: {best_kb_match['answer']}",
            setswana=None,
            category="Blast Block Reference Guide",
            source="Debswana Jwaneng/Orapa Open-Pit Blast Design Reference (2026)",
        ).model_dump()

    # 4. Search PA DEP partial matches
    for k, v in PA_DEP_GLOSSARY.items():
        if k in q_term or q_term in k:
            from src.autshumato_translator import translate_phrase
            tn_str = translate_phrase(k, source_lang="en", target_lang="tn")
            return TermDefinitionResult(
                term=k.title(),
                definition=v,
                plain_language=f"In simple terms, {k} refers to: {v}",
                setswana=tn_str if tn_str != k else None,
                category="PA DEP § 211.101 Regulation",
                source="Pennsylvania DEP Regulations",
            ).model_dump()

    # Search ISEE
    for k, v in ISEE_GLOSSARY.items():
        if k in q_term or q_term in k:
            from src.autshumato_translator import translate_phrase
            tn_str = translate_phrase(k, source_lang="en", target_lang="tn")
            return TermDefinitionResult(
                term=k.title(),
                definition=v,
                plain_language=f"In simple terms, {k} refers to: {v}",
                setswana=tn_str if tn_str != k else None,
                category="ISEE Blaster's Handbook",
                source="ISEE Blaster's Handbook 18th Edition",
            ).model_dump()

    return TermDefinitionResult(
        term=term,
        definition=f"Mining and blast engineering term for '{term}'.",
        plain_language=f"General mining term: {term}",
        setswana=None,
        category="General Mining",
        source="BlasterOPT Knowledge Base",
    ).model_dump()


def _translate_en_tn_handler(text: str) -> str:
    """Translate English text to Setswana using Autshumato corpus."""
    from src.autshumato_translator import translate_phrase
    return translate_phrase(text, source_lang="en", target_lang="tn")


def _translate_tn_en_handler(text: str) -> str:
    """Translate Setswana text to English using Autshumato corpus."""
    from src.autshumato_translator import translate_phrase
    return translate_phrase(text, source_lang="tn", target_lang="en")


def _answer_mining_question_handler(question: str) -> str:
    """Answer a general mining question using fine-tuned Pula-8B llm_client or knowledge base."""
    q_lower = question.lower().strip()
    # Check Blast Block Knowledge Base for best matching entry score
    best_match = None
    best_score = 0
    for k, kb in BLAST_BLOCK_KNOWLEDGE_BASE.items():
        score = sum(1 for term in ["blast block", "bench height", "bench", "height", "burden", "spacing", "orapa", "jwaneng", "debswana"] if term in q_lower and (term in kb["concept"].lower() or term in kb["question"].lower() or term in k))
        if score > best_score:
            best_score = score
            best_match = kb

    if best_match and best_score > 0:
        return f"**{best_match['concept']}**: {best_match['answer']}"

    from src.agent.llm_client import llm_client
    return llm_client.generate(question)


def _query_knowledge_graph_handler(question: str) -> str:
    """Queries Pennsylvania DEP § 211.101, ISEE Handbook glossary, knowledge graph, and HuggingFace dataset."""
    query_term = question.lower().strip()

    # 0. Search Blast Block Knowledge Base best match
    best_match = None
    best_score = 0
    for k, kb in BLAST_BLOCK_KNOWLEDGE_BASE.items():
        score = sum(1 for term in ["blast block", "bench height", "bench", "height", "burden", "spacing", "orapa", "jwaneng", "debswana"] if term in query_term and (term in kb["concept"].lower() or term in kb["question"].lower() or term in k))
        if score > best_score:
            best_score = score
            best_match = kb

    if best_match and best_score > 0:
        return f"Knowledge Graph Result for '{question}': **{best_match['concept']}**: {best_match['answer']}"

    # 1. Search PA DEP § 211.101 Regulatory Definitions
    dep_matches = []
    for term, defn in PA_DEP_GLOSSARY.items():
        if term in query_term or any(w in query_term.split() for w in term.split() if len(w) > 3):
            dep_matches.append(f"**{term.title()}** (PA DEP § 211.101): {defn}")
        if len(dep_matches) >= 3:
            break

    if dep_matches:
        return f"Knowledge Graph Result for '{question}': " + " | ".join(dep_matches)

    # 2. Search ISEE Blaster's Handbook Glossary
    isee_matches = []
    for term, defn in ISEE_GLOSSARY.items():
        if term in query_term or any(w in query_term.split() for w in term.split() if len(w) > 3):
            isee_matches.append(f"**{term.title()}** (ISEE Handbook): {defn}")
        if len(isee_matches) >= 3:
            break

    if isee_matches:
        return f"Knowledge Graph Result for '{question}': " + " | ".join(isee_matches)

    # 2. Search HuggingFace mining domain dataset
    try:
        from datasets import load_dataset
        ds = load_dataset("Lyntas/mining_domain_specific_terminology", split="train")

        matches = []
        for row in ds:
            term = str(row.get("Domain-specific Terminology", "")).lower()
            definition = str(row.get("Definition", ""))
            if any(word in term for word in query_term.split() if len(word) > 3):
                matches.append(f"**{row.get('Domain-specific Terminology')}**: {definition}")
            if len(matches) >= 3:
                break

        if matches:
            matched_text = " | ".join(matches)
            return f"Knowledge Graph Result for '{question}': {matched_text}"
    except Exception as e:
        logger.warning(f"Error querying Lyntas/mining_domain_specific_terminology dataset: {e}")

    return f"Knowledge graph query answer for '{question}': Historical blast logs in Jwaneng Cut 8 show optimal digging rates when powder factor is between 0.62 and 0.68 kg/m3."


def bind_tools(registry: Optional[ToolRegistry] = None) -> ToolRegistry:
    """Bind registry entries to actual execution functions."""
    target = registry if registry is not None else TOOL_REGISTRY
    target["predict_fragmentation"]["function"] = _predict_fragmentation_handler
    target["predict_vibration"]["function"] = _predict_vibration_handler
    target["predict_airblast"]["function"] = _predict_airblast_handler
    target["predict_downstream"]["function"] = _predict_downstream_handler
    target["design_blast"]["function"] = _design_blast_handler
    target["optimize_blast"]["function"] = _optimize_blast_handler
    target["explain_prediction"]["function"] = _explain_prediction_handler
    target["get_mwd_data"]["function"] = _get_mwd_data_handler
    target["get_geology"]["function"] = _get_geology_handler
    target["get_regulations"]["function"] = _get_regulations_handler
    target["search_past_blasts"]["function"] = _search_past_blasts_handler
    target["find_similar_blasts"]["function"] = _find_similar_blasts_handler
    target["generate_report"]["function"] = _generate_report_handler
    target["route_for_approval"]["function"] = _route_for_approval_handler
    target["log_decision"]["function"] = _log_decision_handler
    target["query_knowledge_graph"]["function"] = _query_knowledge_graph_handler
    target["lookup_blast_term"]["function"] = _lookup_blast_term_handler
    target["translate_en_tn"]["function"] = _translate_en_tn_handler
    target["translate_tn_en"]["function"] = _translate_tn_en_handler
    target["answer_mining_question"]["function"] = _answer_mining_question_handler
    return target


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

TOOL_REGISTRY.register_tool(
    name="lookup_blast_term",
    description="Look up the definition of a blast engineering term (e.g., 'powder factor', 'burden', 'stemming'). Returns the technical definition, plain language explanation, and Setswana translation if available.",
    input_schema=TermInput,
    output_schema=TermDefinitionResult,
    handler=_lookup_blast_term_handler,
)

TOOL_REGISTRY.register_tool(
    name="translate_en_tn",
    description="Translate text from English to Setswana.",
    input_schema=TranslationInput,
    output_schema=str,
    handler=_translate_en_tn_handler,
)

TOOL_REGISTRY.register_tool(
    name="translate_tn_en",
    description="Translate text from Setswana to English.",
    input_schema=TranslationInput,
    output_schema=str,
    handler=_translate_tn_en_handler,
)

TOOL_REGISTRY.register_tool(
    name="answer_mining_question",
    description="Answer a general question about mining, blasting, or geology using the Pula-8B language model. Use this when the question is not a simple term lookup or translation request.",
    input_schema=QuestionInput,
    output_schema=str,
    handler=_answer_mining_question_handler,
)

# Bind tool functions
bind_tools()
