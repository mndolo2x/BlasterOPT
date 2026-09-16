"""
Botswana Mining Regulatory Compliance Module for BlastOpt Botswana.

Implements regulatory validation and compliance reporting under Botswana's Mines, Quarries,
Works and Machinery Act (Cap. 44:02) and Data Protection Act. Validates peak particle velocity (PPV),
airblast overpressure (dBL), flyrock range, and stemming confinement limits.
"""

import os
import json
import logging
from fpdf import FPDF
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "data/processed/regulatory_limits.json"


def load_regulatory_limits(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, float]:
    """
    Loads Botswana's mining regulatory limits (PPV, airblast dBL, flyrock range, stemming) from a JSON config file.

    Botswana Regulatory Framework Context:
    --------------------------------------
    - Mines, Quarries, Works and Machinery Act (Cap. 44:02): Establishes legal duties for mine safety,
      ground vibration control (PPV limits), airblast noise overpressure mitigation, and flyrock containment.
    - Data Protection Act of Botswana: Mandates secure handling and anonymization of site-specific blasting telematics
      and geological block model data.
    - Typical Limits: Maximum PPV <= 10.0 mm/s (minimum detectable ~0.1 mm/s), airblast <= 120-125 dB,
      maximum flyrock range <= 250 m.

    Parameters:
    -----------
    config_path : str, default="data/processed/regulatory_limits.json"
        Path to JSON regulatory limits configuration file.

    Returns:
    --------
    Dict[str, float]
        Dictionary of active regulatory threshold limits.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading regulatory config file ({config_path}): {e}")

    # Default fallback regulatory limits
    return {
        "act": "Mines, Quarries, Works and Machinery Act (Cap. 44:02) & Data Protection Act of Botswana",
        "max_ppv_mms": 10.0,
        "min_ppv_mms": 0.1,
        "max_airblast_dbl": 120.0,
        "max_flyrock_m": 250.0,
        "min_stemming_m": 2.5,
        "max_powder_factor_kg_m3": 1.20,
        "d50_target_min_mm": 100.0,
        "d50_target_max_mm": 350.0,
    }


def check_compliance(
    blast_params: Dict[str, float],
    predictions: Dict[str, float],
    custom_limits: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluates whether a proposed blast design and predicted outcomes comply with Botswana regulatory limits.

    Parameters:
    -----------
    blast_params : Dict[str, float]
        Input blast design parameters (stemming_m, powder_factor_kg_m3, burden_m, spacing_m).
    predictions : Dict[str, float]
        Predicted blast outcomes (ppv_mms, airblast_dbl, flyrock_m, d50_mm).
    custom_limits : Dict[str, float], optional
        Optional custom regulatory limits overriding defaults.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing `is_compliant` boolean, `violations` list, and `recommendations` list.
    """
    limits = custom_limits if custom_limits is not None else load_regulatory_limits()

    violations = []
    recommendations = []

    # 1. Peak Particle Velocity (PPV) Check
    ppv_val = float(predictions.get("ppv_mms", predictions.get("pred_ppv_mms", 5.0)))
    max_ppv = float(limits.get("max_ppv_mms", 10.0))
    min_ppv = float(limits.get("min_ppv_mms", 0.1))

    if ppv_val > max_ppv:
        violations.append(
            f"Ground Vibration (PPV = {ppv_val:.2f} mm/s) exceeds maximum regulatory limit ({max_ppv:.1f} mm/s)."
        )
        recommendations.append(
            "Reduce maximum charge per delay or increase inter-hole electronic delay intervals to mitigate PPV."
        )

    # 2. Airblast Overpressure (dBL) Check
    airblast_val = float(predictions.get("airblast_dbl", predictions.get("pred_airblast_dbl", 115.0)))
    max_airblast = float(limits.get("max_airblast_dbl", 120.0))

    if airblast_val > max_airblast:
        violations.append(
            f"Airblast Overpressure ({airblast_val:.1f} dBL) exceeds regulatory limit ({max_airblast:.1f} dBL)."
        )
        recommendations.append(
            "Increase stemming length or improve stemming material quality to prevent gas venting to atmosphere."
        )

    # 3. Flyrock Distance Check
    flyrock_val = float(predictions.get("flyrock_m", predictions.get("pred_flyrock_m", 100.0)))
    max_flyrock = float(limits.get("max_flyrock_m", 250.0))

    if flyrock_val > max_flyrock:
        violations.append(
            f"Flyrock Range ({flyrock_val:.1f} m) exceeds maximum allowable safety boundary ({max_flyrock:.1f} m)."
        )
        recommendations.append(
            "Reduce powder factor or increase burden/stemming confinement ratio."
        )

    # 4. Stemming Confinement Check
    stemming_val = float(blast_params.get("stemming_m", 5.0))
    min_stemming = float(limits.get("min_stemming_m", 2.5))

    if stemming_val < min_stemming:
        violations.append(
            f"Stemming Length ({stemming_val:.2f} m) is below mandatory minimum confinement ({min_stemming:.1f} m)."
        )
        recommendations.append(
            f"Increase stemming length to at least {min_stemming:.1f} m to comply with pit safety rules."
        )

    is_compliant = len(violations) == 0

    return {
        "is_compliant": is_compliant,
        "violations": violations,
        "recommendations": recommendations,
        "active_limits": limits,
    }


class ComplianceReportPDF(FPDF):
    """Custom FPDF class for Regulatory Submission Reports."""

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 100, 0) if getattr(self, "is_compliant", True) else self.set_text_color(200, 0, 0)
        self.cell(0, 10, "BOTSWANA MINING REGULATORY COMPLIANCE REPORT", border=False, new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, "Mines, Quarries, Works and Machinery Act (Cap. 44:02)", border=False, new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} | Generated by BlastOpt Botswana Regulatory Compliance Engine", align="C")


def generate_compliance_report(
    blast_params: Dict[str, float],
    predictions: Dict[str, float],
    output_path: str = "data/processed/compliance_report.pdf",
) -> str:
    """
    Generates an official PDF compliance submission report for Botswana Department of Mines audits.

    Parameters:
    -----------
    blast_params : Dict[str, float]
        Blast design parameter dictionary.
    predictions : Dict[str, float]
        Predicted outcomes dictionary.
    output_path : str, default="data/processed/compliance_report.pdf"
        Output PDF file path.

    Returns:
    --------
    str
        Path to generated PDF report file.
    """
    comp_res = check_compliance(blast_params, predictions)

    pdf = ComplianceReportPDF()
    pdf.is_compliant = comp_res["is_compliant"]
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Section 1: Executive Compliance Status
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "1. Executive Regulatory Status", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 11)
    if comp_res["is_compliant"]:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 7, "STATUS: FULLY COMPLIANT WITH BOTSWANA LAWS", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 7, "STATUS: NON-COMPLIANT - VIOLATIONS DETECTED", new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # Section 2: Blast Design & Outcome Evaluation
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. Blast Design & Environmental Outcome Evaluation", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"- Stemming Length: {blast_params.get('stemming_m', 5.0):.2f} m (Min limit: {comp_res['active_limits'].get('min_stemming_m', 2.5):.1f} m)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"- Ground Vibration (PPV): {predictions.get('ppv_mms', 5.0):.2f} mm/s (Max limit: {comp_res['active_limits'].get('max_ppv_mms', 10.0):.1f} mm/s)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"- Airblast Overpressure: {predictions.get('airblast_dbl', 115.0):.1f} dBL (Max limit: {comp_res['active_limits'].get('max_airblast_dbl', 120.0):.1f} dBL)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"- Flyrock Range: {predictions.get('flyrock_m', 100.0):.1f} m (Max limit: {comp_res['active_limits'].get('max_flyrock_m', 250.0):.1f} m)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Section 3: Specific Violations & Recommendations
    if comp_res["violations"]:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 7, "3. Detected Regulatory Violations", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for v in comp_res["violations"]:
            pdf.multi_cell(0, 5, f"- {v}")
        pdf.ln(2)

    if comp_res["recommendations"]:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 7, "4. Engineering Recommendations", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for r in comp_res["recommendations"]:
            pdf.multi_cell(0, 5, f"- {r}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    pdf.output(output_path)
    return output_path
