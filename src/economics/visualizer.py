"""
Economics Visualizer Module.
Generates Plotly interactive charts for the Economic Dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional
from src.economics.margin_calculator import EconomicResult


def plot_cost_breakdown_stacked(results: List[EconomicResult], top_n: int = 10) -> go.Figure:
    """
    Generates a stacked bar chart showing the 5 cost breakdown components for top_n designs.
    """
    if not results:
        fig = go.Figure()
        fig.add_annotation(text="No economic results available", showarrow=False)
        return fig

    sub_results = results[:top_n]
    design_ids = [f"Design {r.design_id}" for r in sub_results]

    db_costs = [r.cost.drill_and_blast_usd_per_t for r in sub_results]
    dig_costs = [r.cost.digging_usd_per_t for r in sub_results]
    haul_costs = [r.cost.hauling_usd_per_t for r in sub_results]
    crush_costs = [r.cost.crushing_usd_per_t for r in sub_results]
    proc_costs = [r.cost.processing_usd_per_t for r in sub_results]

    fig = go.Figure(data=[
        go.Bar(name="Drill & Blast", x=design_ids, y=db_costs, marker_color="#1f77b4"),
        go.Bar(name="Digging", x=design_ids, y=dig_costs, marker_color="#ff7f0e"),
        go.Bar(name="Hauling", x=design_ids, y=haul_costs, marker_color="#2ca02c"),
        go.Bar(name="Crushing", x=design_ids, y=crush_costs, marker_color="#d62728"),
        go.Bar(name="Processing", x=design_ids, y=proc_costs, marker_color="#9467bd"),
    ])

    fig.update_layout(
        barmode="stack",
        title="<b>Mine-to-Mill Unit Cost Breakdown ($/t)</b>",
        xaxis_title="Design Candidate",
        yaxis_title="Unit Cost ($/t)",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_margin_vs_powder_factor(results: List[EconomicResult], selected_id: Optional[int] = None) -> go.Figure:
    """
    Generates a scatter plot of Margin ($/t) vs. Powder Factor (kg/m3) highlighting Pareto front candidates.
    """
    if not results:
        fig = go.Figure()
        fig.add_annotation(text="No economic results available", showarrow=False)
        return fig

    pfs = [float(r.design_params.get("powder_factor_kg_m3", 0.65)) for r in results]
    margins = [r.margin_usd_per_t for r in results]
    d80s = [float(r.design_params.get("d80_mm", 250.0)) for r in results]
    ids = [r.design_id for r in results]

    # Ensure sizes are positive numbers
    min_m = min(margins)
    sizes = [max(8.0, 10.0 + (m - min_m) * 2.0) if len(margins) > 1 and max(margins) > min_m else 12.0 for m in margins]

    df = pd.DataFrame({"Design": ids, "PowderFactor": pfs, "Margin": margins, "D80": d80s, "Size": sizes})

    fig = px.scatter(
        df,
        x="PowderFactor",
        y="Margin",
        color="D80",
        size="Size",
        hover_data=["Design"],
        title="<b>Margin ($/t) vs. Powder Factor (kg/m³)</b>",
        labels={"PowderFactor": "Powder Factor (kg/m³)", "Margin": "Net Margin ($/t)", "D80": "D80 (mm)"},
        color_continuous_scale="Viridis",
    )

    if selected_id is not None:
        sel_row = df[df["Design"] == selected_id]
        if not sel_row.empty:
            fig.add_trace(go.Scatter(
                x=sel_row["PowderFactor"],
                y=sel_row["Margin"],
                mode="markers",
                marker=dict(size=18, color="red", symbol="star"),
                name="Selected #1 Winner",
            ))

    fig.update_layout(template="plotly_white", height=400)
    return fig


def plot_revenue_vs_cost(results: List[EconomicResult]) -> go.Figure:
    """
    Generates a Revenue vs. Cost scatter plot with break-even line ($Revenue = Cost$).
    """
    if not results:
        fig = go.Figure()
        fig.add_annotation(text="No economic results available", showarrow=False)
        return fig

    costs = [r.cost.total_usd_per_t for r in results]
    revenues = [r.revenue.revenue_usd_per_t for r in results]
    ids = [f"Design {r.design_id}" for r in results]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=costs,
        y=revenues,
        mode="markers",
        marker=dict(size=10, color=revenues, colorscale="Plotly3", showscale=True),
        text=ids,
        name="Designs",
    ))

    # Add 1:1 Break-even line
    min_val = min(min(costs), min(revenues)) * 0.9
    max_val = max(max(costs), max(revenues)) * 1.1
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="gray", dash="dash"),
        name="Break-even Line",
    ))

    fig.update_layout(
        title="<b>Revenue vs. Total Cost ($/t)</b>",
        xaxis_title="Total Unit Cost ($/t)",
        yaxis_title="Revenue ($/t)",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_npv_bars(results: List[EconomicResult], top_n: int = 10) -> go.Figure:
    """
    Generates an NPV bar chart sorted by total NPV ($).
    """
    if not results:
        fig = go.Figure()
        fig.add_annotation(text="No economic results available", showarrow=False)
        return fig

    sorted_res = sorted(results, key=lambda r: r.npv_usd, reverse=True)[:top_n]
    design_ids = [f"Design {r.design_id}" for r in sorted_res]
    npvs = [r.npv_usd / 1e6 for r in sorted_res]  # $ Millions

    fig = px.bar(
        x=design_ids,
        y=npvs,
        labels={"x": "Design Candidate", "y": "Net Present Value ($ Millions)"},
        title="<b>NPV per Design ($ Millions)</b>",
        color=npvs,
        color_continuous_scale="Blues",
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


def plot_sensitivity_tornado(result: EconomicResult) -> go.Figure:
    """
    Generates a Sensitivity Tornado Chart showing how net margin ($/t) changes if key variables change by +/-10%.
    """
    base_margin = result.margin_usd_per_t
    grade = result.revenue.grade
    price = result.revenue.commodity_price
    db_cost = result.cost.drill_and_blast_usd_per_t

    # Grade -10% / +10%
    rev_grade_low = result.revenue.tonnes * (grade * 0.9) * (result.revenue.recovery_pct / 100.0) * price / result.revenue.tonnes
    margin_grade_low = rev_grade_low - result.cost.total_usd_per_t
    rev_grade_high = result.revenue.tonnes * (grade * 1.1) * (result.revenue.recovery_pct / 100.0) * price / result.revenue.tonnes
    margin_grade_high = rev_grade_high - result.cost.total_usd_per_t

    # Price -10% / +10%
    rev_price_low = result.revenue.tonnes * grade * (result.revenue.recovery_pct / 100.0) * (price * 0.9) / result.revenue.tonnes
    margin_price_low = rev_price_low - result.cost.total_usd_per_t
    rev_price_high = result.revenue.tonnes * grade * (result.revenue.recovery_pct / 100.0) * (price * 1.1) / result.revenue.tonnes
    margin_price_high = rev_price_high - result.cost.total_usd_per_t

    # D&B Cost +10% / -10%
    cost_db_high = result.cost.total_usd_per_t + db_cost * 0.10
    margin_db_high = result.revenue.revenue_usd_per_t - cost_db_high
    cost_db_low = result.cost.total_usd_per_t - db_cost * 0.10
    margin_db_low = result.revenue.revenue_usd_per_t - cost_db_low

    variables = ["Ore Grade (±10%)", "Commodity Price (±10%)", "D&B Cost (±10%)"]
    low_deltas = [margin_grade_low - base_margin, margin_price_low - base_margin, margin_db_high - base_margin]
    high_deltas = [margin_grade_high - base_margin, margin_price_high - base_margin, margin_db_low - base_margin]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=variables,
        x=low_deltas,
        name="-10% Change / Cost Increase",
        orientation="h",
        marker=dict(color="#d62728"),
    ))
    fig.add_trace(go.Bar(
        y=variables,
        x=high_deltas,
        name="+10% Change / Cost Decrease",
        orientation="h",
        marker=dict(color="#2ca02c"),
    ))

    fig.update_layout(
        title=f"<b>Sensitivity Tornado Chart (Base Margin: ${base_margin:.2f}/t)</b>",
        xaxis_title="Change in Net Margin ($/t)",
        yaxis_title="Variable",
        barmode="overlay",
        template="plotly_white",
        height=350,
    )
    return fig
