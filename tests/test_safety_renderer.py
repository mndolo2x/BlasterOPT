"""
Unit tests for safety renderer functions.
"""

import pytest
import numpy as np
import plotly.graph_objects as go
from src.safety.dust import DustModel
from src.safety.toxic_gases import GasModel
from src.safety.noise_prediction import NoiseModel
from src.safety.environmental_impact import EnvironmentalAssessment
from src.safety.risk_analysis import BlastRiskAnalyzer
from src.safety.renderer import (
    SafetyRenderer,
    plot_dust_dispersion,
    plot_gas_concentration,
    plot_noise_attenuation,
    plot_risk_matrix,
    plot_environmental_breakdown,
)


def test_safety_renderer_class():
    renderer = SafetyRenderer()
    fig = renderer.plot_dust_plume(pm10_concentration_ug_m3=120.0, distance_m=300.0)
    assert isinstance(fig, go.Figure)


def test_plot_dust_dispersion():
    model = DustModel()
    dists = np.linspace(100, 1000, 10)
    fig = plot_dust_dispersion(model, dists)
    assert isinstance(fig, go.Figure)


def test_plot_gas_concentration():
    model = GasModel("ANFO")
    times = np.linspace(60, 1800, 10)
    fig = plot_gas_concentration(model, times)
    assert isinstance(fig, go.Figure)


def test_plot_noise_attenuation():
    model = NoiseModel(charge_per_delay_kg=150.0, depth_of_burial_m=3.0)
    dists = np.linspace(100, 1000, 10)
    fig = plot_noise_attenuation(model, dists)
    assert isinstance(fig, go.Figure)


def test_plot_risk_matrix():
    analyzer = BlastRiskAnalyzer()
    fig = plot_risk_matrix(analyzer)
    assert isinstance(fig, go.Figure)


def test_plot_environmental_breakdown():
    receptors = {"village": 800.0}
    params = {"explosive_mass_kg": 500.0}
    eia = EnvironmentalAssessment(blast_params=params, receptor_distances=receptors)
    fig = plot_environmental_breakdown(eia)
    assert isinstance(fig, go.Figure)
