"""
Unit tests for Debswana Enterprise Integrations module (SAP, Deswik, Surpac).
"""

import pytest
from src.integrations import (
    connect_to_sap,
    connect_to_deswik,
    connect_to_surpac,
    push_to_sap,
)


def test_connect_to_sap_handles_gracefully():
    """Test connect_to_sap handles connection and credentials gracefully without raising unhandled exceptions."""
    res = connect_to_sap()

    assert isinstance(res, dict)
    assert res["system"] == "SAP ERP"
    assert "status" in res
    assert "retrieved_costs" in res
    assert "explosive_price_usd_kg" in res["retrieved_costs"]
    assert "last_sync" in res


def test_connect_to_deswik_handles_gracefully():
    """Test connect_to_deswik retrieves mine planning data gracefully."""
    res = connect_to_deswik()

    assert isinstance(res, dict)
    assert "Deswik" in res["system"]
    assert "retrieved_mine_plan" in res
    assert "bench_id" in res["retrieved_mine_plan"]


def test_connect_to_surpac_handles_gracefully():
    """Test connect_to_surpac retrieves geological block model data gracefully."""
    res = connect_to_surpac()

    assert isinstance(res, dict)
    assert "Surpac" in res["system"]
    assert "retrieved_geology" in res
    assert "rock_type" in res["retrieved_geology"]


def test_push_to_sap_executes_successfully():
    """Test push_to_sap returns valid status response structure when pushing post-blast cost metrics."""
    cost_data = {
        "blast_id": "BLAST_TEST_99",
        "drilling_cost_usd_t": 1.25,
        "explosive_cost_usd_t": 1.85,
        "total_cost_usd_t": 5.10,
    }

    res = push_to_sap(cost_data)

    assert isinstance(res, dict)
    assert res["target_system"] == "SAP ERP"
    assert res["blast_id"] == "BLAST_TEST_99"
    assert "status" in res
    assert "timestamp" in res
