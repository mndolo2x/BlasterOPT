"""
Unit tests for Ollama Health Check and Verification Module (src/agent/ollama_health.py).
"""

import pytest
from unittest.mock import patch, MagicMock
from src.agent.ollama_health import (
    check_ollama_installed,
    check_ollama_running,
    check_required_models,
    check_model_generation,
    list_ollama_models,
    test_ollama_generation,
    full_ollama_health_check,
)


@patch("shutil.which")
@patch("subprocess.run")
def test_check_ollama_installed_success(mock_run, mock_which):
    """Test check_ollama_installed when binary is installed."""
    mock_which.return_value = "/usr/local/bin/ollama"
    mock_run.return_value = MagicMock(stdout="ollama version is 0.1.24\n", stderr="")

    res = check_ollama_installed()
    assert res["installed"] is True
    assert res["path"] == "/usr/local/bin/ollama"
    assert "0.1.24" in res["version"]
    assert res["error"] is None


@patch("shutil.which")
def test_check_ollama_installed_not_found(mock_which):
    """Test check_ollama_installed when binary is missing."""
    mock_which.return_value = None

    res = check_ollama_installed()
    assert res["installed"] is False
    assert res["path"] is None
    assert res["version"] is None
    assert "not found" in res["error"]


@patch("requests.get")
def test_check_ollama_running_success(mock_get):
    """Test check_ollama_running when server responds 200 OK."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp

    res = check_ollama_running(base_url="http://localhost:11434")
    assert res["running"] is True
    assert res["base_url"] == "http://localhost:11434"
    assert res["response_time_ms"] >= 0.0
    assert res["error"] is None


@patch("requests.get")
def test_check_ollama_running_offline(mock_get):
    """Test check_ollama_running when server is offline."""
    import requests
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

    res = check_ollama_running(base_url="http://localhost:11434")
    assert res["running"] is False
    assert res["base_url"] == "http://localhost:11434"
    assert res["response_time_ms"] >= 0.0
    assert "offline" in res["error"].lower() or "connection" in res["error"].lower()


@patch("requests.get")
def test_check_required_models(mock_get):
    """Test check_required_models evaluating presence and missing models."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {"name": "llama3.1:8b"},
            {"name": "mistral:7b"},
        ]
    }
    mock_get.return_value = mock_resp

    res = check_required_models(required_models=["llama3.1:8b", "llama3.2:3b"])
    assert res["all_present"] is False
    assert "llama3.1:8b" in res["present"]
    assert "llama3.2:3b" in res["missing"]
    assert "llama3.1:8b" in res["available"]


@patch("requests.get")
def test_list_ollama_models(mock_get):
    """Test list_ollama_models returning model tags."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {"name": "llama3:8b", "size": 4661224676},
            {"name": "mistral:7b", "size": 4109865159},
        ]
    }
    mock_get.return_value = mock_resp

    res = list_ollama_models()
    assert res["count"] == 2
    assert "llama3:8b" in res["models"]
    assert "mistral:7b" in res["models"]
    assert res["error"] is None


@patch("requests.post")
def test_check_model_generation_success(mock_post):
    """Test check_model_generation returning successful working response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "OK"}
    mock_post.return_value = mock_resp

    res = check_model_generation(model_name="llama3.1:8b")
    assert res["working"] is True
    assert res["model"] == "llama3.1:8b"
    assert res["response"] == "OK"
    assert res["response_time_ms"] >= 0.0
    assert res["error"] is None


@patch("requests.post")
def test_test_ollama_generation_success(mock_post):
    """Test test_ollama_generation returning successful text response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "Hello from Llama 3!"}
    mock_post.return_value = mock_resp

    res = test_ollama_generation(model="llama3:8b")
    assert res["success"] is True
    assert res["response"] == "Hello from Llama 3!"
    assert res["latency_ms"] >= 0.0
    assert res["error"] is None


@patch("src.agent.ollama_health.check_ollama_installed")
@patch("src.agent.ollama_health.check_ollama_running")
@patch("src.agent.ollama_health.list_ollama_models")
@patch("src.agent.ollama_health.test_ollama_generation")
def test_full_ollama_health_check_aggregated(mock_gen, mock_list, mock_run, mock_inst):
    """Test full_ollama_health_check aggregating all status metrics."""
    mock_inst.return_value = {"installed": True, "path": "/usr/bin/ollama", "version": "0.1.24", "error": None}
    mock_run.return_value = {"running": True, "host": "http://localhost:11434", "status_code": 200, "error": None}
    mock_list.return_value = {"models": ["llama3:8b"], "details": [], "count": 1, "error": None}
    mock_gen.return_value = {"success": True, "response": "OK", "latency_ms": 12.0, "error": None}

    res = full_ollama_health_check(required_models=["llama3:8b"])
    assert res["healthy"] is True
    assert res["installed"] is True
    assert res["running"] is True
    assert "llama3:8b" in res["models"]
    assert len(res["missing_models"]) == 0
    assert res["generation_working"] is True
    assert res["total_latency_ms"] < 5000.0
