"""
Unit tests for Ollama Health Check and Verification Module (src/agent/ollama_health.py).
"""

import pytest
import requests
from unittest.mock import patch, MagicMock
from src.agent.ollama_health import (
    check_ollama_installed,
    check_ollama_running,
    check_required_models,
    check_model_generation,
    list_ollama_models,
    test_ollama_generation as ollama_gen_fn,
    full_health_check,
    full_ollama_health_check,
    check_offline_capability,
)
from src.agent.llm_config import OllamaClient, OllamaCloudClient


@patch("shutil.which")
@patch("subprocess.run")
def test_check_ollama_installed_success(mock_run, mock_which):
    """Test check_ollama_installed when binary is installed and returns dict with correct keys."""
    mock_which.return_value = "/usr/local/bin/ollama"
    mock_run.return_value = MagicMock(stdout="ollama version is 0.1.24\n", stderr="")

    res = check_ollama_installed()
    assert isinstance(res, dict)
    assert set(res.keys()) == {"installed", "path", "version", "error"}
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
def test_check_model_generation_timeout_graceful(mock_post):
    """Test check_model_generation handles timeouts gracefully."""
    mock_post.side_effect = requests.exceptions.Timeout("Request timed out after 30s")

    res = check_model_generation(model_name="llama3.1:8b")
    assert res["working"] is False
    assert res["model"] == "llama3.1:8b"
    assert res["response"] is None
    assert "timed out" in res["error"].lower()


@patch("requests.post")
def test_test_ollama_generation_success(mock_post):
    """Test test_ollama_generation returning successful text response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "Hello from Llama 3!"}
    mock_post.return_value = mock_resp

    res = ollama_gen_fn(model="llama3:8b")
    assert res["success"] is True
    assert res["response"] == "Hello from Llama 3!"
    assert res["latency_ms"] >= 0.0
    assert res["error"] is None


@patch("src.agent.ollama_health.check_ollama_installed")
@patch("src.agent.ollama_health.check_ollama_running")
@patch("src.agent.ollama_health.check_required_models")
@patch("src.agent.ollama_health.check_model_generation")
def test_full_health_check_status_healthy(mock_gen, mock_req, mock_run, mock_inst):
    """Test full_health_check returning 'healthy' when everything works."""
    mock_inst.return_value = {"installed": True, "path": "/usr/bin/ollama", "version": "0.1.24", "error": None}
    mock_run.return_value = {"running": True, "base_url": "http://localhost:11434", "response_time_ms": 10.0, "error": None}
    mock_req.return_value = {"all_present": True, "present": ["llama3.1:8b", "llama3.2:3b"], "missing": [], "available": ["llama3.1:8b", "llama3.2:3b"], "error": None}
    mock_gen.return_value = {"working": True, "model": "llama3.1:8b", "response": "OK", "response_time_ms": 150.0, "error": None}

    res = full_health_check()
    assert res["overall_status"] == "healthy"
    assert res["can_use_offline_llm"] is True
    assert len(res["recommendations"]) == 0


@patch("src.agent.ollama_health.check_ollama_installed")
@patch("src.agent.ollama_health.check_ollama_running")
@patch("src.agent.ollama_health.check_required_models")
@patch("src.agent.ollama_health.check_model_generation")
def test_full_health_check_status_degraded(mock_gen, mock_req, mock_run, mock_inst):
    """Test full_health_check returning 'degraded' when models are missing."""
    mock_inst.return_value = {"installed": True, "path": "/usr/bin/ollama", "version": "0.1.24", "error": None}
    mock_run.return_value = {"running": True, "base_url": "http://localhost:11434", "response_time_ms": 10.0, "error": None}
    mock_req.return_value = {"all_present": False, "present": ["llama3.1:8b"], "missing": ["llama3.2:3b"], "available": ["llama3.1:8b"], "error": None}
    mock_gen.return_value = {"working": True, "model": "llama3.1:8b", "response": "OK", "response_time_ms": 150.0, "error": None}

    res = full_health_check()
    assert res["overall_status"] == "degraded"
    assert res["can_use_offline_llm"] is True
    assert len(res["recommendations"]) >= 1


@patch("src.agent.ollama_health.full_health_check")
def test_check_offline_capability(mock_full):
    """Test check_offline_capability when offline LLM is available vs unavailable."""
    mock_full.return_value = {
        "can_use_offline_llm": True,
        "recommendations": []
    }
    res_avail = check_offline_capability()
    assert res_avail["available"] is True
    assert "Ollama is running" in res_avail["reason"]

    mock_full.return_value = {
        "can_use_offline_llm": False,
        "recommendations": ["Start Ollama service."]
    }
    res_unavail = check_offline_capability()
    assert res_unavail["available"] is False
    assert "Start Ollama service" in res_unavail["reason"]


@patch("src.agent.ollama_health._check_cloud_fallback")
@patch("src.agent.ollama_health.check_ollama_installed")
def test_full_health_check_status_not_installed(mock_inst, mock_cloud):
    """Test full_health_check returning 'not_installed' when Ollama and cloud are missing."""
    mock_inst.return_value = {"installed": False, "path": None, "version": None, "error": "Not found"}
    mock_cloud.return_value = (False, None)

    res = full_health_check()
    assert res["overall_status"] == "not_installed"
    assert res["can_use_offline_llm"] is False
    assert "Install Ollama" in res["recommendations"][0]


@patch("src.agent.ollama_health._check_cloud_fallback")
@patch("src.agent.ollama_health.check_ollama_installed")
@patch("src.agent.ollama_health.check_ollama_running")
def test_full_health_check_status_offline(mock_run, mock_inst, mock_cloud):
    """Test full_health_check returning 'offline' when server is not running and cloud unavailable."""
    mock_inst.return_value = {"installed": True, "path": "/usr/bin/ollama", "version": "0.1.24", "error": None}
    mock_run.return_value = {"running": False, "base_url": "http://localhost:11434", "response_time_ms": 0.0, "error": "Connection refused"}
    mock_cloud.return_value = (False, None)

    res = full_health_check()
    assert res["overall_status"] == "offline"
    assert res["can_use_offline_llm"] is False
    assert any("Start Ollama service" in rec for rec in res["recommendations"])


def test_huggingface_cloud_client_generation():
    """Test OllamaCloudClient using Hugging Face Inference API / fallback."""
    client = OllamaCloudClient(model_id="meta-llama/Llama-3.1-8B-Instruct")
    res = client.generate("What is powder factor?")
    assert isinstance(res, str)
    assert len(res) > 0
    assert "Hugging Face" in res or "Domain Analysis" in res


@patch("src.agent.ollama_client.full_health_check")
def test_ollama_client_generate_with_fallback(mock_full_health):
    """Test OllamaClient.generate_with_fallback falls back to cloud when Ollama is unavailable."""
    mock_full_health.return_value = {
        "can_use_offline_llm": False,
        "recommendations": ["Ollama server is offline."]
    }

    client = OllamaClient(base_url="http://localhost:11434")
    assert client.is_available is False

    mock_fallback = MagicMock(return_value="Cloud Response")
    res = client.generate_with_fallback("Test prompt", cloud_fallback=mock_fallback)
    assert res == "Cloud Response"
    mock_fallback.assert_called_once_with("Test prompt")
