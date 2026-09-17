"""
Ollama Health Check and Integration Verification Module for BlasterOPT.

Verifies that Ollama local LLM runner is installed, running, has required models,
and can generate text responses within strict timeout constraints (<= 5 seconds).
"""

import shutil
import subprocess
import requests
import logging
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 5.0  # Strict 5 second timeout limit for health checks


def check_ollama_installed() -> Dict[str, Any]:
    """
    Checks if Ollama executable binary is installed on the system.

    Returns:
    --------
    Dict[str, Any]
        {
            "installed": bool,
            "path": Optional[str],
            "version": Optional[str],
            "error": Optional[str]
        }
    """
    path = shutil.which("ollama")
    if not path:
        return {
            "installed": False,
            "path": None,
            "version": None,
            "error": "Ollama binary 'ollama' not found in system PATH."
        }

    try:
        proc = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT
        )
        version_str = proc.stdout.strip() or proc.stderr.strip()
        return {
            "installed": True,
            "path": path,
            "version": version_str if version_str else "Unknown",
            "error": None
        }
    except subprocess.TimeoutExpired:
        return {
            "installed": True,
            "path": path,
            "version": None,
            "error": "Subprocess 'ollama --version' timed out after 5 seconds."
        }
    except Exception as e:
        return {
            "installed": True,
            "path": path,
            "version": None,
            "error": f"Error running 'ollama --version': {e}"
        }


def check_ollama_running(host: str = DEFAULT_OLLAMA_HOST) -> Dict[str, Any]:
    """
    Checks if the local Ollama API server is running and responding to GET requests.

    Parameters:
    -----------
    host : str, default="http://localhost:11434"
        Ollama server base URL endpoint.

    Returns:
    --------
    Dict[str, Any]
        {
            "running": bool,
            "host": str,
            "status_code": Optional[int],
            "error": Optional[str]
        }
    """
    try:
        resp = requests.get(host, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            return {
                "running": True,
                "host": host,
                "status_code": 200,
                "error": None
            }
        return {
            "running": False,
            "host": host,
            "status_code": resp.status_code,
            "error": f"Ollama server returned HTTP status code {resp.status_code}"
        }
    except requests.exceptions.Timeout:
        return {
            "running": False,
            "host": host,
            "status_code": None,
            "error": "Connection to Ollama server timed out after 5 seconds."
        }
    except requests.exceptions.ConnectionError:
        return {
            "running": False,
            "host": host,
            "status_code": None,
            "error": f"Could not connect to Ollama server at '{host}'. Server is offline."
        }
    except Exception as e:
        return {
            "running": False,
            "host": host,
            "status_code": None,
            "error": f"Error checking Ollama status: {e}"
        }


def list_ollama_models(host: str = DEFAULT_OLLAMA_HOST) -> Dict[str, Any]:
    """
    Lists all models currently available/pulled in local Ollama server.

    Parameters:
    -----------
    host : str, default="http://localhost:11434"
        Ollama server base URL endpoint.

    Returns:
    --------
    Dict[str, Any]
        {
            "models": List[str],
            "details": List[Dict[str, Any]],
            "count": int,
            "error": Optional[str]
        }
    """
    url = f"{host.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            models_list = data.get("models", [])
            model_names = [m.get("name", "") for m in models_list if m.get("name")]
            return {
                "models": model_names,
                "details": models_list,
                "count": len(model_names),
                "error": None
            }
        return {
            "models": [],
            "details": [],
            "count": 0,
            "error": f"Ollama /api/tags returned status code {resp.status_code}"
        }
    except Exception as e:
        return {
            "models": [],
            "details": [],
            "count": 0,
            "error": f"Error fetching Ollama models: {e}"
        }


def test_ollama_generation(
    model: str = "llama3:8b",
    host: str = DEFAULT_OLLAMA_HOST,
    prompt: str = "Hi"
) -> Dict[str, Any]:
    """
    Tests text generation via local Ollama server with strict timeout.

    Parameters:
    -----------
    model : str, default="llama3:8b"
        Name of model to test generation.
    host : str, default="http://localhost:11434"
        Ollama server base URL endpoint.
    prompt : str, default="Hi"
        Minimal test prompt.

    Returns:
    --------
    Dict[str, Any]
        {
            "success": bool,
            "response": Optional[str],
            "latency_ms": float,
            "error": Optional[str]
        }
    """
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    start_time = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
        latency_ms = (time.time() - start_time) * 1000.0

        if resp.status_code == 200:
            res_data = resp.json()
            gen_text = res_data.get("response", "")
            return {
                "success": True,
                "response": gen_text,
                "latency_ms": round(latency_ms, 2),
                "error": None
            }
        return {
            "success": False,
            "response": None,
            "latency_ms": round(latency_ms, 2),
            "error": f"Ollama /api/generate returned HTTP {resp.status_code}: {resp.text[:100]}"
        }
    except requests.exceptions.Timeout:
        latency_ms = (time.time() - start_time) * 1000.0
        return {
            "success": False,
            "response": None,
            "latency_ms": round(latency_ms, 2),
            "error": f"Generation for model '{model}' timed out after 5 seconds."
        }
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000.0
        return {
            "success": False,
            "response": None,
            "latency_ms": round(latency_ms, 2),
            "error": f"Generation test error: {e}"
        }


def full_ollama_health_check(
    required_models: Optional[List[str]] = None,
    host: str = DEFAULT_OLLAMA_HOST
) -> Dict[str, Any]:
    """
    Executes complete Ollama health check suite (installed, running, model availability, test generation)
    ensuring overall execution completes within <= 5 seconds.

    Parameters:
    -----------
    required_models : Optional[List[str]], optional
        List of required model names (e.g. ["llama3:8b"]).
    host : str, default="http://localhost:11434"
        Ollama server base URL endpoint.

    Returns:
    --------
    Dict[str, Any]
        Aggregated health status dictionary.
    """
    start_time = time.time()

    inst_res = check_ollama_installed()
    run_res = check_ollama_running(host=host)

    if not run_res["running"]:
        total_time_ms = (time.time() - start_time) * 1000.0
        return {
            "healthy": False,
            "installed": inst_res["installed"],
            "running": False,
            "version": inst_res.get("version"),
            "host": host,
            "models": [],
            "missing_models": required_models or [],
            "generation_working": False,
            "total_latency_ms": round(total_time_ms, 2),
            "summary": "Ollama server is not running.",
            "error": run_res.get("error") or inst_res.get("error")
        }

    models_res = list_ollama_models(host=host)
    avail_models = models_res.get("models", [])

    missing = []
    if required_models:
        for req in required_models:
            if not any(req in m for m in avail_models):
                missing.append(req)

    gen_working = False
    gen_err = None
    if avail_models:
        test_model = avail_models[0]
        gen_res = test_ollama_generation(model=test_model, host=host)
        gen_working = gen_res["success"]
        gen_err = gen_res.get("error")

    total_time_ms = (time.time() - start_time) * 1000.0
    is_healthy = inst_res["installed"] and run_res["running"] and (len(missing) == 0) and gen_working

    return {
        "healthy": is_healthy,
        "installed": inst_res["installed"],
        "running": True,
        "version": inst_res.get("version"),
        "host": host,
        "models": avail_models,
        "missing_models": missing,
        "generation_working": gen_working,
        "total_latency_ms": round(total_time_ms, 2),
        "summary": "Ollama offline LLM runner is fully operational." if is_healthy else "Ollama issue detected.",
        "error": gen_err or models_res.get("error")
    }
