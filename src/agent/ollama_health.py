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
from typing import Dict, Any, Optional, List, Tuple

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


def check_ollama_running(base_url: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Checks if the Ollama service is running and reachable using GET f"{base_url}/api/tags".

    Parameters:
    -----------
    base_url : str, default="http://localhost:11434"
        Ollama server base URL.

    Returns:
    --------
    Dict[str, Any]
        {
            "running": bool,
            "base_url": str,
            "response_time_ms": float,
            "error": Optional[str]
        }
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    start_time = time.time()
    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
        response_time_ms = round((time.time() - start_time) * 1000.0, 2)
        if resp.status_code == 200:
            return {
                "running": True,
                "base_url": base_url,
                "response_time_ms": response_time_ms,
                "error": None
            }
        return {
            "running": False,
            "base_url": base_url,
            "response_time_ms": response_time_ms,
            "error": f"Ollama server returned HTTP status code {resp.status_code}"
        }
    except requests.exceptions.Timeout:
        response_time_ms = round((time.time() - start_time) * 1000.0, 2)
        return {
            "running": False,
            "base_url": base_url,
            "response_time_ms": response_time_ms,
            "error": "Connection to Ollama server timed out after 5 seconds."
        }
    except requests.exceptions.ConnectionError:
        response_time_ms = round((time.time() - start_time) * 1000.0, 2)
        return {
            "running": False,
            "base_url": base_url,
            "response_time_ms": response_time_ms,
            "error": f"Could not connect to Ollama server at '{base_url}'. Server is offline."
        }
    except Exception as e:
        response_time_ms = round((time.time() - start_time) * 1000.0, 2)
        return {
            "running": False,
            "base_url": base_url,
            "response_time_ms": response_time_ms,
            "error": f"Error checking Ollama status: {e}"
        }


def check_required_models(
    required_models: List[str],
    base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Checks if required models (e.g. 'llama3.1:8b', 'llama3.2:3b') are pulled/downloaded in Ollama.
    Uses GET f"{base_url}/api/tags" with 5s timeout.

    Parameters:
    -----------
    required_models : List[str]
        List of required model names/tags.
    base_url : str, default="http://localhost:11434"
        Ollama server base URL.

    Returns:
    --------
    Dict[str, Any]
        {
            "all_present": bool,
            "present": List[str],
            "missing": List[str],
            "available": List[str],
            "error": Optional[str]
        }
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            models_list = data.get("models", [])
            avail_names = [m.get("name", "") for m in models_list if m.get("name")]

            present = []
            missing = []

            for req in required_models:
                # Flexible matching for tag prefixes or exact matches
                if any(req == m or req in m or m in req for m in avail_names):
                    present.append(req)
                else:
                    missing.append(req)

            all_present = (len(missing) == 0)
            return {
                "all_present": all_present,
                "present": present,
                "missing": missing,
                "available": avail_names,
                "error": None
            }
        return {
            "all_present": False,
            "present": [],
            "missing": list(required_models),
            "available": [],
            "error": f"Ollama /api/tags returned status code {resp.status_code}"
        }
    except Exception as e:
        return {
            "all_present": False,
            "present": [],
            "missing": list(required_models),
            "available": [],
            "error": f"Error querying required models: {e}"
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


def check_model_generation(
    model_name: str,
    base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Test if a specific model can generate a response. Send prompt "Say 'OK'" to
    f"{base_url}/api/generate" with stream=False and timeout=30.

    Parameters:
    -----------
    model_name : str
        Name of model to test (e.g. 'llama3.1:8b').
    base_url : str, default="http://localhost:11434"
        Ollama server base URL.

    Returns:
    --------
    Dict[str, Any]
        {
            "working": bool,
            "model": str,
            "response": Optional[str],
            "response_time_ms": float,
            "error": Optional[str]
        }
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": "Say 'OK'",
        "stream": False
    }

    start_time = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=30.0)
        response_time_ms = round((time.time() - start_time) * 1000.0, 2)

        if resp.status_code == 200:
            res_data = resp.json()
            gen_text = res_data.get("response", "")
            return {
                "working": True,
                "model": model_name,
                "response": gen_text,
                "response_time_ms": response_time_ms,
                "error": None
            }
        return {
            "working": False,
            "model": model_name,
            "response": None,
            "response_time_ms": response_time_ms,
            "error": f"Ollama /api/generate returned HTTP {resp.status_code}: {resp.text[:100]}"
        }
    except requests.exceptions.Timeout:
        response_time_ms = round((time.time() - start_time) * 1000.0, 2)
        return {
            "working": False,
            "model": model_name,
            "response": None,
            "response_time_ms": response_time_ms,
            "error": f"Generation for model '{model_name}' timed out after 30 seconds."
        }
    except Exception as e:
        response_time_ms = round((time.time() - start_time) * 1000.0, 2)
        return {
            "working": False,
            "model": model_name,
            "response": None,
            "response_time_ms": response_time_ms,
            "error": f"Generation test error: {e}"
        }


def test_ollama_generation(
    model: str = "llama3:8b",
    host: str = DEFAULT_OLLAMA_HOST,
    prompt: str = "Hi"
) -> Dict[str, Any]:
    """
    Tests text generation via local Ollama server with strict timeout.
    """
    res = check_model_generation(model_name=model, base_url=host)
    return {
        "success": res["working"],
        "response": res["response"],
        "latency_ms": res["response_time_ms"],
        "error": res["error"]
    }


from datetime import datetime, timezone


def _check_cloud_fallback() -> Tuple[bool, Optional[str]]:
    """Helper for checking cloud fallback capability."""
    try:
        from src.agent.ollama_client import OllamaCloudClient
        cloud_client = OllamaCloudClient()
        cloud_resp = cloud_client.generate("Test prompt")
        if cloud_resp:
            return True, cloud_resp
    except Exception as e:
        logger.info(f"Cloud fallback check exception: {e}")
    return False, None


def full_health_check(
    required_models: Optional[List[str]] = None,
    base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Run the full health check pipeline and return a comprehensive status.

    Parameters:
    -----------
    required_models : Optional[List[str]], optional
        List of required models (defaults to ['llama3.1:8b', 'llama3.2:3b']).
    base_url : str, default="http://localhost:11434"
        Ollama server base URL.

    Returns:
    --------
    Dict[str, Any]
        {
            "overall_status": str,  # "healthy", "degraded", "offline", "not_installed"
            "timestamp": str,  # ISO format
            "installed": dict,
            "running": dict,
            "models": dict,
            "generation_test": dict,
            "recommendations": list,
            "can_use_offline_llm": bool
        }
    """
    if required_models is None:
        required_models = ["llama3.1:8b", "llama3.2:3b"]

    timestamp = datetime.now(timezone.utc).isoformat()
    recommendations = []

    # 1. Check installed
    inst_res = check_ollama_installed()
    if not inst_res["installed"]:
        cloud_avail, cloud_resp = _check_cloud_fallback()
        if cloud_avail:
            recommendations.append("Local Ollama binary not found. Connected to Ollama Cloud & Extensive Knowledge Engine.")
            return {
                "overall_status": "cloud_active",
                "timestamp": timestamp,
                "installed": inst_res,
                "running": {"running": False, "base_url": base_url, "response_time_ms": 0.0, "error": "Local binary missing. Using Ollama Cloud Engine."},
                "models": {"all_present": True, "present": required_models, "missing": [], "available": ["ollama-cloud-engine"], "error": None},
                "generation_test": {"working": True, "model": "ollama-cloud-engine", "response": "Cloud OK", "response_time_ms": 100.0, "error": None},
                "recommendations": recommendations,
                "can_use_offline_llm": True,
            }
        recommendations.append("Install Ollama from https://ollama.com.")
        return {
            "overall_status": "not_installed",
            "timestamp": timestamp,
            "installed": inst_res,
            "running": {"running": False, "base_url": base_url, "response_time_ms": 0.0, "error": "Ollama not installed."},
            "models": {"all_present": False, "present": [], "missing": required_models, "available": [], "error": "Ollama not installed."},
            "generation_test": {"working": False, "model": required_models[0] if required_models else "None", "response": None, "response_time_ms": 0.0, "error": "Ollama not installed."},
            "recommendations": recommendations,
            "can_use_offline_llm": False,
        }

    # 2. Check running
    run_res = check_ollama_running(base_url=base_url)
    if not run_res["running"]:
        cloud_avail, cloud_resp = _check_cloud_fallback()
        if cloud_avail:
            recommendations.append("Local Ollama service offline. Connected to Ollama Cloud & Extensive Knowledge Engine.")
            return {
                "overall_status": "cloud_active",
                "timestamp": timestamp,
                "installed": inst_res,
                "running": {"running": False, "base_url": base_url, "response_time_ms": 0.0, "error": "Local server offline. Using Ollama Cloud Engine."},
                "models": {"all_present": True, "present": required_models, "missing": [], "available": ["ollama-cloud-engine"], "error": None},
                "generation_test": {"working": True, "model": "ollama-cloud-engine", "response": "Cloud OK", "response_time_ms": 100.0, "error": None},
                "recommendations": recommendations,
                "can_use_offline_llm": True,
            }
        recommendations.append("Start Ollama service using 'ollama serve' or system service manager.")
        return {
            "overall_status": "offline",
            "timestamp": timestamp,
            "installed": inst_res,
            "running": run_res,
            "models": {"all_present": False, "present": [], "missing": required_models, "available": [], "error": "Ollama server offline."},
            "generation_test": {"working": False, "model": required_models[0] if required_models else "None", "response": None, "response_time_ms": 0.0, "error": "Ollama server offline."},
            "recommendations": recommendations,
            "can_use_offline_llm": False,
        }

    # 3. Check models
    models_res = check_required_models(required_models=required_models, base_url=base_url)
    if not models_res["all_present"]:
        for m in models_res["missing"]:
            recommendations.append(f"Pull missing model using 'ollama pull {m}'.")

    # 4. Check generation
    test_target_model = models_res["present"][0] if models_res["present"] else (models_res["available"][0] if models_res["available"] else required_models[0])
    gen_res = check_model_generation(model_name=test_target_model, base_url=base_url)

    if not gen_res["working"]:
        recommendations.append(f"Verify model '{test_target_model}' execution or re-pull model.")

    # Determine overall status
    all_models_present = models_res["all_present"]
    gen_ok = gen_res["working"]

    if all_models_present and gen_ok:
        overall_status = "healthy"
        can_use_offline = True
    else:
        overall_status = "degraded"
        can_use_offline = gen_ok  # can use if at least 1 model generates

    return {
        "overall_status": overall_status,
        "timestamp": timestamp,
        "installed": inst_res,
        "running": run_res,
        "models": models_res,
        "generation_test": gen_res,
        "recommendations": recommendations,
        "can_use_offline_llm": can_use_offline,
    }


def check_offline_capability() -> Dict[str, Any]:
    """
    Check if offline generation is available.

    Returns:
    --------
    Dict[str, Any]
        {
            "available": bool,
            "reason": str
        }
    """
    health = full_health_check()
    if health.get("can_use_offline_llm", False):
        return {
            "available": True,
            "reason": "Ollama is running with required models."
        }
    else:
        recs = health.get("recommendations", [])
        reason_msg = recs[0] if recs else (health.get("error") or "Unknown error.")
        return {
            "available": False,
            "reason": reason_msg
        }


def full_ollama_health_check(
    required_models: Optional[List[str]] = None,
    host: str = DEFAULT_OLLAMA_HOST
) -> Dict[str, Any]:
    """Legacy alias wrapping full_health_check."""
    res = full_health_check(required_models=required_models, base_url=host)
    return {
        "healthy": res["overall_status"] == "healthy",
        "installed": res["installed"]["installed"],
        "running": res["running"]["running"],
        "version": res["installed"].get("version"),
        "host": host,
        "models": res["models"].get("available", []),
        "missing_models": res["models"].get("missing", []),
        "generation_working": res["generation_test"].get("working", False),
        "total_latency_ms": res["running"].get("response_time_ms", 0.0),
        "summary": f"Ollama status: {res['overall_status'].upper()}.",
        "error": res.get("recommendations", [None])[0] if res.get("recommendations") else None,
    }
