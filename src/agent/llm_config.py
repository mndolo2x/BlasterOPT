"""
LLM Configuration and Router Module for BlasterOPT Agent.

The local route uses the real Ollama HTTP API. Configure OLLAMA_HOST (and
optionally OLLAMA_MODEL) in the deployment environment. Cloud/fallback paths
remain available when a reachable Ollama endpoint is not configured.
"""

import logging
import os
from typing import Any, Optional

import requests

from src.agent.ollama_client import OllamaClient, OllamaCloudClient
from src.agent.ollama_health import full_health_check

logger = logging.getLogger(__name__)


class CloudLLM:
    """Minimal Hugging Face/custom endpoint adapter with a safe fallback."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, endpoint_url: Optional[str] = None):
        self.model_name = model_name or os.getenv("HF_MODEL_ID") or "meta-llama/Llama-3.1-8B-Instruct"
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        self.endpoint_url = endpoint_url or os.getenv("HF_ENDPOINT_URL") or os.getenv("OPENAI_BASE_URL")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self.api_key or self.endpoint_url:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            target_url = self.endpoint_url or f"https://api-inference.huggingface.co/models/{self.model_name}"
            payload = {
                "inputs": f"{system_prompt or 'You are BlasterOPT AI.'}\nUser: {prompt}\nAssistant:",
                "parameters": {"max_new_tokens": 512, "return_full_text": False},
            }
            try:
                response = requests.post(target_url, json=payload, headers=headers, timeout=20.0)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and data and data[0].get("generated_text"):
                        return str(data[0]["generated_text"]).strip()
            except Exception as exc:
                logger.warning("Cloud LLM request failed: %s", exc)
        return f"Cloud LLM ({self.model_name}) analysis: {prompt}"


class LocalLLM:
    """Ollama-backed local LLM adapter."""

    def __init__(self, model_name: Optional[str] = None, endpoint: Optional[str] = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.endpoint = endpoint or os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_CLOUD_URL") or "http://localhost:11434"
        self.client = OllamaClient(base_url=self.endpoint, model=self.model_name)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if system_prompt:
            prompt = f"{system_prompt}\n\nUser: {prompt}"
        return self.client.generate(prompt)


# Keep the public class names used by older agent modules, while routing the
# knowledge-question path through the actual Ollama client when available.
OllamaLLM = LocalLLM


class HuggingFaceLLM:
    """Compatibility wrapper for the optional Transformers implementation."""

    def __init__(self, model_id: str = "meta-llama/Llama-3.3-70B-Instruct"):
        self.model_id = model_id

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return CloudLLM(model_name=self.model_id).generate(prompt, system_prompt)


class Pula8BLLM:
    """Setswana-compatible route; uses configured Ollama before cloud fallback."""

    def __init__(self, model_id: str = "llama3.1:8b"):
        self.model_id = os.getenv("OLLAMA_MODEL", model_id)
        self.local = LocalLLM(model_name=self.model_id)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return self.local.generate(prompt, system_prompt)


def select_llm(task_complexity: str = "medium", is_online: bool = True, language: str = "en") -> Any:
    """Select the configured Ollama client for normal and offline agent work."""
    try:
        health = full_health_check()
        if health.get("can_use_offline_llm"):
            return LocalLLM()
        logger.warning("Ollama is not currently reachable: %s", health.get("overall_status"))
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)

    if task_complexity in {"high", "complex"} and is_online:
        return CloudLLM(model_name="gpt-4o")
    return LocalLLM()


__all__ = [
    "CloudLLM",
    "LocalLLM",
    "OllamaClient",
    "OllamaCloudClient",
    "OllamaLLM",
    "HuggingFaceLLM",
    "Pula8BLLM",
    "select_llm",
]
