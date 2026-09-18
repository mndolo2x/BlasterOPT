"""
Ollama Client Wrapper Module for BlasterOPT Conversational Agent.

Provides an interface to interact with local Ollama LLM models, enabling
offline inference in remote pit floor environments without cloud connectivity.
"""

import logging
from typing import Optional, Callable
import requests
from src.agent.ollama_health import full_health_check

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Wrapper for the Ollama API for the BlasterOPT agent.
    Handles model selection, retries, and graceful degradation to cloud fallback.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self.base_url = base_url
        self.model = model
        self.is_available = False
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if Ollama is available and capable of offline generation."""
        try:
            health = full_health_check(base_url=self.base_url)
            self.is_available = health.get("can_use_offline_llm", False)
        except Exception as e:
            logger.warning(f"Error checking Ollama availability: {e}")
            self.is_available = False

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a response from the local Ollama model."""
        if not self.is_available:
            raise RuntimeError("Ollama is not available. Cannot generate offline.")

        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens}
        }

        try:
            resp = requests.post(url, json=payload, timeout=30.0)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "").strip()
            raise RuntimeError(f"Ollama API returned HTTP status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise RuntimeError(f"Failed to generate response from Ollama model '{self.model}': {e}")

    def generate_with_fallback(
        self, prompt: str, cloud_fallback: Optional[Callable[[str], str]] = None
    ) -> str:
        """Try local Ollama first, fall back to cloud if unavailable."""
        if self.is_available:
            try:
                return self.generate(prompt)
            except Exception as e:
                logger.warning(f"Ollama generation failed ({e}). Attempting cloud fallback...")
                if cloud_fallback:
                    return cloud_fallback(prompt)
                raise
        elif cloud_fallback:
            return cloud_fallback(prompt)
        else:
            raise RuntimeError("No LLM available.")
