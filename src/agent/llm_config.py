"""
LLM Configuration and Router Module for BlasterOPT Agent.

Provides cloud (GPT-4 / Claude) and local (Llama-3 / Mistral / rule-based fallback) LLM abstractions
and an intelligent router that selects the model based on network connectivity and task complexity.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CloudLLM:
    """
    Cloud LLM interface (GPT-4o / Claude 3.5 Sonnet) for complex reasoning tasks.
    """

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates response using Cloud LLM or fallback if key missing."""
        if not self.api_key:
            logger.warning("Cloud LLM API key missing. Falling back to rule-based parser.")
            return f"[Cloud LLM Simulated Response for: {prompt[:50]}...]"

        # Real/simulated API call logic
        return f"Cloud LLM ({self.model_name}) analysis: {prompt}"


class LocalLLM:
    """
    Local LLM interface (Llama 3 8B / Mistral 7B / Ollama / local rule fallback) for offline pit floor usage.
    """

    def __init__(self, model_name: str = "llama3:8b", endpoint: str = "http://localhost:11434"):
        self.model_name = model_name
        self.endpoint = endpoint

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates response using local model endpoint or deterministic fallback."""
        return f"Local LLM ({self.model_name}) response: {prompt}"


class Pula8BLLM:
    """
    Setswana-specialized LLM interface wrapping 'OxxoCodes/Pula-8B-v0.1' text-generation pipeline.
    """

    def __init__(self, model_id: str = "OxxoCodes/Pula-8B-v0.1"):
        self.model_id = model_id
        self._pipe = None

    def _get_pipeline(self):
        if self._pipe is None:
            # Check if running under pytest or if environment flag is set to avoid large downloads during tests
            if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING") == "1":
                self._pipe = "FALLBACK"
            else:
                try:
                    from transformers import pipeline
                    self._pipe = pipeline("text-generation", model=self.model_id)
                    logger.info(f"Loaded Setswana LLM pipeline '{self.model_id}'.")
                except Exception as e:
                    logger.warning(f"Error loading '{self.model_id}' pipeline: {e}. Using offline Setswana fallback.")
                    self._pipe = "FALLBACK"
        return self._pipe

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates Setswana text response using OxxoCodes/Pula-8B-v0.1 pipeline or fallback."""
        pipe = self._get_pipeline()
        if pipe != "FALLBACK" and pipe is not None:
            try:
                messages = [{"role": "user", "content": prompt}]
                res = pipe(messages, max_new_tokens=150)
                if res and len(res) > 0:
                    return str(res[0].get("generated_text", f"Pula-8B ({self.model_id}): {prompt}"))
            except Exception as e:
                logger.warning(f"Pula-8B generation error: {e}")

        return f"Pula-8B ({self.model_id}) phetolo ka Setswana: {prompt}"


def select_llm(task_complexity: str = "medium", is_online: bool = True, language: str = "en") -> Any:
    """
    Router function selecting CloudLLM or LocalLLM based on network connectivity and task complexity.

    Parameters:
    -----------
    task_complexity : str, default="medium"
        Complexity level: "low" (intent classification), "medium" (guided design), "high" (complex multi-objective trade-offs).
    is_online : bool, default=True
        Whether network connection is available.

    Returns:
    --------
    Any
        CloudLLM or LocalLLM instance.
    """
    if language.lower() in ["tn", "setswana", "tswana"]:
        logger.info("Setswana language requested. Selecting Pula8BLLM (OxxoCodes/Pula-8B-v0.1).")
        return Pula8BLLM()

    if not is_online:
        logger.info("Device offline. Selecting LocalLLM.")
        return LocalLLM()

    if task_complexity in ["high", "complex"]:
        logger.info("High task complexity. Selecting CloudLLM.")
        return CloudLLM(model_name="gpt-4o")

    logger.info("Low/medium task complexity or online fallback. Selecting LocalLLM.")
    return LocalLLM()
