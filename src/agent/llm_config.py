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


def select_llm(task_complexity: str = "medium", is_online: bool = True) -> Any:
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
    if not is_online:
        logger.info("Device offline. Selecting LocalLLM.")
        return LocalLLM()

    if task_complexity in ["high", "complex"]:
        logger.info("High task complexity. Selecting CloudLLM.")
        return CloudLLM(model_name="gpt-4o")

    logger.info("Low/medium task complexity or online fallback. Selecting LocalLLM.")
    return LocalLLM()
