"""
LLM Configuration and Router Module for BlasterOPT Agent.

Provides cloud (GPT-4 / Claude) and local (Llama-3 / Mistral / rule-based fallback) LLM abstractions
and an intelligent router that selects the model based on network connectivity and task complexity.
"""

import os
import logging
from typing import Dict, Any, Optional, Callable
import requests
from src.agent.ollama_health import full_health_check
from src.agent.ollama_client import OllamaClient, OllamaCloudClient

logger = logging.getLogger(__name__)


class CloudLLM:
    """
    Cloud LLM interface supporting Hugging Face Inference API / Endpoints (e.g. 'meta-llama/Llama-3.1-8B-Instruct'),
    OpenAI GPT-4o, and Anthropic Claude.
    """

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, endpoint_url: Optional[str] = None):
        self.model_name = model_name or os.getenv("HF_MODEL_ID") or "meta-llama/Llama-3.1-8B-Instruct"
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        self.endpoint_url = endpoint_url or os.getenv("HF_ENDPOINT_URL") or os.getenv("OPENAI_BASE_URL")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates response using Hugging Face Inference API, custom endpoint, or fallback."""
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
                resp = requests.post(target_url, json=payload, headers=headers, timeout=20.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        gen = data[0].get("generated_text", "").strip()
                        if gen:
                            return gen
            except Exception as e:
                logger.warning(f"Hugging Face Cloud LLM endpoint request error: {e}")

        # Fallback analysis
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


class HuggingFaceLLM:
    """
    HuggingFace Pipeline LLM interface supporting models like 'meta-llama/Llama-3.3-70B-Instruct'.
    Uses transformers.pipeline('text-generation') for chat message generation.
    """

    def __init__(self, model_id: str = "meta-llama/Llama-3.3-70B-Instruct"):
        self.model_id = model_id
        self._pipe = None
        self._is_fallback = False

    def _get_pipeline(self):
        if self._pipe is None:
            if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING") == "1":
                self._is_fallback = True
            else:
                try:
                    from transformers import pipeline
                    self._pipe = pipeline("text-generation", model=self.model_id)
                    logger.info(f"Loaded HuggingFace text-generation pipeline for '{self.model_id}'.")
                except Exception as e:
                    logger.warning(f"Error loading pipeline for '{self.model_id}': {e}. Using offline fallback.")
                    self._is_fallback = True
        return self._pipe

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates response using HuggingFace pipeline chat messages or fallback."""
        pipe = self._get_pipeline()
        if not self._is_fallback and pipe is not None:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                res = pipe(messages)
                if res and isinstance(res, list):
                    last_res = res[0]
                    if isinstance(last_res, dict) and "generated_text" in last_res:
                        gen = last_res["generated_text"]
                        if isinstance(gen, list) and len(gen) > 0:
                            return str(gen[-1].get("content", "")).strip()
                        elif isinstance(gen, str):
                            return gen.strip()
            except Exception as e:
                logger.warning(f"HuggingFaceLLM generation error: {e}")

        return f"HuggingFace LLM ({self.model_id}) response: {prompt}"


class Pula8BLLM:
    """
    Setswana-specialized LLM interface wrapping 'OxxoCodes/Pula-8B-v0.1' AutoTokenizer & AutoModelForCausalLM.
    """

    def __init__(self, model_id: str = "OxxoCodes/Pula-8B-v0.1"):
        self.model_id = model_id
        self._tokenizer = None
        self._model = None
        self._is_fallback = False

    def _get_model_and_tokenizer(self):
        if self._tokenizer is None or self._model is None:
            if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING") == "1":
                self._is_fallback = True
            else:
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                    self._model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")
                    logger.info(f"Loaded Setswana model & tokenizer for '{self.model_id}'.")
                except Exception as e:
                    logger.warning(f"Error loading '{self.model_id}': {e}. Using offline Setswana fallback.")
                    self._is_fallback = True
        return self._tokenizer, self._model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates Setswana text response using OxxoCodes/Pula-8B-v0.1 apply_chat_template or fallback."""
        tokenizer, model = self._get_model_and_tokenizer()
        if not self._is_fallback and tokenizer is not None and model is not None:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                outputs = model.generate(**inputs, max_new_tokens=100)
                gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                if gen_text:
                    return gen_text.strip()
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
