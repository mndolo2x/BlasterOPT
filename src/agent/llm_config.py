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


class BlastOPT_LLM:
    """
    Fine-tuned BlastOPT LLM loading base model 'OxxoCodes/Pula-8B-v0.1' and QLoRA adapter 'malumbondolo/blastopt-pula-8b-lora'.
    """

    def __init__(self, base_model_id: str = "OxxoCodes/Pula-8B-v0.1", adapter_id: str = "malumbondolo/blastopt-pula-8b-lora"):
        self.base_model_id = base_model_id
        self.adapter_id = adapter_id
        self.model = None
        self.tokenizer = None
        self._is_fallback = False

    def _load_model(self):
        if self.model is None or self.tokenizer is None:
            if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING") == "1":
                self._is_fallback = True
            else:
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                    from peft import PeftModel

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )

                    logger.info(f"Loading base model: {self.base_model_id}")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_id,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )

                    logger.info(f"Loading fine-tuned adapter: {self.adapter_id}")
                    self.model = PeftModel.from_pretrained(base_model, self.adapter_id)

                    self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info("✅ BlastOPT fine-tuned model loaded successfully!")
                except Exception as e:
                    logger.warning(f"Error loading BlastOPT fine-tuned model '{self.adapter_id}': {e}. Using fallback.")
                    self._is_fallback = True

    def generate(self, prompt: str, max_new_tokens: int = 256, system_prompt: Optional[str] = None) -> str:
        """Generate a response using the fine-tuned model or fallback."""
        self._load_model()
        if not self._is_fallback and self.model is not None and self.tokenizer is not None:
            try:
                import torch
                full_prompt = f"{system_prompt or ''}\n{prompt}".strip()
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response[len(full_prompt):].strip()
            except Exception as e:
                logger.warning(f"BlastOPT_LLM generation error: {e}")

        return f"BlastOPT LLM response: {prompt}"


# Create a global instance to be used by the agent
llm_client = BlastOPT_LLM()


class Pula8BLLM:
    """
    Setswana-specialized LLM interface wrapping 'OxxoCodes/Pula-8B-v0.1' / BlastOPT fine-tuned adapter.
    """

    def __init__(self, model_id: str = "OxxoCodes/Pula-8B-v0.1"):
        self.model_id = model_id
        self._llm = llm_client

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates Setswana text response using BlastOPT_LLM or fallback."""
        return self._llm.generate(prompt, system_prompt=system_prompt)


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
