"""
Fine-Tuned Pula-8B LLM Client Module for BlasterOPT / BlastOpt Botswana Agent.

Loads the base model (OxxoCodes/Pula-8B-v0.1) and fine-tuned LoRA adapter (mndolo2x/blastopt-pula-8b-lora)
from Hugging Face using BitsAndBytesConfig 4-bit QLoRA quantization.
Model loading is lazy and non-blocking to ensure instant Streamlit application startup.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BlastOPTLLM:
    """
    Wrapper for the fine-tuned Pula-8B model for the BlastOPT agent.
    Loads the base model + LoRA adapter from Hugging Face lazily on demand.
    """

    def __init__(self, adapter_id: str = "mndolo2x/blastopt-pula-8b-lora"):
        self.base_model_id = "OxxoCodes/Pula-8B-v0.1"
        self.adapter_id = adapter_id
        self.model = None
        self.tokenizer = None
        self._is_fallback = False
        self._load_attempted = False

    def _load_model(self) -> None:
        """Load the base model and LoRA adapter lazily in 4-bit precision."""
        if self._load_attempted:
            return

        self._load_attempted = True

        if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING") == "1":
            self._is_fallback = True
            logger.info("Test environment detected. Using fallback mode for BlastOPTLLM.")
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            # Check if online download is explicitly allowed or attempt local load first
            allow_download = os.getenv("ENABLE_ONLINE_HF_DOWNLOAD", "false").lower() in ["1", "true", "yes"]

            logger.info(f"Loading base model: {self.base_model_id}")
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=not allow_download,
                )
            except Exception as loc_err:
                if not allow_download:
                    logger.info(f"Local files not cached for {self.base_model_id}. Set ENABLE_ONLINE_HF_DOWNLOAD=true to download.")
                    self._is_fallback = True
                    return
                raise loc_err

            logger.info(f"Loading fine-tuned adapter: {self.adapter_id}")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_id)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id, trust_remote_code=True, local_files_only=not allow_download
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ BlastOPT fine-tuned model loaded.")
            logger.info("✅ BlastOPT fine-tuned model loaded.")
        except Exception as e:
            print(f"⚠️ Failed to load fine-tuned model: {e}")
            logger.warning(f"⚠️ Failed to load fine-tuned model: {e}")
            self.model = None
            self.tokenizer = None
            self._is_fallback = True

    def is_available(self) -> bool:
        """Returns True if the fine-tuned model and tokenizer are successfully loaded."""
        self._load_model()
        return self.model is not None and self.tokenizer is not None

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate a response. Falls back to a safe message if model unavailable."""
        if not self.is_available():
            return "The offline language model is currently unavailable. Please check your connection and try again."

        try:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
        except Exception as e:
            logger.warning(f"Error during BlastOPTLLM text generation: {e}")
            return "The offline language model encountered an error during response generation. Please try again."


# Global singleton
llm_client = BlastOPTLLM()
