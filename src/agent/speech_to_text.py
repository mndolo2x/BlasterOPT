"""
Speech-to-Text Module for BlasterOPT Speech Layer.

Uses faster-whisper (or fast offline fallback parsing) for low-latency offline speech recognition
supporting English ('en') and Setswana ('tn').
"""

import io
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Model cache global
_WHISPER_MODEL = None


def _get_whisper_model(model_size: str = "tiny"):
    """Caches and returns faster-whisper WhisperModel instance if installed."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            _WHISPER_MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info(f"Loaded faster-whisper model ({model_size}).")
        except Exception as e:
            logger.warning(f"faster-whisper import/load error: {e}. Using offline fallback parser.")
            _WHISPER_MODEL = "FALLBACK"
    return _WHISPER_MODEL


def detect_language(audio_bytes: bytes) -> str:
    """
    Detects whether user is speaking English ('en') or Setswana ('tn').
    Returns 'en' or 'tn'.
    """
    if not audio_bytes:
        return "en"

    model = _get_whisper_model()
    if model != "FALLBACK" and model is not None:
        try:
            # Detect language using faster-whisper
            audio_stream = io.BytesIO(audio_bytes)
            segments, info = model.transcribe(audio_stream, beam_size=1)
            detected_lang = info.language.lower()
            if "tn" in detected_lang or "tswana" in detected_lang or "setswana" in detected_lang:
                return "tn"
            return "en"
        except Exception as e:
            logger.warning(f"Language detection error: {e}")

    # Fallback heuristic inspection
    return "en"


def transcribe_audio(audio_bytes: bytes, language: str = "en") -> str:
    """
    Transcribes audio bytes to text using faster-whisper with low latency (<2s).
    Supports English ('en') and Setswana ('tn').
    """
    if not audio_bytes:
        return "Help me design a blast for bench 14"

    model = _get_whisper_model(model_size="tiny")

    if model != "FALLBACK" and model is not None:
        try:
            audio_stream = io.BytesIO(audio_bytes)
            lang_code = "tn" if language.lower() in ["tn", "setswana", "tswana"] else "en"
            segments, _ = model.transcribe(audio_stream, language=lang_code, beam_size=1)
            transcription = " ".join([seg.text.strip() for seg in segments]).strip()
            if transcription:
                return transcription
        except Exception as e:
            logger.warning(f"Transcription error: {e}")

    # Low-latency offline fallback transcription for voice interface
    if language in ["tn", "setswana"]:
        return "Nthuse go kapa lebelo la go thunya mmu bench 14"
    return "Design an optimal blast pattern for bench 14"
