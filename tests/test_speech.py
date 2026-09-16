"""
Unit tests for Speech Layer modules (speech_to_text, text_to_speech, voice_interface).
"""

import pytest
from src.agent.speech_to_text import transcribe_audio, detect_language
from src.agent.text_to_speech import synthesize_speech, list_voices
from src.agent.voice_interface import process_voice_turn, start_voice_session


def test_transcribe_audio_returns_non_empty_string():
    """Test transcribe_audio returns a non-empty string for empty or sample audio bytes."""
    # Test empty audio bytes
    text_empty = transcribe_audio(b"")
    assert isinstance(text_empty, str)
    assert len(text_empty) > 0

    # Test sample audio bytes in English
    sample_wav = b"RIFF....WAVEfmt ....data...."
    text_en = transcribe_audio(sample_wav, language="en")
    assert isinstance(text_en, str)
    assert len(text_en) > 0

    # Test sample audio bytes in Setswana
    text_tn = transcribe_audio(sample_wav, language="tn")
    assert isinstance(text_tn, str)
    assert len(text_tn) > 0


def test_detect_language():
    """Test detect_language returns valid language code 'en' or 'tn'."""
    lang = detect_language(b"")
    assert lang in ["en", "tn"]


def test_synthesize_speech_returns_non_empty_bytes():
    """Test synthesize_speech returns non-empty bytes."""
    audio = synthesize_speech("Optimal blast design generated for bench 14.")
    assert isinstance(audio, bytes)
    assert len(audio) > 0


def test_list_voices():
    """Test list_voices returns voice list for English and Setswana."""
    voices_en = list_voices("en")
    assert isinstance(voices_en, list)
    assert len(voices_en) >= 1

    voices_tn = list_voices("tn")
    assert isinstance(voices_tn, list)
    assert len(voices_tn) >= 1


def test_process_voice_turn():
    """Test process_voice_turn returns dict with 'text', 'audio', and 'language'."""
    sample_wav = b"RIFF....WAVEfmt ....data...."
    session_id = start_voice_session(user_id="VOICE_TEST_USER")

    res = process_voice_turn(audio_bytes=sample_wav, user_id="VOICE_TEST_USER", session_id=session_id)

    assert isinstance(res, dict)
    assert "text" in res
    assert "audio" in res
    assert "language" in res
    assert isinstance(res["text"], str)
    assert isinstance(res["audio"], bytes)
    assert len(res["audio"]) > 0
    assert res["language"] in ["en", "tn"]
