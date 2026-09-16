"""
Unit tests for Autshumato Parallel Corpora Translator module (src/autshumato_translator.py).
"""

import pytest
from src.autshumato_translator import (
    load_autshumato_corpus,
    search_parallel_corpus,
    translate_phrase,
)


def test_load_autshumato_corpus():
    """Test loading parallel English-Setswana text pairs from local Autshumato corpus files."""
    corpus = load_autshumato_corpus()
    assert isinstance(corpus, list)
    assert len(corpus) > 0

    en_sample, tn_sample = corpus[0]
    assert isinstance(en_sample, str)
    assert isinstance(tn_sample, str)
    assert len(en_sample) > 0
    assert len(tn_sample) > 0


def test_search_parallel_corpus():
    """Test searching Autshumato parallel corpus for matching English/Setswana phrases."""
    matches = search_parallel_corpus("time to build", source_lang="en", limit=5)
    assert isinstance(matches, list)
    assert len(matches) >= 1

    en_match, tn_match = matches[0]
    assert "time to build" in en_match.lower()
    assert len(tn_match) > 0


def test_translate_phrase():
    """Test translate_phrase returns matched translation or fallback."""
    translated_tn = translate_phrase("The time to build has arrived .", source_lang="en", target_lang="tn")
    assert isinstance(translated_tn, str)
    assert len(translated_tn) > 0

    # Test fallback for unmapped custom technical query
    fallback_res = translate_phrase("Custom non-existent technical sentence 12345", source_lang="en", target_lang="tn")
    assert fallback_res == "Custom non-existent technical sentence 12345"
