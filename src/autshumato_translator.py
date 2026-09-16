"""
Autshumato Parallel Corpora Translator and Search Engine for BlastOpt Botswana.

Provides English <-> Setswana parallel phrase matching and translation search powered by the
NWU-CTexT Autshumato English-Setswana parallel corpus (located in `data/raw/autshumato/`).
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

AUTSHUMATO_DIR = "data/raw/autshumato"

# Cached memory parallel dictionary
_PARALLEL_CORPUS_CACHE: List[Tuple[str, str]] = []


def load_autshumato_corpus(data_dir: str = AUTSHUMATO_DIR) -> List[Tuple[str, str]]:
    """
    Loads parallel English-Setswana text pairs from Autshumato corpora files into memory cache.

    Returns:
    --------
    List[Tuple[str, str]]
        List of parallel tuples `(english_sentence, setswana_sentence)`.
    """
    global _PARALLEL_CORPUS_CACHE
    if _PARALLEL_CORPUS_CACHE:
        return _PARALLEL_CORPUS_CACHE

    pairs = []

    # Primary files to inspect
    files_to_load = [
        ("Corpus.DACB3.BilingualData_Translated.2.0.0.en.txt", "Corpus.DACB3.BilingualData_Translated.2.0.0.tn.txt"),
        ("Corpus.DACB3.BilingualData_ReliableSources.2.0.0.en.txt", "Corpus.DACB3.BilingualData_ReliableSources.2.0.0.tn.txt"),
    ]

    for en_fname, tn_fname in files_to_load:
        en_path = os.path.join(data_dir, en_fname)
        tn_path = os.path.join(data_dir, tn_fname)

        if os.path.exists(en_path) and os.path.exists(tn_path):
            try:
                with open(en_path, "r", encoding="utf-8") as f_en, open(tn_path, "r", encoding="utf-8") as f_tn:
                    for line_en, line_tn in zip(f_en, f_tn):
                        clean_en = line_en.strip()
                        clean_tn = line_tn.strip()
                        if clean_en and clean_tn:
                            pairs.append((clean_en, clean_tn))
            except Exception as e:
                logger.warning(f"Error loading Autshumato file pair ({en_fname}): {e}")

    _PARALLEL_CORPUS_CACHE = pairs
    logger.info(f"Loaded {len(pairs)} Autshumato English-Setswana parallel sentence pairs.")
    return pairs


def search_parallel_corpus(
    query: str,
    source_lang: str = "en",
    limit: int = 5,
) -> List[Tuple[str, str]]:
    """
    Searches the Autshumato parallel corpus for occurrences matching a query string.

    Parameters:
    -----------
    query : str
        Search query keyword or phrase.
    source_lang : str, default="en"
        Source language to query ('en' or 'tn').
    limit : int, default=5
        Maximum number of matched parallel pairs to return.

    Returns:
    --------
    List[Tuple[str, str]]
        Matched parallel tuples `(english_sentence, setswana_sentence)`.
    """
    corpus = load_autshumato_corpus()
    if not corpus or not query:
        return []

    clean_query = query.strip().lower()
    matches = []

    for en_sent, tn_sent in corpus:
        target_sent = en_sent.lower() if source_lang.lower() == "en" else tn_sent.lower()
        if clean_query in target_sent:
            matches.append((en_sent, tn_sent))
            if len(matches) >= limit:
                break

    return matches


def translate_phrase(
    phrase: str,
    source_lang: str = "en",
    target_lang: str = "tn",
) -> str:
    """
    Translates a phrase using exact or fuzzy sentence alignment from the Autshumato parallel corpus.
    Falls back to original phrase if no match is found.

    Parameters:
    -----------
    phrase : str
        Input phrase to translate.
    source_lang : str, default="en"
        Source language code ('en' or 'tn').
    target_lang : str, default="tn"
        Target language code ('en' or 'tn').

    Returns:
    --------
    str
        Translated phrase or fallback.
    """
    matches = search_parallel_corpus(phrase, source_lang=source_lang, limit=1)
    if matches:
        en_match, tn_match = matches[0]
        return tn_match if target_lang.lower() in ["tn", "setswana", "tswana"] else en_match

    return phrase
