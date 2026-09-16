"""
Unit tests for Internationalization (i18n) translation dictionaries and helpers.
"""

import pytest
from src.i18n import TRANSLATIONS, get_translation


def test_translation_dictionaries_have_identical_keys():
    """Test that English ('en') and Setswana ('tn') translation dictionaries contain identical key sets."""
    en_keys = set(TRANSLATIONS["en"].keys())
    tn_keys = set(TRANSLATIONS["tn"].keys())

    missing_in_tn = en_keys - tn_keys
    missing_in_en = tn_keys - en_keys

    assert len(missing_in_tn) == 0, f"Keys present in 'en' but missing in 'tn': {missing_in_tn}"
    assert len(missing_in_en) == 0, f"Keys present in 'tn' but missing in 'en': {missing_in_en}"


def test_get_translation_helper():
    """Test get_translation function returns localized strings for English and Setswana."""
    # English lookup
    title_en = get_translation("app_title", lang="en")
    assert title_en == "BlastOpt Botswana"

    # Setswana lookup
    title_tn = get_translation("app_title", lang="tn")
    assert "Setswana" in title_tn or "Motheo" in title_tn

    # Fallback behavior
    fallback_val = get_translation("non_existent_key", lang="tn")
    assert fallback_val == "non_existent_key"
