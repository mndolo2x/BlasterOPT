"""
Unit tests for Knowledge Tools (lookup_blast_term, translate_en_tn, translate_tn_en, answer_mining_question).
"""

import pytest
from src.agent.tool_registry import TOOL_REGISTRY


def test_lookup_blast_term_tool():
    """Test lookup_blast_term tool returns term definition and plain language explanation."""
    res_powder = TOOL_REGISTRY.execute_tool("lookup_blast_term", {"term": "powder factor"})
    assert isinstance(res_powder, dict)
    assert "term" in res_powder
    assert "definition" in res_powder
    assert "plain_language" in res_powder

    res_anfo = TOOL_REGISTRY.execute_tool("lookup_blast_term", {"term": "ANFO"})
    assert "ammonium nitrate" in res_anfo["definition"].lower()


def test_translate_en_tn_tool():
    """Test translate_en_tn tool converts English text to Setswana."""
    res_trans = TOOL_REGISTRY.execute_tool("translate_en_tn", {"text": "The time to build has arrived ."})
    assert isinstance(res_trans, str)
    assert len(res_trans) > 0


def test_translate_tn_en_tool():
    """Test translate_tn_en tool converts Setswana text to English."""
    res_trans = TOOL_REGISTRY.execute_tool("translate_tn_en", {"text": "Nako ya go aga e gorogile ."})
    assert isinstance(res_trans, str)
    assert len(res_trans) > 0


def test_blast_block_knowledge_base():
    """Test lookup_blast_term and answer_mining_question retrieve blast block reference guide concepts."""
    res_block = TOOL_REGISTRY.execute_tool("lookup_blast_term", {"term": "what is a blast block"})
    assert isinstance(res_block, dict)
    assert "blast block" in res_block["term"].lower() or "blast block" in res_block["definition"].lower()

    res_burden = TOOL_REGISTRY.execute_tool("lookup_blast_term", {"term": "what is a burden"})
    assert isinstance(res_burden, dict)
    assert "burden" in res_burden["term"].lower()
    assert "perpendicular distance" in res_burden["definition"].lower()

    res_spacing = TOOL_REGISTRY.execute_tool("lookup_blast_term", {"term": "what is spacing"})
    assert isinstance(res_spacing, dict)
    assert "spacing" in res_spacing["term"].lower()

    res_bench = TOOL_REGISTRY.execute_tool("answer_mining_question", {"question": "What is a typical Debswana bench height for a blast block?"})
    assert isinstance(res_bench, str)
    assert "orapa" in res_bench.lower() or "15 m" in res_bench.lower() or "jwaneng" in res_bench.lower()


def test_answer_mining_question_tool():
    """Test answer_mining_question tool generates response using Pula-8B LLM."""
    res_ans = TOOL_REGISTRY.execute_tool("answer_mining_question", {"question": "What is the optimal powder factor for kimberlite?"})
    assert isinstance(res_ans, str)
    assert len(res_ans) > 0
