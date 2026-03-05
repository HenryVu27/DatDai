"""Tests for the orchestrator pipeline logic (no LLM calls)."""
import json
import pytest
from app.chat import (
    _parse_orchestrator_response,
    _should_force_retrieval,
    SYSTEM_BASE,
    ORCHESTRATOR_TOOLS_SECTION,
    ORCHESTRATOR_INSTRUCTIONS,
    LEGAL_KEYWORDS,
)


class TestParseOrchestratorResponse:
    def test_valid_json(self):
        raw = json.dumps({
            "reasoning": "User asks about Dieu 15",
            "actions": [{"tool": "lookup_specific_dieu", "doc_id": "ldd2024", "dieu": "Dieu 15"}],
            "complexity": "simple",
            "summary_update": None,
            "direct_response": None,
        })
        result = _parse_orchestrator_response(raw)
        assert result["complexity"] == "simple"
        assert len(result["actions"]) == 1
        assert result["actions"][0]["tool"] == "lookup_specific_dieu"

    def test_json_in_markdown_fences(self):
        raw = '```json\n{"reasoning": "test", "actions": [], "complexity": "simple", "direct_response": "Hi"}\n```'
        result = _parse_orchestrator_response(raw)
        assert result["direct_response"] == "Hi"
        assert result["actions"] == []

    def test_invalid_json_becomes_direct_response(self):
        raw = "Xin chao, toi co the giup gi cho ban?"
        result = _parse_orchestrator_response(raw)
        assert result["direct_response"] == raw
        assert result["actions"] == []

    def test_missing_keys_get_defaults(self):
        raw = json.dumps({"reasoning": "test"})
        result = _parse_orchestrator_response(raw)
        assert result["actions"] == []
        assert result["complexity"] == "simple"
        assert result["summary_update"] is None
        assert result["direct_response"] is None

    def test_multiple_actions(self):
        raw = json.dumps({
            "reasoning": "Need search and amendment check",
            "actions": [
                {"tool": "search_legal_docs", "query": "gia dat", "filters": {"doc_ids": ["nd71"]}},
                {"tool": "lookup_amendment", "target_doc": "nd71"},
            ],
            "complexity": "complex",
            "summary_update": None,
            "direct_response": None,
        })
        result = _parse_orchestrator_response(raw)
        assert len(result["actions"]) == 2
        assert result["complexity"] == "complex"


class TestLegalKeywordGuardrail:
    def test_legal_keyword_triggers_on_dieu(self):
        assert LEGAL_KEYWORDS.search("Dieu 79 Luat Dat Dai")

    def test_legal_keyword_triggers_on_nghi_dinh(self):
        assert LEGAL_KEYWORDS.search("ND 102 quy dinh gi")

    def test_legal_keyword_triggers_on_quyen(self):
        assert LEGAL_KEYWORDS.search("Quyen su dung dat la gi")

    def test_legal_keyword_triggers_on_thu_hoi(self):
        assert LEGAL_KEYWORDS.search("Thu hoi dat de lam duong")

    def test_legal_keyword_triggers_on_so_do(self):
        assert LEGAL_KEYWORDS.search("Lam so do can gi")

    def test_no_trigger_on_greeting(self):
        assert not LEGAL_KEYWORDS.search("Xin chao")

    def test_no_trigger_on_thanks(self):
        assert not LEGAL_KEYWORDS.search("Cam on ban nhieu")

    def test_should_force_retrieval_legal_direct(self):
        decision = {"direct_response": "Day la cau tra loi", "actions": []}
        assert _should_force_retrieval("Dieu 79 luat dat dai quy dinh gi", decision)

    def test_should_not_force_retrieval_greeting(self):
        decision = {"direct_response": "Xin chao!", "actions": []}
        assert not _should_force_retrieval("Xin chao", decision)

    def test_should_not_force_when_actions_present(self):
        decision = {"direct_response": None, "actions": [{"tool": "search"}]}
        assert not _should_force_retrieval("Dieu 79", decision)

    def test_should_not_force_when_no_direct_response(self):
        decision = {"direct_response": None, "actions": []}
        assert not _should_force_retrieval("Dieu 79", decision)


class TestSystemPromptStructure:
    def test_base_has_xml_tags(self):
        assert "<identity>" in SYSTEM_BASE
        assert "</identity>" in SYSTEM_BASE
        assert "<boundaries>" in SYSTEM_BASE
        assert "<legal-hierarchy>" in SYSTEM_BASE

    def test_tools_section_has_all_tools(self):
        assert "search_legal_docs" in ORCHESTRATOR_TOOLS_SECTION
        assert "lookup_amendment" in ORCHESTRATOR_TOOLS_SECTION
        assert "lookup_specific_dieu" in ORCHESTRATOR_TOOLS_SECTION

    def test_tools_section_has_retrieval_policy(self):
        assert "<retrieval-policy>" in ORCHESTRATOR_TOOLS_SECTION
        assert "BAT KY cau hoi lien quan den phap luat" in ORCHESTRATOR_TOOLS_SECTION

    def test_instructions_has_examples(self):
        assert "<examples>" in ORCHESTRATOR_INSTRUCTIONS
        assert "lookup_specific_dieu" in ORCHESTRATOR_INSTRUCTIONS

    def test_static_prefix_ordering(self):
        id_pos = SYSTEM_BASE.index("<identity>")
        bound_pos = SYSTEM_BASE.index("<boundaries>")
        hier_pos = SYSTEM_BASE.index("<legal-hierarchy>")
        assert id_pos < bound_pos < hier_pos
