"""Tests for the orchestrator pipeline logic (no LLM calls)."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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


class TestBackgroundTaskReliability:
    """H1: Background tasks must log errors and retry once on failure."""

    @pytest.mark.asyncio
    async def test_safe_background_retries_once_on_failure(self):
        from app.chat import _safe_background

        call_count = 0

        async def failing_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")

        await _safe_background(failing_then_succeeds, label="test_task")
        assert call_count == 2  # tried once, failed, retried once, succeeded

    @pytest.mark.asyncio
    async def test_safe_background_logs_permanent_failure(self, caplog):
        import logging
        from app.chat import _safe_background

        async def always_fails():
            raise RuntimeError("permanent error")

        with caplog.at_level(logging.ERROR, logger="app.chat"):
            # Should not raise
            await _safe_background(always_fails, label="test_task")

        assert any("permanent" in msg.lower() or "failed" in msg.lower() for msg in caplog.messages)

    @pytest.mark.asyncio
    async def test_safe_background_succeeds_first_try(self):
        from app.chat import _safe_background

        call_count = 0

        async def succeeds():
            nonlocal call_count
            call_count += 1

        await _safe_background(succeeds, label="test_task")
        assert call_count == 1


class TestConvStateWiring:
    """C3, H3: Conv state is read at start and updated after each turn."""

    @pytest.mark.asyncio
    @patch("app.chat.conv_memory")
    @patch("app.chat.rewrite_query")
    @patch("app.chat.db")
    async def test_conv_state_read_at_start_of_turn(
        self, mock_db, mock_rewrite, mock_memory
    ):
        from app.chat import handle_message

        mock_db.create_session = MagicMock()
        mock_db.get_messages = MagicMock(return_value=[])
        mock_db.get_latest_summary = MagicMock(return_value=None)
        mock_db.get_conv_state = MagicMock(return_value={"topic": "test"})
        mock_db.add_messages_batch = MagicMock()
        mock_memory.read = MagicMock(return_value={"topic": "test"})

        mock_rewrite.return_value = {
            "is_in_scope": False,  # Short circuit after rewrite
            "standalone_query": "test",
            "search_queries": [],
            "filters": {"doc_ids": None, "dieu": None},
        }
        mock_db.add_message = MagicMock()

        await handle_message("sess1", "test message")

        mock_memory.read.assert_called_once_with("sess1")

    @pytest.mark.asyncio
    @patch("app.chat.conv_memory")
    @patch("app.chat.rewrite_query")
    @patch("app.chat.db")
    async def test_conv_state_passed_to_rewrite_query(
        self, mock_db, mock_rewrite, mock_memory
    ):
        from app.chat import handle_message

        state = {"topic": "bồi thường", "danh_sach": ["1. A", "2. B", "3. C"]}

        mock_db.create_session = MagicMock()
        mock_db.get_messages = MagicMock(return_value=[])
        mock_db.get_latest_summary = MagicMock(return_value=None)
        mock_db.add_message = MagicMock()
        mock_memory.read = MagicMock(return_value=state)

        mock_rewrite.return_value = {
            "is_in_scope": False,
            "standalone_query": "test",
            "search_queries": [],
            "filters": {"doc_ids": None, "dieu": None},
        }
        mock_db.add_message = MagicMock()

        await handle_message("sess1", "giai thich truong hop 3")

        call_kwargs = mock_rewrite.call_args.kwargs
        assert call_kwargs.get("conv_state") == state
