"""Tests for conversational memory extractor."""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestConvMemoryExtract:

    @pytest.mark.asyncio
    @patch("app.memory.conv_memory.llm")
    async def test_returns_parsed_json_from_llm(self, mock_llm):
        from app.memory.conv_memory import extract_state

        mock_llm.generate = AsyncMock(return_value=json.dumps({
            "chu_de": "bồi thường thu hồi đất",
            "danh_sach": ["1. Trường hợp A", "2. Trường hợp B", "3. Chi phí đầu tư"],
        }))

        result = await extract_state(
            response="...response text...",
            prev_state=None,
        )

        assert result["chu_de"] == "bồi thường thu hồi đất"
        assert len(result["danh_sach"]) == 3

    @pytest.mark.asyncio
    @patch("app.memory.conv_memory.llm")
    async def test_passes_previous_state_to_llm(self, mock_llm):
        from app.memory.conv_memory import extract_state

        prev = {"chu_de": "giá đất", "van_ban": "NĐ 71"}
        mock_llm.generate = AsyncMock(return_value=json.dumps(prev))

        await extract_state(response="new response", prev_state=prev)

        call_args = mock_llm.generate.call_args
        prompt = call_args.kwargs.get("prompt") or call_args.args[0]
        assert "giá đất" in prompt or "NĐ 71" in prompt

    @pytest.mark.asyncio
    @patch("app.memory.conv_memory.llm")
    async def test_returns_none_on_llm_failure(self, mock_llm):
        from app.memory.conv_memory import extract_state

        mock_llm.generate = AsyncMock(side_effect=Exception("API error"))

        result = await extract_state(response="some response", prev_state=None)
        assert result is None

    @pytest.mark.asyncio
    @patch("app.memory.conv_memory.llm")
    async def test_returns_none_on_invalid_json(self, mock_llm):
        from app.memory.conv_memory import extract_state

        mock_llm.generate = AsyncMock(return_value="not json at all")

        result = await extract_state(response="some response", prev_state=None)
        assert result is None

    @pytest.mark.asyncio
    @patch("app.memory.conv_memory.llm")
    async def test_accepts_any_json_shape(self, mock_llm):
        from app.memory.conv_memory import extract_state

        # Procedure-style state
        state = {
            "chu_de": "thủ tục cấp sổ đỏ",
            "buoc_dang_ban": 3,
            "co_quan": "UBND",
        }
        mock_llm.generate = AsyncMock(return_value=json.dumps(state))

        result = await extract_state(response="...", prev_state=None)
        assert result["buoc_dang_ban"] == 3


class TestConvMemoryReadUpdate:

    @pytest.mark.asyncio
    @patch("app.memory.conv_memory.db")
    @patch("app.memory.conv_memory.llm")
    async def test_update_writes_extracted_state_to_db(self, mock_llm, mock_db):
        from app.memory.conv_memory import update

        mock_llm.generate = AsyncMock(return_value=json.dumps({"topic": "test"}))
        mock_db.get_conv_state.return_value = None
        mock_db.upsert_conv_state = MagicMock()

        await update(session_id="sess1", response="response text", turn=2)

        mock_db.upsert_conv_state.assert_called_once()
        args = mock_db.upsert_conv_state.call_args
        assert args[0][0] == "sess1"
        assert args[0][1]["topic"] == "test"

    @pytest.mark.asyncio
    @patch("app.memory.conv_memory.db")
    @patch("app.memory.conv_memory.llm")
    async def test_update_skips_db_write_when_extraction_fails(self, mock_llm, mock_db):
        from app.memory.conv_memory import update

        mock_llm.generate = AsyncMock(side_effect=Exception("fail"))
        mock_db.get_conv_state.return_value = None
        mock_db.upsert_conv_state = MagicMock()

        await update(session_id="sess1", response="response", turn=1)

        mock_db.upsert_conv_state.assert_not_called()

    @patch("app.memory.conv_memory.db")
    def test_read_returns_db_value(self, mock_db):
        from app.memory.conv_memory import read

        mock_db.get_conv_state.return_value = {"topic": "test"}
        result = read("sess1")
        assert result["topic"] == "test"

    @patch("app.memory.conv_memory.db")
    def test_read_returns_none_when_no_state(self, mock_db):
        from app.memory.conv_memory import read

        mock_db.get_conv_state.return_value = None
        assert read("sess1") is None
