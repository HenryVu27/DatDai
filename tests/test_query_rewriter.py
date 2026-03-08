"""Tests for query_rewriter module."""
import json
from unittest.mock import AsyncMock, MagicMock, patch, patch as sync_patch

import pytest
from app.rag.query_rewriter import (
    extract_filters_regex, VALID_DOC_IDS, validate_filters,
    rewrite_query, parse_rewrite_response, REWRITER_SYSTEM_PROMPT,
)


class TestExtractFiltersRegex:
    """Existing regex extraction, renamed."""

    def test_extracts_doc_id_ldd(self):
        result = extract_filters_regex("luat dat dai 2024 quy dinh gi")
        assert result["doc_ids"] == ["ldd2024"]

    def test_extracts_doc_id_nd(self):
        result = extract_filters_regex("ND 102 dieu 15")
        assert result["doc_ids"] == ["nd102"]

    def test_extracts_dieu(self):
        result = extract_filters_regex("dieu 79 luat dat dai")
        assert result["dieu"] == "Dieu 79"

    def test_no_match(self):
        result = extract_filters_regex("xin chao")
        assert result["doc_ids"] is None
        assert result["dieu"] is None

    def test_multiple_doc_ids(self):
        result = extract_filters_regex("ND 49 sua doi ND 102")
        assert "nd49" in result["doc_ids"]
        assert "nd102" in result["doc_ids"]


class TestValidateFilters:
    """Validate LLM-extracted filters against known values."""

    def test_valid_doc_ids_pass(self):
        result = validate_filters({"doc_ids": ["nd102", "ldd2024"], "dieu": "Dieu 15"})
        assert result["doc_ids"] == ["nd102", "ldd2024"]
        assert result["dieu"] == "Dieu 15"

    def test_invalid_doc_ids_stripped(self):
        result = validate_filters({"doc_ids": ["nd102", "nd999"], "dieu": None})
        assert result["doc_ids"] == ["nd102"]

    def test_all_invalid_doc_ids_becomes_none(self):
        result = validate_filters({"doc_ids": ["fake_doc"], "dieu": None})
        assert result["doc_ids"] is None

    def test_invalid_dieu_format_stripped(self):
        result = validate_filters({"doc_ids": None, "dieu": "article 15"})
        assert result["dieu"] is None

    def test_valid_dieu_formats(self):
        assert validate_filters({"doc_ids": None, "dieu": "Dieu 15"})["dieu"] == "Dieu 15"
        assert validate_filters({"doc_ids": None, "dieu": "Dieu 5a"})["dieu"] == "Dieu 5a"

    def test_null_inputs(self):
        result = validate_filters({"doc_ids": None, "dieu": None})
        assert result["doc_ids"] is None
        assert result["dieu"] is None

    def test_empty_doc_ids_becomes_none(self):
        result = validate_filters({"doc_ids": [], "dieu": None})
        assert result["doc_ids"] is None


class TestParseRewriteResponse:
    """Parse LLM JSON output into validated rewrite result."""

    def test_valid_json(self):
        raw = json.dumps({
            "standalone_query": "quyen su dung dat theo Luat Dat Dai 2024",
            "search_queries": ["quyen nguoi su dung dat", "quyen loi su dung dat"],
            "filters": {"doc_ids": ["ldd2024"], "dieu": None},
        })
        result = parse_rewrite_response(raw, "quyen su dung dat")
        assert result["standalone_query"] == "quyen su dung dat theo Luat Dat Dai 2024"
        assert len(result["search_queries"]) == 2
        assert result["filters"]["doc_ids"] == ["ldd2024"]

    def test_json_in_markdown_fences(self):
        raw = '```json\n{"standalone_query": "test", "search_queries": ["a"], "filters": {"doc_ids": null, "dieu": null}}\n```'
        result = parse_rewrite_response(raw, "test")
        assert result["standalone_query"] == "test"

    def test_invalid_json_falls_back(self):
        result = parse_rewrite_response("not json at all", "original query")
        assert result["standalone_query"] == "original query"
        assert result["search_queries"] == []
        assert result["filters"]["doc_ids"] is None

    def test_invalid_doc_ids_stripped(self):
        raw = json.dumps({
            "standalone_query": "test",
            "search_queries": [],
            "filters": {"doc_ids": ["nd999"], "dieu": None},
        })
        result = parse_rewrite_response(raw, "test")
        assert result["filters"]["doc_ids"] is None

    def test_missing_fields_get_defaults(self):
        raw = json.dumps({"standalone_query": "test"})
        result = parse_rewrite_response(raw, "test")
        assert result["search_queries"] == []
        assert result["filters"] == {"doc_ids": None, "dieu": None}

    def test_empty_standalone_falls_back(self):
        raw = json.dumps({"standalone_query": "", "search_queries": [], "filters": {}})
        result = parse_rewrite_response(raw, "original")
        assert result["standalone_query"] == "original"


class TestRewriteQuery:
    """Integration test for the full rewrite_query function."""

    @pytest.mark.asyncio
    @patch("app.rag.query_rewriter.llm")
    async def test_calls_llm_and_parses(self, mock_llm):
        mock_llm.generate = AsyncMock(return_value=json.dumps({
            "standalone_query": "giay chung nhan quyen su dung dat (so do) can nhung gi",
            "search_queries": ["thu tuc cap giay chung nhan quyen su dung dat", "ho so xin cap so do"],
            "filters": {"doc_ids": ["ldd2024"], "dieu": None},
        }))
        result = await rewrite_query("lam so do can gi", summary=None, recent_messages=[])
        assert "giay chung nhan" in result["standalone_query"]
        assert len(result["search_queries"]) == 2
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.rag.query_rewriter.llm")
    async def test_passes_history_to_llm(self, mock_llm):
        mock_llm.generate = AsyncMock(return_value=json.dumps({
            "standalone_query": "thoi han su dung dat o tai do thi theo Luat Dat Dai 2024",
            "search_queries": ["thoi han giao dat o", "thoi han su dung dat o do thi"],
            "filters": {"doc_ids": ["ldd2024"], "dieu": None},
        }))
        history = [
            {"role": "user", "content": "Dat o tai do thi co thoi han bao lau?"},
            {"role": "assistant", "content": "Theo Dieu 172 Luat Dat Dai 2024, dat o la loai dat su dung on dinh lau dai."},
        ]
        result = await rewrite_query("Con dat nong nghiep thi sao?", summary=None, recent_messages=history)
        call_kwargs = mock_llm.generate.call_args
        prompt_text = call_kwargs.kwargs.get("prompt") or call_kwargs.args[0]
        assert "do thi" in prompt_text or "nong nghiep" in prompt_text

    @pytest.mark.asyncio
    @patch("app.rag.query_rewriter.llm")
    async def test_falls_back_on_llm_error(self, mock_llm):
        mock_llm.generate = AsyncMock(side_effect=Exception("API error"))
        result = await rewrite_query("ND 102 dieu 15", summary=None, recent_messages=[])
        assert result["standalone_query"] == "ND 102 dieu 15"
        assert result["filters"]["doc_ids"] == ["nd102"]
        assert result["filters"]["dieu"] == "Dieu 15"

    @pytest.mark.asyncio
    @patch("app.rag.query_rewriter.llm")
    async def test_falls_back_on_timeout(self, mock_llm):
        import asyncio
        mock_llm.generate = AsyncMock(side_effect=asyncio.TimeoutError())
        result = await rewrite_query("so do la gi", summary=None, recent_messages=[])
        assert result["standalone_query"] == "so do la gi"


class TestRewriterPrompt:
    """Verify the system prompt contains required instructions."""

    def test_has_known_doc_ids(self):
        assert "ldd2024" in REWRITER_SYSTEM_PROMPT
        assert "nd102" in REWRITER_SYSTEM_PROMPT

    def test_has_json_output_instruction(self):
        assert "JSON" in REWRITER_SYSTEM_PROMPT

    def test_has_pronoun_resolution_instruction(self):
        assert "dai tu" in REWRITER_SYSTEM_PROMPT.lower() or "pronoun" in REWRITER_SYSTEM_PROMPT.lower()


class TestMultiQuerySearch:
    """Test that search_legal_docs handles multiple queries."""

    @pytest.mark.asyncio
    @patch("app.rag.retriever.llm")
    @patch("app.rag.retriever._get_store")
    @patch("app.rag.retriever._get_reranker")
    async def test_multi_query_merges_results(self, mock_reranker, mock_store_fn, mock_llm):
        mock_reranker.return_value = None  # no reranker

        store = MagicMock()
        mock_store_fn.return_value = store

        # Each query returns different chunks
        store.search_hybrid.side_effect = [
            [{"chunk_id": "c1", "text": "chunk 1", "score": 0.9}],
            [{"chunk_id": "c2", "text": "chunk 2", "score": 0.8}],
            [{"chunk_id": "c1", "text": "chunk 1", "score": 0.85}],  # duplicate
        ]

        # Mock fetch_full_dieu for _expand_full_dieu
        store.fetch_full_dieu.return_value = []

        mock_llm.embed = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768, [0.3] * 768])

        from app.rag.retriever import search_legal_docs
        results = await search_legal_docs(
            query="main query",
            search_queries=["variant 1", "variant 2"],
        )

        # Should have 2 unique chunks (c1 deduplicated)
        chunk_ids = [c["chunk_id"] for c in results]
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids
        assert len([c for c in chunk_ids if c == "c1"]) == 1

        # Should have called embed once with all 3 queries
        assert mock_llm.embed.call_count == 1
