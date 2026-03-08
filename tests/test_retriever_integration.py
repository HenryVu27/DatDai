"""Integration tests for the RAG retriever pipeline.

Requires GEMINI_API_KEY and QDRANT_URL in .env.
Skipped automatically if not available.
"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

HAS_API_KEY = bool(os.getenv("GEMINI_API_KEY"))
HAS_QDRANT = bool(os.getenv("QDRANT_URL")) and os.getenv("QDRANT_URL") != ":memory:"


@pytest.mark.skipif(
    not HAS_API_KEY or not HAS_QDRANT,
    reason="Needs GEMINI_API_KEY and QDRANT_URL",
)
class TestRetrieverIntegration:

    def test_basic_retrieval(self):
        from app.rag import search

        result = asyncio.get_event_loop().run_until_complete(
            search("quyền sử dụng đất là gì")
        )
        assert len(result["chunks"]) > 0
        assert any(
            "đất" in c.get("text", c.get("content", "")).lower()
            for c in result["chunks"]
        )

    def test_filter_extraction(self):
        from app.rag.query_rewriter import extract_filters

        filters = extract_filters("Điều 5 Nghị định 102 quy định gì")
        assert filters["doc_ids"] == ["nd102"]
        assert filters["dieu"] == "Điều 5"

    def test_filter_extraction_luat_dat_dai(self):
        from app.rag.query_rewriter import extract_filters

        filters = extract_filters("Luật đất đai 2024 quy định thế nào về thu hồi đất")
        assert filters["doc_ids"] == ["ldd2024"]

    def test_filter_extraction_no_match(self):
        from app.rag.query_rewriter import extract_filters

        filters = extract_filters("bồi thường khi thu hồi đất")
        assert filters["doc_ids"] is None
        assert filters["dieu"] is None

    def test_build_context(self):
        from app.rag.retriever import build_context

        chunks = [
            {"metadata_str": "Doc 1 | Dieu 1", "text": "Content 1"},
            {"metadata_str": "Doc 2 | Dieu 2", "text": "Content 2"},
        ]
        context = build_context(chunks)
        assert "[Nguon 1: Doc 1 | Dieu 1]" in context
        assert "[Nguon 2: Doc 2 | Dieu 2]" in context
        assert "Content 1" in context
        assert "---" in context


class TestQueryRewriterUnit:
    """Unit tests that don't need API keys."""

    def test_extract_filters_multiple_patterns(self):
        from app.rag.query_rewriter import extract_filters

        # Vietnamese with diacritics
        filters = extract_filters("Điều 10 nghị định 88 về bồi thường")
        assert "nd88" in filters["doc_ids"]
        assert filters["dieu"] == "Điều 10"

    def test_extract_filters_ascii(self):
        from app.rag.query_rewriter import extract_filters

        # ASCII without diacritics
        filters = extract_filters("dieu 15 nghi dinh 103")
        assert "nd103" in filters["doc_ids"]
        assert filters["dieu"] == "Điều 15"

    def test_extract_filters_nghi_quyet(self):
        from app.rag.query_rewriter import extract_filters

        filters = extract_filters("Nghị quyết 254 nói gì")
        assert "nq254" in filters["doc_ids"]


class TestRerankerFallback:
    """C1: Reranker threshold eliminating all results must fall back to vector score."""

    @pytest.mark.asyncio
    @pytest.mark.anyio
    @pytest.mark.asyncio
    async def test_falls_back_when_all_below_threshold(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        with patch("app.rag.retriever.llm") as mock_llm, \
             patch("app.rag.retriever._get_store") as mock_store_fn, \
             patch("app.rag.retriever._get_reranker") as mock_reranker_fn:
            from app.rag.retriever import search_legal_docs

            reranker = MagicMock()
            # All candidates score below 0.3 threshold
            reranker.rerank = AsyncMock(return_value=[
                {"chunk_id": "c1", "text": "text 1", "score": 0.1},
                {"chunk_id": "c2", "text": "text 2", "score": 0.2},
            ])
            mock_reranker_fn.return_value = reranker

            store = MagicMock()
            store.search_hybrid.return_value = [
                {"chunk_id": "c1", "text": "text 1", "score": 0.9},
                {"chunk_id": "c2", "text": "text 2", "score": 0.8},
            ]
            store.fetch_full_dieu.return_value = []
            mock_store_fn.return_value = store

            mock_llm.embed = AsyncMock(return_value=[[0.1] * 768])

            results = await search_legal_docs("some query")

            # Must return results even though all were below threshold
            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_normal_results_above_threshold_returned(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        with patch("app.rag.retriever.llm") as mock_llm, \
             patch("app.rag.retriever._get_store") as mock_store_fn, \
             patch("app.rag.retriever._get_reranker") as mock_reranker_fn:
            from app.rag.retriever import search_legal_docs

            reranker = MagicMock()
            reranker.rerank = AsyncMock(return_value=[
                {"chunk_id": "c1", "text": "text 1", "score": 0.9},
            ])
            mock_reranker_fn.return_value = reranker

            store = MagicMock()
            store.search_hybrid.return_value = [{"chunk_id": "c1", "text": "text 1", "score": 0.9}]
            store.fetch_full_dieu.return_value = []
            mock_store_fn.return_value = store

            mock_llm.embed = AsyncMock(return_value=[[0.1] * 768])

            results = await search_legal_docs("some query")
            assert results[0]["score"] == 0.9


class TestFilterRetryLogging:
    """C4: Filter-retry without filters must log a warning."""

    @pytest.mark.asyncio
    async def test_logs_warning_on_filter_retry(self, caplog):
        import logging
        from unittest.mock import AsyncMock, MagicMock, patch
        with patch("app.rag.retriever.llm") as mock_llm, \
             patch("app.rag.retriever._get_store") as mock_store_fn, \
             patch("app.rag.retriever._get_reranker") as mock_reranker_fn:
            from app.rag.retriever import search_legal_docs

            mock_reranker_fn.return_value = None

            store = MagicMock()
            # First call (with filter) returns empty; second call (no filter) returns results
            store.search_hybrid.side_effect = [
                [],
                [{"chunk_id": "c1", "text": "text", "score": 0.9}],
            ]
            store.fetch_full_dieu.return_value = []
            mock_store_fn.return_value = store

            mock_llm.embed = AsyncMock(return_value=[[0.1] * 768])

            with caplog.at_level(logging.WARNING, logger="app.rag.retriever"):
                await search_legal_docs("query", doc_ids=["nd102"])

            assert any("filter" in msg.lower() or "retry" in msg.lower() for msg in caplog.messages)


class TestDieuExpansionCap:
    """M4: Dieu expansion must not exceed MAX_EXPANSION_CHUNKS total."""

    def test_expansion_capped(self):
        from unittest.mock import MagicMock, patch
        with patch("app.rag.retriever._get_store") as mock_store_fn:
            from app.rag.retriever import _expand_full_dieu

            store = MagicMock()
            # Each Dieu has 20 sibling chunks
            store.fetch_full_dieu.return_value = [
                {"chunk_id": f"sib_{i}", "text": f"sibling {i}"} for i in range(20)
            ]
            mock_store_fn.return_value = store

            base_chunks = [
                {"chunk_id": "c1", "doc_id": "ldd2024", "dieu": "Dieu 79", "score": 0.9},
                {"chunk_id": "c2", "doc_id": "ldd2024", "dieu": "Dieu 80", "score": 0.8},
                {"chunk_id": "c3", "doc_id": "ldd2024", "dieu": "Dieu 81", "score": 0.7},
            ]

            result = _expand_full_dieu(base_chunks, store)

            # Must not exceed base + MAX_EXPANSION_CHUNKS
            from app.config import RAG_MAX_EXPANSION_CHUNKS
            assert len(result) <= len(base_chunks) + RAG_MAX_EXPANSION_CHUNKS
