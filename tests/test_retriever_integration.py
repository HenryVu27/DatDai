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
