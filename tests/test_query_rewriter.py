"""Tests for query_rewriter module."""
import pytest
from app.rag.query_rewriter import extract_filters_regex, VALID_DOC_IDS, validate_filters


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
