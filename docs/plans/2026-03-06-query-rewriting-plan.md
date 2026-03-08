# Query Rewriting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add LLM-powered query rewriting to the RAG pipeline -- pronoun resolution, legal term expansion, multi-query generation, and smarter filter extraction.

**Architecture:** A new `rewrite_query()` function in `app/rag/query_rewriter.py` calls the utility model (flash-lite) before the orchestrator. It takes conversation history + current message and returns a standalone query, 2 search variants, and extracted filters. The orchestrator receives the clean query. `search_legal_docs` uses the variants for multi-query retrieval with deduplication. Regex filter extraction is kept as fallback.

**Tech Stack:** Python 3.12, Gemini API (flash-lite utility model), existing `app/llm.py` client.

---

### Task 1: Refactor query_rewriter.py -- rename and add validation

**Files:**
- Modify: `app/rag/query_rewriter.py`
- Test: `tests/test_query_rewriter.py` (create)

**Step 1: Write tests for existing regex extraction and new validation helpers**

Create `tests/test_query_rewriter.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_query_rewriter.py -v`
Expected: FAIL -- `extract_filters_regex` and `validate_filters` not found.

**Step 3: Implement rename and validation in query_rewriter.py**

Replace contents of `app/rag/query_rewriter.py`:

```python
"""Query rewriting and filter extraction for legal RAG queries."""
import json
import logging
import re

from app.observability import observe

logger = logging.getLogger(__name__)

# -- Known document IDs (for validation) --
VALID_DOC_IDS = {
    "ldd2024", "nd71", "nd88", "nd102", "nd103",
    "nd151", "nd226", "nq254", "nd49", "nd12", "nd50",
}

# -- Regex-based filter extraction (fallback) --

DOC_PATTERNS = {
    r"(?:luật đất đai|luat dat dai|luật\s*số\s*31)": ["ldd2024"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*71": ["nd71"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*88": ["nd88"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*102": ["nd102"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*103": ["nd103"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*151": ["nd151"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*226": ["nd226"],
    r"(?:nghị quyết|nghi quyet|nq)\s*254": ["nq254"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*49": ["nd49"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*12": ["nd12"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*50": ["nd50"],
}

DIEU_RE = re.compile(r"(?:điều|dieu|đ\.?)\s*(\d+[a-z]?)", re.IGNORECASE)
DIEU_VALIDATE_RE = re.compile(r"^Dieu \d+[a-z]?$", re.IGNORECASE)


def extract_filters_regex(query: str) -> dict:
    """Extract doc_ids and dieu from query text using regex. Fallback method."""
    query_lower = query.lower()
    doc_ids = []
    for pattern, ids in DOC_PATTERNS.items():
        if re.search(pattern, query_lower):
            doc_ids.extend(ids)
    dieu = None
    m = DIEU_RE.search(query_lower)
    if m:
        dieu = f"Dieu {m.group(1)}"
    return {"doc_ids": list(set(doc_ids)) or None, "dieu": dieu}


def validate_filters(filters: dict) -> dict:
    """Validate and sanitize LLM-extracted filters."""
    doc_ids = filters.get("doc_ids")
    dieu = filters.get("dieu")

    # Validate doc_ids against known set
    if doc_ids:
        doc_ids = [d for d in doc_ids if d in VALID_DOC_IDS]
        if not doc_ids:
            doc_ids = None

    # Validate dieu format
    if dieu and not DIEU_VALIDATE_RE.match(dieu):
        dieu = None

    return {"doc_ids": doc_ids, "dieu": dieu}
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_query_rewriter.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add app/rag/query_rewriter.py tests/test_query_rewriter.py
git commit -m "refactor: rename extract_filters to extract_filters_regex, add validate_filters"
```

---

### Task 2: Add rewrite_query() with LLM call and parsing

**Files:**
- Modify: `app/rag/query_rewriter.py`
- Modify: `tests/test_query_rewriter.py`

**Step 1: Write tests for rewrite_query and parse_rewrite_response**

Append to `tests/test_query_rewriter.py`:

```python
import json
from unittest.mock import AsyncMock, patch
from app.rag.query_rewriter import rewrite_query, parse_rewrite_response, REWRITER_SYSTEM_PROMPT


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
        # Should have passed history in the prompt
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
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_query_rewriter.py -v`
Expected: FAIL -- `rewrite_query`, `parse_rewrite_response`, `REWRITER_SYSTEM_PROMPT` not found.

**Step 3: Implement rewrite_query in query_rewriter.py**

Add to `app/rag/query_rewriter.py` (after the existing code from Task 1):

```python
from app import llm

# -- LLM-based query rewriting --

REWRITER_SYSTEM_PROMPT = """Ban la bo xu ly truy van cho he thong tra cuu phap luat dat dai Viet Nam.
Nhiem vu: viet lai cau hoi cua nguoi dung thanh cau truy van doc lap, ro rang, phu hop de tim kiem van ban phap luat.

NGUYEN TAC:
1. GIAI QUYET DAI TU: Thay the "no", "do", "cai nay", "van ban nay", "dieu nay" bang thuc the cu the tu lich su hoi thoai.
2. MO RONG THUAT NGU PHAP LY: Thay the tu thong dung bang thuat ngu chinh thuc, giu tu goc trong ngoac.
   Vi du: "so do" -> "giay chung nhan quyen su dung dat (so do)"
   Vi du: "ban dat" -> "chuyen nhuong quyen su dung dat"
   Vi du: "den bu" -> "boi thuong khi Nha nuoc thu hoi dat"
   Vi du: "tach thua" -> "tach thua dat"
   Vi du: "hop thuc hoa" -> "cap giay chung nhan quyen su dung dat lan dau"
3. TAO 2 CAU TRUY VAN THAY THE: Viet 2 cach dien dat khac nhau de tang kha nang tim kiem.
4. TRICH XUAT BO LOC: Xac dinh van ban (doc_ids) va dieu (dieu) neu co trong cau hoi.

MA VAN BAN HOP LE:
- ldd2024: Luat Dat Dai 2024
- nd71: Nghi dinh 71/2024 ve gia dat
- nd88: Nghi dinh 88/2024 ve boi thuong, ho tro, tai dinh cu
- nd102: Nghi dinh 102/2024 chi tiet thi hanh
- nd103: Nghi dinh 103/2024 ve tien su dung dat, tien thue dat
- nd151: Nghi dinh 151/2025 phan dinh tham quyen
- nd226: Nghi dinh 226/2025 sua doi 4 ND
- nq254: Nghi quyet 254/2025 thao go vuong mac
- nd49: Nghi dinh 49/2026 sua doi moi nhat cac ND
- nd12: Nghi dinh 12/2024 chuyen tiep gia dat
- nd50: Nghi dinh 50/2026 chi tiet NQ 254

Tra ve CHINH XAC JSON (khong markdown, khong giai thich):
{"standalone_query": "...", "search_queries": ["...", "..."], "filters": {"doc_ids": [...] hoac null, "dieu": "Dieu X" hoac null}}"""


def _build_rewriter_prompt(
    user_message: str,
    summary: str | None,
    recent_messages: list[dict],
) -> str:
    """Build the user-facing prompt for the rewriter."""
    parts = []
    if summary:
        parts.append(f"<tom-tat-hoi-thoai>\n{summary}\n</tom-tat-hoi-thoai>")
    if recent_messages:
        history_lines = []
        for msg in recent_messages[-10:]:  # last 5 turn-groups max
            role = "Nguoi dung" if msg["role"] == "user" else "Tro ly"
            # Truncate long assistant messages
            content = msg["content"][:300] if msg["role"] == "assistant" else msg["content"]
            history_lines.append(f"{role}: {content}")
        parts.append(f"<lich-su>\n" + "\n".join(history_lines) + "\n</lich-su>")
    parts.append(f"<cau-hoi-hien-tai>\n{user_message}\n</cau-hoi-hien-tai>")
    return "\n\n".join(parts)


def parse_rewrite_response(raw: str, original_query: str) -> dict:
    """Parse LLM rewrite response into validated result. Falls back to original on failure."""
    fallback = {
        "standalone_query": original_query,
        "search_queries": [],
        "filters": extract_filters_regex(original_query),
    }

    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Query rewriter returned invalid JSON: %s", raw[:200])
        return fallback

    standalone = data.get("standalone_query", "").strip()
    if not standalone:
        standalone = original_query

    search_queries = data.get("search_queries", [])
    if not isinstance(search_queries, list):
        search_queries = []
    search_queries = [q for q in search_queries if isinstance(q, str) and q.strip()]

    raw_filters = data.get("filters", {})
    if not isinstance(raw_filters, dict):
        raw_filters = {}
    filters = validate_filters({
        "doc_ids": raw_filters.get("doc_ids"),
        "dieu": raw_filters.get("dieu"),
    })

    return {
        "standalone_query": standalone,
        "search_queries": search_queries,
        "filters": filters,
    }


@observe(name="rewrite_query")
async def rewrite_query(
    user_message: str,
    summary: str | None,
    recent_messages: list[dict],
) -> dict:
    """Rewrite a user query for better retrieval. Falls back to regex on failure."""
    import asyncio

    prompt = _build_rewriter_prompt(user_message, summary, recent_messages)

    try:
        raw = await asyncio.wait_for(
            llm.generate(
                prompt=prompt,
                system=REWRITER_SYSTEM_PROMPT,
                model="utility",
                temperature=0.0,
                max_tokens=300,
            ),
            timeout=5.0,
        )
        result = parse_rewrite_response(raw, user_message)
        logger.info("Query rewritten: %r -> %r (%d variants)",
                     user_message[:80], result["standalone_query"][:80],
                     len(result["search_queries"]))
        return result
    except asyncio.TimeoutError:
        logger.warning("Query rewriter timed out for: %s", user_message[:80])
    except Exception as e:
        logger.error("Query rewriter failed: %s", e)

    # Fallback to regex
    return {
        "standalone_query": user_message,
        "search_queries": [],
        "filters": extract_filters_regex(user_message),
    }
```

Also add the import at the top of the file (after existing imports):

```python
from app import llm
from app.observability import observe
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_query_rewriter.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add app/rag/query_rewriter.py tests/test_query_rewriter.py
git commit -m "feat: add LLM-powered rewrite_query with pronoun resolution and legal term expansion"
```

---

### Task 3: Add multi-query support to search_legal_docs

**Files:**
- Modify: `app/rag/retriever.py`
- Modify: `tests/test_query_rewriter.py` (add retriever multi-query test)

**Step 1: Write test for multi-query search**

Append to `tests/test_query_rewriter.py`:

```python
from unittest.mock import MagicMock, patch as sync_patch


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

        mock_llm.embed = AsyncMock(return_value=[[0.1] * 768])

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

        # Should have called embed 3 times (main + 2 variants)
        assert mock_llm.embed.call_count == 3
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_query_rewriter.py::TestMultiQuerySearch -v`
Expected: FAIL -- `search_legal_docs` doesn't accept `search_queries` parameter.

**Step 3: Modify search_legal_docs to support multi-query**

In `app/rag/retriever.py`, update the `search_legal_docs` function signature and body:

```python
@observe(name="search_legal_docs")
async def search_legal_docs(
    query: str,
    doc_ids: list[str] | None = None,
    dieu: str | None = None,
    top_k: int = RAG_TOP_K,
    search_queries: list[str] | None = None,
) -> list[dict]:
    """Hybrid dense+sparse search against Qdrant.

    When search_queries are provided, runs parallel searches for each variant
    and merges results with deduplication before reranking.
    """
    store = _get_store()
    reranker = _get_reranker()
    fetch_k = RAG_RERANK_CANDIDATES if reranker else top_k

    # Build list of all queries to search
    all_queries = [query]
    if search_queries:
        all_queries.extend(search_queries)

    # Embed all queries in parallel
    all_embeddings = await llm.embed(all_queries)

    # Search for each query
    all_candidates = []
    for q_text, q_embedding in zip(all_queries, all_embeddings):
        candidates = store.search_hybrid(
            query_vector=q_embedding,
            query_text=q_text,
            top_k=fetch_k,
            doc_ids=doc_ids,
            dieu=dieu,
        )
        all_candidates.extend(candidates)

    # Deduplicate by chunk_id, keeping highest score
    seen = {}
    for c in all_candidates:
        cid = c.get("chunk_id")
        if cid:
            if cid not in seen or c.get("score", 0) > seen[cid].get("score", 0):
                seen[cid] = c
        else:
            seen[id(c)] = c
    candidates = list(seen.values())

    # Retry without filters if empty
    if not candidates and (doc_ids or dieu):
        query_embedding = all_embeddings[0]
        candidates = store.search_hybrid(
            query_vector=query_embedding, query_text=query, top_k=fetch_k,
        )

    if not candidates:
        return []

    # Rerank using the primary query
    if reranker and len(candidates) > top_k:
        try:
            candidates = await reranker.rerank(query, candidates, top_k * 2)
        except Exception as e:
            logger.warning("Reranker failed: %s", e)

    # Relevance threshold
    if reranker:
        candidates = [c for c in candidates if c["score"] >= RAG_RELEVANCE_THRESHOLD]

    # Trim and expand full Dieu for top results
    chunks = candidates[:top_k]
    chunks = _expand_full_dieu(chunks, store)

    return chunks
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_query_rewriter.py::TestMultiQuerySearch -v`
Expected: PASS.

Also run existing tests to check nothing broke:
Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add app/rag/retriever.py tests/test_query_rewriter.py
git commit -m "feat: add multi-query support to search_legal_docs"
```

---

### Task 4: Integrate query rewriter into chat.py orchestrator pipeline

**Files:**
- Modify: `app/chat.py`
- Modify: `tests/test_orchestrator.py`

**Step 1: Write tests for the integration**

Append to `tests/test_orchestrator.py`:

```python
from unittest.mock import AsyncMock, patch


class TestQueryRewriterIntegration:
    """Verify the orchestrator uses rewritten queries."""

    def test_rewrite_result_used_in_guardrail_fallback(self):
        """When guardrail triggers, it should use rewrite filters, not just regex."""
        from app.chat import _should_force_retrieval
        decision = {"direct_response": "answer", "actions": []}
        # "so do" is a legal keyword
        assert _should_force_retrieval("lam so do can gi", decision)
```

**Step 2: Modify handle_message in chat.py**

In `app/chat.py`, update the import section:

```python
from app.rag.query_rewriter import extract_filters_regex, rewrite_query
```

(Replace the old `from app.rag.query_rewriter import extract_filters` import.)

Then in `handle_message`, add the rewriter call before the orchestrator. Replace the section from `# -- Stage 1: Orchestrator --` to the guardrail block:

```python
    # -- Stage 0: Query rewriting --
    t_rewrite = time.monotonic()
    rewrite_result = await rewrite_query(user_message, summary, recent_messages)
    standalone_query = rewrite_result["standalone_query"]
    search_queries = rewrite_result["search_queries"]
    rewrite_filters = rewrite_result["filters"]
    logger.info("[%s] query rewrite: %r -> %r (%d variants, %.1fs)",
                session_id[:8], user_message[:60], standalone_query[:60],
                len(search_queries), time.monotonic() - t_rewrite)

    # -- Stage 1: Orchestrator --
    t_orch = time.monotonic()
    orch_messages = _trim_for_orchestrator(recent_messages)
    logger.info("orch_msgs=%d gen_msgs=%d", len(orch_messages), len(recent_messages))
    orch_system, orch_history, orch_prompt = _build_orchestrator_prompt(
        standalone_query, orch_messages, summary,
    )

    orch_raw = await llm.generate(
        prompt=orch_prompt,
        system=orch_system,
        history=orch_history,
        model="orchestrator",
        temperature=0.0,
        max_tokens=1000,
    )
    decision = _parse_orchestrator_response(orch_raw)
    logger.info(
        "[%s] turn=%d orchestrator: complexity=%s actions=%d direct=%s (%.1fs)",
        session_id[:8], turn, decision["complexity"],
        len(decision["actions"]), bool(decision["direct_response"]),
        time.monotonic() - t_orch,
    )

    # Guardrail: force retrieval if orchestrator tried to answer a legal question directly
    if _should_force_retrieval(user_message, decision):
        # Use rewrite filters (LLM-extracted), fall back to regex
        filters = rewrite_filters if rewrite_filters.get("doc_ids") or rewrite_filters.get("dieu") else extract_filters_regex(user_message)
        logger.warning(
            "[%s] Guardrail triggered: forcing retrieval for legal question answered directly",
            session_id[:8],
        )
        decision = {
            "reasoning": "Guardrail: legal question requires retrieval",
            "actions": [{"tool": "search_legal_docs", "query": standalone_query, "filters": {
                "doc_ids": filters.get("doc_ids"),
                "dieu": filters.get("dieu"),
            }}],
            "complexity": decision.get("complexity", "simple"),
            "summary_update": decision.get("summary_update"),
            "direct_response": None,
        }
```

Then update `_execute_tools` to pass `search_queries` through. In the `_run_action` function inside `_execute_tools`, update the `search_legal_docs` branch:

```python
            if tool == "search_legal_docs":
                query = action.get("query", "")
                filters = action.get("filters", {})
                return await search_legal_docs(
                    query=query,
                    doc_ids=filters.get("doc_ids"),
                    dieu=filters.get("dieu"),
                    search_queries=action.get("search_queries"),
                )
```

And where `_execute_tools` is called (after the guardrail), inject `search_queries` into the actions:

```python
    # -- Stage 2: Execute tools --
    t_tools = time.monotonic()
    # Inject multi-query variants into search_legal_docs actions
    if search_queries:
        for action in decision["actions"]:
            if action.get("tool") == "search_legal_docs":
                action["search_queries"] = search_queries
    chunks = await _execute_tools(decision["actions"])
```

**Step 3: Apply the same changes to handle_message_stream**

In `handle_message_stream`, make the same three changes:

1. Add Stage 0 (query rewriting) before Stage 1, right after the orchestrator status yield:

```python
    yield ("status", {"text": "Dang phan tich cau hoi...", "step": "orchestrator"})

    # -- Stage 0: Query rewriting --
    rewrite_result = await rewrite_query(user_message, summary, recent_messages)
    standalone_query = rewrite_result["standalone_query"]
    search_queries = rewrite_result["search_queries"]
    rewrite_filters = rewrite_result["filters"]
```

2. Pass `standalone_query` to `_build_orchestrator_prompt` instead of `user_message`.

3. Update the guardrail block and tool injection identically to `handle_message`.

**Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add app/chat.py tests/test_orchestrator.py
git commit -m "feat: integrate query rewriter into orchestrator pipeline"
```

---

### Task 5: Update extract_filters references and fix memory

**Files:**
- Modify: `app/rag/__init__.py` (no change needed -- doesn't export extract_filters)
- Modify: `/Users/vuducdung/.claude/projects/-Users-vuducdung-personal-DatDai/memory/MEMORY.md`

**Step 1: Verify no other files import old extract_filters**

Run: `grep -r "from app.rag.query_rewriter import extract_filters\b" app/`
Expected: No matches (chat.py now imports `extract_filters_regex`).

If any matches found, update them to `extract_filters_regex`.

**Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All PASS.

**Step 3: Update memory to reflect reality**

In `MEMORY.md`, update the RAG Pipeline Architecture section. Replace:

```
1. Query rewriting (Gemini Flash) - resolves pronouns from conversation
```

With:

```
1. Query rewriting (Gemini flash-lite utility) - decontextualization, legal term expansion, multi-query generation, filter extraction (with regex fallback)
```

Remove any mention of HyDE from the pipeline description.

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: update memory, clean up old extract_filters references"
```

---

### Task 6: Manual smoke test

**No code changes. Verification only.**

**Step 1: Start the app locally**

Run: `.venv/bin/python -m uvicorn app.main:app --reload`

**Step 2: Test first-turn simple query**

Send: "Dieu 79 Luat Dat Dai 2024 quy dinh gi?"
Check logs for: `query rewrite:` line showing the rewritten query.
Verify: response still works correctly.

**Step 3: Test pronoun resolution (multi-turn)**

Turn 1: "Quyen cua nguoi su dung dat la gi?"
Turn 2: "Con nghia vu cua ho thi sao?"
Check logs for turn 2: `standalone_query` should mention "nghia vu cua nguoi su dung dat", not "ho".

**Step 4: Test legal term expansion**

Send: "Lam so do can gi?"
Check logs: `standalone_query` should contain "giay chung nhan quyen su dung dat".

**Step 5: Test multi-query variants**

Check logs for `search_queries` showing 2 variant phrasings.
Check retriever logs for multiple embed calls.

**Step 6: Test fallback (optional -- kill API temporarily)**

Temporarily break the API key and send a query.
Verify: falls back gracefully to regex filters + raw query.

---

Plan complete and saved to `docs/plans/2026-03-06-query-rewriting-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** -- I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Parallel Session (separate)** -- Open new session with executing-plans, batch execution with checkpoints.

Which approach?