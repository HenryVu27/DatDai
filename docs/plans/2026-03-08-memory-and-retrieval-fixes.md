# Memory and Retrieval Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 13 identified loopholes in the conversational memory and retrieval pipeline, validated against SOTA research (MemGPT, Mem0, SeCom ICLR 2025, LangMem).

**Architecture:** Seven sequential tasks covering: (1) rewriter context window fix, (2) retrieval failure fallbacks, (3) background task reliability, (4–7) a new schema-free conversation state layer. Each task is independently testable and committable.

**Tech Stack:** Python 3.12, pytest, asyncio, SQLite/PostgreSQL (psycopg2), Gemini via `app/llm`.

**Test runner:** `.venv/bin/python -m pytest`

---

## Bug Reference

| ID | Severity | Short description |
|----|----------|-------------------|
| C1 | Critical | Reranker threshold can eliminate ALL results — no fallback |
| C2 | Critical | Rewriter truncates last assistant message to 300 chars — ordinal refs fail |
| C3 | Critical | No structured state for resolving ordinal/pronoun follow-ups |
| C4 | Critical | Filter-retry silently returns wrong document when filter produces 0 results |
| H1 | High     | Fire-and-forget summary/title tasks — silent data loss on crash |
| H3 | High     | State/summary extraction only from turn 5+ — early turns have no memory |
| M2 | Medium   | `max_tokens=300` for rewriter output — JSON truncated mid-query |
| M3 | Medium   | Reverse char trim can silently exceed 30k budget |
| M4 | Medium   | Dieu expansion unbounded — context string can balloon |

---

## Task 1: Fix Rewriter Context Window

**Fixes:** C2, M2

**Files:**
- Modify: `app/rag/query_rewriter.py:108-125` (`_build_rewriter_prompt`)
- Modify: `app/rag/query_rewriter.py:196` (`max_tokens=300`)
- Modify: `app/rag/query_rewriter.py:75-105` (add ordinal resolution rule to system prompt)
- Test: `tests/test_query_rewriter.py`

**Background:** The rewriter truncates all assistant messages to 300 chars (`query_rewriter.py:121`). When a user asks "explain case 3", the list containing case 3 is in the prior assistant response and gets cut off. The rewriter then can only guess, producing a vague query that returns 0 vector search results. Fix: keep the most recent assistant message up to 1500 chars; keep older ones at 400 chars. Also raise `max_tokens` from 300 to 500 so JSON output isn't cut mid-query.

### Step 1: Write the failing tests

Add to `tests/test_query_rewriter.py`:

```python
class TestRewriterContextWindow:
    """Verify recency-weighted assistant message truncation."""

    def test_last_assistant_message_gets_1500_chars(self):
        from app.rag.query_rewriter import _build_rewriter_prompt
        long_response = "X" * 2000
        messages = [
            {"role": "user", "content": "question 1"},
            {"role": "assistant", "content": long_response},
        ]
        prompt = _build_rewriter_prompt("follow up", summary=None, recent_messages=messages)
        # Last assistant message should appear with up to 1500 chars
        assert "X" * 1500 in prompt
        assert "X" * 1501 not in prompt

    def test_older_assistant_message_gets_400_chars(self):
        from app.rag.query_rewriter import _build_rewriter_prompt
        old_response = "Y" * 800
        recent_response = "Z" * 100
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": old_response},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": recent_response},
        ]
        prompt = _build_rewriter_prompt("follow up", summary=None, recent_messages=messages)
        # Older assistant message truncated to 400
        assert "Y" * 400 in prompt
        assert "Y" * 401 not in prompt
        # Most recent assistant message kept in full (100 chars < 1500)
        assert "Z" * 100 in prompt

    def test_user_messages_not_truncated(self):
        from app.rag.query_rewriter import _build_rewriter_prompt
        long_user = "A" * 1000
        messages = [{"role": "user", "content": long_user}]
        prompt = _build_rewriter_prompt("follow up", summary=None, recent_messages=messages)
        assert "A" * 1000 in prompt

    def test_rewriter_system_prompt_has_ordinal_resolution_rule(self):
        from app.rag.query_rewriter import REWRITER_SYSTEM_PROMPT
        # Must instruct model to resolve ordinal references from history
        lowered = REWRITER_SYSTEM_PROMPT.lower()
        assert any(kw in lowered for kw in ["so thu tu", "ordinal", "truong hop", "muc"])
```

### Step 2: Run to verify they fail

```bash
.venv/bin/python -m pytest tests/test_query_rewriter.py::TestRewriterContextWindow -v
```

Expected: 4 FAILures (truncation logic not yet updated).

### Step 3: Implement the fix

In `app/rag/query_rewriter.py`, find `_build_rewriter_prompt` and replace the assistant truncation line:

**Old** (`query_rewriter.py:119-122`):
```python
    if recent_messages:
        history_lines = []
        for msg in recent_messages[-10:]:  # last 5 turn-groups max
            role = "Nguoi dung" if msg["role"] == "user" else "Tro ly"
            content = msg["content"][:300] if msg["role"] == "assistant" else msg["content"]
            history_lines.append(f"{role}: {content}")
```

**New:**
```python
    if recent_messages:
        history_lines = []
        windowed = recent_messages[-10:]
        # Find index of the last assistant message for recency-weighted truncation
        last_asst_idx = max(
            (i for i, m in enumerate(windowed) if m["role"] == "assistant"),
            default=-1,
        )
        for i, msg in enumerate(windowed):
            role = "Nguoi dung" if msg["role"] == "user" else "Tro ly"
            if msg["role"] == "assistant":
                limit = 1500 if i == last_asst_idx else 400
                content = msg["content"][:limit]
            else:
                content = msg["content"]
            history_lines.append(f"{role}: {content}")
```

Also change `max_tokens=300` to `max_tokens=500` at `query_rewriter.py:197`.

Add an ordinal resolution rule to `REWRITER_SYSTEM_PROMPT`. After the existing rule 2 (MO RONG THUAT NGU PHAP LY), add:

```
3. GIAI QUYET SO THU TU: Neu nguoi dung nhac den "truong hop 1", "muc 3", "buoc 2",
   "dieu do", hay so thu tu bat ky, hay tim chinh xac noi dung cua muc do trong
   <lich-su> va dua noi dung thuc te vao standalone_query.
   Vi du: user hoi "giai thich truong hop 3" va lich su co "3. Chi phi dau tu con lai..."
   -> standalone_query: "Chi phi dau tu vao dat con lai duoc boi thuong khi Nha nuoc thu hoi dat"
```

Renumber the existing rules 3, 4, 5 to 4, 5, 6.

### Step 4: Run tests to verify they pass

```bash
.venv/bin/python -m pytest tests/test_query_rewriter.py -v
```

Expected: All pass (26 existing + 4 new = 30).

### Step 5: Commit

```bash
git add app/rag/query_rewriter.py tests/test_query_rewriter.py
git commit -m "fix(rewriter): recency-weighted context window and ordinal resolution rule

- Last assistant message: 1500 chars (was 300) — fixes ordinal ref failure
- Older assistant messages: 400 chars
- max_tokens: 300 -> 500 to prevent JSON truncation
- Add explicit ordinal resolution instruction to system prompt

Fixes C2, M2"
```

---

## Task 2: Fix Retrieval Failure Modes

**Fixes:** C1, C4, M3, M4

**Files:**
- Modify: `app/rag/retriever.py:87-112` (filter-retry logging + reranker fallback)
- Modify: `app/chat.py:211-217` (forward char trim fix)
- Modify: `app/rag/retriever.py:165-182` (cap Dieu expansion)
- Test: `tests/test_retriever_integration.py`

### Step 1: Write the failing tests

Add to `tests/test_retriever_integration.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRerankerFallback:
    """C1: Reranker threshold eliminating all results must fall back to vector score."""

    @pytest.mark.asyncio
    @patch("app.rag.retriever.llm")
    @patch("app.rag.retriever._get_store")
    @patch("app.rag.retriever._get_reranker")
    async def test_falls_back_when_all_below_threshold(self, mock_reranker_fn, mock_store_fn, mock_llm):
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
    @patch("app.rag.retriever.llm")
    @patch("app.rag.retriever._get_store")
    @patch("app.rag.retriever._get_reranker")
    async def test_normal_results_above_threshold_returned(self, mock_reranker_fn, mock_store_fn, mock_llm):
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
    @patch("app.rag.retriever.llm")
    @patch("app.rag.retriever._get_store")
    @patch("app.rag.retriever._get_reranker")
    async def test_logs_warning_on_filter_retry(self, mock_reranker_fn, mock_store_fn, mock_llm, caplog):
        import logging
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

    @patch("app.rag.retriever._get_store")
    def test_expansion_capped(self, mock_store_fn):
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
```

### Step 2: Run to verify they fail

```bash
.venv/bin/python -m pytest tests/test_retriever_integration.py::TestRerankerFallback tests/test_retriever_integration.py::TestFilterRetryLogging tests/test_retriever_integration.py::TestDieuExpansionCap -v
```

Expected: FAILures (fallback not implemented, warning not logged, cap not enforced).

### Step 3: Implement the fixes

**Fix C1 — reranker fallback** in `app/rag/retriever.py`:

Replace the threshold block (around line 104-106):
```python
# OLD:
if reranker:
    candidates = [c for c in candidates if c["score"] >= RAG_RELEVANCE_THRESHOLD]
```

```python
# NEW:
if reranker:
    above = [c for c in candidates if c["score"] >= RAG_RELEVANCE_THRESHOLD]
    if above:
        candidates = above
    else:
        # All below threshold — fall back to top-k by raw vector score, no threshold
        logger.warning(
            "All %d reranked candidates scored below threshold %.2f — "
            "falling back to top-%d by vector score",
            len(candidates), RAG_RELEVANCE_THRESHOLD, top_k,
        )
        candidates = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)
```

**Fix C4 — filter retry warning** in `app/rag/retriever.py`:

Replace the silent retry block (around line 87-92):
```python
# OLD:
if not candidates and (doc_ids or dieu):
    query_embedding = all_embeddings[0]
    candidates = store.search_hybrid(
        query_vector=query_embedding, query_text=query, top_k=fetch_k,
    )
```

```python
# NEW:
if not candidates and (doc_ids or dieu):
    logger.warning(
        "Filter produced 0 results (doc_ids=%s, dieu=%s) — retrying without filters. "
        "Results may be from a different document.",
        doc_ids, dieu,
    )
    query_embedding = all_embeddings[0]
    candidates = store.search_hybrid(
        query_vector=query_embedding, query_text=query, top_k=fetch_k,
    )
```

**Fix M4 — cap Dieu expansion** in `app/config.py`, add:
```python
RAG_MAX_EXPANSION_CHUNKS = 12   # Max sibling chunks added by _expand_full_dieu
```

In `app/rag/retriever.py`, update `_expand_full_dieu` to enforce the cap:
```python
def _expand_full_dieu(chunks: list[dict], store: KnowledgeStore) -> list[dict]:
    """For top-scoring chunks, fetch sibling chunks from same Dieu. Capped at RAG_MAX_EXPANSION_CHUNKS."""
    from app.config import RAG_MAX_EXPANSION_CHUNKS
    seen_ids = {c.get("chunk_id") for c in chunks}
    expansion = []

    for chunk in chunks[:3]:
        if len(expansion) >= RAG_MAX_EXPANSION_CHUNKS:
            break
        doc_id = chunk.get("doc_id", "")
        dieu = chunk.get("dieu", "")
        if not doc_id or not dieu:
            continue
        siblings = store.fetch_full_dieu(doc_id, dieu)
        for sib in siblings:
            if len(expansion) >= RAG_MAX_EXPANSION_CHUNKS:
                break
            cid = sib.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                expansion.append(sib)

    return list(chunks) + expansion
```

**Fix M3 — forward char trim** in `app/chat.py`, replace `_assemble_conversation_context` trim loop (lines 211-217):

```python
# OLD (reverse iteration can overshoot budget):
total_chars = 0
trimmed = []
for msg in reversed(msgs):
    total_chars += len(msg["content"])
    if total_chars > CONTEXT_MAX_CHARS:
        break
    trimmed.insert(0, msg)
```

```python
# NEW (forward pass — include messages until budget would be exceeded):
total_chars = 0
trimmed = []
for msg in reversed(msgs):
    msg_len = len(msg["content"])
    if total_chars + msg_len > CONTEXT_MAX_CHARS:
        break
    total_chars += msg_len
    trimmed.insert(0, msg)
```

Apply the same fix to `_trim_for_orchestrator` (same pattern, lines 232-236).

### Step 4: Run tests to verify they pass

```bash
.venv/bin/python -m pytest tests/test_retriever_integration.py -v
```

Expected: All pass.

### Step 5: Commit

```bash
git add app/rag/retriever.py app/chat.py app/config.py tests/test_retriever_integration.py
git commit -m "fix(retrieval): reranker fallback, filter-retry warning, char trim, expansion cap

- C1: Fall back to top-k by vector score when all reranked candidates below threshold
- C4: Log explicit warning on filter-retry to surface wrong-document risk
- M3: Fix forward char trim to not overshoot budget
- M4: Cap Dieu expansion at RAG_MAX_EXPANSION_CHUNKS (12)

Fixes C1, C4, M3, M4"
```

---

## Task 3: Background Task Reliability

**Fixes:** H1

**Files:**
- Modify: `app/chat.py:485-487` and `app/chat.py:629-630` (summary update)
- Modify: `app/chat.py:541-542` (title generation)
- Test: `tests/test_orchestrator.py`

**Background:** Three `asyncio.create_task(...)` calls fire-and-forget with no error handling. If the process crashes or the DB connection drops after the response is sent, the summary and title writes are silently lost. Fix: wrap background coroutines in a helper that logs errors and retries once.

### Step 1: Write the failing test

Add to `tests/test_orchestrator.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio


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

        await _safe_background(failing_then_succeeds(), label="test_task")
        assert call_count == 2  # tried once, failed, retried once, succeeded

    @pytest.mark.asyncio
    async def test_safe_background_logs_permanent_failure(self, caplog):
        import logging
        from app.chat import _safe_background

        async def always_fails():
            raise RuntimeError("permanent error")

        with caplog.at_level(logging.ERROR, logger="app.chat"):
            # Should not raise
            await _safe_background(always_fails(), label="test_task")

        assert any("permanent" in msg.lower() or "failed" in msg.lower() for msg in caplog.messages)

    @pytest.mark.asyncio
    async def test_safe_background_succeeds_first_try(self):
        from app.chat import _safe_background

        call_count = 0

        async def succeeds():
            nonlocal call_count
            call_count += 1

        await _safe_background(succeeds(), label="test_task")
        assert call_count == 1
```

### Step 2: Run to verify they fail

```bash
.venv/bin/python -m pytest tests/test_orchestrator.py::TestBackgroundTaskReliability -v
```

Expected: FAIL (`_safe_background` does not exist yet).

### Step 3: Implement

Add `_safe_background` to `app/chat.py` after the imports section:

```python
async def _safe_background(coro, label: str) -> None:
    """Run a coroutine as a background task. Logs errors and retries once."""
    import asyncio
    try:
        await coro
    except Exception as e:
        logger.error("Background task '%s' failed: %s — retrying once", label, e)
        try:
            await coro
        except Exception as e2:
            logger.error("Background task '%s' failed permanently: %s", label, e2)
```

Replace the two fire-and-forget summary writes in `handle_message` and `handle_message_stream`:

```python
# OLD:
asyncio.create_task(asyncio.to_thread(db.upsert_summary, session_id, decision["summary_update"], turn))

# NEW:
asyncio.create_task(_safe_background(
    asyncio.to_thread(db.upsert_summary, session_id, decision["summary_update"], turn),
    label="upsert_summary",
))
```

Replace the fire-and-forget title generation in all 4 call sites:

```python
# OLD:
asyncio.create_task(_generate_title(session_id, user_message, response))

# NEW:
asyncio.create_task(_safe_background(
    _generate_title(session_id, user_message, response),
    label="generate_title",
))
```

### Step 4: Run tests to verify they pass

```bash
.venv/bin/python -m pytest tests/test_orchestrator.py -v
```

Expected: All pass.

### Step 5: Commit

```bash
git add app/chat.py tests/test_orchestrator.py
git commit -m "fix(chat): reliable background tasks with error logging and single retry

- Add _safe_background() helper: logs errors, retries once, never raises
- Wrap summary upsert and title generation in _safe_background
- Prevents silent data loss on transient DB failures

Fixes H1"
```

---

## Task 4: Conversation State — DB Layer

**Fixes:** C3, H3 (foundation)

**Files:**
- Modify: `app/db.py` (add `conv_state` table to both schemas, add `upsert_conv_state` / `get_conv_state`)
- Test: `tests/test_db.py` (new file)

**Background:** We need a single-row-per-session table to hold a free-form JSON state blob. Updated after every assistant turn (not just turn 5+). This is the storage layer for Task 5.

### Step 1: Write the failing tests

Create `tests/test_db.py`:

```python
"""Tests for db.py conversation state functions."""
import json
import pytest
from app import db


@pytest.fixture(autouse=True)
def fresh_session(tmp_path, monkeypatch):
    """Use a temp SQLite DB for each test."""
    import app.db as db_module
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db_module, "_use_pg", False)
    monkeypatch.setattr("app.config.DB_PATH", db_path)
    # Reset module-level sqlite state
    yield
    try:
        import os
        os.unlink(db_path)
    except FileNotFoundError:
        pass


class TestConvState:
    def test_get_returns_none_when_no_state(self):
        db.create_session("sess1")
        result = db.get_conv_state("sess1")
        assert result is None

    def test_upsert_then_get(self):
        db.create_session("sess1")
        state = {"current_topic": "bồi thường", "last_list": ["1. A", "2. B", "3. C"]}
        db.upsert_conv_state("sess1", state, turn=1)
        result = db.get_conv_state("sess1")
        assert result is not None
        assert result["current_topic"] == "bồi thường"
        assert result["last_list"] == ["1. A", "2. B", "3. C"]

    def test_upsert_overwrites_previous_state(self):
        db.create_session("sess1")
        db.upsert_conv_state("sess1", {"topic": "first"}, turn=1)
        db.upsert_conv_state("sess1", {"topic": "second", "extra": "data"}, turn=2)
        result = db.get_conv_state("sess1")
        assert result["topic"] == "second"
        assert result["extra"] == "data"

    def test_state_is_free_form_json(self):
        db.create_session("sess1")
        # Any JSON shape is valid — no fixed schema
        state = {
            "chu_de": "thủ tục cấp sổ đỏ",
            "buoc_dang_ban": 3,
            "co_quan": "UBND cấp huyện",
            "ho_so": ["CMND", "đơn xin cấp"],
        }
        db.upsert_conv_state("sess1", state, turn=1)
        result = db.get_conv_state("sess1")
        assert result["buoc_dang_ban"] == 3
        assert result["ho_so"] == ["CMND", "đơn xin cấp"]

    def test_delete_session_removes_state(self):
        db.create_session("sess1")
        db.upsert_conv_state("sess1", {"x": 1}, turn=1)
        db.delete_session("sess1")
        # Session deleted — recreate and check state is gone
        db.create_session("sess1")
        assert db.get_conv_state("sess1") is None
```

### Step 2: Run to verify they fail

```bash
.venv/bin/python -m pytest tests/test_db.py -v
```

Expected: FAIL (`get_conv_state` and `upsert_conv_state` do not exist, `conv_state` table not in schema).

### Step 3: Implement

In `app/db.py`, add the `conv_state` table to `_PG_SCHEMA`:

```sql
CREATE TABLE IF NOT EXISTS conv_state (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id),
    state      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

And to `_SQLITE_SCHEMA`:

```sql
CREATE TABLE IF NOT EXISTS conv_state (
    session_id TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

Also update `delete_session` in `db.py` to delete conv_state first (before deleting the session):

```python
# Add this line before the sessions DELETE in both branches:
cur.execute(f"DELETE FROM conv_state WHERE session_id = {p}", (session_id,))
```

Add the two new functions at the end of `app/db.py`:

```python
def get_conv_state(session_id: str) -> dict | None:
    """Return the current conversation state for a session, or None."""
    db = _get_db()
    p = _ph()
    sql = f"SELECT state FROM conv_state WHERE session_id = {p}"
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, (session_id,))
            row = cur.fetchone()
        _release_db(db)
        return json.loads(row[0]) if row else None
    else:
        row = db.execute(sql, (session_id,)).fetchone()
        _release_db(db)
        return json.loads(row["state"]) if row else None


def upsert_conv_state(session_id: str, state: dict, turn: int) -> None:
    """Upsert the conversation state for a session (one row per session)."""
    db = _get_db()
    p = _ph()
    now = datetime.now().isoformat()
    state_json = json.dumps(state, ensure_ascii=False)
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(
                f"""INSERT INTO conv_state (session_id, state, updated_at)
                    VALUES ({p}, {p}, {p})
                    ON CONFLICT (session_id) DO UPDATE
                    SET state = EXCLUDED.state, updated_at = EXCLUDED.updated_at""",
                (session_id, state_json, now),
            )
        db.commit()
    else:
        db.execute(
            f"""INSERT OR REPLACE INTO conv_state (session_id, state, updated_at)
                VALUES ({p}, {p}, {p})""",
            (session_id, state_json, now),
        )
        db.commit()
    _release_db(db)
```

**Note for PostgreSQL migration:** If using Supabase, run this SQL once:
```sql
CREATE TABLE IF NOT EXISTS conv_state (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id),
    state      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### Step 4: Run tests to verify they pass

```bash
.venv/bin/python -m pytest tests/test_db.py -v
```

Expected: All 5 pass.

### Step 5: Commit

```bash
git add app/db.py tests/test_db.py
git commit -m "feat(db): add conv_state table for schema-free conversation state

- Single row per session, free-form JSON
- upsert_conv_state / get_conv_state functions
- Cascade delete when session deleted
- Both SQLite and PostgreSQL supported

Foundation for C3, H3"
```

---

## Task 5: Conversation State — Extractor

**Fixes:** C3, H3

**Files:**
- Create: `app/memory/__init__.py`
- Create: `app/memory/conv_memory.py`
- Test: `tests/test_conv_memory.py` (new file)

**Background:** After each assistant turn, a background LLM call extracts a free-form JSON state from the response. The LLM decides what fields matter (numbered lists, procedure steps, cited laws, active topic). The extractor takes the previous state as input and produces an updated state — enabling Mem0-style incremental updates rather than wholesale replacement. Non-blocking, retryable.

### Step 1: Write the failing tests

Create `tests/test_conv_memory.py`:

```python
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
```

### Step 2: Run to verify they fail

```bash
.venv/bin/python -m pytest tests/test_conv_memory.py -v
```

Expected: FAIL (module does not exist).

### Step 3: Implement

Create `app/memory/__init__.py`:
```python
```
(empty)

Create `app/memory/conv_memory.py`:

```python
"""Schema-free conversation state extractor and store.

After each assistant turn, extract_state() extracts a compact JSON object
from the response. The LLM decides what fields to keep — no fixed schema.
The extractor takes the previous state as input, enabling incremental updates.

Public API:
  read(session_id) -> dict | None
  update(session_id, response, turn) -> None  (async, call as background task)
"""
import json
import logging

from app import db, llm

logger = logging.getLogger(__name__)

_EXTRACTOR_SYSTEM = """Ban la bo trich xuat trang thai hoi thoai cho chatbot phap luat dat dai.

Nhiem vu: Doc cau tra loi cua tro ly va cap nhat trang thai hoi thoai.
Trang thai giup giai quyet cac cau hoi tiep theo nhu "truong hop 3", "buoc 2", "dieu do".

Quy tac:
- Giu lai thong tin tu trang thai truoc (neu co) tru khi chu de thay doi hoan toan
- Luon ghi lai danh sach so thu tu neu co trong cau tra loi (day du noi dung moi muc)
- Ghi lai phap luat dang duoc ban (dieu luat, van ban)
- Ghi lai chu de chinh dang thao luan
- Su dung khoa tu do — khong co schema co dinh — chi ghi nhung gi se huu ich de giai quyet cau hoi tiep theo

Tra ve CHINH XAC mot JSON object (khong markdown, khong giai thich).
Neu khong co thong tin dang ke, tra ve {}.

Vi du dau ra cho hoi thoai ve danh sach:
{"chu_de": "boi thuong khi Nha nuoc thu hoi dat", "danh_sach_truong_hop": ["1. ...", "2. ...", "3. Chi phi dau tu con lai..."], "van_ban": "Dieu 101, 107 LDD2024"}

Vi du dau ra cho hoi thoai ve thu tuc:
{"chu_de": "thu tuc cap so do lan dau", "cac_buoc": ["1. Chuan bi ho so", "2. Nop tai UBND", "3. Nhan ket qua"], "co_quan": "UBND cap huyen"}"""


async def extract_state(response: str, prev_state: dict | None) -> dict | None:
    """Extract conversation state from an assistant response.

    Returns a free-form JSON dict, or None if extraction fails.
    """
    prev_block = ""
    if prev_state:
        prev_json = json.dumps(prev_state, ensure_ascii=False)
        prev_block = f"<trang-thai-truoc>\n{prev_json}\n</trang-thai-truoc>\n\n"

    prompt = (
        f"{prev_block}"
        f"<cau-tra-loi-moi>\n{response[:3000]}\n</cau-tra-loi-moi>\n\n"
        "Cap nhat trang thai hoi thoai:"
    )

    try:
        raw = await llm.generate(
            prompt=prompt,
            system=_EXTRACTOR_SYSTEM,
            model="utility",
            temperature=0.0,
            max_tokens=400,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(l for l in lines[1:] if not l.strip().startswith("```"))
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("State extractor returned invalid JSON: %s", raw[:100] if 'raw' in dir() else "?")
        return None
    except Exception as e:
        logger.error("State extractor failed: %s", e)
        return None


def read(session_id: str) -> dict | None:
    """Return the current conversation state for a session, or None."""
    return db.get_conv_state(session_id)


async def update(session_id: str, response: str, turn: int) -> None:
    """Extract state from the latest response and persist it.

    Designed to run as a background task — never raises.
    """
    try:
        prev_state = db.get_conv_state(session_id)
        new_state = await extract_state(response, prev_state)
        if new_state is not None:
            db.upsert_conv_state(session_id, new_state, turn)
            logger.debug("Conv state updated for session %s turn %d", session_id[:8], turn)
    except Exception as e:
        logger.error("Conv state update failed for session %s: %s", session_id[:8], e)
```

### Step 4: Run tests to verify they pass

```bash
.venv/bin/python -m pytest tests/test_conv_memory.py -v
```

Expected: All 9 pass.

### Step 5: Commit

```bash
git add app/memory/__init__.py app/memory/conv_memory.py tests/test_conv_memory.py
git commit -m "feat(memory): schema-free conversation state extractor

- extract_state(): LLM extracts free-form JSON from each assistant response
- State captures numbered lists, procedure steps, cited laws, active topic
- Takes previous state as input for incremental updates (Mem0 pattern)
- read() / update() public API; update() is silent on failure
- No fixed schema — LLM decides what fields are relevant per conversation

Implements C3, H3 (storage layer)"
```

---

## Task 6: Conversation State — Rewriter Integration

**Fixes:** C3

**Files:**
- Modify: `app/rag/query_rewriter.py:108-125` (`_build_rewriter_prompt` signature + body)
- Modify: `app/rag/query_rewriter.py:178-215` (`rewrite_query` signature)
- Test: `tests/test_query_rewriter.py`

**Background:** The rewriter now receives `conv_state` as an optional parameter. It is injected as a `<bo-nho>` XML block in the prompt, placed before the conversation history. The rewriter system prompt already has the ordinal resolution rule (added in Task 1).

### Step 1: Write the failing tests

Add to `tests/test_query_rewriter.py`:

```python
class TestRewriterConvState:
    """Verify conv_state is injected into rewriter prompt."""

    def test_conv_state_injected_into_prompt(self):
        from app.rag.query_rewriter import _build_rewriter_prompt

        state = {
            "chu_de": "bồi thường",
            "danh_sach": ["1. Trường hợp A", "2. B", "3. Chi phí đầu tư"],
        }
        prompt = _build_rewriter_prompt(
            "giai thich truong hop 3",
            summary=None,
            recent_messages=[],
            conv_state=state,
        )
        assert "bo-nho" in prompt or "Chi phí đầu tư" in prompt

    def test_conv_state_none_does_not_error(self):
        from app.rag.query_rewriter import _build_rewriter_prompt

        prompt = _build_rewriter_prompt(
            "some query",
            summary=None,
            recent_messages=[],
            conv_state=None,
        )
        assert "some query" in prompt

    @pytest.mark.asyncio
    @patch("app.rag.query_rewriter.llm")
    async def test_rewrite_query_accepts_conv_state(self, mock_llm):
        from app.rag.query_rewriter import rewrite_query
        import json

        mock_llm.generate = AsyncMock(return_value=json.dumps({
            "standalone_query": "Chi phí đầu tư vào đất còn lại được bồi thường",
            "search_queries": ["chi phi dau tu con lai boi thuong"],
            "filters": {"doc_ids": None, "dieu": None},
        }))

        state = {"danh_sach": ["1. A", "2. B", "3. Chi phí đầu tư"]}
        result = await rewrite_query(
            "giai thich truong hop 3",
            summary=None,
            recent_messages=[],
            conv_state=state,
        )
        assert "Chi phí" in result["standalone_query"]
```

### Step 2: Run to verify they fail

```bash
.venv/bin/python -m pytest tests/test_query_rewriter.py::TestRewriterConvState -v
```

Expected: FAIL (`conv_state` parameter not accepted yet).

### Step 3: Implement

Update `_build_rewriter_prompt` signature and body in `app/rag/query_rewriter.py`:

```python
def _build_rewriter_prompt(
    user_message: str,
    summary: str | None,
    recent_messages: list[dict],
    conv_state: dict | None = None,       # <-- new
) -> str:
    parts = []
    if summary:
        parts.append(f"<tom-tat-hoi-thoai>\n{summary}\n</tom-tat-hoi-thoai>")
    if conv_state:
        import json
        state_text = json.dumps(conv_state, ensure_ascii=False, indent=2)
        parts.append(f"<bo-nho>\n{state_text}\n</bo-nho>")
    if recent_messages:
        # ... (existing recency-weighted history block from Task 1)
    parts.append(f"<cau-hoi-hien-tai>\n{user_message}\n</cau-hoi-hien-tai>")
    return "\n\n".join(parts)
```

Update `rewrite_query` signature:

```python
async def rewrite_query(
    user_message: str,
    summary: str | None,
    recent_messages: list[dict],
    conv_state: dict | None = None,       # <-- new
) -> dict:
    prompt = _build_rewriter_prompt(user_message, summary, recent_messages, conv_state)
    # ... rest unchanged
```

### Step 4: Run tests to verify they pass

```bash
.venv/bin/python -m pytest tests/test_query_rewriter.py -v
```

Expected: All pass (30 existing + 3 new = 33).

### Step 5: Commit

```bash
git add app/rag/query_rewriter.py tests/test_query_rewriter.py
git commit -m "feat(rewriter): inject conv_state as <bo-nho> context block

- _build_rewriter_prompt and rewrite_query now accept optional conv_state dict
- State injected before history, enabling ordinal reference resolution
- conv_state=None is backwards-compatible, no behaviour change

Fixes C3 (rewriter side)"
```

---

## Task 7: Conversation State — chat.py Wiring

**Fixes:** C3, H3

**Files:**
- Modify: `app/chat.py:408-558` (`handle_message`)
- Modify: `app/chat.py:571-701` (`handle_message_stream`)
- Test: `tests/test_orchestrator.py`

**Background:** Wire the `ConversationMemory` module into the main chat pipeline. At turn start, read the state. After response is saved, fire a background update. The state flows into `rewrite_query`. Since extraction runs from turn 1 (not turn 5+), early ordinal references are handled from the second turn onwards.

### Step 1: Write the failing tests

Add to `tests/test_orchestrator.py`:

```python
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
```

### Step 2: Run to verify they fail

```bash
.venv/bin/python -m pytest tests/test_orchestrator.py::TestConvStateWiring -v
```

Expected: FAIL (`conv_memory` not imported in `chat.py`).

### Step 3: Implement

At the top of `app/chat.py`, add the import:

```python
from app.memory import conv_memory
```

In `handle_message` (and `handle_message_stream`), after assembling conversation context:

```python
# After this line:
recent_messages, summary = await _assemble_conversation_context(history, session_id)

# Add:
conv_state = conv_memory.read(session_id)
```

Update the `rewrite_query` call to pass `conv_state`:

```python
rewrite_result = await rewrite_query(user_message, summary, recent_messages, conv_state)
```

After saving messages (after `db.add_messages_batch`), fire background state update:

```python
asyncio.create_task(_safe_background(
    conv_memory.update(session_id, response, turn),
    label="conv_state_update",
))
```

Do this in both `handle_message` and `handle_message_stream`.

### Step 4: Run all tests to verify

```bash
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_chunker.py
```

Expected: All pass.

### Step 5: Commit

```bash
git add app/chat.py tests/test_orchestrator.py
git commit -m "feat(chat): wire conversation state into pipeline

- Read conv_state at turn start, pass to rewrite_query
- Fire background state update after every assistant turn (not just turn 5+)
- State extracted from turn 1 onwards — early ordinal refs now resolvable
- Background update wrapped in _safe_background for reliability

Closes C3, H3"
```

---

## Final Verification

Run the full test suite to confirm no regressions:

```bash
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_chunker.py
```

Expected output: All tests pass. Verify these test classes exist and pass:
- `TestRewriterContextWindow` (Task 1)
- `TestRerankerFallback`, `TestFilterRetryLogging`, `TestDieuExpansionCap` (Task 2)
- `TestBackgroundTaskReliability` (Task 3)
- `TestConvState` (Task 4)
- `TestConvMemoryExtract`, `TestConvMemoryReadUpdate` (Task 5)
- `TestRewriterConvState` (Task 6)
- `TestConvStateWiring` (Task 7)

---

## What's Not In This Plan (Deliberate Deferral)

| Issue | Reason deferred |
|---|---|
| H2: Orchestrator context asymmetry | Design tradeoff — orchestrator intentionally has a smaller budget to reduce latency. Revisit if routing decisions degrade in long conversations. |
| M1: Multi-query reranking | Low ROI vs. complexity. Reranker already uses the primary standalone_query which is now correctly resolved via state. |
| L1: Turn uniqueness DB constraint | Requires schema migration with index. Low failure probability in practice. Defer to next DB schema revision. |
