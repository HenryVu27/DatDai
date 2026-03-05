# Agentic Context Engineering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the sequential always-retrieve RAG pipeline with an orchestrator-first agentic architecture where an LLM decides when and how to retrieve, with proper context engineering.

**Architecture:** Two-stage pipeline. Stage 1: orchestrator (Flash 3) analyzes conversation and decides tool calls. Stage 2: generator (Pro 3.1 or Flash 3) produces answer using retrieved context. Rolling summaries generated as orchestrator byproduct.

**Tech Stack:** Python 3.12, FastAPI, Gemini 3.x/2.5 models, Qdrant, Jina reranker, SQLite.

---

### Task 1: Update config.py with new model constants and context parameters

**Files:**
- Modify: `app/config.py`

**Step 1: Update config.py**

Replace model constants and RAG flags with new agentic configuration:

```python
"""Application configuration."""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Model IDs -- primary (Gemini 3.x preview)
ORCHESTRATOR_MODEL = "gemini-3-flash-preview"
GENERATOR_PRO_MODEL = "gemini-3.1-pro-preview"
GENERATOR_FLASH_MODEL = "gemini-3-flash-preview"
UTILITY_MODEL = "gemini-3.1-flash-lite-preview"
EMBEDDING_MODEL = "gemini-embedding-001"

# Model IDs -- fallbacks (stable Gemini 2.5)
FALLBACK_PRO_MODEL = "gemini-2.5-pro"
FALLBACK_FLASH_MODEL = "gemini-2.5-flash"
FALLBACK_UTILITY_MODEL = "gemini-2.5-flash-lite"

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
DB_PATH = os.path.join(DATA_DIR, "chat.db")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = "dat_dai_law"

# Retrieval
RAG_TOP_K = 8
RAG_RERANK_CANDIDATES = 20
RAG_RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
RAG_RELEVANCE_THRESHOLD = 0.15
RAG_USE_RERANKER = True

# Context management
CONTEXT_MAX_TURN_GROUPS = 6
CONTEXT_MAX_CHARS = 30_000
```

**Step 2: Verify import works**

Run: `.venv/bin/python -c "from app.config import ORCHESTRATOR_MODEL, GENERATOR_PRO_MODEL; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/config.py
git commit -m "config: update model IDs to Gemini 3.x, simplify RAG flags for agentic pipeline"
```

---

### Task 2: Rewrite llm.py with new model routing and fallback logic

**Files:**
- Modify: `app/llm.py`

**Step 1: Rewrite llm.py**

```python
"""Gemini LLM client with model routing and fallback."""
import logging
from google import genai
from google.genai import types

from app.config import (
    GEMINI_API_KEY, EMBEDDING_MODEL,
    ORCHESTRATOR_MODEL, GENERATOR_PRO_MODEL, GENERATOR_FLASH_MODEL, UTILITY_MODEL,
    FALLBACK_PRO_MODEL, FALLBACK_FLASH_MODEL, FALLBACK_UTILITY_MODEL,
)

logger = logging.getLogger(__name__)

_client = None

# Model role -> (primary, fallback)
MODEL_MAP = {
    "orchestrator": (ORCHESTRATOR_MODEL, FALLBACK_FLASH_MODEL),
    "pro": (GENERATOR_PRO_MODEL, FALLBACK_PRO_MODEL),
    "flash": (GENERATOR_FLASH_MODEL, FALLBACK_FLASH_MODEL),
    "utility": (UTILITY_MODEL, FALLBACK_UTILITY_MODEL),
}


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


async def generate(
    prompt: str,
    system: str = "",
    history: list[dict] | None = None,
    model: str = "flash",
    temperature: float = 0.3,
    max_tokens: int = 4000,
) -> str:
    """Generate a response using Gemini with automatic fallback."""
    client = get_client()
    primary, fallback = MODEL_MAP.get(model, (GENERATOR_FLASH_MODEL, FALLBACK_FLASH_MODEL))

    contents = []
    if history:
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    config = types.GenerateContentConfig(
        system_instruction=system if system else None,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    for model_id in (primary, fallback):
        try:
            response = client.models.generate_content(
                model=model_id, contents=contents, config=config,
            )
            return response.text or ""
        except Exception as e:
            if model_id == primary:
                logger.warning("Primary model %s failed, trying fallback %s: %s", primary, fallback, e)
                continue
            logger.error("Fallback model %s also failed: %s", fallback, e)
            raise

    return ""


def embed(texts: list[str]) -> list[list[float]]:
    """Create embeddings using Gemini embedding model."""
    client = get_client()
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
    return [e.values for e in result.embeddings]
```

Note: `classify_complexity()` is removed -- the orchestrator handles this.

**Step 2: Verify import works**

Run: `.venv/bin/python -c "from app.llm import generate, embed, MODEL_MAP; print(MODEL_MAP)"`
Expected: prints the model map dict without errors

**Step 3: Commit**

```bash
git add app/llm.py
git commit -m "llm: add Gemini 3.x models with fallback routing, remove classify_complexity"
```

---

### Task 3: Update db.py for per-turn summary storage

**Files:**
- Modify: `app/db.py`

**Step 1: Update db.py**

The schema already supports summaries with `covers_through_turn`. The only change needed is adding an `upsert_summary` that updates the latest summary rather than always inserting, since the orchestrator may update the summary every turn (but not always).

Add this function after `get_latest_summary`:

```python
def upsert_summary(session_id: str, summary: str, covers_through_turn: int) -> None:
    """Insert or update the rolling summary for a session.

    The orchestrator generates summary_update on some turns. We keep only one
    active summary per session and update it in place.
    """
    db = get_db()
    existing = db.execute(
        "SELECT id FROM summaries WHERE session_id = ? ORDER BY covers_through_turn DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if existing:
        db.execute(
            "UPDATE summaries SET summary = ?, covers_through_turn = ?, created_at = ? WHERE id = ?",
            (summary, covers_through_turn, datetime.now().isoformat(), existing["id"]),
        )
    else:
        db.execute(
            "INSERT INTO summaries (session_id, summary, covers_through_turn, created_at) VALUES (?, ?, ?, ?)",
            (session_id, summary, covers_through_turn, datetime.now().isoformat()),
        )
    db.commit()
    db.close()
```

**Step 2: Verify**

Run: `.venv/bin/python -c "from app.db import upsert_summary; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/db.py
git commit -m "db: add upsert_summary for orchestrator-driven rolling summaries"
```

---

### Task 4: Simplify query_rewriter.py -- keep only extract_filters

**Files:**
- Modify: `app/rag/query_rewriter.py`

**Step 1: Remove rewrite_query, keep extract_filters**

```python
"""Filter extraction for legal RAG queries."""
import re

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

DIEU_RE = re.compile(r"(?:điều|dieu|đ\.?)\s*(\d+)", re.IGNORECASE)


def extract_filters(query: str) -> dict:
    """Extract doc_ids and dieu from query text."""
    query_lower = query.lower()
    doc_ids = []
    for pattern, ids in DOC_PATTERNS.items():
        if re.search(pattern, query_lower):
            doc_ids.extend(ids)
    dieu = None
    m = DIEU_RE.search(query_lower)
    if m:
        dieu = f"Điều {m.group(1)}"
    return {"doc_ids": list(set(doc_ids)) or None, "dieu": dieu}
```

**Step 2: Verify**

Run: `.venv/bin/python -c "from app.rag.query_rewriter import extract_filters; print(extract_filters('Dieu 15 ND 102'))"`
Expected: `{'doc_ids': ['nd102'], 'dieu': 'Dieu 15'}` (or similar)

**Step 3: Commit**

```bash
git add app/rag/query_rewriter.py
git commit -m "query_rewriter: remove rewrite_query, keep extract_filters only (orchestrator handles rewriting)"
```

---

### Task 5: Delete hyde.py and query_decomposer.py

**Files:**
- Delete: `app/rag/hyde.py`
- Delete: `app/rag/query_decomposer.py`

**Step 1: Delete files**

```bash
rm app/rag/hyde.py app/rag/query_decomposer.py
```

**Step 2: Verify no remaining imports**

Run: `grep -rn "from app.rag.hyde\|from app.rag.query_decomposer\|import hyde\|import query_decomposer" app/`

Should return only hits in `app/rag/retriever.py` which we rewrite in Task 6.

**Step 3: Commit**

```bash
git add -A app/rag/hyde.py app/rag/query_decomposer.py
git commit -m "delete hyde.py and query_decomposer.py (orchestrator replaces both)"
```

---

### Task 6: Simplify cross_ref.py -- keep data maps, remove auto-expansion

**Files:**
- Modify: `app/rag/cross_ref.py`

**Step 1: Simplify to data + targeted lookup**

Keep `AMENDS_MAP`, `AMENDED_BY`, `ND_NUM_TO_DOC_ID`, and the per-document lookup functions. Remove `expand_with_cross_refs` (the auto-expansion loop) and `parse_cross_refs`. The orchestrator will call `lookup_amendment` explicitly.

```python
"""Cross-reference data for Vietnamese land law documents.

Maps amendment relationships between documents. Used by the orchestrator
to decide when to fetch amendment-related chunks.
"""

# Which docs amend which (doc_id -> list of target doc_ids it amends)
AMENDS_MAP = {
    "nd49": ["nd71", "nd88", "nd102", "nd151", "nd226"],
    "nd226": ["nd71", "nd88", "nd102", "nd151"],
    "nd50": ["nd103"],
}

# Reverse: which docs is a given doc amended BY
AMENDED_BY: dict[str, list[str]] = {}
for _src, _targets in AMENDS_MAP.items():
    for _t in _targets:
        AMENDED_BY.setdefault(_t, []).append(_src)

# Which docs detail/implement which
IMPLEMENTS_MAP = {
    "nd71": ["ldd2024"],
    "nd88": ["ldd2024"],
    "nd102": ["ldd2024"],
    "nd103": ["ldd2024"],
    "nd151": ["ldd2024"],
    "nd50": ["nq254"],
    "nd12": ["ldd2013"],
}

# Map ND numbers found in text to doc_ids
ND_NUM_TO_DOC_ID = {
    "71": "nd71", "88": "nd88", "102": "nd102", "103": "nd103",
    "151": "nd151", "226": "nd226", "49": "nd49", "50": "nd50",
    "12": "nd12", "254": "nq254",
}


def get_amendment_doc_ids(doc_id: str) -> list[str]:
    """Get doc_ids that amend the given doc."""
    return AMENDED_BY.get(doc_id, [])


def get_amended_doc_ids(doc_id: str) -> list[str]:
    """Get doc_ids that the given doc amends."""
    return AMENDS_MAP.get(doc_id, [])
```

**Step 2: Verify**

Run: `.venv/bin/python -c "from app.rag.cross_ref import AMENDED_BY; print(AMENDED_BY)"`
Expected: prints the reverse amendment map

**Step 3: Commit**

```bash
git add app/rag/cross_ref.py
git commit -m "cross_ref: simplify to data maps only, remove auto-expansion (orchestrator decides)"
```

---

### Task 7: Rewrite retriever.py as tool functions

**Files:**
- Modify: `app/rag/retriever.py`

**Step 1: Rewrite retriever.py**

Replace the monolithic `Retriever` class with individual tool functions that the orchestrator calls. Keep reranking as a post-step.

```python
"""RAG tool functions for the orchestrator.

Each function is a discrete tool the orchestrator can invoke.
Results are merged and reranked in the orchestrator pipeline.
"""
import logging
from app import llm
from app.config import RAG_RERANK_CANDIDATES, RAG_RERANKER_MODEL, RAG_RELEVANCE_THRESHOLD, RAG_USE_RERANKER, RAG_TOP_K
from app.rag.knowledge_store import KnowledgeStore
from app.rag.reranker import CrossEncoderReranker
from app.rag.cross_ref import AMENDED_BY
from app.rag.amendment_index import get_amendment_chunk_ids, get_all_amendment_chunk_ids_to

logger = logging.getLogger(__name__)

# Singletons
_store: KnowledgeStore | None = None
_reranker: CrossEncoderReranker | None = None


def _get_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store


def _get_reranker() -> CrossEncoderReranker | None:
    global _reranker
    if _reranker is None and RAG_USE_RERANKER:
        try:
            _reranker = CrossEncoderReranker(model_name=RAG_RERANKER_MODEL)
        except Exception as e:
            logger.warning("Reranker init failed: %s", e)
    return _reranker


async def search_legal_docs(
    query: str,
    doc_ids: list[str] | None = None,
    dieu: str | None = None,
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """Hybrid dense+sparse search against Qdrant.

    The orchestrator crafts the query and optional filters.
    Returns reranked chunks.
    """
    store = _get_store()

    # Embed the query
    query_embedding = llm.embed([query])[0]

    # Hybrid search
    fetch_k = RAG_RERANK_CANDIDATES if _get_reranker() else top_k
    candidates = store.search_hybrid(
        query_vector=query_embedding,
        query_text=query,
        top_k=fetch_k,
        doc_ids=doc_ids,
        dieu=dieu,
    )

    # Retry without filters if empty
    if not candidates and (doc_ids or dieu):
        candidates = store.search_hybrid(
            query_vector=query_embedding, query_text=query, top_k=fetch_k,
        )

    if not candidates:
        return []

    # Rerank
    reranker = _get_reranker()
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


def lookup_amendment(
    target_doc: str,
    source_doc: str | None = None,
    dieu: str | None = None,
) -> list[dict]:
    """Look up amendment relationships and fetch related chunks.

    If source_doc is given, fetches amendments from source_doc -> target_doc.
    If source_doc is None, fetches all amendments to target_doc.
    If dieu is given, filters to chunks matching that dieu.
    """
    store = _get_store()

    if source_doc:
        chunk_ids = get_amendment_chunk_ids(source_doc, target_doc)
    else:
        chunk_ids = get_all_amendment_chunk_ids_to(target_doc)

    if not chunk_ids:
        # Fallback: fetch from amending docs by metadata
        amenders = AMENDED_BY.get(target_doc, [])
        if source_doc:
            amenders = [source_doc] if source_doc in amenders else []
        chunks = []
        for amender_id in amenders:
            chunks.extend(store.fetch_by_metadata(doc_id=amender_id, dieu=dieu, limit=5))
        return chunks

    chunks = store.fetch_by_chunk_ids(chunk_ids)

    # Filter by dieu if specified
    if dieu and chunks:
        filtered = [c for c in chunks if c.get("dieu") == dieu]
        if filtered:
            chunks = filtered

    return chunks


def lookup_specific_dieu(doc_id: str, dieu: str) -> list[dict]:
    """Fetch all chunks for a specific Dieu from a specific document.

    Uses metadata lookup (no embedding needed). Returns all Khoan sub-chunks.
    """
    store = _get_store()
    return store.fetch_full_dieu(doc_id, dieu)


def _expand_full_dieu(chunks: list[dict], store: KnowledgeStore) -> list[dict]:
    """For top-scoring chunks, fetch all sibling chunks from same Dieu."""
    seen_ids = {c.get("chunk_id") for c in chunks}
    expansion = []

    for chunk in chunks[:3]:
        doc_id = chunk.get("doc_id", "")
        dieu = chunk.get("dieu", "")
        if not doc_id or not dieu:
            continue
        siblings = store.fetch_full_dieu(doc_id, dieu)
        for sib in siblings:
            cid = sib.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                expansion.append(sib)

    return list(chunks) + expansion


def build_context(chunks: list[dict]) -> str:
    """Format chunks into a context string for the generator."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata_str", "")
        content = chunk.get("text", chunk.get("content", ""))
        parts.append(f"[Nguon {i}: {source}]\n{content}")
    return "\n\n---\n\n".join(parts)


def extract_sources(chunks: list[dict]) -> list[dict]:
    """Extract deduplicated source references from chunks."""
    sources, seen = [], set()
    for chunk in chunks:
        key = f"{chunk.get('doc_name', '')}|{chunk.get('dieu', '')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "doc_name": chunk.get("doc_name", ""),
                "chapter": chunk.get("chapter", ""),
                "dieu": chunk.get("dieu", ""),
                "dieu_title": chunk.get("dieu_title", ""),
            })
    return sources
```

**Step 2: Verify imports work**

Run: `.venv/bin/python -c "from app.rag.retriever import search_legal_docs, lookup_amendment, lookup_specific_dieu, build_context; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/rag/retriever.py
git commit -m "retriever: rewrite as discrete tool functions for orchestrator"
```

---

### Task 8: Update rag/__init__.py to expose new tool functions

**Files:**
- Modify: `app/rag/__init__.py`

**Step 1: Update __init__.py**

```python
"""RAG package -- exposes tool functions for the orchestrator."""
from app.rag.retriever import (
    search_legal_docs,
    lookup_amendment,
    lookup_specific_dieu,
    build_context,
    extract_sources,
)
```

**Step 2: Verify**

Run: `.venv/bin/python -c "from app.rag import search_legal_docs, lookup_amendment, lookup_specific_dieu; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/rag/__init__.py
git commit -m "rag/__init__: expose discrete tool functions"
```

---

### Task 9: Rewrite chat.py -- orchestrator-based pipeline

This is the core task. Rewrite the entire chat orchestration.

**Files:**
- Modify: `app/chat.py`

**Step 1: Write the new chat.py**

```python
"""Agentic chat orchestrator.

Two-stage pipeline:
1. Orchestrator (Flash 3) decides what tools to call
2. Generator (Pro 3.1 / Flash 3) produces the answer
"""
import asyncio
import json
import logging
import time
import uuid

from app import db, llm
from app.config import CONTEXT_MAX_TURN_GROUPS, CONTEXT_MAX_CHARS
from app.rag.retriever import (
    search_legal_docs, lookup_amendment, lookup_specific_dieu,
    build_context, extract_sources,
)
from app.rag.query_rewriter import extract_filters
from app.rag.citation_check import verify_citations

logger = logging.getLogger(__name__)

# ── System prompts (static, XML-tagged) ──────────────────────────

SYSTEM_BASE = """<identity>
Ban la chuyen gia tu van phap luat dat dai Viet Nam. Ban tra loi cau hoi dua tren cac van ban phap luat duoc cung cap.
Tra loi bang tieng Viet, ro rang, de hieu cho nguoi dan thuong.
</identity>

<boundaries>
1. Chi tra loi dua tren noi dung van ban phap luat duoc cung cap trong phan "Tai lieu tham khao"
2. Trich dan cu the so dieu, khoan, diem va ten van ban khi tra loi
3. Neu thong tin khong co trong tai lieu, noi ro "Toi khong tim thay thong tin nay trong cac van ban hien co"
4. Khi co nhieu van ban lien quan, neu ro moi quan he giua chung (vi du: Luat quy dinh chung, Nghi dinh huong dan chi tiet)
5. Neu cau hoi mo ho, hoi lai de lam ro truoc khi tra loi
</boundaries>

<legal-hierarchy>
Luat Dat Dai 2024 (31/2024/QH15) - luat goc
ND 71/2024 - gia dat [sua doi boi: ND 226, ND 49]
ND 88/2024 - boi thuong, ho tro, tai dinh cu [sua doi boi: ND 226, ND 49]
ND 102/2024 - chi tiet thi hanh [sua doi boi: ND 226, ND 49]
ND 103/2024 - tien su dung dat, tien thue dat [sua doi boi: ND 50]
ND 151/2025 - phan dinh tham quyen [sua doi boi: ND 226, ND 49]
ND 226/2025 - sua doi 4 ND [sua doi boi: ND 49]
NQ 254/2025 - thao go vuong mac
ND 49/2026 - sua doi moi nhat cac ND
ND 12/2024 - chuyen tiep gia dat
ND 50/2026 - chi tiet NQ 254 ve tien su dung dat, tien thue dat
Thu tu uu tien: ND moi nhat > ND cu > Luat goc
</legal-hierarchy>"""

ORCHESTRATOR_TOOLS_SECTION = """
<tools>
Ban co the goi cac tool sau de tra cuu thong tin. Tra ve JSON hop le.

1. search_legal_docs: Tim kiem van ban phap luat bang ngon ngu tu nhien.
   Params: {"query": "cau truy van", "filters": {"doc_ids": ["nd102"], "dieu": "Dieu 15"}}
   - query: viet ro rang, day du ngu canh, khong dung dai tu
   - filters: tuy chon, chi dinh khi biet chinh xac van ban/dieu

2. lookup_amendment: Tra cuu cac sua doi giua cac nghi dinh.
   Params: {"target_doc": "nd102", "source_doc": "nd49", "dieu": "Dieu 15"}
   - target_doc: bat buoc - van ban bi sua doi
   - source_doc: tuy chon - van ban sua doi (neu biet)
   - dieu: tuy chon - so dieu cu the

3. lookup_specific_dieu: Lay toan bo noi dung mot dieu cu the.
   Params: {"doc_id": "ldd2024", "dieu": "Dieu 79"}
   - Dung khi biet chinh xac dieu va van ban can tra cuu

KHONG goi tool khi:
- Nguoi dung chao hoi, cam on, noi chuyen xa giao
- Cau tra loi da co trong lich su hoi thoai gan day
- Chi can lam ro hoac hoi lai cau hoi cua nguoi dung
- Nguoi dung hoi ve noi dung ban vua tra loi
</tools>"""

ORCHESTRATOR_INSTRUCTIONS = """
<instructions>
Phan tich tin nhan cua nguoi dung va quyet dinh hanh dong.
Tra ve CHINH XAC mot JSON object (khong markdown, khong giai thich) voi format:
{
  "reasoning": "suy nghi ngan ve nhung gi nguoi dung can",
  "actions": [
    {"tool": "ten_tool", ...params}
  ],
  "complexity": "simple" hoac "complex",
  "summary_update": "ban tom tat cap nhat" hoac null,
  "direct_response": "cau tra loi truc tiep" hoac null
}

Quy tac:
- Neu co the tra loi truc tiep (chao hoi, cam on, lam ro), dat direct_response va actions=[]
- Neu can tra cuu, dat actions voi cac tool call phu hop
- Co the goi nhieu tool cung luc (vd: search + lookup_amendment)
- complexity: "complex" cho cau hoi phap ly kho, so sanh, nhieu van ban. "simple" cho con lai
- summary_update: chi dat khi cuoc hoi thoai da tien trien dang ke (4+ luot). Tom tat phai BAO GOM summary truoc do va bo sung noi dung moi
- Khi viet query cho search_legal_docs: viet ro rang, khong dung dai tu, bao gom ngu canh tu hoi thoai
</instructions>"""


# ── Context assembly ─────────────────────────────────────────────

def _assemble_conversation_context(history: list[dict], session_id: str) -> tuple[list[dict], str | None]:
    """Build trimmed conversation context and retrieve summary.

    Returns (recent_messages, summary_text).
    """
    summary_row = db.get_latest_summary(session_id)
    summary_text = summary_row["summary"] if summary_row else None

    # Convert to simple dicts
    msgs = [{"role": r["role"], "content": r["content"]} for r in history]

    # Atomic turn-group trimming: keep last N complete pairs
    # A turn-group = (user msg, assistant msg)
    max_msgs = CONTEXT_MAX_TURN_GROUPS * 2
    if len(msgs) > max_msgs:
        msgs = msgs[-max_msgs:]

    # Trim by character budget
    total_chars = 0
    trimmed = []
    for msg in reversed(msgs):
        total_chars += len(msg["content"])
        if total_chars > CONTEXT_MAX_CHARS:
            break
        trimmed.insert(0, msg)

    # Ensure we don't start with an assistant message (orphaned)
    if trimmed and trimmed[0]["role"] == "assistant":
        trimmed = trimmed[1:]

    return trimmed, summary_text


def _build_orchestrator_prompt(
    user_message: str,
    recent_messages: list[dict],
    summary: str | None,
) -> tuple[str, list[dict]]:
    """Build system prompt and history for the orchestrator call.

    Returns (system_prompt, history_for_llm).
    """
    system = SYSTEM_BASE + ORCHESTRATOR_TOOLS_SECTION + ORCHESTRATOR_INSTRUCTIONS

    # Build volatile context as part of the user message
    volatile_parts = []
    if summary:
        volatile_parts.append(f"<conversation-summary>\n{summary}\n</conversation-summary>")

    # The recent_messages become the LLM history, current message is the prompt
    history = recent_messages  # These are already trimmed

    # Current user message with any volatile context
    if volatile_parts:
        prompt = "\n".join(volatile_parts) + f"\n\n{user_message}"
    else:
        prompt = user_message

    return system, history, prompt


def _build_generator_prompt(
    user_message: str,
    context: str,
    recent_messages: list[dict],
    summary: str | None,
) -> tuple[str, list[dict], str]:
    """Build system prompt, history, and user prompt for the generator.

    Returns (system_prompt, history_for_llm, user_prompt).
    """
    system = SYSTEM_BASE

    history = []
    if summary:
        history.append({"role": "user", "content": f"<conversation-summary>\n{summary}\n</conversation-summary>"})
        history.append({"role": "model", "content": "Da ghi nhan."})
    history.extend(recent_messages)

    if context:
        prompt = f"""<retrieved-context>
{context}
</retrieved-context>

<user-question>
{user_message}
</user-question>

Hay tra loi dua tren tai lieu tham khao o tren. Trich dan cu the dieu, khoan, ten van ban."""
    else:
        prompt = f"""<user-question>
{user_message}
</user-question>

Tra loi dua tren noi dung da thao luan trong cuoc hoi thoai."""

    return system, history, prompt


# ── Orchestrator ─────────────────────────────────────────────────

def _parse_orchestrator_response(text: str) -> dict:
    """Parse the orchestrator's JSON response, handling edge cases."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Orchestrator returned invalid JSON, treating as direct response: %s", text[:200])
        return {
            "reasoning": "Failed to parse orchestrator output",
            "actions": [],
            "complexity": "simple",
            "summary_update": None,
            "direct_response": text,
        }

    # Ensure required keys with defaults
    result.setdefault("reasoning", "")
    result.setdefault("actions", [])
    result.setdefault("complexity", "simple")
    result.setdefault("summary_update", None)
    result.setdefault("direct_response", None)
    return result


async def _execute_tools(actions: list[dict]) -> list[dict]:
    """Execute orchestrator tool calls in parallel. Returns merged chunks."""
    if not actions:
        return []

    async def _run_action(action: dict) -> list[dict]:
        tool = action.get("tool", "")
        try:
            if tool == "search_legal_docs":
                query = action.get("query", "")
                filters = action.get("filters", {})
                return await search_legal_docs(
                    query=query,
                    doc_ids=filters.get("doc_ids"),
                    dieu=filters.get("dieu"),
                )
            elif tool == "lookup_amendment":
                return lookup_amendment(
                    target_doc=action.get("target_doc", ""),
                    source_doc=action.get("source_doc"),
                    dieu=action.get("dieu"),
                )
            elif tool == "lookup_specific_dieu":
                return lookup_specific_dieu(
                    doc_id=action.get("doc_id", ""),
                    dieu=action.get("dieu", ""),
                )
            else:
                logger.warning("Unknown tool: %s", tool)
                return []
        except Exception as e:
            logger.error("Tool %s failed: %s", tool, e)
            return []

    results = await asyncio.gather(*[_run_action(a) for a in actions])

    # Merge and deduplicate
    seen_ids = set()
    merged = []
    for chunk_list in results:
        for chunk in chunk_list:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(chunk)
            elif not cid:
                merged.append(chunk)

    return merged


# ── Main entry point ─────────────────────────────────────────────

async def handle_message(session_id: str, user_message: str) -> dict:
    """Process a user message through the orchestrator pipeline."""
    t_start = time.monotonic()

    # Ensure session exists
    db.create_session(session_id)

    # Get conversation history and turn count
    history = db.get_messages(session_id)
    turn = db.get_turn_count(session_id) + 1

    # Save user message
    db.add_message(session_id, turn, "user", user_message)

    # Assemble conversation context
    recent_messages, summary = _assemble_conversation_context(history, session_id)

    # ── Stage 1: Orchestrator ──
    t_orch = time.monotonic()
    orch_system, orch_history, orch_prompt = _build_orchestrator_prompt(
        user_message, recent_messages, summary,
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

    # Save summary update if provided
    if decision["summary_update"]:
        db.upsert_summary(session_id, decision["summary_update"], turn)

    # ── Direct response path ──
    if decision["direct_response"]:
        response = decision["direct_response"]
        db.add_message(session_id, turn, "assistant", response, [])

        if turn == 1:
            await _generate_title(session_id, user_message, response)

        return {"answer": response, "sources": [], "session_id": session_id}

    # ── Stage 2: Execute tools ──
    t_tools = time.monotonic()
    chunks = await _execute_tools(decision["actions"])
    logger.info("[%s] tools returned %d chunks (%.1fs)", session_id[:8], len(chunks), time.monotonic() - t_tools)

    context = build_context(chunks)
    sources = extract_sources(chunks)

    # ── Stage 3: Generator ──
    t_gen = time.monotonic()
    model = "pro" if decision["complexity"] == "complex" else "flash"
    gen_system, gen_history, gen_prompt = _build_generator_prompt(
        user_message, context, recent_messages, summary,
    )

    response = await llm.generate(
        prompt=gen_prompt,
        system=gen_system,
        history=gen_history,
        model=model,
        temperature=0.3,
        max_tokens=8000,
    )
    logger.info("[%s] generator (%s): %d chars (%.1fs)", session_id[:8], model, len(response), time.monotonic() - t_gen)

    # Citation verification
    response, cite_meta = verify_citations(response, chunks)
    if cite_meta.get("unverified"):
        logger.warning("[%s] Unverified citations: %s", session_id[:8], cite_meta["unverified"])

    # Save assistant response
    db.add_message(session_id, turn, "assistant", response, sources)

    # Auto-generate title for new sessions
    if turn == 1:
        await _generate_title(session_id, user_message, response)

    logger.info("[%s] total turn time: %.1fs", session_id[:8], time.monotonic() - t_start)

    return {"answer": response, "sources": sources, "session_id": session_id}


async def _generate_title(session_id: str, question: str, answer: str) -> None:
    """Auto-generate a session title from first exchange."""
    try:
        prompt = f"""Tao tieu de ngan gon (duoi 50 ky tu) cho cuoc hoi thoai bat dau voi cau hoi sau.
Chi tra ve tieu de, khong giai thich, khong dau ngoac kep.

Cau hoi: {question[:200]}

Tieu de:"""
        title = await llm.generate(prompt, model="utility", temperature=0.0, max_tokens=60)
        if title and len(title.strip()) > 3:
            db.update_session_title(session_id, title.strip()[:80])
    except Exception:
        pass


def create_new_session() -> str:
    """Create a new chat session and return its ID."""
    session_id = str(uuid.uuid4())
    db.create_session(session_id)
    return session_id
```

**Step 2: Verify import**

Run: `.venv/bin/python -c "from app.chat import handle_message, create_new_session; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/chat.py
git commit -m "chat: rewrite with orchestrator-first agentic pipeline

- Orchestrator (Flash 3) decides tool calls per turn
- Generator (Pro 3.1/Flash 3) produces answer with retrieved context
- Direct response path for greetings/follow-ups (no retrieval)
- Rolling summaries as orchestrator byproduct
- Parallel tool execution
- XML-tagged system prompts with static prefix"
```

---

### Task 10: Update main.py to remove stale imports

**Files:**
- Modify: `app/main.py`

**Step 1: Check main.py**

The current `main.py` imports `chat` and `db` which are still valid. No changes needed unless the import of `chat.handle_message` signature changed -- but it hasn't (still takes `session_id, user_message`, returns dict). Verify it works.

Run: `.venv/bin/python -c "from app.main import app; print('FastAPI app OK')"`
Expected: `FastAPI app OK`

If it works, skip this task (no commit needed).

---

### Task 11: Update frontend -- localStorage session persistence

**Files:**
- Modify: `app/static/index.html`

**Step 1: Add localStorage persistence**

Add these changes to the `<script>` section:

At the top of the script (after variable declarations), add session restoration:

```javascript
// Restore last session from localStorage
const savedSessionId = localStorage.getItem('datdai_session_id');
if (savedSessionId) {
  // Will be loaded after sessions list is fetched
  currentSessionId = savedSessionId;
}
```

In `handleSubmit`, after setting `currentSessionId`:

```javascript
// Set session ID from server response
if (!currentSessionId && data.session_id) {
  currentSessionId = data.session_id;
  localStorage.setItem('datdai_session_id', currentSessionId);
}
```

In `switchSession`:

```javascript
async function switchSession(id, title) {
  currentSessionId = id;
  localStorage.setItem('datdai_session_id', id);
  // ... rest unchanged
```

In `newChat`:

```javascript
async function newChat() {
  currentSessionId = null;
  localStorage.removeItem('datdai_session_id');
  // ... rest unchanged
```

In the `// ===== Init =====` section at the bottom, update to restore session:

```javascript
// ===== Init =====
loadSessions().then(() => {
  if (currentSessionId) {
    // Restore last session
    const title = document.querySelector(`.session-item.active`)?.textContent || 'Cuoc hoi thoai';
    switchSession(currentSessionId, title);
  }
});
```

**Step 2: Test manually**

Open the app in browser, start a conversation, refresh the page. Should resume on the same session.

**Step 3: Commit**

```bash
git add app/static/index.html
git commit -m "frontend: persist session ID in localStorage, restore on refresh"
```

---

### Task 12: Integration test -- verify full pipeline

**Files:**
- Create: `tests/test_orchestrator.py`

**Step 1: Write integration test**

```python
"""Tests for the orchestrator pipeline logic (no LLM calls)."""
import json
import pytest
from app.chat import (
    _parse_orchestrator_response,
    _assemble_conversation_context,
    SYSTEM_BASE,
    ORCHESTRATOR_TOOLS_SECTION,
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

    def test_static_prefix_ordering(self):
        # identity should come before tools, tools before instructions
        id_pos = SYSTEM_BASE.index("<identity>")
        bound_pos = SYSTEM_BASE.index("<boundaries>")
        hier_pos = SYSTEM_BASE.index("<legal-hierarchy>")
        assert id_pos < bound_pos < hier_pos
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_orchestrator.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_orchestrator.py
git commit -m "tests: add orchestrator parsing and prompt structure tests"
```

---

### Task 13: Smoke test with live LLM (manual)

**Step 1: Start the server**

```bash
cd /Users/vuducdung/personal/DatDai && .venv/bin/uvicorn app.main:app --reload --port 8000
```

**Step 2: Test scenarios via curl**

Test 1 -- Greeting (should use direct_response, no retrieval):
```bash
curl -s http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "Xin chao"}' | python3 -m json.tool
```
Expected: response with empty sources, fast response time.

Test 2 -- Specific legal question (should trigger search):
```bash
curl -s http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "Dieu 79 Luat Dat Dai 2024 quy dinh gi?", "session_id": "test-session-1"}' | python3 -m json.tool
```
Expected: response with sources citing Dieu 79 LDD 2024.

Test 3 -- Follow-up (should use existing context or targeted lookup):
```bash
curl -s http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "Con dieu 80 thi sao?", "session_id": "test-session-1"}' | python3 -m json.tool
```
Expected: response about Dieu 80 with appropriate sources. The orchestrator should craft a specific query rather than a pronoun-filled one.

Test 4 -- Amendment query (should trigger lookup_amendment):
```bash
curl -s http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "ND 49 sua doi gi cua ND 102?"}' | python3 -m json.tool
```
Expected: response with sources from ND 49 and ND 102.

**Step 3: Check server logs**

Look for orchestrator decisions in logs:
- `orchestrator: complexity=... actions=... direct=...`
- `tools returned N chunks`
- `generator (pro/flash): ...`

Verify that greetings don't trigger tool calls and legal questions do.

**Step 4: Commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: smoke test adjustments for agentic pipeline"
```

---

## Summary of Changes

| Task | File | Action |
|------|------|--------|
| 1 | app/config.py | Edit -- new models, simplified flags |
| 2 | app/llm.py | Rewrite -- fallback routing, remove classify_complexity |
| 3 | app/db.py | Edit -- add upsert_summary |
| 4 | app/rag/query_rewriter.py | Simplify -- keep extract_filters only |
| 5 | app/rag/hyde.py, query_decomposer.py | Delete |
| 6 | app/rag/cross_ref.py | Simplify -- data maps only |
| 7 | app/rag/retriever.py | Rewrite -- discrete tool functions |
| 8 | app/rag/__init__.py | Edit -- expose new functions |
| 9 | app/chat.py | Rewrite -- orchestrator pipeline |
| 10 | app/main.py | Verify (likely no changes) |
| 11 | app/static/index.html | Edit -- localStorage persistence |
| 12 | tests/test_orchestrator.py | Create -- unit tests |
| 13 | Manual | Smoke test with live LLM |
