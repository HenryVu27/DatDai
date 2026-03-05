"""
RAG engine with hybrid retrieval for Vietnamese legal documents.
Dense (Gemini embeddings) + Sparse (TF) + Qdrant native RRF fusion.
"""
import json
import os
import re
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector, Prefetch, FusionQuery, Fusion,
)

from app.config import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    RAG_TOP_K, RAG_RERANK_CANDIDATES, PROJECT_ROOT,
)
from app import llm

_client = None
_vocab = {}

VOCAB_FILE = os.path.join(PROJECT_ROOT, "data", "vocab.json")


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_URL and QDRANT_URL != ":memory:":
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        else:
            _client = QdrantClient(location=":memory:")
    return _client


def _load_vocab() -> dict:
    global _vocab
    if not _vocab and os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            _vocab = json.load(f)
    return _vocab


def _tokenize_vi(text: str) -> list[str]:
    """Simple Vietnamese tokenizer -- splits on whitespace and punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", " ", text)
    return [w for w in text.split() if len(w) > 1]


def text_to_sparse(text: str, vocab: dict | None = None) -> SparseVector:
    """Convert text to sparse vector using term frequency."""
    v = vocab or _load_vocab()
    if not v:
        return SparseVector(indices=[], values=[])
    tokens = _tokenize_vi(text)
    counts = Counter(tokens)
    indices = []
    values = []
    for token, count in sorted(counts.items()):
        if token in v:
            indices.append(v[token])
            values.append(float(count))
    return SparseVector(indices=indices, values=values)


def _metadata_boost(query: str, metadata: dict) -> float:
    """Boost score based on metadata match."""
    boost = 0.0
    query_lower = query.lower()

    dieu_matches = re.findall(r"(?:điều|dieu)\s*(\d+)", query_lower)
    for dieu_num in dieu_matches:
        if f"Điều {dieu_num}" in metadata.get("dieu", ""):
            boost += 0.3

    doc_name = metadata.get("doc_name", "").lower()
    if "luật đất đai" in query_lower or "luat dat dai" in query_lower:
        if "luat dat dai" in doc_name:
            boost += 0.15
    for n in ["71", "88", "102", "103", "151", "226", "49"]:
        if f"nghị định {n}" in query_lower or f"nghi dinh {n}" in query_lower:
            if n in doc_name:
                boost += 0.2

    return boost


async def rewrite_query(query: str, history: list[dict] | None = None) -> str:
    """Rewrite query with conversation context for better retrieval."""
    if not history or len(history) < 2:
        return query

    recent = history[-6:]
    context = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in recent)

    prompt = f"""Viet lai cau hoi sau de ro rang hon, giai quyet dai tu va tham chieu ngam.
Giu nguyen y dinh goc. Tra ve CHI cau hoi da viet lai, khong giai thich.

Lich su hoi thoai gan day:
{context}

Cau hoi hien tai: {query}

Cau hoi da viet lai:"""

    try:
        rewritten = await llm.generate(prompt, model="flash", temperature=0.0, max_tokens=200)
        rewritten = rewritten.strip().strip('"').strip("'")
        if 10 < len(rewritten) < 500:
            return rewritten
    except Exception:
        pass
    return query


def search(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """Hybrid search using Qdrant native RRF fusion."""
    client = get_client()

    try:
        collection_info = client.get_collection(QDRANT_COLLECTION)
        if collection_info.points_count == 0:
            return []
    except Exception:
        return []

    # Dense query vector
    query_embedding = llm.embed([query])[0]

    # Sparse query vector
    sparse_vec = text_to_sparse(query)

    prefetch_limit = min(RAG_RERANK_CANDIDATES, collection_info.points_count)

    # Build prefetch list
    prefetches = [
        Prefetch(
            query=query_embedding,
            using="dense",
            limit=prefetch_limit,
        ),
    ]

    # Only add sparse prefetch if we have a vocabulary
    if sparse_vec.indices:
        prefetches.append(
            Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=prefetch_limit,
            ),
        )

    # Qdrant native RRF fusion
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=prefetches,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k * 2,  # Get more for metadata boosting
        with_payload=True,
    ).points

    if not results:
        return []

    # Apply metadata boost and re-rank
    candidates = []
    for point in results:
        payload = point.payload
        boost = _metadata_boost(query, payload)
        candidates.append({
            "content": payload.get("text", ""),
            "metadata": {
                "doc_name": payload.get("doc_name", ""),
                "chapter": payload.get("chapter", ""),
                "dieu": payload.get("dieu", ""),
                "dieu_title": payload.get("dieu_title", ""),
                "metadata_str": payload.get("metadata_str", ""),
            },
            "score": point.score + boost,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]


def build_context(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("metadata_str", "")
        parts.append(f"[Nguon {i}: {source}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def extract_sources(chunks: list[dict]) -> list[dict]:
    """Extract unique source references from chunks."""
    sources = []
    seen = set()
    for chunk in chunks:
        meta = chunk["metadata"]
        key = f"{meta.get('doc_name', '')}|{meta.get('dieu', '')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "doc_name": meta.get("doc_name", ""),
                "chapter": meta.get("chapter", ""),
                "dieu": meta.get("dieu", ""),
                "dieu_title": meta.get("dieu_title", ""),
            })
    return sources
