"""RAG tool functions for the orchestrator.

Each function is a discrete tool the orchestrator can invoke.
Results are merged and reranked in the orchestrator pipeline.
"""
import logging
from app import llm
from app.config import RAG_RERANK_CANDIDATES, RAG_RELEVANCE_THRESHOLD, RAG_USE_RERANKER, RAG_TOP_K, RAG_MAX_EXPANSION_CHUNKS
from app.rag.knowledge_store import KnowledgeStore
from app.rag.reranker import CrossEncoderReranker
from app.rag.cross_ref import AMENDED_BY
from app.rag.amendment_index import get_amendment_chunk_ids, get_all_amendment_chunk_ids_to
from app.observability import observe

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
            _reranker = CrossEncoderReranker()
        except Exception as e:
            logger.warning("Reranker init failed: %s", e)
    return _reranker


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
        logger.warning(
            "Filter produced 0 results (doc_ids=%s, dieu=%s) — retrying without filters. "
            "Results may be from a different document.",
            doc_ids, dieu,
        )
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

    # Trim and expand full Dieu for top results
    chunks = candidates[:top_k]
    chunks = _expand_full_dieu(chunks, store)

    return chunks


@observe(name="lookup_amendment")
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


@observe(name="lookup_specific_dieu")
def lookup_specific_dieu(doc_id: str, dieu: str) -> list[dict]:
    """Fetch all chunks for a specific Dieu from a specific document.

    Uses metadata lookup (no embedding needed). Returns all Khoan sub-chunks.
    """
    store = _get_store()
    return store.fetch_full_dieu(doc_id, dieu)


def _expand_full_dieu(chunks: list[dict], store: KnowledgeStore) -> list[dict]:
    """For top-scoring chunks, fetch sibling chunks from same Dieu. Capped at RAG_MAX_EXPANSION_CHUNKS."""
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
