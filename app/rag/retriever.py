"""Main retrieval pipeline orchestrator."""
import logging
from app import llm
from app.config import (
    RAG_TOP_K, RAG_TOP_K_COMPLEX, RAG_RERANK_CANDIDATES, RAG_RERANKER_MODEL,
    RAG_RELEVANCE_THRESHOLD, RAG_USE_HYDE, RAG_USE_RERANKER, RAG_USE_QUERY_REWRITE,
)
from app.rag.knowledge_store import KnowledgeStore
from app.rag.query_rewriter import extract_filters, rewrite_query
from app.rag.query_decomposer import decompose_query
from app.rag.amendment_index import detect_amendment_query, get_amendment_chunk_ids, get_all_amendment_chunk_ids_to
from app.rag.reranker import CrossEncoderReranker
from app.rag.cross_ref import expand_with_cross_refs
from app.rag import hyde

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self):
        self._store = KnowledgeStore()
        self._reranker: CrossEncoderReranker | None = None
        if RAG_USE_RERANKER:
            try:
                self._reranker = CrossEncoderReranker(model_name=RAG_RERANKER_MODEL)
            except Exception as e:
                logger.warning("Reranker init failed, running without: %s", e)

    async def retrieve(self, query: str, history: list[dict] | None = None, top_k: int = RAG_TOP_K) -> dict:
        # Step 1: Extract filters from original query
        filters = extract_filters(query)

        # Step 1b: Adaptive top_k for complex/multi-doc queries
        doc_count = len(filters.get("doc_ids") or [])
        if doc_count >= 2 or top_k > RAG_TOP_K:
            top_k = max(top_k, RAG_TOP_K_COMPLEX)

        # Step 2: Rewrite query (resolve pronouns from conversation)
        search_query = query
        rewritten = None
        if RAG_USE_QUERY_REWRITE:
            search_query = await rewrite_query(query, history)
            if search_query != query:
                rewritten = search_query

        # Step 3: Decompose into sub-queries
        sub_queries = await decompose_query(search_query)

        # Step 4: Retrieve for each sub-query + merge
        all_candidates = []
        seen_ids = set()

        for sq in sub_queries:
            sq_filters = extract_filters(sq)
            # Merge filters: use sub-query filters if present, else original
            effective_doc_ids = sq_filters.get("doc_ids") or filters.get("doc_ids")
            effective_dieu = sq_filters.get("dieu") or filters.get("dieu")

            # Get embedding (HyDE or direct)
            query_embedding = None
            if RAG_USE_HYDE:
                query_embedding = await hyde.get_hyde_embedding(sq)
            if query_embedding is None:
                query_embedding = llm.embed([sq])[0]

            # Hybrid search
            fetch_k = RAG_RERANK_CANDIDATES if self._reranker else top_k
            candidates = self._store.search_hybrid(
                query_vector=query_embedding,
                query_text=sq,
                top_k=fetch_k,
                doc_ids=effective_doc_ids,
                dieu=effective_dieu,
            )

            # Retry without filters if empty
            if not candidates and (effective_doc_ids or effective_dieu):
                candidates = self._store.search_hybrid(
                    query_vector=query_embedding, query_text=sq, top_k=fetch_k,
                )

            # Deduplicate across sub-queries
            for c in candidates:
                cid = c.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_candidates.append(c)

        # Step 5: Amendment index lookup (deterministic, no embedding)
        amendment_result = detect_amendment_query(search_query)
        if amendment_result:
            src_doc, tgt_doc = amendment_result
            if src_doc:
                chunk_ids = get_amendment_chunk_ids(src_doc, tgt_doc)
            else:
                chunk_ids = get_all_amendment_chunk_ids_to(tgt_doc)

            if chunk_ids:
                # Fetch these chunks by ID via metadata
                for cid in chunk_ids:
                    if cid not in seen_ids:
                        # We don't have a fetch-by-chunk-id method, so we'll
                        # rely on the cross-ref expansion to cover these.
                        # Instead, log what the index found for debugging.
                        pass
                logger.info("Amendment index found %d chunk_ids for %s->%s",
                            len(chunk_ids), src_doc or "all", tgt_doc)

        if not all_candidates:
            return {"chunks": [], "sources": [], "rewritten_query": rewritten,
                    "filters_used": filters, "sub_queries": sub_queries}

        # Step 6: Rerank merged candidates
        if self._reranker and len(all_candidates) > top_k:
            try:
                all_candidates = await self._reranker.rerank(search_query, all_candidates, top_k * 2)
            except Exception as e:
                logger.warning("Reranker failed: %s", e)

        # Step 7: Relevance threshold
        if self._reranker:
            all_candidates = [c for c in all_candidates if c["score"] >= RAG_RELEVANCE_THRESHOLD]

        # Step 8: Trim
        chunks = all_candidates[:top_k]

        # Step 9: Full-Dieu expansion (fetch sibling chunks for top-scoring articles)
        pre_dieu = len(chunks)
        chunks = self._expand_full_dieu(chunks, seen_ids)

        # Step 10: Cross-reference expansion
        pre_expand = len(chunks)
        try:
            chunks = await expand_with_cross_refs(chunks, self._store, top_k)
        except Exception as e:
            logger.warning("Cross-ref expansion failed: %s", e)

        sources = _extract_sources(chunks)

        logger.info(
            "RAG result: query=%r rewritten=%r sub_queries=%d filters=%s "
            "candidates=%d reranked=%d dieu_expanded=%d->%d xref_expanded=%d->%d sources=%d",
            query[:80], rewritten, len(sub_queries), filters,
            len(all_candidates) if all_candidates else 0,
            top_k, pre_dieu, pre_expand - (pre_expand - len(chunks)),
            pre_expand, len(chunks), len(sources),
        )

        return {"chunks": chunks, "sources": sources, "rewritten_query": rewritten,
                "filters_used": filters, "sub_queries": sub_queries}

    def _expand_full_dieu(self, chunks: list[dict], seen_ids: set) -> list[dict]:
        """For top-scoring chunks, fetch all sibling chunks from the same Dieu."""
        expansion = []
        # Only expand top 3 chunks to avoid over-fetching
        for chunk in chunks[:3]:
            doc_id = chunk.get("doc_id", "")
            dieu = chunk.get("dieu", "")
            if not doc_id or not dieu:
                continue
            siblings = self._store.fetch_full_dieu(doc_id, dieu)
            for sib in siblings:
                cid = sib.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    expansion.append(sib)

        if expansion:
            logger.info("Full-Dieu expansion added %d sibling chunks", len(expansion))

        return list(chunks) + expansion


def _extract_sources(chunks: list[dict]) -> list[dict]:
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


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata_str", "")
        content = chunk.get("text", chunk.get("content", ""))
        parts.append(f"[Nguon {i}: {source}]\n{content}")
    return "\n\n---\n\n".join(parts)
