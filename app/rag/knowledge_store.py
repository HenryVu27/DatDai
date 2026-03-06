"""Qdrant client wrapper for search operations at runtime."""
import logging
import re
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, Fusion, FusionQuery,
    MatchAny, MatchValue, Prefetch, SparseVector,
)

from app.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from app.storage import load_vocab

logger = logging.getLogger(__name__)


class KnowledgeStore:
    def __init__(self):
        self._client: QdrantClient | None = None
        self._vocab = load_vocab()

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            if QDRANT_URL and QDRANT_URL != ":memory:":
                self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
            else:
                self._client = QdrantClient(location=":memory:")
        return self._client

    @property
    def has_sparse(self) -> bool:
        return len(self._vocab) > 0

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(
            r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
            " ", text,
        )
        return [w for w in text.split() if len(w) > 1]

    def _text_to_sparse(self, text: str) -> SparseVector:
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        indices, values = [], []
        for token, count in sorted(counts.items()):
            if token in self._vocab:
                indices.append(self._vocab[token])
                values.append(float(count))
        return SparseVector(indices=indices, values=values)

    def search_hybrid(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 20,
        doc_ids: list[str] | None = None,
        dieu: str | None = None,
    ) -> list[dict]:
        """Hybrid dense+sparse search with RRF fusion.
        Returns list of payload dicts with 'score' key added.
        """
        client = self._get_client()
        try:
            info = client.get_collection(QDRANT_COLLECTION)
            if info.points_count == 0:
                return []
        except Exception:
            return []

        qdrant_filter = self._build_filter(doc_ids=doc_ids, dieu=dieu)
        sparse_vec = self._text_to_sparse(query_text)
        prefetch_limit = min(top_k * 3, info.points_count)

        prefetches = [
            Prefetch(query=query_vector, using="dense", limit=prefetch_limit),
        ]
        if sparse_vec.indices:
            prefetches.append(
                Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit),
            )

        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=prefetches,
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        ).points

        return [{**point.payload, "score": point.score} for point in results]

    def fetch_by_metadata(self, doc_id: str = None, dieu: str = None, limit: int = 5) -> list[dict]:
        """Fetch chunks by exact metadata match (no embedding needed)."""
        client = self._get_client()
        try:
            info = client.get_collection(QDRANT_COLLECTION)
            if info.points_count == 0:
                return []
        except Exception:
            return []

        conditions = []
        if doc_id:
            conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if dieu:
            conditions.append(FieldCondition(key="dieu", match=MatchValue(value=dieu)))
        if not conditions:
            return []

        results = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(must=conditions),
            limit=limit,
            with_payload=True,
        )[0]

        return [{**point.payload, "score": 0.0} for point in results]

    def fetch_full_dieu(self, doc_id: str, dieu: str, limit: int = 20) -> list[dict]:
        """Fetch ALL chunks belonging to the same (doc_id, dieu).

        Used to expand a high-scoring chunk to its full article context.
        A single Dieu may be split into multiple chunks (by Khoan).
        """
        client = self._get_client()
        try:
            info = client.get_collection(QDRANT_COLLECTION)
            if info.points_count == 0:
                return []
        except Exception:
            return []

        conditions = [
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            FieldCondition(key="dieu", match=MatchValue(value=dieu)),
        ]

        results = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(must=conditions),
            limit=limit,
            with_payload=True,
        )[0]

        return [{**point.payload, "score": 0.0} for point in results]

    def fetch_by_chunk_ids(self, chunk_ids: list[str], limit: int = 50) -> list[dict]:
        """Fetch chunks by their chunk_id field using Qdrant scroll with MatchAny."""
        if not chunk_ids:
            return []
        client = self._get_client()
        try:
            info = client.get_collection(QDRANT_COLLECTION)
            if info.points_count == 0:
                return []
        except Exception:
            return []

        results = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids)),
            ]),
            limit=limit,
            with_payload=True,
        )[0]

        return [{**point.payload, "score": 0.0} for point in results]

    def _build_filter(self, doc_ids=None, dieu=None) -> Filter | None:
        conditions = []
        if doc_ids:
            conditions.append(FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)))
        if dieu:
            conditions.append(FieldCondition(key="dieu", match=MatchValue(value=dieu)))
        return Filter(must=conditions) if conditions else None
