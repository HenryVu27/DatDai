"""Cross-encoder reranker using FastEmbed ONNX inference."""
import asyncio
import logging
import math

from app.observability import observe

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self._model = TextCrossEncoder(model_name=model_name)
        logger.info("CrossEncoderReranker loaded (model=%s)", model_name)

    @observe(name="rerank")
    async def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        documents = [c.get("text", c.get("content", "")) for c in candidates]
        raw_scores = await asyncio.to_thread(
            lambda: list(self._model.rerank(query, documents))
        )
        scores = [1 / (1 + math.exp(-s)) for s in raw_scores]
        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [{**c, "score": s} for s, c in scored[:top_k]]
