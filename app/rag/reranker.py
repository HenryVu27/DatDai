"""LLM-based reranker using Gemini Flash API."""
import json
import logging
import re

from app.observability import observe

logger = logging.getLogger(__name__)

_RERANK_PROMPT = """\
You are a relevance scoring engine for Vietnamese legal documents.

Given a user query and a list of candidate text passages, score each passage's relevance to the query on a scale from 0.0 to 1.0.

Respond ONLY with a JSON array of numbers (the scores), in the same order as the passages. Example: [0.9, 0.2, 0.7]

User query: {query}

Passages:
{passages}"""


class CrossEncoderReranker:
    def __init__(self, **kwargs):
        logger.info("Gemini Flash reranker initialized")

    @observe(name="rerank")
    async def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []

        from app import llm

        documents = [c.get("text", c.get("content", "")) for c in candidates]
        passages = "\n\n".join(
            f"[{i}] {doc[:500]}" for i, doc in enumerate(documents)
        )

        prompt = _RERANK_PROMPT.format(query=query, passages=passages)

        try:
            response = await llm.generate(
                prompt=prompt, model="utility", temperature=0.0, max_tokens=256,
            )
            scores = self._parse_scores(response, len(candidates))
        except Exception as e:
            logger.warning("Gemini reranker failed: %s", e)
            return candidates[:top_k]

        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [{**c, "score": s} for s, c in scored[:top_k]]

    @staticmethod
    def _parse_scores(response: str, expected: int) -> list[float]:
        match = re.search(r"\[[\d\s.,]+\]", response)
        if not match:
            raise ValueError(f"Could not parse scores from: {response[:200]}")
        scores = json.loads(match.group())
        if len(scores) != expected:
            raise ValueError(f"Expected {expected} scores, got {len(scores)}")
        return [max(0.0, min(1.0, float(s))) for s in scores]
