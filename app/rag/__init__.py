"""RAG package public API."""
from app.rag.retriever import Retriever, build_context

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


async def search(query: str, history: list[dict] | None = None, top_k: int = 8) -> dict:
    return await get_retriever().retrieve(query, history=history, top_k=top_k)
