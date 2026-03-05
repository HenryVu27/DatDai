"""RAG package -- exposes tool functions for the orchestrator."""
from app.rag.retriever import (
    search_legal_docs,
    lookup_amendment,
    lookup_specific_dieu,
    build_context,
    extract_sources,
)
