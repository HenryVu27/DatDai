# RAG Pipeline Improvements for Vietnamese Land Law

## Problem
Single-pass retrieval (8 chunks) cannot cover broad multi-document questions.
LLM hallucinates article citations when source text is insufficient.

## 5 Independent Improvements

### 1. Query Decomposition (`app/rag/query_decomposer.py`)
- Gemini Flash splits complex questions into 2-4 sub-queries
- Each sub-query gets its own retrieval pass
- Merge + deduplicate chunks before reranking
- Query rewrite (pronoun resolution) happens BEFORE decomposition
- Static document list + amendment hierarchy injected into prompt

### 2. Structured Amendment Index (`data/amendment_index.json` + `app/rag/amendment_index.py`)
- Built at chunk time: parse amendment decrees for Dieu-level modifications
- Runtime: deterministic lookup for "what changed" questions
- Keyed by (amending_doc, amended_doc) -> list of Dieu references

### 3. Full-Dieu Retrieval (modify `knowledge_store.py`)
- When chunk scores high, fetch ALL sibling chunks with same (doc_id, dieu)
- New `fetch_full_dieu()` method on KnowledgeStore
- Ensures LLM sees complete articles, not fragments

### 4. Citation Verification (`app/rag/citation_check.py`)
- Regex-extract "Dieu X Khoan Y" from LLM response
- Verify each citation exists in context chunks
- Append disclaimer for unverifiable citations

### 5. Adaptive top_k (modify `retriever.py` + `chat.py`)
- Complex/multi-doc: top_k=16; simple: top_k=8
- Based on complexity classifier + count of detected doc_ids

## Pipeline Flow
```
Query rewrite -> Decompose -> Per-sub-query retrieval -> Merge ->
Full-Dieu expand -> Cross-ref expand -> Rerank -> LLM -> Citation check
```
