# Observability & RAG Evaluation Design

## Overview

Add Langfuse-based observability to the DatDai chatbot for operational monitoring and RAG quality evaluation.

## Architecture

```
User request --> FastAPI /chat endpoint
                    |
                    v
            Langfuse @observe (root trace)
                    |
        +-----------+-----------+------- ... ------+
        |           |           |                   |
   query_rewrite  hyde    retrieval+rerank    llm_generation
   (span)        (span)   (span)              (generation)
        |           |           |                   |
        v           v           v                   v
    All inputs/outputs/latencies sent to Langfuse Cloud
                    |
                    v
            Langfuse Dashboard
            - Trace viewer (per-request breakdown)
            - Latency charts
            - Token usage
            - Auto heuristic scores
            - Manual/batch LLM evaluation
```

## Platform

Langfuse Cloud (free tier: 50k observations/month, ~5-6k full requests/month).

## Integration Approach

Decorate each RAG pipeline function with `@observe()`. Langfuse automatically captures inputs, outputs, latency, and nests them into a trace tree. For LLM calls, use `@observe(as_type="generation")` to additionally track model name, token counts, and cost.

### Files to modify

- `app/chat.py` -- root trace on `handle_message`
- `app/rag/query_rewriter.py` -- span for query rewriting
- `app/rag/hyde.py` -- span for HyDE generation
- `app/rag/retriever.py` -- span for retrieval + reranking
- `app/rag/cross_ref.py` -- span for cross-reference expansion
- `app/rag/reranker.py` -- span for reranking detail
- `app/llm.py` -- generation span for all Gemini calls
- `app/config.py` -- add Langfuse env vars
- `app/main.py` -- add feedback endpoint

### New files

- `scripts/evaluate_rag.py` -- batch LLM-as-judge evaluation script

## Tracing Scope (all 9 components)

1. Full request trace -- total latency
2. Query rewriting -- input query, rewritten query, latency
3. HyDE -- hypothetical passage generated, latency
4. Retrieval -- search query, results count, doc IDs, scores, latency
5. Reranking -- pre/post ordering, scores, latency
6. Cross-reference expansion -- amendment chunks fetched
7. LLM generation -- prompt, response, model, token counts, latency
8. Model routing decision -- which model selected
9. User feedback -- thumbs up/down from frontend

## Automatic Heuristic Scores (per request, zero LLM cost)

Logged as Langfuse scores on each trace:
- `retrieval_count` -- number of chunks returned
- `reranker_top_score` -- highest reranker confidence
- `reranker_mean_score` -- average reranker score
- `response_length` -- character count of answer
- `model_used` -- pro or flash
- `has_sources` -- whether response cited sources

## Batch LLM Evaluation (manual)

Script `scripts/evaluate_rag.py`:
1. Pulls recent traces from Langfuse API
2. Runs Gemini as judge (groundedness, relevance, completeness)
3. Posts scores back to Langfuse

Run periodically, not per-request.

## User Feedback

New `/feedback` POST endpoint. Frontend adds thumbs up/down buttons below each AI response. Sends trace ID + score to Langfuse.

## Secrets

New env vars:
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`

Added to: `.env` (local), GCP Secret Manager, Cloud Run `--set-secrets`.

Langfuse host defaults to `https://cloud.langfuse.com`.

## Cost

Free tier covers the expected traffic. No additional infrastructure needed.
