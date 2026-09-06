# DatDai

A question-answering assistant over Vietnamese land law, deployed on Google Cloud Run.

Ask a question in Vietnamese and get an answer grounded in the actual statute text, with citations back to the specific Điều and Khoản it came from. The corpus is Luật Đất Đai 2024 plus the eleven nghị định and nghị quyết that implement and amend it.

## Why it is not a plain RAG chatbot

Legal documents amend each other. A naive retriever will happily quote a clause that a later decree replaced, and the answer looks perfectly confident. Two pieces of the system exist to stop that:

**An amendment graph.** `app/rag/cross_ref.py` encodes which decree amends which and which implements which. `app/rag/amendment_index.py` detects amendment-shaped questions and resolves the relevant chunks across documents, so a question about a clause pulls in the decrees that changed it.

**A citation verifier.** `app/rag/citation_check.py` extracts every legal citation from the model's answer (Điều X, Khoản Y Điều X, Điểm Z Khoản Y Điều X, in both Unicode and ASCII spellings), checks each one against the chunks that were actually retrieved, and appends a disclaimer when any citation cannot be verified. It costs zero extra model calls.

## Architecture

```
question
  → query rewriter        regex filter extraction + LLM rewrite, validated
  → hybrid retrieval      Qdrant dense + sparse, RRF fusion
  → rerank                Gemini Flash scores passages 0.0-1.0
  → amendment expansion   pulls cross-referenced chunks when relevant
  → generation            model tier chosen by question complexity
  → citation check        verifies every Điều cited against retrieved chunks
  → answer + sources
```

Conversation memory uses rolling summaries rather than replaying full history. Every stage is traced.

| Component | Choice |
|---|---|
| API | FastAPI, SSE streaming, slowapi rate limiting |
| Models | Gemini 3 Flash / 3.1 Pro / 3.1 Flash Lite, with Gemini 2.5 fallbacks |
| Embeddings | `gemini-embedding-001` |
| Vector store | Qdrant, dense + sparse with RRF |
| Persistence | Supabase Postgres, connection-pooled |
| Tracing | Langfuse |
| Hosting | Cloud Run, deployed by GitHub Actions |

## Evaluation

`scripts/evaluate_rag.py` pulls recent traces out of Langfuse, scores each one with Gemini as judge on groundedness, relevance and completeness, and writes the scores back onto the trace. It skips traces that already carry a score, so it is safe to re-run.

This evaluates the system on real questions people actually asked, rather than on a fixed test set that stops being representative.

## Deployment

`.github/workflows/deploy.yml` builds the image, pushes it to Artifact Registry pinned to the commit SHA, and deploys to Cloud Run. Authentication uses Workload Identity Federation, so there is no long-lived service account key. Every secret comes from Secret Manager. The service has a startup probe against `/health` and runs with CPU boost.

Before launch I audited the service against itself and wrote up the findings in [`docs/plans/2026-03-06-deployment-audit-report.md`](docs/plans/2026-03-06-deployment-audit-report.md). It found, among other things, that a hung primary model would block for its full timeout before the fallback was even attempted, doubling the worst case, and that a single request opened six to eight fresh Postgres connections. Both are fixed.

## Known gaps

- No test or eval threshold gates the deploy. The workflow goes straight from checkout to build.
- `QDRANT_URL` still defaults to `:memory:`, so a missing environment variable starts a healthy-looking service that returns nothing. This was flagged in the audit and is not yet fixed.
- Auth is designed (`docs/plans/2026-03-06-auth-design.md`) but not built. The service is public with IP-based rate limiting only.
- The corpus is built once by `scripts/build_index.py` from static markdown. There is no incremental ingestion, which matters for a body of law that keeps being amended.

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY
python scripts/build_index.py
uvicorn app.main:app --reload
```

Then open http://localhost:8000.
