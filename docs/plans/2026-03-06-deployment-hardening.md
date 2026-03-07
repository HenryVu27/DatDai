# Deployment Hardening -- Issues & Fixes

## Context

After deploying to Cloud Run + Supabase, a deep audit revealed several production-readiness issues. The reranker model download problem (HuggingFace 429 rate limits causing 40s+ retry loops) was already fixed by pre-caching in the Docker image. This document covers the remaining issues.

## Architecture Reference

- Compute: GCP Cloud Run (us-central1, 1GB RAM, 1 vCPU, concurrency 10)
- Chat DB: Supabase PostgreSQL
- Reference data: Supabase Storage (vocab.json, amendment_index.json)
- Vector DB: Qdrant Cloud
- LLM: Gemini API (3.x-preview primary, 2.5 fallback)
- Reranker: jinaai/jina-reranker-v2-base-multilingual (ONNX, ~500MB in Docker image)

---

## Issue 1: No timeouts on external API calls

**Severity:** HIGH
**Impact:** A single slow/hung network call blocks the entire request indefinitely. Cloud Run's 120s timeout is the only backstop, but by then the user has already given up.

### 1a: Gemini API calls have no timeout

**File:** `app/llm.py` -- `generate()` function
**Problem:** `client.models.generate_content()` has no timeout parameter. If Gemini is slow or unresponsive, the request hangs.
**Fix:** Pass `timeout` parameter to the Gemini client or wrap calls with `asyncio.wait_for()`. Target: 45s for pro model, 30s for flash.

### 1b: Qdrant queries have no timeout

**File:** `app/rag/knowledge_store.py` -- `search_hybrid()`, `fetch_by_chunk_ids()`, `fetch_by_metadata()`, `fetch_full_dieu()`
**Problem:** `client.query_points()` and `client.scroll()` have no timeout. Qdrant Cloud network issues = hung request.
**Fix:** Set `timeout` on QdrantClient initialization (e.g., `QdrantClient(url=..., timeout=15)`).

### 1c: Supabase Storage downloads have no timeout

**File:** `app/storage.py` -- `_load_json()`
**Problem:** `storage.from_(_BUCKET).download(remote_path)` has no timeout.
**Fix:** Set timeout on the Supabase client or use httpx with explicit timeout. Less critical since files are cached after first load.

---

## Issue 2: No database connection pooling

**Severity:** HIGH
**Impact:** Every DB function call creates a new PostgreSQL connection and closes it after. With concurrency=10, this means up to 10+ new connections per second during load. Supabase PgBouncer helps, but the app-side overhead of connect/close per call is wasteful and slow.

**File:** `app/db.py` -- `_get_pg()` called by every function
**Problem:** Each of the 8 public functions (create_session, add_message, get_messages, etc.) calls `_get_db()` which calls `psycopg2.connect()` every time.
**Fix:** Use `psycopg2.pool.ThreadedConnectionPool` with min=1, max=5 connections. Get/put connections from pool instead of connect/close.

---

## Issue 3: No health check endpoint

**Severity:** MEDIUM
**Impact:** Cloud Run routes traffic as soon as the container starts listening. The first user request triggers cold start initialization (reranker model loading into memory, Qdrant connection, Supabase connection). This means the first request after scale-from-zero takes 10-15s+ even with the Docker cache fix.

**File:** `app/main.py`
**Problem:** No `/health` or `/ready` endpoint exists.
**Fix:** Add a `GET /health` endpoint. Optionally, add a startup probe that pre-warms dependencies (load reranker, connect to Qdrant, connect to Supabase) before accepting traffic. Cloud Run supports startup probes via `--startup-cpu-boost` and startup probe config.

---

## Issue 4: No environment variable validation at startup

**Severity:** MEDIUM
**Impact:** If `GEMINI_API_KEY` or `SUPABASE_DB_URL` is missing, the app starts fine but fails on the first request with a confusing error deep in the call stack. User gets "Loi xu ly cau hoi" with no useful information.

**File:** `app/config.py`
**Problem:** All env vars default to empty string. No validation.
**Fix:** Add a `validate_config()` function called at app startup that checks required vars (`GEMINI_API_KEY`, `SUPABASE_DB_URL`, `QDRANT_URL`, `QDRANT_API_KEY`) and raises `RuntimeError` with a clear message if any are missing. Only enforce in production (check for a `ENVIRONMENT` env var or detect Cloud Run via `K_SERVICE`).

---

## Issue 5: Memory pressure -- 1GB with concurrency 10

**Severity:** HIGH
**Impact:** Reranker model alone is ~500MB in memory. With 10 concurrent requests, each running Gemini SDK + Qdrant client + JSON processing, the container can OOM. Cloud Run kills OOM containers silently -- users see intermittent failures.

**File:** `.github/workflows/deploy.yml` -- `--memory 1Gi --concurrency 10`
**Fix:** Either:
- (A) Bump memory to 2Gi (still within free tier for low traffic), keep concurrency 10
- (B) Keep 1Gi, reduce concurrency to 4-5
- Recommendation: 2Gi + concurrency 5 (safest)

---

## Issue 6: Frontend silently swallows errors

**Severity:** MEDIUM
**Impact:** When `/sessions` or `/sessions/{id}/messages` fails (network error, 500, etc.), the empty catch block silently drops the error. User sees an empty session list or empty chat with no feedback.

### 6a: loadSessions() silently fails

**File:** `app/static/index.html` -- `loadSessions()` function (~line 501)
**Problem:** `catch(e) {}` -- empty error handler.
**Fix:** Show a subtle error message or retry indicator in the sidebar.

### 6b: switchSession() silently fails

**File:** `app/static/index.html` -- `switchSession()` function (~line 529)
**Problem:** `catch(e) {}` -- empty error handler.
**Fix:** Show "Could not load messages" in the chat area instead of blank.

### 6c: sources JSON.parse can throw

**File:** `app/static/index.html` -- `switchSession()` (~line 533)
**Problem:** `JSON.parse(m.sources)` -- no try-catch. Malformed JSON crashes the whole message rendering.
**Fix:** Wrap in try-catch, default to empty array.

---

## Issue 7: Dockerfile model download not verified

**Severity:** MEDIUM
**Impact:** If the HuggingFace download fails during `docker build`, the build still succeeds (Python prints the error but exits 0). At runtime, the reranker falls back to downloading -- hitting the same rate limit issue we already fixed.

**File:** `Dockerfile` -- line 15
**Current:**
```dockerfile
RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')"
```
**Fix:** Add verification that the model files actually exist after download:
```dockerfile
RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; m = TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual'); assert m is not None" || exit 1
```

---

## Priority Order for Fixes

1. **Issue 5** -- Memory/concurrency (deploy.yml change only, immediate)
2. **Issue 1** -- API timeouts (prevents hung requests)
3. **Issue 2** -- Connection pooling (performance under load)
4. **Issue 3** -- Health check + startup warmup (better cold start UX)
5. **Issue 4** -- Env var validation (fail fast)
6. **Issue 7** -- Dockerfile verification (build safety)
7. **Issue 6** -- Frontend error handling (UX polish)

## Files to Modify

| File | Changes |
|------|---------|
| `.github/workflows/deploy.yml` | Memory 2Gi, concurrency 5 |
| `app/llm.py` | Add timeout to generate() calls |
| `app/rag/knowledge_store.py` | Add timeout to QdrantClient init |
| `app/storage.py` | Add timeout to Supabase download |
| `app/db.py` | Replace per-call connections with ThreadedConnectionPool |
| `app/main.py` | Add /health endpoint, call validate_config() on startup |
| `app/config.py` | Add validate_config() function |
| `app/static/index.html` | Error feedback on session/message loading, try-catch JSON.parse |
| `Dockerfile` | Verify model download with assertion |
