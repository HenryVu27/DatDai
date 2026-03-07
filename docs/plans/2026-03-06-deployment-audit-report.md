# Deployment Audit -- Validation Report

**Date:** 2026-03-06
**Scope:** Verify hardening document claims, validate deployment design, find additional issues.

---

## Part 1: Hardening Document Issue Verification

### Issue 1: No timeouts on external API calls -- CONFIRMED

**1a: Gemini API calls have no timeout -- CONFIRMED**
- `app/llm.py:59` -- `client.models.generate_content()` has no timeout parameter.
- The fallback loop (lines 57-68) means a hung primary can block for its full duration before even attempting fallback. Worst case: two consecutive timeouts = 2x the hang time.
- Additionally, `llm.embed()` (line 76) also has no timeout -- the hardening doc missed this. A hung embed call blocks the entire search pipeline.

**1b: Qdrant queries have no timeout -- CONFIRMED**
- `app/rag/knowledge_store.py:26` -- `QdrantClient(url=..., api_key=...)` has no `timeout` parameter.
- All four query methods (`search_hybrid`, `fetch_by_metadata`, `fetch_full_dieu`, `fetch_by_chunk_ids`) are affected.

**1c: Supabase Storage downloads have no timeout -- CONFIRMED**
- `app/storage.py:28` -- `storage.from_(_BUCKET).download(remote_path)` has no timeout.
- Mitigated by `@lru_cache` (only hits once per process), but that first hit during cold start is unprotected.

### Issue 2: No database connection pooling -- CONFIRMED

- `app/db.py:86` -- `psycopg2.connect(SUPABASE_DB_URL)` called fresh on every `_get_pg()` invocation.
- Every public function (`create_session`, `add_message`, `get_messages`, etc.) calls `_get_db()` -> `_get_pg()` -> new connection.
- `handle_message()` in `app/chat.py` calls at minimum: `create_session`, `get_messages`, `get_turn_count`, `add_message` (user), `get_latest_summary`, `add_message` (assistant) = **6 connections per request**.
- With `upsert_summary` and `_generate_title` -> `update_session_title`, it can be **8+ connections per request**.
- The hardening doc is accurate. This is wasteful even with PgBouncer.

### Issue 3: No health check endpoint -- CONFIRMED

- `app/main.py` -- no `/health` or `/ready` endpoint exists.
- The only routes are: `POST /chat`, `GET /sessions`, `GET /sessions/{id}/messages`, `POST /sessions`, `GET /`.
- The hardening doc's description of first-request cold start penalty is accurate.

### Issue 4: No environment variable validation at startup -- CONFIRMED

- `app/config.py:8` -- `GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")` -- defaults to empty string.
- Same for `SUPABASE_DB_URL` (line 28), `SUPABASE_URL` (line 32), `SUPABASE_SERVICE_KEY` (line 33), `QDRANT_API_KEY` (line 37).
- `QDRANT_URL` defaults to `":memory:"` (line 36), which means a missing env var silently creates an in-memory Qdrant with zero documents -- the app starts but returns empty results for every query. This is arguably worse than a crash.

### Issue 5: Memory pressure -- 1GB with concurrency 10 -- CONFIRMED

- `.github/workflows/deploy.yml:50-54` -- `--memory 1Gi` and `--concurrency 10`.
- Reranker model (`jinaai/jina-reranker-v2-base-multilingual`) is ~500MB.
- Each concurrent request runs: embedding (Gemini SDK), Qdrant query, reranker inference (CPU-bound via `asyncio.to_thread`), Gemini generation. Heavy memory pressure.
- The hardening doc's recommendation of 2Gi + concurrency 5 is reasonable.

### Issue 6: Frontend silently swallows errors -- CONFIRMED

**6a: loadSessions() -- CONFIRMED**
- `app/static/index.html:513` -- `catch(e) {}` -- completely empty.

**6b: switchSession() -- CONFIRMED**
- `app/static/index.html:536` -- `catch(e) {}` -- completely empty.

**6c: sources JSON.parse can throw -- CONFIRMED**
- `app/static/index.html:533` -- `JSON.parse(m.sources)` with no try-catch.
- However, this is slightly less severe than stated: if `m.sources` is already a parsed object (JSON response), `JSON.parse` on an object would fail. Looking at the `/sessions/{id}/messages` endpoint (main.py:62), it returns raw DB rows where `sources` is stored as a JSON string in the DB (db.py:155 -- `json.dumps(sources)`). So yes, it's a string that needs parsing, and malformed data will crash rendering.

### Issue 7: Dockerfile model download not verified -- PARTIALLY CONFIRMED

- `Dockerfile:15` -- The `RUN python -c "..."` command.
- The hardening doc claims the build succeeds even if download fails. This is **partially overstated**: if the Python import fails (e.g., network error during download), the `RUN` command will return non-zero and Docker will fail the build. The `TextCrossEncoder()` constructor raises on download failure.
- However, there is a legitimate edge case: if the download succeeds but the model files are corrupted/incomplete, the constructor might not detect it. The suggested assertion `assert m is not None` doesn't actually help with that -- it only checks the object exists, not that the model is functional.
- A better verification would be to run a test inference: `m.rerank("test", ["test"])`.

---

## Part 2: Deployment Design Validation

### Dockerfile vs Design Document

| Design claim | Actual | Match? |
|---|---|---|
| Base image: python:3.12-slim | `FROM python:3.12-slim` | YES |
| New dependencies: psycopg2-binary, supabase | In requirements.txt | YES |
| Entrypoint: uvicorn app.main:app --host 0.0.0.0 --port $PORT | CMD uses `${PORT:-8080}` with default | YES (better -- has fallback) |

### GitHub Actions Workflow vs Design Document

| Design claim | Actual | Match? |
|---|---|---|
| Trigger: push to main | `on: push: branches: [main]` | YES |
| Authenticate via WIF (OIDC) | `google-github-actions/auth@v2` with WIF | YES |
| No service account JSON keys | Uses `vars.GCP_SERVICE_ACCOUNT` (SA email, not key) | YES |
| GitHub vars: GCP_PROJECT_ID, GCP_WORKLOAD_IDENTITY_PROVIDER | Both referenced | YES |

**Discrepancy found:** The design document (line 121-123) lists only 2 GitHub repository variables: `GCP_PROJECT_ID` and `GCP_WORKLOAD_IDENTITY_PROVIDER`. But the workflow also uses `${{ vars.GCP_SERVICE_ACCOUNT }}` (deploy.yml:28) -- a **third** variable not documented in the design.

### Secrets (6 secrets) vs Design Document

| Secret (design) | In deploy.yml --set-secrets? |
|---|---|
| GEMINI_API_KEY | YES |
| QDRANT_URL | YES |
| QDRANT_API_KEY | YES |
| SUPABASE_DB_URL | YES |
| SUPABASE_URL | YES |
| SUPABASE_SERVICE_KEY | YES |

All 6 secrets are correctly referenced in deploy.yml:56.

### .dockerignore vs Design Document

Design says excludes: `data/, .env, .venv/, __pycache__/, .git/, tests/, scripts/`

Actual .dockerignore:
```
.env, .venv/, venv/, data/, tests/, scripts/, docs/,
__pycache__/, *.pyc, .git/, .gitignore, .DS_Store, models/, *.md
```

The actual .dockerignore is **more comprehensive** than the design document describes. It additionally excludes: `venv/`, `docs/`, `*.pyc`, `.gitignore`, `.DS_Store`, `models/`, `*.md`. This is fine -- the design understated what's excluded but the implementation is better.

---

## Part 3: NEW Issues Not Covered in Hardening Document

### NEW-1: XSS via session_id path parameter (MEDIUM)

`app/main.py:61` -- `GET /sessions/{session_id}/messages` takes `session_id` as a path parameter and passes it directly to `db.get_messages()`. While the DB query uses parameterized queries (safe from SQL injection), a malicious session_id is not validated. The `session_id` flows through to the response without sanitization. In the frontend, session IDs are used in URLs (`fetch(\`/sessions/${id}/messages\`)`), but since they come from the server's own `/sessions` response, this is low risk in practice. **Low severity** -- UUID format is not enforced but exploitation path is limited.

### NEW-2: Error message leaks internal details (MEDIUM)

`app/main.py:51` -- `raise HTTPException(status_code=500, detail=f"Loi xu ly cau hoi: {str(e)}")`.
The raw exception string is sent to the client. This can leak:
- Database connection strings or host info from psycopg2 errors
- Gemini API error messages containing project/key metadata
- File paths from Python tracebacks
- Qdrant cluster info from connection errors

Should return a generic error and log the details server-side.

### NEW-3: No rate limiting (HIGH)

No rate limiting exists at any level:
- No middleware in `app/main.py`
- No Cloud Run-level rate limiting in deploy.yml
- A single user can send unlimited requests, each triggering multiple Gemini API calls (orchestrator + generator + optional title generation = up to 3 LLM calls per request)
- **Cost risk:** With Gemini Pro at ~$1.25/1M input tokens, an adversary could run up costs by sending complex queries in a loop. The `max_tokens=8000` on the generator amplifies this.

### NEW-4: Blocking synchronous DB calls in async handlers (MEDIUM)

`app/chat.py` -- `handle_message()` is an `async` function, but all `db.*` calls are synchronous (psycopg2 is blocking). Calls like `db.create_session()`, `db.get_messages()`, `db.add_message()` block the event loop.
- `app/main.py:55-57` -- `list_sessions()` and `get_session_messages()` are sync `def` functions (correct -- FastAPI runs these in a threadpool).
- But `chat_endpoint()` (line 41) is `async def`, and `chat.handle_message()` calls blocking DB functions directly on the event loop.
- With concurrency=10, multiple requests can be blocked waiting on DB I/O serially rather than concurrently.

### NEW-5: Reranker inference blocks event loop for other concurrent requests (LOW-MEDIUM)

`app/rag/reranker.py:19` -- `await asyncio.to_thread(lambda: list(self._model.rerank(...)))` correctly offloads to a thread. However, `asyncio.to_thread` uses the default executor (ThreadPoolExecutor). On Cloud Run with 1 vCPU, multiple concurrent reranker calls will compete for CPU. This is mitigated by the concurrency limit but worth noting.

### NEW-6: `_get_storage_client()` creates a new Supabase client on every call (LOW)

`app/storage.py:14-20` -- `_get_storage_client()` creates a new `create_client()` every time it's called. Since `_load_json` is behind `@lru_cache`, this only happens once per cached function, so in practice it's called at most 2 times (vocab + amendment_index). Low severity.

### NEW-7: `get_collection()` called on every Qdrant query (LOW-MEDIUM)

`app/rag/knowledge_store.py:66` -- `client.get_collection(QDRANT_COLLECTION)` is called at the start of `search_hybrid()`, `fetch_by_metadata()`, `fetch_full_dieu()`, and `fetch_by_chunk_ids()` to check `points_count == 0`. This adds an extra round-trip to Qdrant Cloud on **every single query**. With a complex question triggering `search_hybrid` + `fetch_full_dieu` (for top 3 results) + potentially `fetch_by_chunk_ids` for amendments, a single user request can hit `get_collection` 4-7 times.

Should cache the collection info or remove the check (an empty result is a fine response).

### NEW-8: Malformed Gemini JSON response handling is fragile (MEDIUM)

`app/chat.py:266-293` -- `_parse_orchestrator_response()` handles `JSONDecodeError` by treating the entire response as a `direct_response`. But:
- If Gemini returns valid JSON with unexpected structure (e.g., `actions` is a string instead of a list), the code will crash later in `_execute_tools` when iterating over actions.
- If `actions` contains entries with missing `tool` keys, `_run_action` handles it (returns `[]`), but missing required params like `query` for `search_legal_docs` would pass empty string to the embed call, wasting API calls.

### NEW-9: User message saved before processing -- no rollback on failure (LOW)

`app/chat.py:361` -- `db.add_message(session_id, turn, "user", user_message)` is called before processing. If the pipeline fails after this point (and the exception propagates to main.py:51), the user message is persisted but no assistant response is saved. On retry, the frontend sends a new request with the same question, creating a **duplicate user message** in the DB. The turn count increments, and the conversation history grows with orphaned user messages.

### NEW-10: Cold start sequence analysis (INFORMATIONAL)

Tracing the exact cold start path:

1. **Container starts** -- Python interpreter loads, uvicorn starts
2. **First request arrives** (e.g., `GET /sessions` or `POST /chat`)
3. For `POST /chat`:
   - `app/config.py` -- module-level `load_dotenv()` + `os.getenv()` -- instant
   - `app/db.py` -- module-level: evaluates `_use_pg = bool(SUPABASE_DB_URL)`, conditionally imports psycopg2 -- fast
   - `chat.handle_message()` -> `db.create_session()` -> first `_get_pg()` call:
     - **Blocking:** `psycopg2.connect()` to Supabase (network round-trip, ~100-500ms)
     - **Blocking:** Executes full `_PG_SCHEMA` DDL (CREATE TABLE IF NOT EXISTS x3, CREATE INDEX x2) -- ~200ms
   - `_get_store()` -> `KnowledgeStore.__init__()`:
     - **Blocking:** `load_vocab()` -> `_load_json()` -> `_get_storage_client()` -> `create_client()` + `download("vocab.json")` from Supabase Storage -- network I/O, ~500-2000ms for 33KB file
   - `KnowledgeStore._get_client()`:
     - **Blocking:** `QdrantClient(url=..., api_key=...)` -- connection setup, ~200-500ms
   - `llm.embed()` -> `get_client()`:
     - First Gemini client creation -- fast (just instantiation)
   - `_get_reranker()` -> `CrossEncoderReranker.__init__()`:
     - **Blocking:** `TextCrossEncoder(model_name=...)` -- loads ~500MB ONNX model from disk into memory -- **3-8 seconds**
   - Then actual processing begins...

   **Total cold start overhead: ~5-12 seconds** on top of normal processing time (which itself is 3-10s for Gemini calls).

### NEW-11: `_generate_title` failure is silently swallowed (LOW)

`app/chat.py:474` -- `except Exception: pass`. If title generation fails, no error is logged. Should at minimum log the error.

### NEW-12: No CORS configuration (LOW)

`app/main.py` -- No CORS middleware. Currently fine since the frontend is served from the same origin. But if the frontend is ever separated (e.g., CDN), API calls will be blocked. Not an issue today.

### NEW-13: Concurrent modification of global singletons (LOW)

`app/rag/retriever.py:22-35` -- `_get_store()` and `_get_reranker()` use module-level globals with no locking. Two concurrent requests arriving simultaneously during cold start could both enter the `if _store is None` branch and create duplicate instances. In Python with the GIL this is unlikely to cause data corruption, but one instance gets immediately orphaned (wasted memory for the reranker's 500MB model). Same pattern in `app/llm.py:26-29`.

### NEW-14: `.dockerignore` excludes `*.md` which could hide issues (INFORMATIONAL)

The `.dockerignore` excludes `*.md` and `docs/`. This is correct for production but means the `CLAUDE.md` and `README.md` files don't end up in the image. This is desired behavior, just noting it's intentional.

---

## Part 4: Recommended Priority Order for Fixes

### Critical (fix before significant traffic)
1. **NEW-3: Rate limiting** -- Unmitigated cost risk. Add at minimum a per-IP rate limit middleware (e.g., slowapi) or Cloud Run-level throttling.
2. **Issue 5: Memory/concurrency** -- Single deploy.yml change. 2Gi + concurrency 5.
3. **Issue 1: API timeouts** -- Add timeouts to Gemini, Qdrant, and Supabase calls. Include `llm.embed()` (missed by hardening doc).

### High (fix soon)
4. **Issue 2: Connection pooling** -- Replace per-call connections with a pool.
5. **NEW-4: Blocking DB calls in async handler** -- Either make DB calls async (aiosqlite is already in requirements.txt but not used for pg) or run them via `asyncio.to_thread()`.
6. **NEW-2: Error message leaks** -- Replace `str(e)` with generic message in HTTP response.
7. **Issue 4: Env var validation** -- Fail fast on missing secrets.

### Medium (fix before production scale)
8. **Issue 3: Health check + startup warmup** -- Add `/health` endpoint and pre-warm on startup.
9. **NEW-7: Qdrant `get_collection` overhead** -- Cache or remove the per-query collection check.
10. **NEW-8: Malformed JSON handling** -- Add type validation to parsed orchestrator response.
11. **Issue 6: Frontend error handling** -- Replace empty catch blocks, wrap JSON.parse.
12. **NEW-9: Duplicate messages on retry** -- Consider idempotency key or deduplication.

### Low (cleanup)
13. **Issue 7: Dockerfile model verification** -- Add test inference to build step.
14. **NEW-11: Silent title generation failure** -- Add logging.
15. **NEW-13: Singleton race condition** -- Add locking or initialize eagerly at startup.
16. **Design doc discrepancy** -- Document `GCP_SERVICE_ACCOUNT` as a third GitHub Actions variable.

---

## Summary

| Category | Count |
|---|---|
| Hardening issues confirmed as-stated | 6 of 7 |
| Hardening issues partially overstated | 1 (Issue 7 -- Dockerfile verification) |
| Hardening issues with missed scope | 1 (Issue 1a -- `embed()` also lacks timeout) |
| New issues found | 14 |
| Design doc discrepancies | 1 (undocumented GCP_SERVICE_ACCOUNT variable) |
