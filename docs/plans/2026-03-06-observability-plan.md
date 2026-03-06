# Langfuse Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Langfuse tracing to all RAG pipeline components, automatic heuristic scores, user feedback endpoint, and batch evaluation script.

**Architecture:** Langfuse Cloud receives traces via `@observe()` decorators on existing functions. Each chat request becomes a trace with nested spans for orchestrator, retrieval, reranking, generation. Heuristic scores are logged automatically. Batch LLM evaluation via a separate script.

**Tech Stack:** langfuse Python SDK, Gemini API (for batch eval), FastAPI

**Important:** The project has no hyde.py -- HyDE is implicit in the orchestrator's query rewriting. `query_rewriter.py` only contains regex-based `extract_filters()`, not an LLM call. The actual traced spans are: orchestrator LLM, tool execution (search/lookup), reranking, generator LLM, citation check, title generation.

---

## Phase 1: Core Langfuse Integration

### Task 1: Add langfuse dependency

**Files:**
- Modify: `requirements.txt:11`

**Step 1: Add langfuse to requirements.txt**

Add this line at the end:

```
langfuse>=2.0.0
```

**Step 2: Install locally**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/pip install langfuse`
Expected: Successful installation

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add langfuse for observability"
```

---

### Task 2: Add Langfuse env vars to config.py

**Files:**
- Modify: `app/config.py:33` (add after SUPABASE_SERVICE_KEY line)

**Step 1: Add Langfuse config**

Add these lines after line 33 (after `SUPABASE_SERVICE_KEY`):

```python
# Langfuse observability
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
```

**Step 2: Verify**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/config.py
git commit -m "config: add Langfuse environment variables"
```

---

### Task 3: Create app/observability.py -- Langfuse initialization

**Files:**
- Create: `app/observability.py`

**Step 1: Write app/observability.py**

This module initializes Langfuse and provides the `@observe` decorator and score helper. When Langfuse keys are not configured (local dev), tracing is silently disabled.

```python
"""Langfuse observability setup.

Provides @observe decorator and score helpers.
When LANGFUSE_PUBLIC_KEY is not set, tracing is disabled (no-op).
"""
import logging
from langfuse.decorators import langfuse_context, observe  # noqa: F401

from app.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

log = logging.getLogger(__name__)

_enabled = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

if _enabled:
    from langfuse import Langfuse

    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
    )
    log.info("Langfuse tracing enabled")
else:
    langfuse = None
    log.info("Langfuse tracing disabled (no keys configured)")


def score_current_trace(name: str, value: float, comment: str = "") -> None:
    """Log a heuristic score on the current trace. No-op if tracing disabled."""
    if not _enabled:
        return
    try:
        langfuse_context.score_current_trace(name=name, value=value, comment=comment)
    except Exception as e:
        log.debug("Failed to log score %s: %s", name, e)


def flush() -> None:
    """Flush pending Langfuse events. Call on shutdown."""
    if langfuse:
        langfuse.flush()
```

**Step 2: Verify import**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.observability import observe, score_current_trace; print('OK')"`
Expected: `OK` (tracing disabled message in logs, no crash)

**Step 3: Commit**

```bash
git add app/observability.py
git commit -m "feat: add Langfuse observability module with graceful disable"
```

---

### Task 4: Add @observe to llm.py (generation spans)

**Files:**
- Modify: `app/llm.py:1-5,38-86,89-98`

**Step 1: Add import and decorators**

Add import at top of `app/llm.py` (after existing imports around line 5):

```python
from app.observability import observe
```

Decorate the `generate` function (line 38) with:

```python
@observe(as_type="generation")
async def generate(
```

Decorate the `embed` function (line 89) with:

```python
@observe(name="embed")
async def embed(texts: list[str]) -> list[list[float]]:
```

Inside `generate`, after the successful `response.text` return (line 73), add metadata update just before the return. Replace the inner try block (lines 65-84) with:

```python
    for model_id in (primary, fallback):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=model_id, contents=contents, config=config,
                ),
                timeout=timeout,
            )
            text = response.text or ""
            # Update Langfuse generation metadata
            try:
                from langfuse.decorators import langfuse_context
                usage = getattr(response, "usage_metadata", None)
                langfuse_context.update_current_observation(
                    model=model_id,
                    usage={
                        "input": getattr(usage, "prompt_token_count", 0) if usage else 0,
                        "output": getattr(usage, "candidates_token_count", 0) if usage else 0,
                    } if usage else None,
                    metadata={"role": model, "was_fallback": model_id != primary},
                )
            except Exception:
                pass
            return text
        except asyncio.TimeoutError:
            logger.warning("Model %s timed out after %ds", model_id, timeout)
            if model_id == primary:
                continue
            raise TimeoutError(f"Gemini generation timed out after {timeout}s")
        except Exception as e:
            if model_id == primary:
                logger.warning("Primary model %s failed, trying fallback %s: %s", primary, fallback, e)
                continue
            logger.error("Fallback model %s also failed: %s", fallback, e)
            raise
```

**Step 2: Verify the app starts**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.llm import generate, embed; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/llm.py
git commit -m "feat: add Langfuse tracing to LLM generate and embed functions"
```

---

### Task 5: Add @observe to retriever.py (retrieval spans)

**Files:**
- Modify: `app/rag/retriever.py:1-14,38-89`

**Step 1: Add import**

Add at top of file (after existing imports, around line 13):

```python
from app.observability import observe
```

**Step 2: Decorate search_legal_docs**

Add `@observe(name="search_legal_docs")` before the function definition at line 38:

```python
@observe(name="search_legal_docs")
async def search_legal_docs(
```

**Step 3: Decorate lookup_amendment**

Add `@observe(name="lookup_amendment")` before line 92:

```python
@observe(name="lookup_amendment")
def lookup_amendment(
```

**Step 4: Decorate lookup_specific_dieu**

Add `@observe(name="lookup_specific_dieu")` before line 131:

```python
@observe(name="lookup_specific_dieu")
def lookup_specific_dieu(doc_id: str, dieu: str) -> list[dict]:
```

**Step 5: Verify**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.rag.retriever import search_legal_docs, lookup_amendment, lookup_specific_dieu; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add app/rag/retriever.py
git commit -m "feat: add Langfuse tracing to retriever functions"
```

---

### Task 6: Add @observe to reranker.py

**Files:**
- Modify: `app/rag/reranker.py:1-6,15`

**Step 1: Add import and decorator**

Add import after line 4:

```python
from app.observability import observe
```

Decorate the `rerank` method at line 15:

```python
    @observe(name="rerank")
    async def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
```

**Step 2: Verify**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.rag.reranker import CrossEncoderReranker; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/rag/reranker.py
git commit -m "feat: add Langfuse tracing to reranker"
```

---

### Task 7: Add @observe to chat.py (root trace + heuristic scores)

**Files:**
- Modify: `app/chat.py:1-23,349-462,465-478`

This is the most important file -- the root trace. Every child span (llm.generate, search_legal_docs, rerank) will nest under this automatically.

**Step 1: Add imports**

Add after the existing imports (around line 12):

```python
from app.observability import observe, score_current_trace
```

**Step 2: Decorate handle_message as root trace**

Replace the function signature at line 349:

```python
@observe(name="handle_message")
async def handle_message(session_id: str, user_message: str) -> dict:
```

**Step 3: Add heuristic scores before the return**

Just before the final return statement (line 462), add scoring:

```python
    # -- Heuristic scores for Langfuse --
    score_current_trace("retrieval_count", float(len(chunks)))
    score_current_trace("response_length", float(len(response)))
    score_current_trace("has_sources", 1.0 if sources else 0.0)
    score_current_trace("model_used", 1.0 if model == "pro" else 0.0, comment=model)

    # Reranker scores (if reranking happened)
    reranked = [c for c in chunks if c.get("score", 0) > 0]
    if reranked:
        scores = [c["score"] for c in reranked]
        score_current_trace("reranker_top_score", max(scores))
        score_current_trace("reranker_mean_score", sum(scores) / len(scores))
```

**Step 4: Decorate _generate_title**

Add `@observe(name="generate_title")` before line 465:

```python
@observe(name="generate_title")
async def _generate_title(session_id: str, question: str, answer: str) -> None:
```

**Step 5: Verify import chain**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.chat import handle_message; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add app/chat.py
git commit -m "feat: add Langfuse root trace and heuristic scores to chat pipeline"
```

---

### Task 8: Add Langfuse flush on shutdown in main.py

**Files:**
- Modify: `app/main.py:1-18`

**Step 1: Add shutdown event**

Add import after existing imports (around line 5):

```python
from contextlib import asynccontextmanager
from app.observability import flush as langfuse_flush
```

Replace the app creation (line 18) with a lifespan handler:

```python
@asynccontextmanager
async def lifespan(app):
    yield
    langfuse_flush()

app = FastAPI(title="Chatbot Luat Dat Dai", lifespan=lifespan)
```

**Step 2: Verify the app starts**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.main import app; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat: flush Langfuse traces on app shutdown"
```

---

## Phase 2: User Feedback

### Task 9: Add /feedback endpoint to main.py

**Files:**
- Modify: `app/main.py`

**Step 1: Add feedback model and endpoint**

Add after the existing endpoint definitions (before the static files section, around line 69):

```python
class FeedbackRequest(BaseModel):
    trace_id: str
    score: int  # 1 = thumbs up, 0 = thumbs down

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    from app.observability import langfuse
    if not langfuse:
        return {"status": "tracing_disabled"}
    langfuse.score(
        trace_id=req.trace_id,
        name="user_feedback",
        value=req.score,
    )
    return {"status": "ok"}
```

**Step 2: Return trace_id from /chat endpoint**

To connect feedback to traces, the chat endpoint needs to return the Langfuse trace ID.

Modify `chat_endpoint` in `app/main.py` (around line 41). After calling `chat.handle_message`, get the current trace ID:

```python
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Cau hoi khong duoc de trong.")

    session_id = request.session_id or chat.create_new_session()

    try:
        result = await chat.handle_message(session_id, request.question)

        # Get Langfuse trace ID for feedback linking
        trace_id = None
        try:
            from langfuse.decorators import langfuse_context
            trace_id = langfuse_context.get_current_trace_id()
        except Exception:
            pass

        return {**result, "trace_id": trace_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi xu ly cau hoi: {str(e)}")
```

Update `ChatResponse` to include optional trace_id:

```python
class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    trace_id: str | None = None
```

**Step 3: Verify**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.main import app; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat: add /feedback endpoint and trace_id in chat response"
```

---

### Task 10: Add feedback buttons to frontend

**Files:**
- Modify: `app/static/index.html`

**Step 1: Add CSS for feedback buttons**

Add these styles inside the `<style>` block (before the `</style>` closing tag, around line 411):

```css
/* Feedback buttons */
.feedback {
  display: flex;
  gap: 0.4rem;
  margin-top: 0.5rem;
}
.feedback-btn {
  background: none;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.25rem 0.5rem;
  cursor: pointer;
  font-size: 0.75rem;
  color: var(--text-muted);
  transition: border-color 0.15s, color 0.15s, background 0.15s;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}
.feedback-btn:hover { border-color: var(--accent); color: var(--text-secondary); }
.feedback-btn.selected { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
```

**Step 2: Modify addMessage to accept traceId and show feedback buttons**

In the JavaScript section, update the `addMessage` function (around line 590) to accept a 4th parameter:

```javascript
function addMessage(role, content, sources, traceId) {
  const w = document.getElementById('welcome');
  if (w) w.remove();

  const div = document.createElement('div');
  div.className = `message ${role}`;
  let html = `<div class="bubble">${formatContent(content)}`;

  if (sources && sources.length > 0) {
    const grouped = {};
    sources.forEach(s => {
      const doc = s.doc_name || '';
      if (!doc) return;
      if (!grouped[doc]) grouped[doc] = [];
      if (s.dieu) grouped[doc].push(s.dieu);
    });
    html += `<div class="sources"><div class="sources-label">Nguồn</div>`;
    Object.entries(grouped).forEach(([doc, dieus]) => {
      let label = doc;
      if (dieus.length > 0) label += ': ' + dieus.join(', ');
      html += `<span class="source-tag">${esc(label)}</span>`;
    });
    html += `</div>`;
  }

  if (role === 'assistant' && traceId) {
    html += `<div class="feedback" data-trace-id="${esc(traceId)}">
      <button class="feedback-btn" onclick="sendFeedback(this, 1)" title="Hữu ích">&#x1F44D;</button>
      <button class="feedback-btn" onclick="sendFeedback(this, 0)" title="Chưa tốt">&#x1F44E;</button>
    </div>`;
  }

  html += `</div>`;
  div.innerHTML = html;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}
```

**Step 3: Add sendFeedback function**

Add this function in the `<script>` section (after the `esc` function, around line 661):

```javascript
async function sendFeedback(btn, score) {
  const container = btn.closest('.feedback');
  const traceId = container.dataset.traceId;
  container.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  try {
    await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trace_id: traceId, score }),
    });
  } catch(e) {}
}
```

**Step 4: Update handleSubmit to pass traceId**

In `handleSubmit` (around line 708), change the addMessage call for assistant:

```javascript
    addMessage('assistant', data.answer, data.sources, data.trace_id);
```

**Step 5: Commit**

```bash
git add app/static/index.html
git commit -m "feat: add thumbs up/down feedback buttons in chat UI"
```

---

## Phase 3: Deployment Config

### Task 11: Add Langfuse secrets to GCP and deploy workflow

**Files:**
- Modify: `.github/workflows/deploy.yml:56`

**Step 1: Update --set-secrets in deploy.yml**

Replace line 56 (the `--set-secrets` line) with:

```yaml
            --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest,QDRANT_URL=QDRANT_URL:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest,SUPABASE_DB_URL=SUPABASE_DB_URL:latest,SUPABASE_URL=SUPABASE_URL:latest,SUPABASE_SERVICE_KEY=SUPABASE_SERVICE_KEY:latest,LANGFUSE_PUBLIC_KEY=LANGFUSE_PUBLIC_KEY:latest,LANGFUSE_SECRET_KEY=LANGFUSE_SECRET_KEY:latest"
```

**Step 2: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "ci: add Langfuse secrets to Cloud Run deployment"
```

**Step 3: Manual -- Create Langfuse account and secrets**

These are manual steps to do in the browser/CLI:

1. Sign up at https://cloud.langfuse.com
2. Create a project named "DatDai"
3. Go to Settings > API Keys, create a key pair
4. Copy the public key and secret key
5. Add to your local `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   ```
6. Add to GCP Secret Manager:
   ```bash
   echo -n "pk-lf-YOUR_KEY" | gcloud secrets create LANGFUSE_PUBLIC_KEY --data-file=-
   echo -n "sk-lf-YOUR_KEY" | gcloud secrets create LANGFUSE_SECRET_KEY --data-file=-
   ```
7. Grant Cloud Run service account access (if needed):
   ```bash
   PROJECT_ID=$(gcloud config get-value project)
   PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
   for SECRET in LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
     gcloud secrets add-iam-policy-binding $SECRET \
       --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
       --role="roles/secretmanager.secretAccessor"
   done
   ```

---

## Phase 4: Batch Evaluation Script

### Task 12: Create scripts/evaluate_rag.py

**Files:**
- Create: `scripts/evaluate_rag.py`

**Step 1: Write the evaluation script**

```python
"""Batch RAG evaluation using Langfuse traces + Gemini as judge.

Usage:
    .venv/bin/python scripts/evaluate_rag.py [--limit 50] [--days 7]

Pulls recent traces from Langfuse, evaluates answer quality with Gemini,
and posts scores back to Langfuse.
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EVAL_PROMPT = """Ban la chuyen gia danh gia chat luong tra loi phap luat. Danh gia cau tra loi sau dua tren tai lieu tham khao.

<retrieved-context>
{context}
</retrieved-context>

<question>
{question}
</question>

<answer>
{answer}
</answer>

Danh gia 3 tieu chi, moi tieu chi cho diem 0-1 (0=kem, 0.5=trung binh, 1=tot):
1. groundedness: Cau tra loi co dua tren tai lieu khong? Co bua dat thong tin khong?
2. relevance: Cau tra loi co dung trong tam cau hoi khong?
3. completeness: Cau tra loi co day du, khong thieu thong tin quan trong khong?

Tra ve CHINH XAC JSON (khong markdown):
{{"groundedness": 0.0, "relevance": 0.0, "completeness": 0.0, "reasoning": "giai thich ngan"}}"""


def get_traces(lf: Langfuse, limit: int, days: int) -> list:
    """Fetch recent traces that haven't been evaluated yet."""
    traces = lf.fetch_traces(
        name="handle_message",
        limit=limit,
        order_by="timestamp",
        order="DESC",
    )
    cutoff = datetime.now() - timedelta(days=days)
    result = []
    for t in traces.data:
        if t.timestamp and t.timestamp.replace(tzinfo=None) < cutoff:
            continue
        # Skip traces that already have eval scores
        existing_scores = {s.name for s in (t.scores or [])}
        if "groundedness" in existing_scores:
            continue
        result.append(t)
    return result


def extract_trace_data(lf: Langfuse, trace_id: str) -> dict | None:
    """Extract question, context, and answer from a trace's observations."""
    trace = lf.fetch_trace(trace_id)
    if not trace:
        return None

    question = ""
    context = ""
    answer = ""

    # The root observation input contains the user_message
    if trace.input:
        question = trace.input.get("user_message", "") if isinstance(trace.input, dict) else str(trace.input)

    # The root observation output contains the answer
    if trace.output:
        answer = trace.output.get("answer", "") if isinstance(trace.output, dict) else str(trace.output)

    # Look through observations for retrieval context
    observations = lf.fetch_observations(trace_id=trace_id)
    for obs in observations.data:
        if obs.name == "search_legal_docs" and obs.output:
            # The output is the list of chunks
            chunks = obs.output if isinstance(obs.output, list) else []
            context_parts = []
            for c in chunks[:8]:
                if isinstance(c, dict):
                    text = c.get("text", c.get("content", ""))
                    source = c.get("metadata_str", "")
                    context_parts.append(f"[{source}] {text}")
            context = "\n---\n".join(context_parts)

    if not question or not answer:
        return None

    return {"question": question, "context": context, "answer": answer}


def evaluate_with_gemini(client: genai.Client, data: dict) -> dict | None:
    """Run Gemini as judge on a single trace."""
    prompt = EVAL_PROMPT.format(**data)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=500),
        )
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
    except Exception as e:
        log.error("Gemini eval failed: %s", e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate RAG quality via Langfuse + Gemini")
    parser.add_argument("--limit", type=int, default=50, help="Max traces to evaluate")
    parser.add_argument("--days", type=int, default=7, help="Look back N days")
    args = parser.parse_args()

    lf = Langfuse()
    gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    log.info("Fetching recent traces (limit=%d, days=%d)...", args.limit, args.days)
    traces = get_traces(lf, args.limit, args.days)
    log.info("Found %d unevaluated traces", len(traces))

    evaluated = 0
    for trace in traces:
        data = extract_trace_data(lf, trace.id)
        if not data:
            log.warning("Skipping trace %s: could not extract data", trace.id)
            continue

        scores = evaluate_with_gemini(gemini, data)
        if not scores:
            continue

        # Post scores back to Langfuse
        for metric in ("groundedness", "relevance", "completeness"):
            if metric in scores:
                lf.score(
                    trace_id=trace.id,
                    name=metric,
                    value=scores[metric],
                    comment=scores.get("reasoning", ""),
                )

        evaluated += 1
        log.info(
            "Trace %s: groundedness=%.1f relevance=%.1f completeness=%.1f",
            trace.id[:8],
            scores.get("groundedness", 0),
            scores.get("relevance", 0),
            scores.get("completeness", 0),
        )

        # Rate limit: ~2 requests/sec
        time.sleep(0.5)

    lf.flush()
    log.info("Done. Evaluated %d/%d traces.", evaluated, len(traces))


if __name__ == "__main__":
    main()
```

**Step 2: Verify the script parses**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "import scripts.evaluate_rag; print('OK')" 2>/dev/null || .venv/bin/python -c "import ast; ast.parse(open('scripts/evaluate_rag.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

**Step 3: Commit**

```bash
git add scripts/evaluate_rag.py
git commit -m "feat: add batch RAG evaluation script using Langfuse + Gemini as judge"
```

---

## Phase 5: Final Verification

### Task 13: Local integration test

**Step 1: Add Langfuse keys to .env (if not already done)**

Add to `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

**Step 2: Start the server locally**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -m uvicorn app.main:app --port 8080`

**Step 3: Send a test request**

In another terminal:
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Dieu 79 Luat Dat Dai quy dinh gi?"}'
```

Expected: Response includes `trace_id` field. Check Langfuse dashboard to see the trace with nested spans.

**Step 4: Test feedback**

```bash
curl -X POST http://localhost:8080/feedback \
  -H "Content-Type: application/json" \
  -d '{"trace_id": "TRACE_ID_FROM_ABOVE", "score": 1}'
```

Expected: `{"status": "ok"}`

**Step 5: Verify Langfuse dashboard**

Go to https://cloud.langfuse.com, open your project. You should see:
- A trace named "handle_message"
- Nested spans: generate (orchestrator), search_legal_docs, embed, rerank, generate (generator)
- Heuristic scores: retrieval_count, reranker_top_score, response_length, etc.
- User feedback score (if you sent the curl above)

---

### Task 14: Commit all and push

**Step 1: Final git status check**

Run: `git status`
Verify no uncommitted changes remain.

**Step 2: Push to main**

Run: `git push origin main`

This triggers GitHub Actions, which will build and deploy with the new Langfuse integration. Remember to create the GCP secrets (Task 11 Step 3) before the deploy workflow runs.
