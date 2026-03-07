# GCP Deployment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy the DatDai chatbot to GCP Cloud Run with Supabase for data persistence and storage.

**Architecture:** Cloud Run (stateless FastAPI container) + Supabase PostgreSQL (chat DB, replaces SQLite) + Supabase Storage (reference files) + GCP Secret Manager (secrets) + GitHub Actions CI/CD. Qdrant Cloud and Gemini API remain unchanged.

**Tech Stack:** Python 3.12, FastAPI, psycopg2, supabase-py, Docker, gcloud CLI, gh CLI

**Important:** Never commit API keys, DB credentials, or .env files. All secrets go through GCP Secret Manager.

---

## Phase 1: Code Changes (Local)

### Task 1: Add new dependencies to requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Add psycopg2-binary and supabase to requirements.txt**

Add these two lines to the end of `requirements.txt`:

```
psycopg2-binary>=2.9.0
supabase>=2.0.0
```

**Step 2: Install and verify**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/pip install psycopg2-binary supabase`
Expected: Successful installation

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add psycopg2-binary and supabase for GCP deployment"
```

---

### Task 2: Add SUPABASE_DB_URL, SUPABASE_URL, SUPABASE_SERVICE_KEY to config.py

**Files:**
- Modify: `app/config.py:27-31`

**Step 1: Update config.py**

Add Supabase config below the existing DB_PATH line (line 27) and update DB_PATH to read from env:

```python
# Database -- prefer Supabase PostgreSQL, fall back to local SQLite
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")
DB_PATH = os.path.join(DATA_DIR, "chat.db")  # SQLite fallback for local dev

# Supabase Storage
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
```

**Step 2: Verify app still starts locally**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.config import SUPABASE_DB_URL, SUPABASE_URL, SUPABASE_SERVICE_KEY; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/config.py
git commit -m "config: add Supabase environment variables"
```

---

### Task 3: Create app/storage.py for Supabase Storage

**Files:**
- Create: `app/storage.py`

**Step 1: Write app/storage.py**

This module downloads reference files from Supabase Storage on first access and caches them in memory. Falls back to local filesystem for local dev (when SUPABASE_URL is not set).

```python
"""Load reference data from Supabase Storage or local filesystem."""
import json
import logging
import os
from functools import lru_cache

from app.config import SUPABASE_URL, SUPABASE_SERVICE_KEY, DATA_DIR

log = logging.getLogger(__name__)

_BUCKET = "datdai-data"


def _get_storage_client():
    """Return Supabase storage client, or None if not configured."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None
    from supabase import create_client
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return client.storage


def _load_json(remote_path: str, local_path: str) -> dict | list:
    """Load JSON from Supabase Storage, falling back to local file."""
    storage = _get_storage_client()
    if storage:
        log.info("Loading %s from Supabase Storage", remote_path)
        data = storage.from_(_BUCKET).download(remote_path)
        return json.loads(data)
    log.info("Loading %s from local filesystem", local_path)
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_vocab() -> dict:
    return _load_json("vocab.json", os.path.join(DATA_DIR, "vocab.json"))


@lru_cache(maxsize=1)
def load_amendment_index() -> dict:
    return _load_json(
        "amendment_index.json",
        os.path.join(DATA_DIR, "amendment_index.json"),
    )
```

**Step 2: Verify it loads locally (no Supabase configured)**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.storage import load_vocab, load_amendment_index; v = load_vocab(); a = load_amendment_index(); print(f'vocab: {len(v)} keys, amendments: {len(a)} keys')"`
Expected: Prints key counts, no errors (falls back to local files)

**Step 3: Commit**

```bash
git add app/storage.py
git commit -m "feat: add storage module for Supabase Storage with local fallback"
```

---

### Task 4: Update knowledge_store.py to use storage.py

**Files:**
- Modify: `app/rag/knowledge_store.py:18,36-38`

**Step 1: Read the current file**

Read `app/rag/knowledge_store.py` to see the full context around lines 18 and 36-38.

**Step 2: Replace local vocab loading with storage.py**

Remove the `VOCAB_FILE` constant and the local file loading in `__init__`. Replace with:

```python
from app.storage import load_vocab
```

In `__init__`, replace the vocab loading block with:

```python
self._vocab = load_vocab()
```

Remove the `import os` if no longer used, and remove the `VOCAB_FILE` constant.

**Step 3: Verify the app still works locally**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.rag.knowledge_store import KnowledgeStore; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add app/rag/knowledge_store.py
git commit -m "refactor: use storage module for vocab loading"
```

---

### Task 5: Update amendment_index.py to use storage.py

**Files:**
- Modify: `app/rag/amendment_index.py:16,29-30,41`

**Step 1: Read the current file**

Read `app/rag/amendment_index.py` to see the full module.

**Step 2: Replace local file loading with storage.py**

Remove `_INDEX_PATH`, `_load_index()`, and the module-level call. Replace with:

```python
from app.storage import load_amendment_index

def _get_index() -> dict:
    return load_amendment_index()
```

Update all functions that reference `_index` to call `_get_index()` instead.

**Step 3: Verify the app still works locally**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.rag.amendment_index import get_amendments; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add app/rag/amendment_index.py
git commit -m "refactor: use storage module for amendment index loading"
```

---

### Task 6: Migrate db.py from SQLite to PostgreSQL with fallback

**Files:**
- Modify: `app/db.py` (full rewrite)

**Step 1: Read current db.py**

Already read. 151 lines, 8 functions, raw sqlite3.

**Step 2: Rewrite db.py with psycopg2 + SQLite fallback**

The new db.py should:
- Use psycopg2 when `SUPABASE_DB_URL` is set (production)
- Fall back to sqlite3 when not set (local dev)
- Keep the same function signatures so no other files need to change
- Use `%s` placeholders for PostgreSQL, `?` for SQLite
- Replace `INSERT OR IGNORE` with `ON CONFLICT DO NOTHING` for PostgreSQL
- Replace `AUTOINCREMENT` with `SERIAL` for PostgreSQL
- Use a module-level connection pool approach for PostgreSQL

```python
"""Database layer for conversation persistence.

Uses PostgreSQL (via psycopg2) when SUPABASE_DB_URL is set,
falls back to SQLite for local development.
"""
import json
import os
import sqlite3
from datetime import datetime

from app.config import DB_PATH, SUPABASE_DB_URL

_use_pg = bool(SUPABASE_DB_URL)

if _use_pg:
    import psycopg2
    import psycopg2.extras

# --- Schema ---

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    turn INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    summary TEXT NOT NULL,
    covers_through_turn INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
"""

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    covers_through_turn INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
"""

_pg_schema_initialized = False


def _get_pg():
    global _pg_schema_initialized
    conn = psycopg2.connect(SUPABASE_DB_URL)
    if not _pg_schema_initialized:
        with conn.cursor() as cur:
            cur.execute(_PG_SCHEMA)
        conn.commit()
        _pg_schema_initialized = True
    return conn


def _get_sqlite():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SQLITE_SCHEMA)
    return conn


def _get_db():
    return _get_pg() if _use_pg else _get_sqlite()


def _ph(name=""):
    """Return placeholder style."""
    return "%s" if _use_pg else "?"


def _rows_to_dicts(conn, cursor):
    """Convert cursor results to list of dicts."""
    if _use_pg:
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    return [dict(r) for r in cursor.fetchall()]


def _row_to_dict(conn, cursor):
    """Convert single cursor result to dict or None."""
    if _use_pg:
        row = cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    row = cursor.fetchone()
    return dict(row) if row else None


def create_session(session_id: str, title: str = "") -> None:
    db = _get_db()
    p = _ph()
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(
                f"INSERT INTO sessions (id, created_at, title) VALUES ({p}, {p}, {p}) ON CONFLICT DO NOTHING",
                (session_id, datetime.now().isoformat(), title),
            )
        db.commit()
    else:
        db.execute(
            f"INSERT OR IGNORE INTO sessions (id, created_at, title) VALUES ({p}, {p}, {p})",
            (session_id, datetime.now().isoformat(), title),
        )
        db.commit()
    db.close()


def add_message(session_id: str, turn: int, role: str, content: str, sources: list | None = None) -> None:
    db = _get_db()
    p = _ph()
    sql = f"INSERT INTO messages (session_id, turn, role, content, sources, created_at) VALUES ({p}, {p}, {p}, {p}, {p}, {p})"
    args = (session_id, turn, role, content, json.dumps(sources or []), datetime.now().isoformat())
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, args)
        db.commit()
    else:
        db.execute(sql, args)
        db.commit()
    db.close()


def get_messages(session_id: str, limit: int = 100) -> list[dict]:
    db = _get_db()
    p = _ph()
    sql = f"SELECT role, content, turn, sources FROM messages WHERE session_id = {p} ORDER BY turn, id LIMIT {p}"
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, (session_id, limit))
            result = _rows_to_dicts(db, cur)
    else:
        rows = db.execute(sql, (session_id, limit)).fetchall()
        result = [dict(r) for r in rows]
    db.close()
    return result


def get_turn_count(session_id: str) -> int:
    db = _get_db()
    p = _ph()
    sql = f"SELECT COALESCE(MAX(turn), 0) as max_turn FROM messages WHERE session_id = {p}"
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, (session_id,))
            row = cur.fetchone()
        db.close()
        return row[0] if row else 0
    else:
        row = db.execute(sql, (session_id,)).fetchone()
        db.close()
        return row["max_turn"] if row else 0


def save_summary(session_id: str, summary: str, covers_through_turn: int) -> None:
    db = _get_db()
    p = _ph()
    sql = f"INSERT INTO summaries (session_id, summary, covers_through_turn, created_at) VALUES ({p}, {p}, {p}, {p})"
    args = (session_id, summary, covers_through_turn, datetime.now().isoformat())
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, args)
        db.commit()
    else:
        db.execute(sql, args)
        db.commit()
    db.close()


def get_latest_summary(session_id: str) -> dict | None:
    db = _get_db()
    p = _ph()
    sql = f"SELECT summary, covers_through_turn FROM summaries WHERE session_id = {p} ORDER BY covers_through_turn DESC LIMIT 1"
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, (session_id,))
            result = _row_to_dict(db, cur)
    else:
        row = db.execute(sql, (session_id,)).fetchone()
        result = dict(row) if row else None
    db.close()
    return result


def upsert_summary(session_id: str, summary: str, covers_through_turn: int) -> None:
    db = _get_db()
    p = _ph()
    now = datetime.now().isoformat()
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(
                f"SELECT id FROM summaries WHERE session_id = {p} ORDER BY covers_through_turn DESC LIMIT 1",
                (session_id,),
            )
            existing = cur.fetchone()
            if existing:
                cur.execute(
                    f"UPDATE summaries SET summary = {p}, covers_through_turn = {p}, created_at = {p} WHERE id = {p}",
                    (summary, covers_through_turn, now, existing[0]),
                )
            else:
                cur.execute(
                    f"INSERT INTO summaries (session_id, summary, covers_through_turn, created_at) VALUES ({p}, {p}, {p}, {p})",
                    (session_id, summary, covers_through_turn, now),
                )
        db.commit()
    else:
        existing = db.execute(
            f"SELECT id FROM summaries WHERE session_id = {p} ORDER BY covers_through_turn DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if existing:
            db.execute(
                f"UPDATE summaries SET summary = {p}, covers_through_turn = {p}, created_at = {p} WHERE id = {p}",
                (summary, covers_through_turn, now, existing["id"]),
            )
        else:
            db.execute(
                f"INSERT INTO summaries (session_id, summary, covers_through_turn, created_at) VALUES ({p}, {p}, {p}, {p})",
                (session_id, summary, covers_through_turn, now),
            )
        db.commit()
    db.close()


def get_sessions() -> list[dict]:
    db = _get_db()
    sql = """SELECT s.id, s.created_at, s.title,
           (SELECT content FROM messages WHERE session_id = s.id AND role = 'user' ORDER BY turn LIMIT 1) as first_message,
           (SELECT MAX(turn) FROM messages WHERE session_id = s.id) as turn_count
           FROM sessions s ORDER BY s.created_at DESC LIMIT 50"""
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql)
            result = _rows_to_dicts(db, cur)
    else:
        rows = db.execute(sql).fetchall()
        result = [dict(r) for r in rows]
    db.close()
    return result


def update_session_title(session_id: str, title: str) -> None:
    db = _get_db()
    p = _ph()
    sql = f"UPDATE sessions SET title = {p} WHERE id = {p}"
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, (title, session_id))
        db.commit()
    else:
        db.execute(sql, (title, session_id))
        db.commit()
    db.close()
```

**Step 3: Verify SQLite fallback still works locally**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/python -c "from app.db import create_session, get_sessions; create_session('test-deploy'); print(get_sessions()[:1])"`
Expected: Prints session data, no errors

**Step 4: Commit**

```bash
git add app/db.py
git commit -m "feat: migrate db.py to support PostgreSQL with SQLite fallback"
```

---

### Task 7: Create Dockerfile

**Files:**
- Create: `Dockerfile`

**Step 1: Write Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system deps for psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

**Step 2: Commit**

```bash
git add Dockerfile
git commit -m "build: add Dockerfile for Cloud Run deployment"
```

---

### Task 8: Create .dockerignore

**Files:**
- Create: `.dockerignore`

**Step 1: Write .dockerignore**

```
.env
.venv/
venv/
data/
tests/
scripts/
docs/
__pycache__/
*.pyc
.git/
.gitignore
.DS_Store
models/
*.md
```

**Step 2: Commit**

```bash
git add .dockerignore
git commit -m "build: add .dockerignore to exclude sensitive and unnecessary files"
```

---

### Task 9: Create GitHub Actions deploy workflow

**Files:**
- Create: `.github/workflows/deploy.yml`

**Step 1: Write the workflow**

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

env:
  PROJECT_ID: ${{ vars.GCP_PROJECT_ID }}
  REGION: us-central1
  SERVICE: datdai
  REPO: datdai-repo

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write  # Required for Workload Identity Federation

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ vars.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ vars.GCP_SERVICE_ACCOUNT }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Configure Docker for Artifact Registry
        run: gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev --quiet

      - name: Build and push Docker image
        run: |
          IMAGE="${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.REPO }}/${{ env.SERVICE }}:${{ github.sha }}"
          docker build -t "$IMAGE" .
          docker push "$IMAGE"

      - name: Deploy to Cloud Run
        run: |
          IMAGE="${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.REPO }}/${{ env.SERVICE }}:${{ github.sha }}"
          gcloud run deploy ${{ env.SERVICE }} \
            --image "$IMAGE" \
            --region ${{ env.REGION }} \
            --platform managed \
            --allow-unauthenticated \
            --memory 1Gi \
            --cpu 1 \
            --max-instances 3 \
            --min-instances 0 \
            --concurrency 10 \
            --timeout 120 \
            --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest,QDRANT_URL=QDRANT_URL:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest,SUPABASE_DB_URL=SUPABASE_DB_URL:latest,SUPABASE_URL=SUPABASE_URL:latest,SUPABASE_SERVICE_KEY=SUPABASE_SERVICE_KEY:latest"
```

**Step 2: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/deploy.yml
git commit -m "ci: add GitHub Actions workflow for Cloud Run deployment"
```

---

### Task 10: Build and test Docker image locally

**Step 1: Build the image**

Run: `cd /Users/vuducdung/personal/DatDai && docker build -t datdai-test .`
Expected: Successful build

**Step 2: Run the container with local env**

Run: `docker run --rm -p 8080:8080 --env-file .env datdai-test`
Expected: Server starts, responds to requests (will use SQLite fallback and local storage fallback since SUPABASE_* vars are not in .env)

Note: The container won't have data/ files since they're excluded. This tests the Supabase Storage fallback path -- it will fail to load vocab/amendment_index since neither Supabase nor local files are available. This is expected. In production, Supabase Storage will serve the files.

**Step 3: Stop the container (Ctrl+C)**

No commit needed -- this is a verification step.

---

## Phase 2: Platform Setup (Manual)

These steps are done in browser/CLI, not code changes.

### Task 11: Set up Supabase project

**Step 1: Create project at supabase.com**
- Sign up / log in
- Create new project, name: `datdai`, region: US East, generate a strong DB password
- Wait for project to provision

**Step 2: Run schema SQL in SQL Editor**

Go to SQL Editor and run:

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    turn INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    summary TEXT NOT NULL,
    covers_through_turn INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
```

**Step 3: Create storage bucket**
- Go to Storage > New bucket
- Name: `datdai-data`, private (not public)

**Step 4: Upload reference files**
- Upload `data/vocab.json` to bucket root
- Upload `data/amendment_index.json` to bucket root

**Step 5: Collect credentials**
- Go to Settings > API: copy Project URL and service_role key
- Go to Settings > Database: copy Connection string (use "Connection pooling" mode, port 6543)
- Save these securely (you'll enter them in GCP Secret Manager next)

---

### Task 12: Set up GCP project

**Step 1: Create or select GCP project**

```bash
gcloud projects create datdai-chatbot --name="DatDai Chatbot"
gcloud config set project datdai-chatbot
```

Or use an existing project: `gcloud config set project YOUR_PROJECT_ID`

**Step 2: Enable required APIs**

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com
```

**Step 3: Create Artifact Registry repository**

```bash
gcloud artifacts repositories create datdai-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="DatDai Docker images"
```

---

### Task 13: Create secrets in GCP Secret Manager

**Step 1: Create each secret**

```bash
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets create GEMINI_API_KEY --data-file=-
echo -n "YOUR_QDRANT_URL" | gcloud secrets create QDRANT_URL --data-file=-
echo -n "YOUR_QDRANT_API_KEY" | gcloud secrets create QDRANT_API_KEY --data-file=-
echo -n "YOUR_SUPABASE_DB_URL" | gcloud secrets create SUPABASE_DB_URL --data-file=-
echo -n "YOUR_SUPABASE_URL" | gcloud secrets create SUPABASE_URL --data-file=-
echo -n "YOUR_SUPABASE_SERVICE_KEY" | gcloud secrets create SUPABASE_SERVICE_KEY --data-file=-
```

Replace `YOUR_*` with actual values from .env and Supabase dashboard.

**Step 2: Verify secrets exist**

```bash
gcloud secrets list
```

Expected: All 6 secrets listed.

---

### Task 14: Set up Workload Identity Federation for GitHub Actions

**Step 1: Create Workload Identity Pool**

```bash
gcloud iam workload-identity-pools create "github-pool" \
  --location="global" \
  --display-name="GitHub Actions Pool"
```

**Step 2: Create OIDC Provider**

```bash
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"
```

**Step 3: Create Service Account**

```bash
gcloud iam service-accounts create github-deploy \
  --display-name="GitHub Actions Deploy"
```

**Step 4: Grant roles to the service account**

```bash
PROJECT_ID=$(gcloud config get-value project)

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-deploy@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-deploy@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-deploy@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-deploy@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

**Step 5: Allow GitHub repo to impersonate the service account**

```bash
PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

gcloud iam service-accounts add-iam-policy-binding \
  "github-deploy@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/henryvu27/DatDai"
```

Note: Replace `henryvu27/DatDai` with your actual GitHub `owner/repo`.

**Step 6: Get the Workload Identity Provider resource name**

```bash
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
echo "projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
```

Save this output -- it goes into the GitHub variable `GCP_WORKLOAD_IDENTITY_PROVIDER`.

---

### Task 15: Configure GitHub repository variables

**Step 1: Add variables via gh CLI or GitHub web UI**

```bash
gh variable set GCP_PROJECT_ID --body "datdai-chatbot"
gh variable set GCP_WORKLOAD_IDENTITY_PROVIDER --body "projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
gh variable set GCP_SERVICE_ACCOUNT --body "github-deploy@datdai-chatbot.iam.gserviceaccount.com"
```

Replace `PROJECT_NUMBER` with the actual number from Task 14 Step 6.

These are non-secret configuration values -- safe to store as GitHub variables.

---

## Phase 3: Deploy and Verify

### Task 16: First deployment

**Step 1: Push to main**

All code changes from Phase 1 should already be committed. Push:

```bash
git push origin main
```

**Step 2: Monitor GitHub Actions**

Go to your repo's Actions tab and watch the deploy workflow run.

**Step 3: Verify Cloud Run service is running**

```bash
gcloud run services describe datdai --region=us-central1 --format='value(status.url)'
```

Visit the URL in a browser. Test a chat query.

---

### Task 17: Map custom domain

**Step 1: Add domain mapping in Cloud Run**

```bash
gcloud run domain-mappings create --service=datdai --domain=datdai.henryvu.io --region=us-central1
```

**Step 2: Add CNAME at DNS registrar**

At your domain registrar, add:
- Type: CNAME
- Name: datdai
- Value: ghs.googlehosted.com

**Step 3: Wait for SSL provisioning**

SSL certificate auto-provisions. Can take 15-30 minutes. Verify:

```bash
gcloud run domain-mappings describe --domain=datdai.henryvu.io --region=us-central1
```

**Step 4: Test**

Visit `https://datdai.henryvu.io` in a browser.
