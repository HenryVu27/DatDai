# GCP Deployment Design

## Overview

Deploy the DatDai chatbot (FastAPI + RAG pipeline) to production using GCP Cloud Run, with Supabase for data persistence and storage.

## Architecture

```
                    GitHub Actions (CI/CD)
                         |
                         v
               GCP Artifact Registry
                    (Docker image)
                         |
                         v
    datdai.henryvu.io --> Cloud Run
                         |
              +----------+----------+
              |          |          |
              v          v          v
         Supabase    Supabase    Qdrant Cloud
        PostgreSQL   Storage     (vector DB)
        (chat DB)   (ref files)     |
                                    v
                              Gemini API

    Secrets: GCP Secret Manager
```

## Components

### Cloud Run (Compute)

- Region: us-central1
- Min instances: 0 (scale to zero)
- Max instances: 3
- Memory: 1GB (reranker model ~500MB)
- CPU: 1 vCPU
- Concurrency: 10
- Timeout: 120s
- Custom domain: datdai.henryvu.io

### Supabase PostgreSQL (Chat DB)

Replaces SQLite. Same schema, different driver.

Tables:
- sessions (id, created_at, title)
- messages (id, session_id, turn, role, content, sources, created_at)
- summaries (id, session_id, summary, covers_through_turn, created_at)

Migration changes in app/db.py:
- sqlite3 -> psycopg2
- ? placeholders -> %s
- INSERT OR IGNORE -> INSERT ... ON CONFLICT DO NOTHING
- AUTOINCREMENT -> SERIAL
- Connection via DATABASE_URL from Secret Manager
- Use Supabase PgBouncer (port 6543) for connection pooling

### Supabase Storage (Reference Files)

Bucket: datdai-data (private)

Files:
- chunks/all_chunks.json (8.6 MB)
- vocab.json (33 KB)
- amendment_index.json (4.5 KB)
- Future document files

New module app/storage.py:
- Downloads files from Supabase Storage on first access
- Caches in memory (lazy loading)
- Existing code updated to use storage.py instead of local filesystem paths

### Qdrant Cloud (Vector DB)

No changes. Already hosted externally.

### Gemini API

No changes. Already a Google service.

## Secrets Management

All secrets stored in GCP Secret Manager, mounted as env vars in Cloud Run.

| Secret Name | Description |
|---|---|
| GEMINI_API_KEY | Gemini API key |
| QDRANT_URL | Qdrant Cloud endpoint |
| QDRANT_API_KEY | Qdrant auth token |
| SUPABASE_DB_URL | PostgreSQL connection string (pooled) |
| SUPABASE_URL | Supabase project URL |
| SUPABASE_SERVICE_KEY | Supabase service role key |

Local dev continues using .env (git-ignored).

GitHub Actions authenticates via Workload Identity Federation (OIDC) -- no service account JSON keys stored anywhere.

## Docker

Base image: python:3.12-slim
New dependencies: psycopg2-binary, supabase
Entrypoint: uvicorn app.main:app --host 0.0.0.0 --port $PORT

.dockerignore excludes: data/, .env, .venv/, __pycache__/, .git/, tests/, scripts/

## CI/CD (GitHub Actions)

Workflow: .github/workflows/deploy.yml
Trigger: push to main

Steps:
1. Checkout code
2. Authenticate to GCP via Workload Identity Federation
3. Build Docker image
4. Push to Artifact Registry
5. Deploy to Cloud Run with secrets from Secret Manager

GitHub repository variables (non-secret):
- GCP_PROJECT_ID
- GCP_WORKLOAD_IDENTITY_PROVIDER

## Custom Domain

- CNAME record: datdai -> ghs.googlehosted.com (at DNS registrar)
- Cloud Run domain mapping: datdai.henryvu.io
- SSL: auto-provisioned by Google (free)

## Cost Estimate

$0-5/month within free tiers for low-moderate traffic:
- Cloud Run: 2M requests/month free
- Supabase: 500MB DB + 1GB storage free
- Secret Manager: 10K access/month free
- Artifact Registry: 500MB free

## Manual Setup Required

### Supabase
1. Create project (free tier, US East region)
2. Run schema SQL in SQL Editor
3. Create storage bucket, upload reference files
4. Copy project URL, service role key, DB connection string

### GCP
1. Create project, enable APIs (Cloud Run, Artifact Registry, Secret Manager, IAM)
2. Create Artifact Registry repo (us-central1)
3. Create 6 secrets in Secret Manager
4. Set up Workload Identity Federation for GitHub Actions
5. Map custom domain after first deploy

### DNS Registrar
1. Add CNAME: datdai -> ghs.googlehosted.com

### GitHub
1. Add repository variables: GCP_PROJECT_ID, GCP_WORKLOAD_IDENTITY_PROVIDER
