"""Application configuration."""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Model IDs -- primary (Gemini 3.x preview)
ORCHESTRATOR_MODEL = "gemini-3-flash-preview"
GENERATOR_PRO_MODEL = "gemini-3.1-pro-preview"
GENERATOR_FLASH_MODEL = "gemini-3-flash-preview"
UTILITY_MODEL = "gemini-3.1-flash-lite-preview"
EMBEDDING_MODEL = "gemini-embedding-001"

# Model IDs -- fallbacks (stable Gemini 2.5)
FALLBACK_PRO_MODEL = "gemini-2.5-pro"
FALLBACK_FLASH_MODEL = "gemini-2.5-flash"
FALLBACK_UTILITY_MODEL = "gemini-2.5-flash-lite"

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
DB_PATH = os.path.join(DATA_DIR, "chat.db")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = "dat_dai_law"

# Retrieval
RAG_TOP_K = 8
RAG_RERANK_CANDIDATES = 20
RAG_RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
RAG_RELEVANCE_THRESHOLD = 0.15
RAG_USE_RERANKER = True

# Context management
CONTEXT_MAX_TURN_GROUPS = 6
CONTEXT_MAX_CHARS = 30_000
