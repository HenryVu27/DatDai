"""Application configuration."""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_PRO_MODEL = "gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

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

# RAG
RAG_TOP_K = 8
RAG_TOP_K_COMPLEX = 16
RAG_RERANK_CANDIDATES = 20
RAG_RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
RAG_RELEVANCE_THRESHOLD = 0.15
RAG_USE_HYDE = True
RAG_USE_RERANKER = True
RAG_USE_QUERY_REWRITE = True

# Context management
CONTEXT_WINDOW_TURNS = 8
CONTEXT_MAX_CHARS = 100_000
SUMMARY_INTERVAL_TURNS = 6
