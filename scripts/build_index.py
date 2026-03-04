"""
Build vector index from chunks using OpenAI embeddings and ChromaDB.
"""
import json
import os
import sys
import time

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

CHUNKS_FILE = os.path.join(PROJECT_ROOT, "data", "chunks", "all_chunks.json")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "data", "chromadb")
COLLECTION_NAME = "dat_dai_law"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 50  # OpenAI embedding API batch limit


def load_chunks() -> list[dict]:
    """Load chunks from JSON file."""
    if not os.path.exists(CHUNKS_FILE):
        print(f"Khong tim thay file chunks: {CHUNKS_FILE}")
        print("Hay chay chunk_documents.py truoc.")
        sys.exit(1)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def create_embeddings(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Create embeddings for a batch of texts using OpenAI API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def main():
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "sk-your-key-here":
        print("Loi: Chua cau hinh OPENAI_API_KEY trong file .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load chunks
    chunks = load_chunks()
    print(f"Da load {len(chunks)} chunks")

    # Setup ChromaDB
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if exists, then recreate
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Da xoa collection cu: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Van ban phap luat dat dai Viet Nam"},
    )

    # Process in batches
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Tao embeddings va luu vao ChromaDB ({total_batches} batches)...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        # Prepare texts for embedding - use content + metadata for better retrieval
        texts = []
        for chunk in batch:
            # Prepend metadata to content for richer embedding
            prefix = chunk.get("metadata_str", "")
            content = chunk.get("content", "")
            texts.append(f"{prefix}\n\n{content}" if prefix else content)

        # Create embeddings with retry
        for attempt in range(3):
            try:
                embeddings = create_embeddings(texts, client)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Loi, thu lai sau 5 giay... ({e})")
                    time.sleep(5)
                else:
                    print(f"  Loi khong the khac phuc: {e}")
                    raise

        # Prepare data for ChromaDB
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        documents = [chunk["content"] for chunk in batch]
        metadatas = [
            {
                "doc_name": chunk.get("doc_name", ""),
                "chapter": chunk.get("chapter", ""),
                "dieu": chunk.get("dieu", ""),
                "dieu_title": chunk.get("dieu_title", ""),
                "khoan": chunk.get("khoan", ""),
                "metadata_str": chunk.get("metadata_str", ""),
            }
            for chunk in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"  Batch {batch_num}/{total_batches}: {len(batch)} chunks")

        # Rate limiting - avoid hitting OpenAI API limits
        if batch_num < total_batches:
            time.sleep(0.5)

    print(f"\nHoan thanh! Da luu {len(chunks)} chunks vao ChromaDB tai: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")

    # Verify
    count = collection.count()
    print(f"Xac nhan: {count} documents trong collection")


if __name__ == "__main__":
    main()
