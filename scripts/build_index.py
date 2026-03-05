"""Build Qdrant vector index from chunks using Gemini embeddings.
Hybrid: dense (Gemini) + sparse (TF with IDF modifier).
Supports resume via --fresh flag to force rebuild.
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, Modifier,
    PointStruct, SparseVector, NamedVector, NamedSparseVector,
    PayloadSchemaType,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

CHUNKS_FILE = os.path.join(PROJECT_ROOT, "data", "chunks", "all_chunks.json")
VOCAB_FILE = os.path.join(PROJECT_ROOT, "data", "vocab.json")
COLLECTION_NAME = "dat_dai_law"
EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 10
WAIT_BETWEEN_BATCHES = 4

import re

DOC_ID_MAP = {
    "luat dat dai": "ldd2024",
    "102": "nd102",
    "103": "nd103",
    "151": "nd151",
    "226": "nd226",
    "254": "nq254",
    "49": "nd49",
    "71": "nd71",
    "88": "nd88",
    "12": "nd12",
    "50": "nd50",
}


def _to_doc_id(doc_name: str) -> str:
    name_lower = doc_name.lower()
    for key, doc_id in DOC_ID_MAP.items():
        if key in name_lower:
            return doc_id
    return name_lower.replace(" ", "_")[:20]


def tokenize_vi(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", " ", text)
    return [w for w in text.split() if len(w) > 1]


def build_vocab(chunks: list[dict]) -> dict:
    """Build vocabulary mapping token -> index from all chunks."""
    all_tokens = set()
    for chunk in chunks:
        tokens = tokenize_vi(chunk.get("content", ""))
        all_tokens.update(tokens)
    vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}
    return vocab


def text_to_sparse(text: str, vocab: dict) -> SparseVector:
    tokens = tokenize_vi(text)
    counts = Counter(tokens)
    indices = []
    values = []
    for token, count in sorted(counts.items()):
        if token in vocab:
            indices.append(vocab[token])
            values.append(float(count))
    return SparseVector(indices=indices, values=values)


def load_chunks() -> list[dict]:
    if not os.path.exists(CHUNKS_FILE):
        print(f"Khong tim thay file chunks: {CHUNKS_FILE}")
        print("Hay chay chunk_documents.py truoc.")
        sys.exit(1)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("Loi: Chua cau hinh GEMINI_API_KEY trong file .env")
        sys.exit(1)

    gemini = genai.Client(api_key=api_key)

    chunks = load_chunks()
    print(f"Da load {len(chunks)} chunks")

    # Build vocabulary for sparse vectors
    print("Xay dung vocabulary...")
    vocab = build_vocab(chunks)
    print(f"Vocabulary: {len(vocab)} tokens")

    # Save vocab for runtime use
    os.makedirs(os.path.dirname(VOCAB_FILE), exist_ok=True)
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # Setup Qdrant cloud
    qdrant_url = os.getenv("QDRANT_URL", ":memory:")
    qdrant_key = os.getenv("QDRANT_API_KEY", "")
    if qdrant_url and qdrant_url != ":memory:":
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key or None)
    else:
        client = QdrantClient(location=":memory:")

    # Get embedding dimension from a test embed
    print("Kiem tra embedding dimension...")
    test_result = gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=["test"],
    )
    dim = len(test_result.embeddings[0].values)
    print(f"Embedding dimension: {dim}")

    # Create collection
    if client.collection_exists(COLLECTION_NAME):
        if "--fresh" in sys.argv:
            client.delete_collection(COLLECTION_NAME)
            print("Da xoa collection cu")
        else:
            info = client.get_collection(COLLECTION_NAME)
            if info.points_count >= len(chunks):
                print(f"Collection da co {info.points_count} points. Da xong!")
                return
            print(f"Collection co {info.points_count} points, can {len(chunks)}. Xay lai...")
            client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=Modifier.IDF),
        },
    )

    # Create payload indexes for filtering
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_name",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="dieu",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    # Index in batches
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Indexing {len(chunks)} chunks ({total_batches} batches)...")

    indexed = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        # Prepare texts for embedding
        texts = []
        for chunk in batch:
            prefix = chunk.get("metadata_str", "")
            content = chunk.get("content", "")
            text = f"{prefix}\n\n{content}" if prefix else content
            if len(text) > 8000:
                text = text[:8000]
            texts.append(text)

        # Get embeddings with retry
        embeddings = None
        for attempt in range(5):
            try:
                result = gemini.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=texts,
                )
                embeddings = [e.values for e in result.embeddings]
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait = 15 * (attempt + 1)
                    print(f"  Rate limit, doi {wait}s... (batch {batch_num})", flush=True)
                    time.sleep(wait)
                elif attempt < 4:
                    wait = 5 * (attempt + 1)
                    print(f"  Loi, thu lai sau {wait}s... ({e})", flush=True)
                    time.sleep(wait)
                else:
                    print(f"  Loi khong khac phuc: {e}", flush=True)
                    print(f"  Da index {indexed} chunks. Chay lai de tiep tuc.")
                    return

        if embeddings is None:
            print(f"  Batch {batch_num} that bai. Da index {indexed}.")
            return

        # Build points
        points = []
        for j, chunk in enumerate(batch):
            sparse = text_to_sparse(chunk.get("content", ""), vocab)
            points.append(PointStruct(
                id=i + j,
                vector={
                    "dense": embeddings[j],
                    "sparse": sparse,
                },
                payload={
                    "text": chunk.get("content", ""),
                    "doc_name": chunk.get("doc_name", ""),
                    "doc_id": _to_doc_id(chunk.get("doc_name", "")),
                    "chapter": chunk.get("chapter", ""),
                    "dieu": chunk.get("dieu", ""),
                    "dieu_title": chunk.get("dieu_title", ""),
                    "khoan": chunk.get("khoan", ""),
                    "metadata_str": chunk.get("metadata_str", ""),
                    "chunk_id": chunk.get("chunk_id", i),
                },
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        indexed += len(batch)
        print(f"  Batch {batch_num}/{total_batches}: {len(batch)} chunks (tong: {indexed})", flush=True)

        if batch_num < total_batches:
            time.sleep(WAIT_BETWEEN_BATCHES)

    info = client.get_collection(COLLECTION_NAME)
    print(f"\nHoan thanh! {info.points_count} points trong Qdrant tai: {qdrant_url}")


if __name__ == "__main__":
    main()
