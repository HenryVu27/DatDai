"""
RAG (Retrieval-Augmented Generation) logic for the land law chatbot.
"""
import os

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

CHROMA_DIR = os.path.join(PROJECT_ROOT, "data", "chromadb")
COLLECTION_NAME = "dat_dai_law"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
TOP_K = 8

SYSTEM_PROMPT = """Ban la chuyen gia tu van phap luat dat dai Viet Nam. Ban co kien thuc sau rong ve he thong van ban phap luat dat dai hien hanh gom:

1. Luat Dat Dai 2024 (Luat so 31/2024/QH15, co hieu luc tu 01/08/2024) - 16 Chuong, 260 Dieu
2. Nghi dinh 71/2024/ND-CP (27/06/2024) - Quy dinh ve gia dat - 5 Chuong, 37 Dieu
3. Nghi dinh 88/2024/ND-CP (15/07/2024) - Boi thuong, ho tro, tai dinh cu khi thu hoi dat - 3 Chuong, 32 Dieu
4. Nghi dinh 102/2024/ND-CP (30/07/2024) - Quy dinh chi tiet thi hanh Luat Dat dai - 10 Chuong, 113 Dieu
5. Nghi dinh 103/2024/ND-CP (30/07/2024) - Tien su dung dat, tien thue dat - 5 Chuong, 54 Dieu
6. Nghi dinh 151/2025/ND-CP (12/06/2025) - Phan dinh tham quyen chinh quyen dia phuong 02 cap - 3 Chuong, 23 Dieu
7. Nghi dinh 226/2025/ND-CP (15/08/2025) - Sua doi bo sung cac ND 71, 88, 101, 102, 112, 151 - 10 Dieu
8. Nghi quyet 254/2025/QH15 - Thao go vuong mac thi hanh Luat Dat dai
9. Nghi dinh 49/2026/ND-CP (31/01/2026) - Sua doi bo sung cac ND thi hanh Luat Dat dai - 5 Chuong, 22 Dieu

Nhiem vu cua ban:
1. Tra loi cau hoi cua nguoi dung bang tieng Viet, ro rang va de hieu
2. LUON trich dan cu the dieu, khoan cua van ban phap luat lien quan
3. Neu co nhieu van ban lien quan, hay liet ke tat ca va giai thich moi lien he
4. Neu khong chac chan hoac thong tin khong co trong tai lieu, hay noi ro rang ban khong tim thay thong tin cu the trong cac van ban da co
5. Khi tra loi, hay sap xep theo thu tu: quy dinh chinh (Luat) -> quy dinh chi tiet (Nghi dinh) -> huong dan cu the

Luu y quan trong ve hieu luc va thu tu uu tien:
- Luat Dat Dai 2024 co hieu luc phap ly CAO NHAT
- ND 49/2026 (31/01/2026) la nghi dinh moi nhat, sua doi ND 71, 88, 101, 102, 151, 226
- ND 226/2025 (15/08/2025) sua doi 6 nghi dinh: 71, 88, 101, 102, 112, 151
- ND 151/2025 (01/07/2025) thay the mot so dieu cua cac nghi dinh truoc, het hieu luc truoc 01/03/2027
- NQ 254/2025 sua doi truc tiep cac ND 71, 88, 101, 102
- Khi co xung dot, uu tien: van ban moi hon > van ban cu, van ban hieu luc phap ly cao > thap"""


class RAGEngine:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-your-key-here":
            raise ValueError("Chua cau hinh OPENAI_API_KEY trong file .env")

        self.openai_client = OpenAI(api_key=api_key)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Search for relevant chunks using similarity search."""
        # Create embedding for query
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        query_embedding = response.data[0].embedding

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return chunks

    def build_context(self, chunks: list[dict]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            source = meta.get("metadata_str", "")
            content = chunk["content"]
            context_parts.append(
                f"[Nguon {i}: {source}]\n{content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def build_sources(self, chunks: list[dict]) -> list[dict]:
        """Extract source references from chunks."""
        sources = []
        seen = set()
        for chunk in chunks:
            meta = chunk["metadata"]
            source_key = f"{meta.get('doc_name', '')}|{meta.get('dieu', '')}"
            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "doc_name": meta.get("doc_name", ""),
                    "chapter": meta.get("chapter", ""),
                    "dieu": meta.get("dieu", ""),
                    "dieu_title": meta.get("dieu_title", ""),
                })
        return sources

    def chat(self, question: str, history: list[dict] | None = None) -> dict:
        """Process a user question and return answer with sources."""
        # Search for relevant chunks
        chunks = self.search(question)
        context = self.build_context(chunks)
        sources = self.build_sources(chunks)

        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history if provided
        if history:
            for msg in history[-6:]:  # Keep last 3 exchanges
                messages.append(msg)

        # Add current question with context
        user_message = f"""Cau hoi cua nguoi dung: {question}

Tai lieu tham khao:
{context}

Hay tra loi cau hoi dua tren cac tai lieu tham khao tren. Trich dan cu the dieu, khoan cua van ban."""

        messages.append({"role": "user", "content": user_message})

        # Get response from GPT
        response = self.openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=2000,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources,
        }
