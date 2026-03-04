"""
FastAPI server for the land law chatbot.
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from app.rag import RAGEngine

app = FastAPI(title="Chatbot Luat Dat Dai")

# Initialize RAG engine
rag_engine = None


@app.on_event("startup")
def startup():
    global rag_engine
    try:
        rag_engine = RAGEngine()
    except Exception as e:
        print(f"Loi khoi tao RAG engine: {e}")
        print("Hay dam bao da chay build_index.py va cau hinh OPENAI_API_KEY")


class ChatRequest(BaseModel):
    question: str
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not rag_engine:
        raise HTTPException(
            status_code=503,
            detail="RAG engine chua san sang. Hay kiem tra OPENAI_API_KEY va ChromaDB.",
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Cau hoi khong duoc de trong.")

    try:
        result = rag_engine.chat(request.question, request.history)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi xu ly cau hoi: {str(e)}")


# Serve static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))
