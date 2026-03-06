"""FastAPI server for the land law chatbot."""
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app import chat, db

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Chatbot Luat Dat Dai")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Qua nhieu yeu cau. Vui long thu lai sau 1 phut."},
    )


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    first_message: str | None
    turn_count: int | None


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    if not chat_request.question.strip():
        raise HTTPException(status_code=400, detail="Cau hoi khong duoc de trong.")

    session_id = chat_request.session_id or chat.create_new_session()

    try:
        result = await chat.handle_message(session_id, chat_request.question)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi xu ly cau hoi: {str(e)}")


@app.get("/sessions")
def list_sessions() -> list[SessionResponse]:
    sessions = db.get_sessions()
    return [SessionResponse(**s) for s in sessions]


@app.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    messages = db.get_messages(session_id)
    return messages


@app.post("/sessions")
def create_session():
    session_id = chat.create_new_session()
    return {"session_id": session_id}


# Serve static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))
