"""FastAPI server for the land law chatbot."""
import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import chat, db

app = FastAPI(title="Chatbot Luat Dat Dai")


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
async def chat_endpoint(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Cau hoi khong duoc de trong.")

    session_id = request.session_id or chat.create_new_session()

    try:
        result = await chat.handle_message(session_id, request.question)
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
