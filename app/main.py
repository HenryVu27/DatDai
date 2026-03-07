"""FastAPI server for the land law chatbot."""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app import chat, db
from app.observability import flush as langfuse_flush, langfuse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm heavy resources at startup so the first request isn't penalised."""
    t0 = time.time()
    logger.info("Startup warmup: begin")

    # DB connection pool + schema init
    try:
        conn = db._get_db()
        db._release_db(conn)
        logger.info("Startup warmup: DB ready")
    except Exception as e:
        logger.warning("Startup warmup: DB init failed: %s", e)

    # Vocab + amendment index (triggers Supabase download, then lru_cache keeps it)
    try:
        from app.storage import load_vocab, load_amendment_index
        await asyncio.to_thread(load_vocab)
        await asyncio.to_thread(load_amendment_index)
        logger.info("Startup warmup: vocab + amendment index loaded")
    except Exception as e:
        logger.warning("Startup warmup: storage load failed: %s", e)

    # KnowledgeStore singleton + Qdrant connection
    try:
        from app.rag.retriever import _get_store
        store = _get_store()
        await asyncio.to_thread(store._get_client)
        logger.info("Startup warmup: Qdrant connected")
    except Exception as e:
        logger.warning("Startup warmup: Qdrant connect failed: %s", e)

    # Reranker model load (from cached ONNX files in Docker image)
    try:
        from app.rag.retriever import _get_reranker
        await asyncio.to_thread(_get_reranker)
        logger.info("Startup warmup: reranker loaded")
    except Exception as e:
        logger.warning("Startup warmup: reranker load failed: %s", e)

    logger.info("Startup warmup: done in %.2fs", time.time() - t0)
    yield
    langfuse_flush()


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Chatbot Luat Dat Dai", lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Quá nhiều yêu cầu. Vui lòng thử lại sau 1 phút."},
    )


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=5000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    trace_id: str | None = None


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    first_message: str | None
    turn_count: int | None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    if not chat_request.question.strip():
        raise HTTPException(status_code=400, detail="Cau hoi khong duoc de trong.")

    session_id = chat_request.session_id or await asyncio.to_thread(chat.create_new_session)

    try:
        result = await chat.handle_message(session_id, chat_request.question)
        # Get Langfuse trace ID for feedback linking
        trace_id = None
        if langfuse:
            try:
                trace_id = langfuse.get_current_trace_id()
            except Exception:
                pass
        return ChatResponse(**result, trace_id=trace_id)
    except Exception:
        logging.getLogger(__name__).exception("Error handling chat question")
        raise HTTPException(status_code=500, detail="Lỗi xử lý câu hỏi. Vui lòng thử lại.")


@app.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream_endpoint(request: Request, chat_request: ChatRequest):
    if not chat_request.question.strip():
        raise HTTPException(status_code=400, detail="Cau hoi khong duoc de trong.")

    session_id = chat_request.session_id or await asyncio.to_thread(chat.create_new_session)

    async def event_generator():
        try:
            async for event_type, data in chat.handle_message_stream(session_id, chat_request.question):
                yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("Error in streaming chat")
            yield f"event: error\ndata: {json.dumps({'detail': 'Lỗi xử lý câu hỏi. Vui lòng thử lại.'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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


class FeedbackRequest(BaseModel):
    trace_id: str
    score: int  # 1 = thumbs up, 0 = thumbs down


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    if not langfuse:
        return {"status": "tracing_disabled"}
    langfuse.score(
        trace_id=req.trace_id,
        name="user_feedback",
        value=req.score,
        data_type="NUMERIC",
    )
    return {"status": "ok"}


# Serve static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))
