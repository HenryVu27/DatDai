"""Chat orchestrator: ties RAG, memory, and LLM together."""
import uuid

from app import db, llm, rag
from app.config import CONTEXT_WINDOW_TURNS, CONTEXT_MAX_CHARS, SUMMARY_INTERVAL_TURNS

SYSTEM_PROMPT = """Ban la chuyen gia tu van phap luat dat dai Viet Nam. Ban tra loi cau hoi dua tren cac van ban phap luat duoc cung cap.

Nguyen tac:
1. Chi tra loi dua tren noi dung van ban phap luat duoc cung cap trong phan "Tai lieu tham khao"
2. Trich dan cu the so dieu, khoan, diem va ten van ban khi tra loi
3. Neu thong tin khong co trong tai lieu, noi ro "Toi khong tim thay thong tin nay trong cac van ban hien co"
4. Tra loi bang tieng Viet, ro rang, de hieu cho nguoi dan thuong
5. Khi co nhieu van ban lien quan, neu ro moi quan he giua chung (vi du: Luat quy dinh chung, Nghi dinh huong dan chi tiet)
6. Neu cau hoi mo ho, hoi lai de lam ro truoc khi tra loi

He thong van ban:
- Luat Dat Dai 2024 (Luat so 31/2024/QH15) - luat goc
- Nghi dinh 71/2024 - ve gia dat
- Nghi dinh 88/2024 - boi thuong, ho tro, tai dinh cu
- Nghi dinh 102/2024 - quy dinh chi tiet thi hanh Luat Dat Dai
- Nghi dinh 103/2024 - tien su dung dat, tien thue dat
- Nghi dinh 151/2025 - phan dinh tham quyen chinh quyen dia phuong
- Nghi dinh 226/2025 - sua doi bo sung cac nghi dinh
- Nghi quyet 254/2025 - thao go vuong mac thi hanh Luat Dat Dai
- Nghi dinh 49/2026 - sua doi bo sung nghi dinh chi tiet Luat Dat Dai"""


async def handle_message(session_id: str, user_message: str) -> dict:
    """Process a user message and return response with sources."""
    # Ensure session exists
    db.create_session(session_id)

    # Get conversation history
    history = db.get_messages(session_id)
    turn = db.get_turn_count(session_id) + 1

    # Save user message
    db.add_message(session_id, turn, "user", user_message)

    # Determine complexity for model routing
    complexity = llm.classify_complexity(user_message)
    model = "pro" if complexity == "complex" else "flash"

    # Rewrite query with conversation context for better retrieval
    search_query = await rag.rewrite_query(user_message, _to_history_dicts(history))

    # Retrieve relevant chunks
    chunks = rag.search(search_query)
    context = rag.build_context(chunks)
    sources = rag.extract_sources(chunks)

    # Build conversation context for LLM
    llm_history = _build_llm_context(history, session_id)

    # Build the prompt with RAG context
    prompt = _build_prompt(user_message, context)

    # Generate response
    response = await llm.generate(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        history=llm_history,
        model=model,
        temperature=0.3,
        max_tokens=4000,
    )

    # Save assistant response
    db.add_message(session_id, turn, "assistant", response, sources)

    # Generate rolling summary if needed
    if turn % SUMMARY_INTERVAL_TURNS == 0:
        await _generate_summary(session_id, history, turn)

    # Auto-generate title for new sessions
    if turn == 1:
        await _generate_title(session_id, user_message, response)

    return {
        "answer": response,
        "sources": sources,
        "session_id": session_id,
    }


def _to_history_dicts(rows: list[dict]) -> list[dict]:
    """Convert DB rows to simple history dicts."""
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def _build_llm_context(history: list[dict], session_id: str) -> list[dict]:
    """Build trimmed conversation context for LLM."""
    # Get rolling summary if available
    summary = db.get_latest_summary(session_id)

    msgs = _to_history_dicts(history)

    # Trim to context window
    if len(msgs) > CONTEXT_WINDOW_TURNS * 2:
        msgs = msgs[-(CONTEXT_WINDOW_TURNS * 2):]

    # Trim by character count
    total_chars = 0
    trimmed = []
    for msg in reversed(msgs):
        total_chars += len(msg["content"])
        if total_chars > CONTEXT_MAX_CHARS:
            break
        trimmed.insert(0, msg)

    # Prepend summary if we trimmed messages
    if summary and len(trimmed) < len(_to_history_dicts(history)):
        trimmed.insert(0, {
            "role": "user",
            "content": f"[Tom tat hoi thoai truoc]: {summary['summary']}",
        })
        trimmed.insert(1, {
            "role": "model",
            "content": "Da hieu, toi se tham khao thong tin nay de tra loi tiep.",
        })

    return trimmed


def _build_prompt(question: str, context: str) -> str:
    """Build the final prompt with RAG context."""
    if context:
        return f"""Tai lieu tham khao:
{context}

---
Cau hoi cua nguoi dung: {question}

Hay tra loi dua tren tai lieu tham khao o tren. Trich dan cu the dieu, khoan, ten van ban."""
    else:
        return f"""Cau hoi cua nguoi dung: {question}

Luu y: Khong tim thay tai lieu lien quan trong co so du lieu. Hay tra loi dua tren kien thuc chung ve phap luat dat dai Viet Nam va luu y nguoi dung rang cau tra loi chua duoc xac minh tu van ban cu the."""


async def _generate_summary(session_id: str, history: list[dict], current_turn: int) -> None:
    """Generate a rolling summary of conversation."""
    try:
        msgs = _to_history_dicts(history)
        recent = msgs[-12:]  # Last 6 exchanges
        convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in recent)

        prompt = f"""Tom tat ngan gon cuoc hoi thoai sau ve phap luat dat dai.
Giu lai cac dieu khoan, van ban da duoc thao luan va cac cau hoi chinh.
Chi tra ve ban tom tat, khong giai thich.

{convo}

Tom tat:"""

        summary = await llm.generate(prompt, model="flash", temperature=0.0, max_tokens=500)
        if summary and len(summary) > 20:
            db.save_summary(session_id, summary.strip(), current_turn)
    except Exception:
        pass


async def _generate_title(session_id: str, question: str, answer: str) -> None:
    """Auto-generate a session title from first exchange."""
    try:
        prompt = f"""Tao tieu de ngan gon (duoi 50 ky tu) cho cuoc hoi thoai bat dau voi cau hoi sau.
Chi tra ve tieu de, khong giai thich, khong dau ngoac kep.

Cau hoi: {question[:200]}

Tieu de:"""

        title = await llm.generate(prompt, model="flash", temperature=0.0, max_tokens=60)
        if title and len(title.strip()) > 3:
            db.update_session_title(session_id, title.strip()[:80])
    except Exception:
        pass


def create_new_session() -> str:
    """Create a new chat session and return its ID."""
    session_id = str(uuid.uuid4())
    db.create_session(session_id)
    return session_id
