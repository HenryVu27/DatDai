"""Agentic chat orchestrator.

Two-stage pipeline:
1. Orchestrator (Flash 3) decides what tools to call
2. Generator (Pro 3.1 / Flash 3) produces the answer
"""
import asyncio
import json
import logging
import re
import time
import uuid

from app import db, llm
from app.config import CONTEXT_MAX_TURN_GROUPS, CONTEXT_MAX_CHARS
from app.rag.retriever import (
    search_legal_docs, lookup_amendment, lookup_specific_dieu,
    build_context, extract_sources,
)
from app.rag.query_rewriter import extract_filters
from app.rag.citation_check import verify_citations

logger = logging.getLogger(__name__)

# -- System prompts (static, XML-tagged) --

SYSTEM_BASE = """<identity>
Ban la chuyen gia tu van phap luat dat dai Viet Nam. Ban tra loi cau hoi dua tren cac van ban phap luat duoc cung cap.
Tra loi bang tieng Viet, ro rang, de hieu cho nguoi dan thuong.
</identity>

<boundaries>
1. Chi tra loi dua tren noi dung van ban phap luat duoc cung cap trong phan "Tai lieu tham khao"
2. Trich dan cu the so dieu, khoan, diem va ten van ban khi tra loi
3. Neu thong tin khong co trong tai lieu, noi ro "Toi khong tim thay thong tin nay trong cac van ban hien co"
4. Khi co nhieu van ban lien quan, neu ro moi quan he giua chung (vi du: Luat quy dinh chung, Nghi dinh huong dan chi tiet)
5. Neu cau hoi mo ho, hoi lai de lam ro truoc khi tra loi
</boundaries>

<legal-hierarchy>
Luat Dat Dai 2024 (31/2024/QH15) - luat goc
ND 71/2024 - gia dat [sua doi boi: ND 226, ND 49]
ND 88/2024 - boi thuong, ho tro, tai dinh cu [sua doi boi: ND 226, ND 49]
ND 102/2024 - chi tiet thi hanh [sua doi boi: ND 226, ND 49]
ND 103/2024 - tien su dung dat, tien thue dat [sua doi boi: ND 50]
ND 151/2025 - phan dinh tham quyen [sua doi boi: ND 226, ND 49]
ND 226/2025 - sua doi 4 ND [sua doi boi: ND 49]
NQ 254/2025 - thao go vuong mac
ND 49/2026 - sua doi moi nhat cac ND
ND 12/2024 - chuyen tiep gia dat
ND 50/2026 - chi tiet NQ 254 ve tien su dung dat, tien thue dat
Thu tu uu tien: ND moi nhat > ND cu > Luat goc
</legal-hierarchy>"""

ORCHESTRATOR_TOOLS_SECTION = """
<tools>
Ban co the goi cac tool sau de tra cuu thong tin. Tra ve JSON hop le.

1. search_legal_docs: Tim kiem van ban phap luat bang ngon ngu tu nhien.
   Params: {"query": "cau truy van", "filters": {"doc_ids": ["nd102"], "dieu": "Dieu 15"}}
   - query: viet ro rang, day du ngu canh, khong dung dai tu
   - filters: tuy chon, chi dinh khi biet chinh xac van ban/dieu

2. lookup_amendment: Tra cuu cac sua doi giua cac nghi dinh.
   Params: {"target_doc": "nd102", "source_doc": "nd49", "dieu": "Dieu 15"}
   - target_doc: bat buoc - van ban bi sua doi
   - source_doc: tuy chon - van ban sua doi (neu biet)
   - dieu: tuy chon - so dieu cu the

3. lookup_specific_dieu: Lay toan bo noi dung mot dieu cu the.
   Params: {"doc_id": "ldd2024", "dieu": "Dieu 79"}
   - Dung khi biet chinh xac dieu va van ban can tra cuu

Chon tool phu hop:
- Biet chinh xac dieu + van ban -> lookup_specific_dieu
- Hoi ve sua doi giua cac ND -> lookup_amendment (+ search_legal_docs neu can them context)
- Cau hoi chung, khong biet dieu cu the -> search_legal_docs
- Cau hoi phuc tap, nhieu van ban -> nhieu tool cung luc
</tools>

<retrieval-policy>
QUY TAC BAT BUOC:
- BAT KY cau hoi lien quan den phap luat, dieu khoan, thu tuc, quyen, nghia vu, dat dai -> PHAI goi it nhat mot tool
- KHONG BAO GIO tu tra loi cau hoi phap luat tu kien thuc cua ban - LUON tra cuu truoc
- direct_response CHI dung cho: chao hoi, cam on, tam biet, noi chuyen xa giao, hoac hoi lai de lam ro cau hoi mo ho
- Khi nguoi dung hoi tiep ve noi dung vua tra loi (lam ro, giai thich them) -> van PHAI goi tool de dam bao chinh xac
- Khi khong chac co can tra cuu khong -> GOI TOOL (an toan hon la tu tra loi sai)
</retrieval-policy>"""

ORCHESTRATOR_INSTRUCTIONS = """
<instructions>
Phan tich tin nhan cua nguoi dung va quyet dinh hanh dong.
Tra ve CHINH XAC mot JSON object (khong markdown, khong giai thich) voi format:
{
  "reasoning": "suy nghi ngan gon",
  "actions": [{"tool": "ten_tool", ...params}],
  "complexity": "simple" hoac "complex",
  "summary_update": "tom tat cap nhat" hoac null,
  "direct_response": "cau tra loi truc tiep" hoac null
}

Quy tac:
- actions va direct_response khong dong thoi co gia tri. Chon mot trong hai.
- Cau hoi phap luat -> actions (KHONG DUOC dung direct_response)
- Chao hoi, cam on, xa giao, hoi lai -> direct_response (KHONG CAN actions)
- complexity: "complex" khi so sanh nhieu van ban, phan tich tinh huong phuc tap, cau hoi lien quan den sua doi. "simple" cho con lai
- summary_update: chi khi cuoc hoi thoai da co 4+ luot trao doi. Tom tat PHAI bao gom summary truoc do va bo sung noi dung moi
- Khi viet query cho search_legal_docs: viet cau truy van day du ngu canh, khong dung dai tu (no, do, nay), khong viet tat
</instructions>

<examples>
INPUT: "Xin chao"
OUTPUT: {"reasoning": "Nguoi dung chao hoi, khong can tra cuu", "actions": [], "complexity": "simple", "summary_update": null, "direct_response": "Xin chao! Toi la chuyen gia tu van Luat Dat Dai. Ban can ho tro gi?"}

INPUT: "Dieu 79 Luat Dat Dai 2024 quy dinh gi?"
OUTPUT: {"reasoning": "Hoi noi dung cu the Dieu 79 LDD 2024, dung lookup_specific_dieu", "actions": [{"tool": "lookup_specific_dieu", "doc_id": "ldd2024", "dieu": "Dieu 79"}], "complexity": "simple", "summary_update": null, "direct_response": null}

INPUT: "Quyen cua nguoi su dung dat la gi?"
OUTPUT: {"reasoning": "Cau hoi chung ve quyen su dung dat, can search", "actions": [{"tool": "search_legal_docs", "query": "quyen cua nguoi su dung dat theo Luat Dat Dai 2024", "filters": {"doc_ids": ["ldd2024"]}}], "complexity": "simple", "summary_update": null, "direct_response": null}

INPUT: "ND 49 sua doi gi cua ND 102?"
OUTPUT: {"reasoning": "Hoi ve sua doi giua 2 ND, can lookup_amendment va search", "actions": [{"tool": "lookup_amendment", "target_doc": "nd102", "source_doc": "nd49"}, {"tool": "search_legal_docs", "query": "Nghi dinh 49/2026 sua doi bo sung Nghi dinh 102/2024 chi tiet thi hanh Luat Dat Dai", "filters": {"doc_ids": ["nd49"]}}], "complexity": "complex", "summary_update": null, "direct_response": null}

INPUT: "Cam on ban"
OUTPUT: {"reasoning": "Cam on, xa giao", "actions": [], "complexity": "simple", "summary_update": null, "direct_response": "Khong co gi! Neu ban co them cau hoi ve Luat Dat Dai, hay hoi bat cu luc nao."}
</examples>"""


# -- Legal keyword guardrail --

LEGAL_KEYWORDS = re.compile(
    r"(?:điều|dieu|đ\.\s*\d|luật|luat|nghị định|nghi dinh|nđ|nd\s*\d+"
    r"|quyền|quyen|thu hồi|thu hoi|bồi thường|boi thuong"
    r"|giấy chứng nhận|giay chung nhan|sử dụng đất|su dung dat"
    r"|tiền thuê|tien thue|chuyển nhượng|chuyen nhuong"
    r"|quy hoạch|quy hoach|giải phóng mặt bằng|giai phong mat bang"
    r"|đất đai|dat dai|thửa đất|thua dat|cấp đất|cap dat|giao đất|giao dat"
    r"|nghị quyết|nghi quyet|khiếu nại|khieu nai|tranh chấp|tranh chap"
    r"|đăng ký|dang ky|sổ đỏ|so do|sổ hồng|so hong)",
    re.IGNORECASE,
)


def _should_force_retrieval(user_message: str, decision: dict) -> bool:
    """Check if a direct_response should be overridden with retrieval."""
    if not decision.get("direct_response"):
        return False
    if decision.get("actions"):
        return False
    return bool(LEGAL_KEYWORDS.search(user_message))


# -- Context assembly --

def _assemble_conversation_context(history: list[dict], session_id: str) -> tuple[list[dict], str | None]:
    """Build trimmed conversation context and retrieve summary.

    Returns (recent_messages, summary_text).
    """
    summary_row = db.get_latest_summary(session_id)
    summary_text = summary_row["summary"] if summary_row else None

    # Convert to simple dicts
    msgs = [{"role": r["role"], "content": r["content"]} for r in history]

    # Atomic turn-group trimming: keep last N complete pairs
    # A turn-group = (user msg, assistant msg)
    max_msgs = CONTEXT_MAX_TURN_GROUPS * 2
    if len(msgs) > max_msgs:
        msgs = msgs[-max_msgs:]

    # Trim by character budget
    total_chars = 0
    trimmed = []
    for msg in reversed(msgs):
        total_chars += len(msg["content"])
        if total_chars > CONTEXT_MAX_CHARS:
            break
        trimmed.insert(0, msg)

    # Ensure we don't start with an assistant message (orphaned)
    if trimmed and trimmed[0]["role"] == "assistant":
        trimmed = trimmed[1:]

    return trimmed, summary_text


def _build_orchestrator_prompt(
    user_message: str,
    recent_messages: list[dict],
    summary: str | None,
) -> tuple[str, list[dict], str]:
    """Build system prompt and history for the orchestrator call.

    Returns (system_prompt, history_for_llm, user_prompt).
    """
    system = SYSTEM_BASE + ORCHESTRATOR_TOOLS_SECTION + ORCHESTRATOR_INSTRUCTIONS

    # Build volatile context as part of the user message
    volatile_parts = []
    if summary:
        volatile_parts.append(f"<conversation-summary>\n{summary}\n</conversation-summary>")

    # The recent_messages become the LLM history, current message is the prompt
    history = recent_messages  # These are already trimmed

    # Current user message with any volatile context
    if volatile_parts:
        prompt = "\n".join(volatile_parts) + f"\n\n{user_message}"
    else:
        prompt = user_message

    return system, history, prompt


def _build_generator_prompt(
    user_message: str,
    context: str,
    recent_messages: list[dict],
    summary: str | None,
) -> tuple[str, list[dict], str]:
    """Build system prompt, history, and user prompt for the generator.

    Returns (system_prompt, history_for_llm, user_prompt).
    """
    system = SYSTEM_BASE

    history = []
    if summary:
        history.append({"role": "user", "content": f"<conversation-summary>\n{summary}\n</conversation-summary>"})
        history.append({"role": "model", "content": "Da ghi nhan."})
    history.extend(recent_messages)

    if context:
        prompt = f"""<retrieved-context>
{context}
</retrieved-context>

<user-question>
{user_message}
</user-question>

Hay tra loi dua tren tai lieu tham khao o tren. Trich dan cu the dieu, khoan, ten van ban."""
    else:
        prompt = f"""<user-question>
{user_message}
</user-question>

Tra loi dua tren noi dung da thao luan trong cuoc hoi thoai."""

    return system, history, prompt


# -- Orchestrator --

def _parse_orchestrator_response(text: str) -> dict:
    """Parse the orchestrator's JSON response, handling edge cases."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Orchestrator returned invalid JSON, treating as direct response: %s", text[:200])
        return {
            "reasoning": "Failed to parse orchestrator output",
            "actions": [],
            "complexity": "simple",
            "summary_update": None,
            "direct_response": text,
        }

    # Ensure required keys with defaults
    result.setdefault("reasoning", "")
    result.setdefault("actions", [])
    result.setdefault("complexity", "simple")
    result.setdefault("summary_update", None)
    result.setdefault("direct_response", None)
    return result


async def _execute_tools(actions: list[dict]) -> list[dict]:
    """Execute orchestrator tool calls in parallel. Returns merged chunks."""
    if not actions:
        return []

    async def _run_action(action: dict) -> list[dict]:
        tool = action.get("tool", "")
        try:
            if tool == "search_legal_docs":
                query = action.get("query", "")
                filters = action.get("filters", {})
                return await search_legal_docs(
                    query=query,
                    doc_ids=filters.get("doc_ids"),
                    dieu=filters.get("dieu"),
                )
            elif tool == "lookup_amendment":
                return lookup_amendment(
                    target_doc=action.get("target_doc", ""),
                    source_doc=action.get("source_doc"),
                    dieu=action.get("dieu"),
                )
            elif tool == "lookup_specific_dieu":
                return lookup_specific_dieu(
                    doc_id=action.get("doc_id", ""),
                    dieu=action.get("dieu", ""),
                )
            else:
                logger.warning("Unknown tool: %s", tool)
                return []
        except Exception as e:
            logger.error("Tool %s failed: %s", tool, e)
            return []

    results = await asyncio.gather(*[_run_action(a) for a in actions])

    # Merge and deduplicate
    seen_ids = set()
    merged = []
    for chunk_list in results:
        for chunk in chunk_list:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(chunk)
            elif not cid:
                merged.append(chunk)

    return merged


# -- Main entry point --

async def handle_message(session_id: str, user_message: str) -> dict:
    """Process a user message through the orchestrator pipeline."""
    t_start = time.monotonic()

    # Ensure session exists
    db.create_session(session_id)

    # Get conversation history and turn count
    history = db.get_messages(session_id)
    turn = db.get_turn_count(session_id) + 1

    # Save user message
    db.add_message(session_id, turn, "user", user_message)

    # Assemble conversation context
    recent_messages, summary = _assemble_conversation_context(history, session_id)

    # -- Stage 1: Orchestrator --
    t_orch = time.monotonic()
    orch_system, orch_history, orch_prompt = _build_orchestrator_prompt(
        user_message, recent_messages, summary,
    )

    orch_raw = await llm.generate(
        prompt=orch_prompt,
        system=orch_system,
        history=orch_history,
        model="orchestrator",
        temperature=0.0,
        max_tokens=1000,
    )
    decision = _parse_orchestrator_response(orch_raw)
    logger.info(
        "[%s] turn=%d orchestrator: complexity=%s actions=%d direct=%s (%.1fs)",
        session_id[:8], turn, decision["complexity"],
        len(decision["actions"]), bool(decision["direct_response"]),
        time.monotonic() - t_orch,
    )

    # Guardrail: force retrieval if orchestrator tried to answer a legal question directly
    if _should_force_retrieval(user_message, decision):
        filters = extract_filters(user_message)
        logger.warning(
            "[%s] Guardrail triggered: forcing retrieval for legal question answered directly",
            session_id[:8],
        )
        decision = {
            "reasoning": "Guardrail: legal question requires retrieval",
            "actions": [{"tool": "search_legal_docs", "query": user_message, "filters": {
                "doc_ids": filters.get("doc_ids"),
                "dieu": filters.get("dieu"),
            }}],
            "complexity": decision.get("complexity", "simple"),
            "summary_update": decision.get("summary_update"),
            "direct_response": None,
        }

    # Save summary update if provided
    if decision["summary_update"]:
        db.upsert_summary(session_id, decision["summary_update"], turn)

    # -- Direct response path --
    if decision["direct_response"]:
        response = decision["direct_response"]
        db.add_message(session_id, turn, "assistant", response, [])

        if turn == 1:
            await _generate_title(session_id, user_message, response)

        return {"answer": response, "sources": [], "session_id": session_id}

    # -- Stage 2: Execute tools --
    t_tools = time.monotonic()
    chunks = await _execute_tools(decision["actions"])
    logger.info("[%s] tools returned %d chunks (%.1fs)", session_id[:8], len(chunks), time.monotonic() - t_tools)

    context = build_context(chunks)
    sources = extract_sources(chunks)

    # -- Stage 3: Generator --
    t_gen = time.monotonic()
    model = "pro" if decision["complexity"] == "complex" else "flash"
    gen_system, gen_history, gen_prompt = _build_generator_prompt(
        user_message, context, recent_messages, summary,
    )

    response = await llm.generate(
        prompt=gen_prompt,
        system=gen_system,
        history=gen_history,
        model=model,
        temperature=0.3,
        max_tokens=8000,
    )
    logger.info("[%s] generator (%s): %d chars (%.1fs)", session_id[:8], model, len(response), time.monotonic() - t_gen)

    # Citation verification
    response, cite_meta = verify_citations(response, chunks)
    if cite_meta.get("unverified"):
        logger.warning("[%s] Unverified citations: %s", session_id[:8], cite_meta["unverified"])

    # Save assistant response
    db.add_message(session_id, turn, "assistant", response, sources)

    # Auto-generate title for new sessions
    if turn == 1:
        await _generate_title(session_id, user_message, response)

    logger.info("[%s] total turn time: %.1fs", session_id[:8], time.monotonic() - t_start)

    return {"answer": response, "sources": sources, "session_id": session_id}


async def _generate_title(session_id: str, question: str, answer: str) -> None:
    """Auto-generate a session title from first exchange."""
    try:
        prompt = f"""Tao tieu de ngan gon (duoi 50 ky tu) cho cuoc hoi thoai bat dau voi cau hoi sau.
Chi tra ve tieu de, khong giai thich, khong dau ngoac kep.

Cau hoi: {question[:200]}

Tieu de:"""
        title = await llm.generate(prompt, model="utility", temperature=0.0, max_tokens=60)
        if title and len(title.strip()) > 3:
            db.update_session_title(session_id, title.strip()[:80])
    except Exception:
        pass


def create_new_session() -> str:
    """Create a new chat session and return its ID."""
    session_id = str(uuid.uuid4())
    db.create_session(session_id)
    return session_id
