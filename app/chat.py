"""Agentic chat orchestrator.

Two-stage pipeline:
1. Orchestrator (Flash 3) decides what tools to call
2. Generator (Pro 3.1 / Flash 3) produces the answer
"""
import asyncio
import json
import logging
import random
import re
import time
import uuid

from app import db, llm
from app.config import CONTEXT_MAX_TURN_GROUPS, CONTEXT_MAX_CHARS, ORCH_MAX_TURN_GROUPS, ORCH_MAX_CHARS
from app.memory import conv_memory
from app.rag.retriever import (
    search_legal_docs, lookup_amendment, lookup_specific_dieu,
    build_context, extract_sources,
)
from app.rag.query_rewriter import extract_filters_regex, rewrite_query
from app.rag.citation_check import verify_citations
from app.observability import observe, score_current_trace

logger = logging.getLogger(__name__)


async def _safe_background(coro_factory, label: str) -> None:
    """Run a coroutine factory as a background task. Logs errors and retries once."""
    try:
        await coro_factory()
    except Exception as e:
        logger.error("Background task '%s' failed: %s — retrying once", label, e)
        try:
            await coro_factory()
        except Exception as e2:
            logger.error("Background task '%s' failed permanently: %s", label, e2)


# -- System prompts (static, XML-tagged) --

SYSTEM_BASE = """<identity>
Bạn là chuyên gia tư vấn pháp luật đất đai Việt Nam. Bạn trả lời câu hỏi dựa trên các văn bản pháp luật được cung cấp.
Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu cho người dân thường.
</identity>

<boundaries>
1. Chỉ trả lời dựa trên nội dung văn bản pháp luật được cung cấp trong phần "Tài liệu tham khảo"
2. Trích dẫn cụ thể số điều, khoản, điểm và tên văn bản khi trả lời
3. Nếu thông tin không có trong tài liệu, nói rõ "Tôi không tìm thấy thông tin này trong các văn bản hiện có"
4. Khi có nhiều văn bản liên quan, nêu rõ mối quan hệ giữa chúng (ví dụ: Luật quy định chung, Nghị định hướng dẫn chi tiết)
5. Nếu câu hỏi mơ hồ, hỏi lại để làm rõ trước khi trả lời
6. KHÔNG bắt đầu câu trả lời bằng lời chào (vd: "Chào bạn", "Xin chào") trừ khi người dùng vừa chào. Đi thẳng vào nội dung trả lời.
</boundaries>

<legal-hierarchy>
Luật Đất Đai 2024 (31/2024/QH15) - luật gốc
NĐ 71/2024 - giá đất [sửa đổi bởi: NĐ 226, NĐ 49]
NĐ 88/2024 - bồi thường, hỗ trợ, tái định cư [sửa đổi bởi: NĐ 226, NĐ 49]
NĐ 102/2024 - chi tiết thi hành [sửa đổi bởi: NĐ 226, NĐ 49]
NĐ 103/2024 - tiền sử dụng đất, tiền thuê đất [sửa đổi bởi: NĐ 50]
NĐ 151/2025 - phân định thẩm quyền [sửa đổi bởi: NĐ 226, NĐ 49]
NĐ 226/2025 - sửa đổi 4 NĐ [sửa đổi bởi: NĐ 49]
NQ 254/2025 - tháo gỡ vướng mắc
NĐ 49/2026 - sửa đổi mới nhất các NĐ
NĐ 12/2024 - chuyển tiếp giá đất
NĐ 50/2026 - chi tiết NQ 254 về tiền sử dụng đất, tiền thuê đất
Thứ tự ưu tiên: NĐ mới nhất > NĐ cũ > Luật gốc
</legal-hierarchy>"""

ORCHESTRATOR_TOOLS_SECTION = """
<tools>
Bạn có thể gọi các tool sau để tra cứu thông tin. Trả về JSON hợp lệ.

1. search_legal_docs: Tìm kiếm văn bản pháp luật bằng ngôn ngữ tự nhiên.
   Params: {"query": "câu truy vấn", "filters": {"doc_ids": ["nd102"], "dieu": "Dieu 15"}}
   - query: viết rõ ràng, đầy đủ ngữ cảnh, không dùng đại từ
   - filters: tùy chọn, chỉ định khi biết chính xác văn bản/điều

2. lookup_amendment: Tra cứu các sửa đổi giữa các nghị định.
   Params: {"target_doc": "nd102", "source_doc": "nd49", "dieu": "Dieu 15"}
   - target_doc: bắt buộc - văn bản bị sửa đổi
   - source_doc: tùy chọn - văn bản sửa đổi (nếu biết)
   - dieu: tùy chọn - số điều cụ thể

3. lookup_specific_dieu: Lấy toàn bộ nội dung một điều cụ thể.
   Params: {"doc_id": "ldd2024", "dieu": "Dieu 79"}
   - Dùng khi biết chính xác điều và văn bản cần tra cứu

Chọn tool phù hợp:
- Biết chính xác điều + văn bản -> lookup_specific_dieu
- Hỏi về sửa đổi giữa các NĐ -> lookup_amendment (+ search_legal_docs nếu cần thêm context)
- Câu hỏi chung, không biết điều cụ thể -> search_legal_docs
- Câu hỏi phức tạp, nhiều văn bản -> nhiều tool cùng lúc
</tools>

<retrieval-policy>
QUY TẮC BẮT BUỘC:
- BẤT KỲ câu hỏi liên quan đến pháp luật, điều khoản, thủ tục, quyền, nghĩa vụ, đất đai -> PHẢI gọi ít nhất một tool
- KHÔNG BAO GIỜ tự trả lời câu hỏi pháp luật từ kiến thức của bạn - LUÔN tra cứu trước
- direct_response CHỈ dùng cho: chào hỏi, cảm ơn, tạm biệt, nói chuyện xã giao, hỏi về khả năng/phạm vi của chatbot, hoặc hỏi lại để làm rõ câu hỏi mơ hồ
- Khi người dùng hỏi tiếp về nội dung vừa trả lời (làm rõ, giải thích thêm) -> vẫn PHẢI gọi tool để đảm bảo chính xác
- Khi không chắc có cần tra cứu không -> GỌI TOOL (an toàn hơn là tự trả lời sai)
</retrieval-policy>"""

ORCHESTRATOR_INSTRUCTIONS = """
<instructions>
Phân tích tin nhắn của người dùng và quyết định hành động.
Trả về CHÍNH XÁC một JSON object (không markdown, không giải thích) với format:
{
  "reasoning": "suy nghĩ ngắn gọn",
  "actions": [{"tool": "ten_tool", ...params}],
  "complexity": "simple" hoặc "complex",
  "summary_update": "tóm tắt cập nhật" hoặc null,
  "direct_response": "câu trả lời trực tiếp" hoặc null
}

Quy tắc:
- actions và direct_response không đồng thời có giá trị. Chọn một trong hai.
- Câu hỏi pháp luật -> actions (KHÔNG ĐƯỢC dùng direct_response)
- Chào hỏi, cảm ơn, xã giao, hỏi về khả năng của chatbot, hỏi lại -> direct_response (KHÔNG CẦN actions)
- complexity: "complex" khi so sánh nhiều văn bản, phân tích tình huống phức tạp, câu hỏi liên quan đến sửa đổi. "simple" cho còn lại
- summary_update: chỉ khi cuộc hội thoại đã có 4+ lượt trao đổi. Tóm tắt PHẢI bao gồm summary trước đó và bổ sung nội dung mới
- Khi viết query cho search_legal_docs: viết câu truy vấn đầy đủ ngữ cảnh, không dùng đại từ (nó, đó, này), không viết tắt
</instructions>

<examples>
INPUT: "Điều 79 Luật Đất Đai 2024 quy định gì?"
OUTPUT: {"reasoning": "Hỏi nội dung cụ thể Điều 79 LĐĐ 2024, dùng lookup_specific_dieu", "actions": [{"tool": "lookup_specific_dieu", "doc_id": "ldd2024", "dieu": "Dieu 79"}], "complexity": "simple", "summary_update": null, "direct_response": null}

INPUT: "Quyền của người sử dụng đất là gì?"
OUTPUT: {"reasoning": "Câu hỏi chung về quyền sử dụng đất, cần search", "actions": [{"tool": "search_legal_docs", "query": "quyền của người sử dụng đất theo Luật Đất Đai 2024", "filters": {"doc_ids": ["ldd2024"]}}], "complexity": "simple", "summary_update": null, "direct_response": null}

INPUT: "NĐ 49 sửa đổi gì của NĐ 102?"
OUTPUT: {"reasoning": "Hỏi về sửa đổi giữa 2 NĐ, cần lookup_amendment và search", "actions": [{"tool": "lookup_amendment", "target_doc": "nd102", "source_doc": "nd49"}, {"tool": "search_legal_docs", "query": "Nghị định 49/2026 sửa đổi bổ sung Nghị định 102/2024 chi tiết thi hành Luật Đất Đai", "filters": {"doc_ids": ["nd49"]}}], "complexity": "complex", "summary_update": null, "direct_response": null}

INPUT: "Bạn có thể làm gì?" hoặc "Bạn hỗ trợ những gì?"
OUTPUT: {"reasoning": "Hỏi về khả năng của chatbot, trả lời trực tiếp không cần tra cứu", "actions": [], "complexity": "simple", "summary_update": null, "direct_response": "Mình có thể giúp bạn tra cứu và giải thích các quy định trong Luật Đất Đai 2024 cùng các nghị định hướng dẫn (NĐ 71, 88, 102, 103, 151, 226, 49, 50...). Bạn có thể hỏi về thủ tục, quyền, nghĩa vụ, bồi thường, giá đất, hay bất kỳ điều khoản cụ thể nào. Bạn muốn hỏi về vấn đề gì?"}
</examples>"""


GENERATOR_INSTRUCTIONS = """
<response-style>
Trả lời tự nhiên, phù hợp với từng câu hỏi cụ thể:
- Câu hỏi đơn giản về một điều/khoản cụ thể → trả lời ngắn gọn bằng văn xuôi, không cần cấu trúc phức tạp
- Câu hỏi hỏi liệt kê nhiều mục (từ 4 mục trở lên) → dùng bullet points để dễ đọc
- Câu hỏi so sánh, phân tích, nhiều văn bản liên quan → có thể dùng cấu trúc rõ ràng
- Trích dẫn điều khoản khi cần thiết, không nhất thiết phải trích dẫn mọi điều tìm được
- Chỉ thêm cảnh báo về sửa đổi khi có xung đột thực sự giữa văn bản cũ và mới
</response-style>"""


_STATUS_READING = [
    "Mình đang đọc câu hỏi của bạn...",
    "Đang phân tích câu hỏi...",
    "Mình đang xem xét câu hỏi này...",
]

_STATUS_SEARCHING = [
    "Mình đang tra cứu tài liệu pháp luật...",
    "Đang tìm kiếm trong văn bản pháp luật...",
    "Mình đang kiểm tra trong tài liệu...",
]

_STATUS_GENERATING = [
    "Mình đang soạn câu trả lời...",
    "Đang tổng hợp thông tin...",
    "Mình đang chuẩn bị câu trả lời...",
    "Đang xem xét và soạn thảo...",
]


def _model_for_complexity(complexity: str) -> tuple[str, float]:
    """Return (model_name, temperature) based on query complexity."""
    if complexity == "complex":
        return "pro", 0.1
    return "flash", 0.0


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


def _should_force_retrieval(user_message: str, decision: dict, standalone_query: str = "") -> bool:
    """Check if a direct_response should be overridden with retrieval."""
    if not decision.get("direct_response"):
        return False
    if decision.get("actions"):
        return False
    return bool(LEGAL_KEYWORDS.search(user_message) or LEGAL_KEYWORDS.search(standalone_query))


# -- Context assembly --

async def _assemble_conversation_context(history: list[dict], session_id: str) -> tuple[list[dict], str | None]:
    """Build trimmed conversation context and retrieve summary.

    Returns (recent_messages, summary_text).
    """
    summary_row = await asyncio.to_thread(db.get_latest_summary, session_id)
    summary_text = summary_row["summary"] if summary_row else None

    # Convert to simple dicts
    msgs = [{"role": r["role"], "content": r["content"]} for r in history]

    # Atomic turn-group trimming: keep last N complete pairs
    # A turn-group = (user msg, assistant msg)
    max_msgs = CONTEXT_MAX_TURN_GROUPS * 2
    if len(msgs) > max_msgs:
        msgs = msgs[-max_msgs:]

    # Trim by character budget (forward pass — include messages until budget would be exceeded)
    total_chars = 0
    trimmed = []
    for msg in reversed(msgs):
        msg_len = len(msg["content"])
        if total_chars + msg_len > CONTEXT_MAX_CHARS:
            break
        total_chars += msg_len
        trimmed.insert(0, msg)

    # Ensure we don't start with an assistant message (orphaned)
    if trimmed and trimmed[0]["role"] == "assistant":
        trimmed = trimmed[1:]

    return trimmed, summary_text


def _trim_for_orchestrator(msgs: list[dict]) -> list[dict]:
    """Re-trim messages to orchestrator's smaller budget."""
    max_msgs = ORCH_MAX_TURN_GROUPS * 2
    trimmed = msgs[-max_msgs:] if len(msgs) > max_msgs else list(msgs)
    total = 0
    result = []
    for msg in reversed(trimmed):
        msg_len = len(msg["content"])
        if total + msg_len > ORCH_MAX_CHARS:
            break
        total += msg_len
        result.insert(0, msg)
    if result and result[0]["role"] == "assistant":
        result = result[1:]
    return result


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
    system = SYSTEM_BASE + GENERATOR_INSTRUCTIONS

    history = []
    if summary:
        history.append({"role": "user", "content": f"<conversation-summary>\n{summary}\n</conversation-summary>"})
        history.append({"role": "model", "content": "Đã ghi nhận."})
    history.extend(recent_messages)

    if context:
        prompt = f"""<retrieved-context>
{context}
</retrieved-context>

<user-question>
{user_message}
</user-question>

Hãy trả lời câu hỏi dựa trên tài liệu tham khảo ở trên. Trích dẫn điều khoản khi cần thiết. Không tự thêm thông tin ngoài tài liệu."""
    else:
        prompt = f"""<user-question>
{user_message}
</user-question>

Tôi không tìm thấy tài liệu liên quan trong cơ sở dữ liệu cho câu hỏi này.
Hãy nói rõ với người dùng rằng bạn không tìm thấy thông tin cụ thể trong các văn bản hiện có.
Gợi ý họ cách hỏi cụ thể hơn, ví dụ: chỉ định số điều, tên văn bản (Luật Đất Đai, NĐ 102...), hoặc mô tả tình huống cụ thể."""

    return system, history, prompt


# -- Orchestrator --

def _parse_orchestrator_response(text: str, user_message: str = "") -> dict:
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
        logger.warning("Orchestrator returned invalid JSON: %s", text[:200])
        # If it looks like truncated orchestrator JSON, fall back to retrieval
        # instead of showing raw JSON to the user
        if text.lstrip().startswith("{"):
            return {
                "reasoning": "Failed to parse orchestrator JSON, falling back to retrieval",
                "actions": [{"tool": "search_legal_docs", "query": user_message, "filters": {}}],
                "complexity": "simple",
                "summary_update": None,
                "direct_response": None,
            }
        return {
            "reasoning": "Failed to parse orchestrator output",
            "actions": [],
            "complexity": "simple",
            "summary_update": None,
            "direct_response": text,
        }

    # Ensure required keys with defaults
    result.setdefault("reasoning", "")
    result.setdefault("complexity", "simple")
    result.setdefault("summary_update", None)
    result.setdefault("direct_response", None)
    result["actions"] = result.get("actions") or []  # guard against explicit null from LLM
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
                    search_queries=action.get("search_queries"),
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

@observe(name="handle_message")
async def handle_message(session_id: str, user_message: str) -> dict:
    """Process a user message through the orchestrator pipeline."""
    t_start = time.monotonic()

    # Ensure session exists
    await asyncio.to_thread(db.create_session, session_id)

    # Get conversation history and derive turn count
    history = await asyncio.to_thread(db.get_messages, session_id)
    turn = max((m["turn"] for m in history), default=0) + 1

    # Assemble conversation context
    recent_messages, summary = await _assemble_conversation_context(history, session_id)
    conv_state = conv_memory.read(session_id)

    # -- Stage 0: Query rewriting --
    t_rewrite = time.monotonic()
    rewrite_result = await rewrite_query(user_message, summary, recent_messages, conv_state=conv_state)

    # Off-topic short-circuit — skip orchestrator entirely
    if not rewrite_result.get("is_in_scope", True):
        off_topic_response = "Tôi chỉ hỗ trợ tra cứu pháp luật đất đai Việt Nam (Luật Đất Đai 2024 và các nghị định hướng dẫn thi hành). Bạn có câu hỏi nào về đất đai không?"
        logger.info("[%s] turn=%d off-topic query, short-circuiting", session_id[:8], turn)
        await asyncio.to_thread(db.add_message, session_id, turn, "user", user_message)
        await asyncio.to_thread(db.add_message, session_id, turn, "assistant", off_topic_response, [])
        if turn == 1:
            asyncio.create_task(_safe_background(
                lambda: _generate_title(session_id, user_message, off_topic_response),
                label="generate_title",
            ))
        return {"answer": off_topic_response, "sources": [], "session_id": session_id}

    standalone_query = rewrite_result["standalone_query"]
    search_queries = rewrite_result["search_queries"]
    rewrite_filters = rewrite_result["filters"]
    logger.info("[%s] query rewrite: %r -> %r (%d variants, %.1fs)",
                session_id[:8], user_message[:60], standalone_query[:60],
                len(search_queries), time.monotonic() - t_rewrite)

    # -- Stage 1: Orchestrator --
    t_orch = time.monotonic()
    orch_messages = _trim_for_orchestrator(recent_messages)
    logger.info("orch_msgs=%d gen_msgs=%d", len(orch_messages), len(recent_messages))
    orch_system, orch_history, orch_prompt = _build_orchestrator_prompt(
        standalone_query, orch_messages, summary,
    )

    orch_raw = await llm.generate(
        prompt=orch_prompt,
        system=orch_system,
        history=orch_history,
        model="orchestrator",
        temperature=0.0,
        max_tokens=1000,
    )
    decision = _parse_orchestrator_response(orch_raw, user_message)
    logger.info(
        "[%s] turn=%d orchestrator: complexity=%s actions=%d direct=%s (%.1fs)",
        session_id[:8], turn, decision["complexity"],
        len(decision["actions"] or []), bool(decision["direct_response"]),
        time.monotonic() - t_orch,
    )

    # Guardrail: force retrieval if orchestrator tried to answer a legal question directly
    if _should_force_retrieval(user_message, decision, standalone_query):
        filters = rewrite_filters if rewrite_filters.get("doc_ids") or rewrite_filters.get("dieu") else extract_filters_regex(user_message)
        logger.warning(
            "[%s] Guardrail triggered: forcing retrieval for legal question answered directly",
            session_id[:8],
        )
        decision = {
            "reasoning": "Guardrail: legal question requires retrieval",
            "actions": [{"tool": "search_legal_docs", "query": standalone_query, "filters": {
                "doc_ids": filters.get("doc_ids"),
                "dieu": filters.get("dieu"),
            }}],
            "complexity": decision.get("complexity", "simple"),
            "summary_update": decision.get("summary_update"),
            "direct_response": None,
        }

    # Save summary update if provided (background -- not needed for current response)
    if decision["summary_update"]:
        asyncio.create_task(_safe_background(
            lambda: asyncio.to_thread(db.upsert_summary, session_id, decision["summary_update"], turn),
            label="upsert_summary",
        ))

    # -- Direct response path --
    if decision["direct_response"]:
        response = decision["direct_response"]
        await asyncio.to_thread(db.add_message, session_id, turn, "user", user_message)
        await asyncio.to_thread(db.add_message, session_id, turn, "assistant", response, [])

        if turn == 1:
            asyncio.create_task(_safe_background(
                lambda: _generate_title(session_id, user_message, response),
                label="generate_title",
            ))

        return {"answer": response, "sources": [], "session_id": session_id}

    # -- Stage 2: Execute tools --
    t_tools = time.monotonic()
    if search_queries:
        for action in decision["actions"]:
            if action.get("tool") == "search_legal_docs":
                action["search_queries"] = search_queries
    chunks = await _execute_tools(decision["actions"])
    logger.info("[%s] tools returned %d chunks (%.1fs)", session_id[:8], len(chunks), time.monotonic() - t_tools)

    context = build_context(chunks)
    sources = extract_sources(chunks)

    # -- Stage 3: Generator --
    t_gen = time.monotonic()
    model, temperature = _model_for_complexity(decision["complexity"])
    gen_system, gen_history, gen_prompt = _build_generator_prompt(
        user_message, context, recent_messages, summary,
    )

    response = await llm.generate(
        prompt=gen_prompt,
        system=gen_system,
        history=gen_history,
        model=model,
        temperature=temperature,
        max_tokens=6000,
    )
    logger.info("[%s] generator (%s): %d chars (%.1fs)", session_id[:8], model, len(response), time.monotonic() - t_gen)

    # Citation verification
    response, cite_meta = verify_citations(response, chunks)
    if cite_meta.get("unverified"):
        logger.warning("[%s] Unverified citations: %s", session_id[:8], cite_meta["unverified"])

    # Save user + assistant messages in a single transaction
    await asyncio.to_thread(db.add_messages_batch, session_id, [
        (turn, "user", user_message, None),
        (turn, "assistant", response, sources),
    ])

    # Background: update conversation state from this response
    asyncio.create_task(_safe_background(
        lambda: conv_memory.update(session_id, response, turn),
        label="conv_state_update",
    ))

    # Auto-generate title for new sessions
    if turn == 1:
        asyncio.create_task(_safe_background(
            lambda: _generate_title(session_id, user_message, response),
            label="generate_title",
        ))

    # -- Heuristic scores for Langfuse --
    score_current_trace("retrieval_count", float(len(chunks)))
    score_current_trace("response_length", float(len(response)))
    score_current_trace("has_sources", 1.0 if sources else 0.0)
    score_current_trace("model_used", 1.0 if model == "pro" else 0.0, comment=model)
    reranked = [c for c in chunks if c.get("score", 0) > 0]
    if reranked:
        scores = [c["score"] for c in reranked]
        score_current_trace("reranker_top_score", max(scores))
        score_current_trace("reranker_mean_score", sum(scores) / len(scores))

    logger.info("[%s] total turn time: %.1fs", session_id[:8], time.monotonic() - t_start)

    return {"answer": response, "sources": sources, "session_id": session_id}


def _get_trace_id() -> str | None:
    """Get current Langfuse trace ID, or None if tracing disabled."""
    from app.observability import langfuse
    if langfuse:
        try:
            return langfuse.get_current_trace_id()
        except Exception:
            pass
    return None


@observe(name="handle_message_stream")
async def handle_message_stream(session_id: str, user_message: str):
    """Async generator yielding (event_type, data_dict) tuples for SSE streaming."""
    t_start = time.monotonic()

    await asyncio.to_thread(db.create_session, session_id)
    history = await asyncio.to_thread(db.get_messages, session_id)
    turn = max((m["turn"] for m in history), default=0) + 1
    recent_messages, summary = await _assemble_conversation_context(history, session_id)
    conv_state = conv_memory.read(session_id)

    yield ("status", {"text": random.choice(_STATUS_READING), "step": "orchestrator"})

    # -- Stage 0: Query rewriting --
    rewrite_result = await rewrite_query(user_message, summary, recent_messages, conv_state=conv_state)

    # Off-topic short-circuit — skip orchestrator entirely
    if not rewrite_result.get("is_in_scope", True):
        off_topic_response = "Tôi chỉ hỗ trợ tra cứu pháp luật đất đai Việt Nam (Luật Đất Đai 2024 và các nghị định hướng dẫn thi hành). Bạn có câu hỏi nào về đất đai không?"
        logger.info("[%s] turn=%d off-topic query, short-circuiting", session_id[:8], turn)
        await asyncio.to_thread(db.add_message, session_id, turn, "user", user_message)
        await asyncio.to_thread(db.add_message, session_id, turn, "assistant", off_topic_response, [])
        if turn == 1:
            asyncio.create_task(_safe_background(
                lambda: _generate_title(session_id, user_message, off_topic_response),
                label="generate_title",
            ))
        yield ("token", {"text": off_topic_response})
        yield ("done", {"session_id": session_id, "trace_id": _get_trace_id()})
        return

    standalone_query = rewrite_result["standalone_query"]
    search_queries = rewrite_result["search_queries"]
    rewrite_filters = rewrite_result["filters"]

    # -- Stage 1: Orchestrator --
    orch_messages = _trim_for_orchestrator(recent_messages)
    orch_system, orch_history, orch_prompt = _build_orchestrator_prompt(
        standalone_query, orch_messages, summary,
    )
    orch_raw = await llm.generate(
        prompt=orch_prompt, system=orch_system, history=orch_history,
        model="orchestrator", temperature=0.0, max_tokens=1500,
    )
    decision = _parse_orchestrator_response(orch_raw, user_message)
    logger.info("[%s] turn=%d orchestrator: complexity=%s actions=%d direct=%s",
                session_id[:8], turn, decision["complexity"],
                len(decision["actions"] or []), bool(decision["direct_response"]))

    # Guardrail
    if _should_force_retrieval(user_message, decision, standalone_query):
        filters = rewrite_filters if rewrite_filters.get("doc_ids") or rewrite_filters.get("dieu") else extract_filters_regex(user_message)
        decision = {
            "reasoning": "Guardrail: legal question requires retrieval",
            "actions": [{"tool": "search_legal_docs", "query": standalone_query, "filters": {
                "doc_ids": filters.get("doc_ids"), "dieu": filters.get("dieu"),
            }}],
            "complexity": decision.get("complexity", "simple"),
            "summary_update": decision.get("summary_update"),
            "direct_response": None,
        }

    if decision["summary_update"]:
        asyncio.create_task(_safe_background(
            lambda: asyncio.to_thread(db.upsert_summary, session_id, decision["summary_update"], turn),
            label="upsert_summary",
        ))

    # -- Direct response path --
    if decision["direct_response"]:
        response = decision["direct_response"]
        await asyncio.to_thread(db.add_message, session_id, turn, "user", user_message)
        await asyncio.to_thread(db.add_message, session_id, turn, "assistant", response, [])
        if turn == 1:
            asyncio.create_task(_safe_background(
                lambda: _generate_title(session_id, user_message, response),
                label="generate_title",
            ))
        # Yield the full direct response as tokens
        yield ("token", {"text": response})
        yield ("done", {"session_id": session_id, "trace_id": _get_trace_id()})
        return

    # -- Stage 2: Execute tools --
    yield ("status", {"text": random.choice(_STATUS_SEARCHING), "step": "retrieval"})
    if search_queries:
        for action in decision["actions"]:
            if action.get("tool") == "search_legal_docs":
                action["search_queries"] = search_queries
    chunks = await _execute_tools(decision["actions"])
    logger.info("[%s] tools returned %d chunks", session_id[:8], len(chunks))

    context = build_context(chunks)
    sources = extract_sources(chunks)
    yield ("sources", {"sources": sources})

    # -- Stage 3: Generator (streaming) --
    yield ("status", {"text": random.choice(_STATUS_GENERATING), "step": "generating"})

    model, temperature = _model_for_complexity(decision["complexity"])
    gen_system, gen_history, gen_prompt = _build_generator_prompt(
        user_message, context, recent_messages, summary,
    )

    full_response_parts = []
    async for chunk_text in llm.generate_stream(
        prompt=gen_prompt, system=gen_system, history=gen_history,
        model=model, temperature=temperature, max_tokens=6000,
    ):
        full_response_parts.append(chunk_text)
        yield ("token", {"text": chunk_text})

    response = "".join(full_response_parts)

    # Citation verification
    response, cite_meta = verify_citations(response, chunks)
    if cite_meta.get("unverified"):
        logger.warning("[%s] Unverified citations: %s", session_id[:8], cite_meta["unverified"])

    # Save messages
    await asyncio.to_thread(db.add_messages_batch, session_id, [
        (turn, "user", user_message, None),
        (turn, "assistant", response, sources),
    ])

    # Background: update conversation state from this response
    asyncio.create_task(_safe_background(
        lambda: conv_memory.update(session_id, response, turn),
        label="conv_state_update",
    ))

    if turn == 1:
        asyncio.create_task(_safe_background(
            lambda: _generate_title(session_id, user_message, response),
            label="generate_title",
        ))

    # Langfuse scores
    score_current_trace("retrieval_count", float(len(chunks)))
    score_current_trace("response_length", float(len(response)))
    score_current_trace("has_sources", 1.0 if sources else 0.0)
    score_current_trace("model_used", 1.0 if model == "pro" else 0.0, comment=model)
    reranked = [c for c in chunks if c.get("score", 0) > 0]
    if reranked:
        scores = [c["score"] for c in reranked]
        score_current_trace("reranker_top_score", max(scores))
        score_current_trace("reranker_mean_score", sum(scores) / len(scores))

    logger.info("[%s] stream total: %.1fs", session_id[:8], time.monotonic() - t_start)
    yield ("done", {"session_id": session_id, "trace_id": _get_trace_id()})


@observe(name="generate_title")
async def _generate_title(session_id: str, question: str, answer: str) -> None:
    """Auto-generate a session title from first exchange."""
    try:
        prompt = f"""Tạo tiêu đề ngắn gọn (dưới 50 ký tự) cho cuộc hội thoại bắt đầu với câu hỏi sau.
Chỉ trả về tiêu đề, không giải thích, không dấu ngoặc kép.

Câu hỏi: {question[:200]}

Tiêu đề:"""
        title = await llm.generate(prompt, model="utility", temperature=0.0, max_tokens=60)
        if title and len(title.strip()) > 3:
            await asyncio.to_thread(db.update_session_title, session_id, title.strip()[:80])
    except Exception:
        pass


def create_new_session() -> str:
    """Create a new chat session and return its ID."""
    session_id = str(uuid.uuid4())
    db.create_session(session_id)
    return session_id
