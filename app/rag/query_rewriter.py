"""Query rewriting and filter extraction for legal RAG queries."""
import json
import logging
import re

from app import llm
from app.observability import observe

logger = logging.getLogger(__name__)

# -- Known document IDs (for validation) --
VALID_DOC_IDS = {
    "ldd2024", "nd71", "nd88", "nd102", "nd103",
    "nd151", "nd226", "nq254", "nd49", "nd12", "nd50",
}

# -- Regex-based filter extraction (fallback) --

DOC_PATTERNS = {
    r"(?:luật đất đai|luat dat dai|luật\s*số\s*31)": ["ldd2024"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*71": ["nd71"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*88": ["nd88"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*102": ["nd102"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*103": ["nd103"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*151": ["nd151"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*226": ["nd226"],
    r"(?:nghị quyết|nghi quyet|nq)\s*254": ["nq254"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*49": ["nd49"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*12": ["nd12"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*50": ["nd50"],
}

DIEU_RE = re.compile(r"(?:điều|dieu|đ\.?)\s*(\d+[a-z]?)", re.IGNORECASE)
DIEU_VALIDATE_RE = re.compile(r"^(?:Điều|Dieu) \d+[a-z]?$", re.IGNORECASE)


def extract_filters_regex(query: str) -> dict:
    """Extract doc_ids and dieu from query text using regex. Fallback method."""
    query_lower = query.lower()
    doc_ids = []
    for pattern, ids in DOC_PATTERNS.items():
        if re.search(pattern, query_lower):
            doc_ids.extend(ids)
    dieu = None
    m = DIEU_RE.search(query_lower)
    if m:
        dieu = f"Điều {m.group(1)}"
    return {"doc_ids": list(set(doc_ids)) or None, "dieu": dieu}


# Backward-compatible alias
extract_filters = extract_filters_regex


def validate_filters(filters: dict) -> dict:
    """Validate and sanitize LLM-extracted filters."""
    doc_ids = filters.get("doc_ids")
    dieu = filters.get("dieu")

    # Validate doc_ids against known set
    if doc_ids:
        doc_ids = [d for d in doc_ids if d in VALID_DOC_IDS]
    if not doc_ids:
        doc_ids = None

    # Validate dieu format
    if dieu and not DIEU_VALIDATE_RE.match(dieu):
        dieu = None

    return {"doc_ids": doc_ids, "dieu": dieu}


# -- LLM-based query rewriting --

REWRITER_SYSTEM_PROMPT = """Ban la bo xu ly truy van cho he thong tra cuu phap luat dat dai Viet Nam.
Nhiem vu: viet lai cau hoi cua nguoi dung thanh cau truy van doc lap, ro rang, phu hop de tim kiem van ban phap luat.

NGUYEN TAC:
1. GIAI QUYET DAI TU: Thay the "no", "do", "cai nay", "van ban nay", "dieu nay" bang thuc the cu the tu lich su hoi thoai.
2. MO RONG THUAT NGU PHAP LY: Thay the tu thong dung bang thuat ngu chinh thuc, giu tu goc trong ngoac.
   Vi du: "so do" -> "giay chung nhan quyen su dung dat (so do)"
   Vi du: "ban dat" -> "chuyen nhuong quyen su dung dat"
   Vi du: "den bu" -> "boi thuong khi Nha nuoc thu hoi dat"
   Vi du: "tach thua" -> "tach thua dat"
   Vi du: "hop thuc hoa" -> "cap giay chung nhan quyen su dung dat lan dau"
3. GIAI QUYET SO THU TU: Neu nguoi dung nhac den "truong hop 1", "muc 3", "buoc 2",
   "dieu do", hay so thu tu bat ky, hay tim chinh xac noi dung cua muc do trong
   <lich-su> va dua noi dung thuc te vao standalone_query.
   Vi du: user hoi "giai thich truong hop 3" va lich su co "3. Chi phi dau tu con lai..."
   -> standalone_query: "Chi phi dau tu vao dat con lai duoc boi thuong khi Nha nuoc thu hoi dat"
4. TAO 2 CAU TRUY VAN THAY THE: Viet 2 cach dien dat khac nhau de tang kha nang tim kiem.
5. TRICH XUAT BO LOC: Xac dinh van ban (doc_ids) va dieu (dieu) neu co trong cau hoi.
6. PHAM VI: Neu cau hoi KHONG lien quan den phap luat dat dai Viet Nam, tra ve is_in_scope: false va cac truong khac de trong.

MA VAN BAN HOP LE:
- ldd2024: Luat Dat Dai 2024
- nd71: Nghi dinh 71/2024 ve gia dat
- nd88: Nghi dinh 88/2024 ve boi thuong, ho tro, tai dinh cu
- nd102: Nghi dinh 102/2024 chi tiet thi hanh
- nd103: Nghi dinh 103/2024 ve tien su dung dat, tien thue dat
- nd151: Nghi dinh 151/2025 phan dinh tham quyen
- nd226: Nghi dinh 226/2025 sua doi 4 ND
- nq254: Nghi quyet 254/2025 thao go vuong mac
- nd49: Nghi dinh 49/2026 sua doi moi nhat cac ND
- nd12: Nghi dinh 12/2024 chuyen tiep gia dat
- nd50: Nghi dinh 50/2026 chi tiet NQ 254

Tra ve CHINH XAC JSON (khong markdown, khong giai thich):
{"is_in_scope": true, "standalone_query": "...", "search_queries": ["...", "..."], "filters": {"doc_ids": [...] hoac null, "dieu": "Dieu X" hoac null}}
Neu ngoai pham vi: {"is_in_scope": false, "standalone_query": "", "search_queries": [], "filters": {"doc_ids": null, "dieu": null}}"""


def _build_rewriter_prompt(
    user_message: str,
    summary: str | None,
    recent_messages: list[dict],
) -> str:
    """Build the user-facing prompt for the rewriter."""
    parts = []
    if summary:
        parts.append(f"<tom-tat-hoi-thoai>\n{summary}\n</tom-tat-hoi-thoai>")
    if recent_messages:
        history_lines = []
        windowed = recent_messages[-10:]
        # Find index of the last assistant message for recency-weighted truncation
        last_asst_idx = max(
            (i for i, m in enumerate(windowed) if m["role"] == "assistant"),
            default=-1,
        )
        for i, msg in enumerate(windowed):
            role = "Nguoi dung" if msg["role"] == "user" else "Tro ly"
            if msg["role"] == "assistant":
                limit = 1500 if i == last_asst_idx else 400
                content = msg["content"][:limit]
            else:
                content = msg["content"]
            history_lines.append(f"{role}: {content}")
        parts.append(f"<lich-su>\n" + "\n".join(history_lines) + "\n</lich-su>")
    parts.append(f"<cau-hoi-hien-tai>\n{user_message}\n</cau-hoi-hien-tai>")
    return "\n\n".join(parts)


def parse_rewrite_response(raw: str, original_query: str) -> dict:
    """Parse LLM rewrite response into validated result. Falls back to original on failure."""
    fallback = {
        "is_in_scope": True,
        "standalone_query": original_query,
        "search_queries": [],
        "filters": extract_filters_regex(original_query),
    }

    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Query rewriter returned invalid JSON: %s", raw[:200])
        return fallback

    is_in_scope = data.get("is_in_scope", True)
    if not is_in_scope:
        return {"is_in_scope": False, "standalone_query": original_query, "search_queries": [], "filters": {"doc_ids": None, "dieu": None}}

    standalone = data.get("standalone_query", "").strip()
    if not standalone:
        standalone = original_query

    search_queries = data.get("search_queries", [])
    if not isinstance(search_queries, list):
        search_queries = []
    search_queries = [q for q in search_queries if isinstance(q, str) and q.strip()]

    raw_filters = data.get("filters", {})
    if not isinstance(raw_filters, dict):
        raw_filters = {}
    filters = validate_filters({
        "doc_ids": raw_filters.get("doc_ids"),
        "dieu": raw_filters.get("dieu"),
    })

    return {
        "is_in_scope": True,
        "standalone_query": standalone,
        "search_queries": search_queries,
        "filters": filters,
    }


@observe(name="rewrite_query")
async def rewrite_query(
    user_message: str,
    summary: str | None,
    recent_messages: list[dict],
) -> dict:
    """Rewrite a user query for better retrieval. Falls back to regex on failure."""
    import asyncio

    prompt = _build_rewriter_prompt(user_message, summary, recent_messages)

    try:
        raw = await asyncio.wait_for(
            llm.generate(
                prompt=prompt,
                system=REWRITER_SYSTEM_PROMPT,
                model="utility",
                temperature=0.0,
                max_tokens=500,
            ),
            timeout=5.0,
        )
        result = parse_rewrite_response(raw, user_message)
        logger.info("Query rewritten: %r -> %r (%d variants)",
                     user_message[:80], result["standalone_query"][:80],
                     len(result["search_queries"]))
        return result
    except asyncio.TimeoutError:
        logger.warning("Query rewriter timed out for: %s", user_message[:80])
    except Exception as e:
        logger.error("Query rewriter failed: %s", e)

    return {
        "is_in_scope": True,
        "standalone_query": user_message,
        "search_queries": [],
        "filters": extract_filters_regex(user_message),
    }
