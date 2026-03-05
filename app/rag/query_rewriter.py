"""Query rewriting and filter extraction for legal RAG."""
import logging
import re
from app import llm

logger = logging.getLogger(__name__)

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

DIEU_RE = re.compile(r"(?:điều|dieu|đ\.?)\s*(\d+)", re.IGNORECASE)


def extract_filters(query: str) -> dict:
    """Extract doc_ids and dieu from query text."""
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


async def rewrite_query(query: str, history: list[dict] | None = None) -> str:
    """Rewrite query with conversation context."""
    if not history or len(history) < 2:
        return query
    recent = history[-6:]
    context = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in recent)
    prompt = f"""Viet lai cau hoi sau de ro rang hon, giai quyet dai tu va tham chieu ngam.
Giu nguyen y dinh goc. Tra ve CHI cau hoi da viet lai, khong giai thich.

Lich su hoi thoai:
{context}

Cau hoi: {query}

Cau hoi da viet lai:"""
    try:
        rewritten = await llm.generate(prompt, model="flash", temperature=0.0, max_tokens=200)
        rewritten = rewritten.strip().strip('"').strip("'")
        if 10 < len(rewritten) < 500:
            return rewritten
    except Exception:
        pass
    return query
