"""Query rewriting and filter extraction for legal RAG queries."""
import json
import logging
import re

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
DIEU_VALIDATE_RE = re.compile(r"^Dieu \d+[a-z]?$", re.IGNORECASE)


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
        dieu = f"Dieu {m.group(1)}"
    return {"doc_ids": list(set(doc_ids)) or None, "dieu": dieu}


# Backward-compatible alias (will be removed in Task 4)
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
