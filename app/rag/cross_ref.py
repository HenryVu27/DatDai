"""Cross-reference resolution for Vietnamese land law documents.

Maps amendment relationships between documents and provides
chunk expansion at retrieval time. This is critical because:
- A Dieu in the Luat is elaborated in NDs
- A Dieu in an ND can be amended by later NDs
- Provisions are scattered across many documents
"""
import re
import logging

logger = logging.getLogger(__name__)

# Which docs amend which (doc_id -> list of target doc_ids it amends)
AMENDS_MAP = {
    "nd49": ["nd71", "nd88", "nd102", "nd151", "nd226"],
    "nd226": ["nd71", "nd88", "nd102", "nd151"],
    "nd50": ["nd103"],
}

# Reverse: which docs is a given doc amended BY
AMENDED_BY: dict[str, list[str]] = {}
for _src, _targets in AMENDS_MAP.items():
    for _t in _targets:
        AMENDED_BY.setdefault(_t, []).append(_src)

# Which docs detail/implement which
IMPLEMENTS_MAP = {
    "nd71": ["ldd2024"],
    "nd88": ["ldd2024"],
    "nd102": ["ldd2024"],
    "nd103": ["ldd2024"],
    "nd151": ["ldd2024"],
    "nd50": ["nq254"],
    "nd12": ["ldd2013"],  # old law, transitional
}

# Map ND numbers found in text to doc_ids
ND_NUM_TO_DOC_ID = {
    "71": "nd71", "88": "nd88", "102": "nd102", "103": "nd103",
    "151": "nd151", "226": "nd226", "49": "nd49", "50": "nd50",
    "12": "nd12", "254": "nq254", "291": "nd103",  # ND 291 = amended ND 103
}

# Regex for inline cross-references
_REF_ND = re.compile(
    r"(?:Điều|điều)\s+(\d+[a-z]?)\s+(?:Nghị định|nghị định|NĐ|Nghi dinh)\s+(?:số\s+)?(\d+)",
    re.IGNORECASE,
)
_REF_LDD = re.compile(
    r"(?:Điều|điều)\s+(\d+[a-z]?)\s+(?:Luật Đất đai|Luật đất đai|Luat Dat Dai|của Luật)",
    re.IGNORECASE,
)
_REF_NQ = re.compile(
    r"(?:Điều|điều)\s+(\d+[a-z]?)\s+(?:Nghị quyết|nghị quyết|NQ)\s+(?:số\s+)?(\d+)",
    re.IGNORECASE,
)


def parse_cross_refs(text: str) -> list[dict]:
    """Extract cross-references from chunk text.

    Returns list of {"doc_id": ..., "dieu": "Dieu N"}.
    """
    refs = []
    seen = set()

    for m in _REF_ND.finditer(text):
        dieu_num, nd_num = m.group(1), m.group(2)
        doc_id = ND_NUM_TO_DOC_ID.get(nd_num)
        if doc_id:
            key = (doc_id, dieu_num)
            if key not in seen:
                seen.add(key)
                refs.append({"doc_id": doc_id, "dieu": f"Điều {dieu_num}"})

    for m in _REF_LDD.finditer(text):
        dieu_num = m.group(1)
        key = ("ldd2024", dieu_num)
        if key not in seen:
            seen.add(key)
            refs.append({"doc_id": "ldd2024", "dieu": f"Điều {dieu_num}"})

    for m in _REF_NQ.finditer(text):
        dieu_num, nq_num = m.group(1), m.group(2)
        if nq_num == "254":
            key = ("nq254", dieu_num)
            if key not in seen:
                seen.add(key)
                refs.append({"doc_id": "nq254", "dieu": f"Điều {dieu_num}"})

    return refs


def get_amendment_doc_ids(doc_id: str) -> list[str]:
    """Get doc_ids that amend the given doc."""
    return AMENDED_BY.get(doc_id, [])


def get_amended_doc_ids(doc_id: str) -> list[str]:
    """Get doc_ids that the given doc amends."""
    return AMENDS_MAP.get(doc_id, [])


async def expand_with_cross_refs(
    chunks: list[dict],
    store,
    top_k: int = 8,
) -> list[dict]:
    """Expand retrieved chunks with related amendment/cross-referenced chunks.

    Strategy:
    1. Parse inline cross-refs from top retrieved chunks
    2. For chunks from base decrees, check amending decrees for same Dieu
    3. Deduplicate and limit total expansion
    """
    if not chunks:
        return chunks

    seen_ids = {c.get("chunk_id") for c in chunks}
    expansion = []

    # 1. Parse inline cross-refs from top chunks
    for chunk in chunks[:5]:
        text = chunk.get("text", chunk.get("content", ""))
        refs = parse_cross_refs(text)
        for ref in refs[:3]:  # limit per chunk
            ref_chunks = store.fetch_by_metadata(
                doc_id=ref["doc_id"],
                dieu=ref["dieu"],
            )
            for rc in ref_chunks:
                cid = rc.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    expansion.append(rc)

    # 2. For base decree chunks, fetch amendments for same Dieu
    for chunk in chunks[:5]:
        doc_id = chunk.get("doc_id", "")
        dieu = chunk.get("dieu", "")
        if not dieu:
            continue

        amenders = get_amendment_doc_ids(doc_id)
        for amender_id in amenders:
            # Search amending doc for chunks that reference this Dieu+doc
            # We can't do text search here, so fetch by same Dieu number
            # (works when amendment preserves the Dieu number)
            related = store.fetch_by_metadata(doc_id=amender_id, dieu=dieu)
            for rc in related:
                cid = rc.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    expansion.append(rc)

    # Limit expansion to avoid overwhelming LLM context
    max_expand = max(top_k // 2, 3)
    result = list(chunks) + expansion[:max_expand]
    return result
