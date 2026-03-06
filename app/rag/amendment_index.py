"""Runtime lookup for the amendment index built by scripts/chunk_md.py.

Loads data/amendment_index.json at startup and provides functions to:
- Look up amendments between specific documents
- Get chunk IDs for amendment-related retrieval
- Detect amendment-related queries from user input
"""
import logging
import re

from app.storage import load_amendment_index

logger = logging.getLogger(__name__)


def _get_index() -> dict:
    return load_amendment_index()


def get_amendments(source_doc: str, target_doc: str) -> list[dict]:
    """Get all amendments from source_doc affecting target_doc.

    Args:
        source_doc: doc_id of the amending decree (e.g. "nd226").
        target_doc: doc_id of the amended decree (e.g. "nd71").

    Returns:
        List of amendment dicts with keys: source_dieu, target_dieu, type, chunk_ids.
        Empty list if no amendments found.
    """
    key = f"{source_doc}->{target_doc}"
    entry = _get_index().get(key)
    if entry:
        return entry["amendments"]
    return []


def get_amendment_chunk_ids(source_doc: str, target_doc: str) -> list[str]:
    """Get chunk_ids of all amendments from source affecting target.

    Args:
        source_doc: doc_id of the amending decree.
        target_doc: doc_id of the amended decree.

    Returns:
        Flat list of chunk_ids (deduplicated, order preserved).
    """
    amendments = get_amendments(source_doc, target_doc)
    seen = set()
    result = []
    for a in amendments:
        for cid in a.get("chunk_ids", []):
            if cid not in seen:
                seen.add(cid)
                result.append(cid)
    return result


def get_all_amendments_to(target_doc: str) -> list[dict]:
    """Get all amendments from any source affecting the given target doc.

    Returns list of amendment dicts (same format as get_amendments),
    with an additional "source_doc" key on each entry.
    """
    results = []
    for key, entry in _get_index().items():
        if entry["target_doc"] == target_doc:
            src = entry["source_doc"]
            for a in entry["amendments"]:
                enriched = dict(a)
                enriched["source_doc"] = src
                results.append(enriched)
    return results


def get_all_amendment_chunk_ids_to(target_doc: str) -> list[str]:
    """Get all chunk_ids of amendments affecting the given target doc."""
    seen = set()
    result = []
    for key, entry in _get_index().items():
        if entry["target_doc"] == target_doc:
            for a in entry["amendments"]:
                for cid in a.get("chunk_ids", []):
                    if cid not in seen:
                        seen.add(cid)
                        result.append(cid)
    return result


# ---------- Query detection ----------

# Known doc_id aliases in user queries
_DOC_ALIASES = {
    # "ND 226" or "NĐ 226" or "nghi dinh 226" etc.
    "226": "nd226", "49": "nd49", "50": "nd50",
    "71": "nd71", "88": "nd88", "102": "nd102", "103": "nd103",
    "151": "nd151", "12": "nd12", "254": "nq254",
    "291": "nd103",
}

# Pattern: "ND/NĐ/Nghi dinh <number>"
_DOC_REF_RE = re.compile(
    r"(?:NĐ|ND|Nghị định|nghi dinh|Nghi dinh|nghị định)\s*(\d+)",
    re.IGNORECASE,
)

# Patterns that indicate asking about amendments
_AMENDMENT_QUERY_PATTERNS = [
    # "ND 226 sua doi gi cua ND 71"
    re.compile(
        r"(?:NĐ|ND|Nghị định|nghi dinh)\s*(\d+)\s+(?:sửa đổi|sua doi|bổ sung|bo sung|thay đổi|thay doi)"
        r".*?(?:NĐ|ND|Nghị định|nghi dinh)\s*(\d+)",
        re.IGNORECASE,
    ),
    # "ND 71 bi sua doi nhu the nao" (passive: target being amended)
    re.compile(
        r"(?:NĐ|ND|Nghị định|nghi dinh)\s*(\d+)\s+(?:bị|bi|được|duoc)\s+(?:sửa đổi|sua doi|bổ sung|bo sung|thay đổi|thay doi)",
        re.IGNORECASE,
    ),
    # "sua doi cua ND 226 doi voi ND 71"
    re.compile(
        r"(?:sửa đổi|sua doi|bổ sung|bo sung).*?(?:NĐ|ND|Nghị định|nghi dinh)\s*(\d+)"
        r".*?(?:đối với|doi voi|của|cua|tới|toi|cho)\s*(?:NĐ|ND|Nghị định|nghi dinh)\s*(\d+)",
        re.IGNORECASE,
    ),
]


def detect_amendment_query(query: str) -> tuple[str, str] | None:
    """Detect if a query asks about amendments between documents.

    Returns:
        (amending_doc_id, amended_doc_id) if both are identified.
        (None, amended_doc_id) if query asks about all amendments TO a doc.
        None if this is not an amendment query.
    """
    # Try specific two-doc patterns first
    for i, pattern in enumerate(_AMENDMENT_QUERY_PATTERNS):
        m = pattern.search(query)
        if not m:
            continue

        if i == 0:
            # "ND X sua doi ... ND Y" -> X amends Y
            src_num, tgt_num = m.group(1), m.group(2)
            src_id = _DOC_ALIASES.get(src_num)
            tgt_id = _DOC_ALIASES.get(tgt_num)
            if src_id and tgt_id:
                logger.info("Detected amendment query: %s -> %s", src_id, tgt_id)
                return (src_id, tgt_id)

        elif i == 1:
            # "ND X bi sua doi" -> target = X, source = None (all amenders)
            tgt_num = m.group(1)
            tgt_id = _DOC_ALIASES.get(tgt_num)
            if tgt_id:
                logger.info("Detected amendment query: all -> %s", tgt_id)
                return (None, tgt_id)

        elif i == 2:
            # "sua doi cua ND X doi voi ND Y" -> X amends Y
            src_num, tgt_num = m.group(1), m.group(2)
            src_id = _DOC_ALIASES.get(src_num)
            tgt_id = _DOC_ALIASES.get(tgt_num)
            if src_id and tgt_id:
                logger.info("Detected amendment query: %s -> %s", src_id, tgt_id)
                return (src_id, tgt_id)

    return None
