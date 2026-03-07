"""Post-process LLM responses to verify legal citations against retrieved chunks."""

import logging
import re

logger = logging.getLogger(__name__)

# Matches "Dieu 7", "Dieu 7", "dieu 44a", etc. (both Unicode and ASCII)
_DIEU_RE = re.compile(
    r"(?:[DdĐđ]i[eề]u)\s+(\d+[a-zA-Z]?)",
    re.IGNORECASE,
)

# Matches "Khoan 1 Dieu 7", "Khoan 2 Dieu 10", etc.
_KHOAN_DIEU_RE = re.compile(
    r"[Kk]ho[aả]n\s+\d+\s+(?:[DdĐđ]i[eề]u)\s+(\d+[a-zA-Z]?)",
    re.IGNORECASE,
)

# Matches "Diem b Khoan 2 Dieu 10", etc.
_DIEM_KHOAN_DIEU_RE = re.compile(
    r"[DdĐđ]i[eể]m\s+[a-zđ]+\s+[Kk]ho[aả]n\s+\d+\s+(?:[DdĐđ]i[eề]u)\s+(\d+[a-zA-Z]?)",
    re.IGNORECASE,
)

_DISCLAIMER = (
    "\n\n*Lưu ý: Một số trích dẫn trong câu trả lời "
    "chưa được xác minh từ văn bản gốc. "
    "Vui lòng kiểm tra lại các điều khoản cụ thể.*"
)


def _extract_dieu_numbers(text: str) -> set[str]:
    """Extract all unique Dieu numbers from text (e.g. {'7', '44a', '10'})."""
    results: set[str] = set()
    for pattern in (_DIEM_KHOAN_DIEU_RE, _KHOAN_DIEU_RE, _DIEU_RE):
        for m in pattern.finditer(text):
            results.add(m.group(1).lower())
    return results


def _chunk_contains_dieu(chunk: dict, dieu_num: str) -> bool:
    """Check whether a chunk references the given Dieu number."""
    # Check the structured 'dieu' field (e.g. "Điều 7")
    dieu_field = chunk.get("dieu", "")
    if dieu_field:
        m = _DIEU_RE.search(dieu_field)
        if m and m.group(1).lower() == dieu_num:
            return True

    # Fallback: search the chunk text body
    body = chunk.get("text") or chunk.get("content") or ""
    for m in _DIEU_RE.finditer(body):
        if m.group(1).lower() == dieu_num:
            return True

    return False


def verify_citations(
    response: str, chunks: list[dict]
) -> tuple[str, dict]:
    """Verify citations in LLM response against context chunks.

    Extracts all legal citations (Dieu X, Khoan Y, Diem Z) from the response,
    checks if they appear in the provided chunks, and appends a disclaimer
    if any citations cannot be verified.

    Returns (possibly_modified_response, verification_metadata).
    """
    cited = _extract_dieu_numbers(response)
    if not cited:
        logger.debug("No citations found in response.")
        return response, {"total": 0, "verified": 0, "unverified": []}

    verified: set[str] = set()
    for dieu_num in cited:
        if any(_chunk_contains_dieu(c, dieu_num) for c in chunks):
            verified.add(dieu_num)

    unverified = sorted(cited - verified)
    meta = {
        "total": len(cited),
        "verified": len(verified),
        "unverified": [f"Dieu {d}" for d in unverified],
    }

    if unverified:
        logger.warning("Unverified citations: %s", meta["unverified"])
        response += _DISCLAIMER

    logger.info(
        "Citation check: %d total, %d verified, %d unverified",
        meta["total"], meta["verified"], len(unverified),
    )
    return response, meta
