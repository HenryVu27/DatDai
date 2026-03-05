"""Cross-reference data for Vietnamese land law documents.

Maps amendment relationships between documents. Used by the orchestrator
to decide when to fetch amendment-related chunks.
"""

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
    "nd12": ["ldd2013"],
}

# Map ND numbers found in text to doc_ids
ND_NUM_TO_DOC_ID = {
    "71": "nd71", "88": "nd88", "102": "nd102", "103": "nd103",
    "151": "nd151", "226": "nd226", "49": "nd49", "50": "nd50",
    "12": "nd12", "254": "nq254",
}


def get_amendment_doc_ids(doc_id: str) -> list[str]:
    """Get doc_ids that amend the given doc."""
    return AMENDED_BY.get(doc_id, [])


def get_amended_doc_ids(doc_id: str) -> list[str]:
    """Get doc_ids that the given doc amends."""
    return AMENDS_MAP.get(doc_id, [])
