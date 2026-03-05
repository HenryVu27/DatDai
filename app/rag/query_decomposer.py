"""Query decomposition for complex Vietnamese land law questions."""
import logging
from app import llm

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """Ban la chuyen gia phap luat dat dai Viet Nam.
Nhiem vu: tach cau hoi phuc tap thanh 2-4 cau hoi con doc lap.
Moi cau hoi con nen tap trung vao 1 chu de / van ban phap luat cu the.

Danh sach van ban:
- Luat Dat Dai 2024 (31/2024/QH15) - luat goc
- ND 71/2024 - gia dat (sua doi boi ND 226, ND 49)
- ND 88/2024 - boi thuong, ho tro, tai dinh cu (sua doi boi ND 226, ND 49)
- ND 102/2024 - chi tiet thi hanh (sua doi boi ND 226, ND 49)
- ND 103/2024 - tien su dung dat (sua doi boi ND 50)
- ND 151/2025 - phan dinh tham quyen (sua doi boi ND 226, ND 49)
- ND 226/2025 - sua doi ND 71, 88, 102, 151
- NQ 254/2025 - thao go vuong mac
- ND 49/2026 - sua doi ND 71, 88, 102, 151, 226
- ND 12/2024 - sua doi ND 44/2014, ND 10/2023 (chuyen tiep)
- ND 50/2026 - chi tiet NQ 254

Quy tac:
- Neu cau hoi don gian (1 chu de), tra ve nguyen van cau hoi.
- Neu phuc tap, tach thanh 2-4 cau hoi con.
- Moi dong 1 cau hoi, khong danh so, khong giai thich.
- Toi da 4 cau hoi.

Cau hoi: {query}

Cac cau hoi con:"""


async def decompose_query(query: str) -> list[str]:
    """Split a complex query into 2-4 sub-queries using Gemini Flash.

    Returns the original query in a list if the question is simple
    or decomposition fails.
    """
    # Skip decomposition for obviously simple queries
    if len(query) < 30 or query.count("?") <= 1 and " va " not in query.lower():
        logger.debug("Simple query, skipping decomposition: %s", query[:60])
        return [query]

    try:
        response = await llm.generate(
            DECOMPOSE_PROMPT.format(query=query),
            model="flash",
            temperature=0.0,
            max_tokens=500,
        )
        sub_queries = [
            line.strip()
            for line in response.strip().splitlines()
            if line.strip()
        ]
        # Cap at 4 sub-queries
        sub_queries = sub_queries[:4]

        if not sub_queries:
            logger.warning("Decomposition returned empty result, using original")
            return [query]

        logger.info("Decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)
        return sub_queries

    except Exception:
        logger.exception("Query decomposition failed, using original query")
        return [query]
