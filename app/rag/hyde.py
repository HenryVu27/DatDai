"""HyDE: Hypothetical Document Embeddings for legal RAG."""
import logging
from app import llm

logger = logging.getLogger(__name__)

HYDE_PROMPT = """Ban la chuyen gia luat dat dai Viet Nam. Hay viet mot doan ngan (100-150 tu)
tra loi cau hoi sau nhu the ban dang trich dan tu van ban luat. Su dung ngon ngu phap ly
chinh thuc, trich dan so dieu, khoan neu co the. Khong can chinh xac, chi can giong van ban luat.

Cau hoi: {query}

Tra loi (van ban phap ly):"""


async def generate_hyde_passage(query: str) -> str | None:
    try:
        passage = await llm.generate(HYDE_PROMPT.format(query=query), model="flash", temperature=0.3, max_tokens=300)
        passage = passage.strip()
        if len(passage) > 30:
            return passage
    except Exception as e:
        logger.warning("HyDE generation failed: %s", e)
    return None


async def get_hyde_embedding(query: str) -> list[float] | None:
    passage = await generate_hyde_passage(query)
    if not passage:
        return None
    try:
        return llm.embed([passage])[0]
    except Exception as e:
        logger.warning("HyDE embedding failed: %s", e)
        return None
