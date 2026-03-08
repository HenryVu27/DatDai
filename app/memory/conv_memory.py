"""Schema-free conversation state extractor and store.

After each assistant turn, extract_state() extracts a compact JSON object
from the response. The LLM decides what fields to keep — no fixed schema.
The extractor takes the previous state as input, enabling incremental updates.

Public API:
  read(session_id) -> dict | None
  update(session_id, response, turn) -> None  (async, call as background task)
"""
import json
import logging

from app import db, llm

logger = logging.getLogger(__name__)

_EXTRACTOR_SYSTEM = """Ban la bo trich xuat trang thai hoi thoai cho chatbot phap luat dat dai.

Nhiem vu: Doc cau tra loi cua tro ly va cap nhat trang thai hoi thoai.
Trang thai giup giai quyet cac cau hoi tiep theo nhu "truong hop 3", "buoc 2", "dieu do".

Quy tac:
- Giu lai thong tin tu trang thai truoc (neu co) tru khi chu de thay doi hoan toan
- Luon ghi lai danh sach so thu tu neu co trong cau tra loi (day du noi dung moi muc)
- Ghi lai phap luat dang duoc ban (dieu luat, van ban)
- Ghi lai chu de chinh dang thao luan
- Su dung khoa tu do — khong co schema co dinh — chi ghi nhung gi se huu ich de giai quyet cau hoi tiep theo

Tra ve CHINH XAC mot JSON object (khong markdown, khong giai thich).
Neu khong co thong tin dang ke, tra ve {}.

Vi du dau ra cho hoi thoai ve danh sach:
{"chu_de": "boi thuong khi Nha nuoc thu hoi dat", "danh_sach_truong_hop": ["1. ...", "2. ...", "3. Chi phi dau tu con lai..."], "van_ban": "Dieu 101, 107 LDD2024"}

Vi du dau ra cho hoi thoai ve thu tuc:
{"chu_de": "thu tuc cap so do lan dau", "cac_buoc": ["1. Chuan bi ho so", "2. Nop tai UBND", "3. Nhan ket qua"], "co_quan": "UBND cap huyen"}"""


async def extract_state(response: str, prev_state: dict | None) -> dict | None:
    """Extract conversation state from an assistant response.

    Returns a free-form JSON dict, or None if extraction fails.
    """
    prev_block = ""
    if prev_state:
        prev_json = json.dumps(prev_state, ensure_ascii=False)
        prev_block = f"<trang-thai-truoc>\n{prev_json}\n</trang-thai-truoc>\n\n"

    prompt = (
        f"{prev_block}"
        f"<cau-tra-loi-moi>\n{response[:3000]}\n</cau-tra-loi-moi>\n\n"
        "Cap nhat trang thai hoi thoai:"
    )

    try:
        raw = await llm.generate(
            prompt=prompt,
            system=_EXTRACTOR_SYSTEM,
            model="utility",
            temperature=0.0,
            max_tokens=400,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(l for l in lines[1:] if not l.strip().startswith("```"))
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("State extractor returned invalid JSON: %s", raw[:100] if 'raw' in dir() else "?")
        return None
    except Exception as e:
        logger.error("State extractor failed: %s", e)
        return None


def read(session_id: str) -> dict | None:
    """Return the current conversation state for a session, or None."""
    return db.get_conv_state(session_id)


async def update(session_id: str, response: str, turn: int) -> None:
    """Extract state from the latest response and persist it.

    Designed to run as a background task — never raises.
    """
    try:
        prev_state = db.get_conv_state(session_id)
        new_state = await extract_state(response, prev_state)
        if new_state is not None:
            db.upsert_conv_state(session_id, new_state, turn)
            logger.debug("Conv state updated for session %s turn %d", session_id[:8], turn)
    except Exception as e:
        logger.error("Conv state update failed for session %s: %s", session_id[:8], e)
