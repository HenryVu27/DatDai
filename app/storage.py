"""Load reference data from Supabase Storage or local filesystem."""
import json
import logging
import os
from functools import lru_cache

from app.config import SUPABASE_URL, SUPABASE_SERVICE_KEY, DATA_DIR

log = logging.getLogger(__name__)

_BUCKET = "datdai-data"


def _get_storage_client():
    """Return Supabase storage client, or None if not configured."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None
    from supabase import create_client
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return client.storage


def _load_json(remote_path: str, local_path: str) -> dict | list:
    """Load JSON from Supabase Storage, falling back to local file."""
    storage = _get_storage_client()
    if storage:
        log.info("Loading %s from Supabase Storage", remote_path)
        data = storage.from_(_BUCKET).download(remote_path)
        return json.loads(data)
    log.info("Loading %s from local filesystem", local_path)
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_vocab() -> dict:
    return _load_json("vocab.json", os.path.join(DATA_DIR, "vocab.json"))


@lru_cache(maxsize=1)
def load_amendment_index() -> dict:
    return _load_json(
        "amendment_index.json",
        os.path.join(DATA_DIR, "amendment_index.json"),
    )
