"""Database layer for conversation persistence.

Uses PostgreSQL (via psycopg2) when SUPABASE_DB_URL is set,
falls back to SQLite for local development.
"""
import json
import os
import sqlite3
from datetime import datetime

from app.config import DB_PATH, SUPABASE_DB_URL

_use_pg = bool(SUPABASE_DB_URL)

if _use_pg:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool

# --- Schema ---

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    turn INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    summary TEXT NOT NULL,
    covers_through_turn INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
"""

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    covers_through_turn INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
"""

_pg_schema_initialized = False
_pg_pool = None


def _get_pg():
    global _pg_schema_initialized, _pg_pool
    if _pg_pool is None:
        _pg_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1, maxconn=5, dsn=SUPABASE_DB_URL
        )
    conn = _pg_pool.getconn()
    if not _pg_schema_initialized:
        with conn.cursor() as cur:
            cur.execute(_PG_SCHEMA)
        conn.commit()
        _pg_schema_initialized = True
    return conn


def _get_sqlite():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SQLITE_SCHEMA)
    return conn


def _get_db():
    return _get_pg() if _use_pg else _get_sqlite()


def _release_db(conn):
    if _use_pg and _pg_pool is not None:
        _pg_pool.putconn(conn)
    else:
        conn.close()


def _ph(name=""):
    """Return placeholder style."""
    return "%s" if _use_pg else "?"


def _rows_to_dicts(conn, cursor):
    """Convert cursor results to list of dicts."""
    if _use_pg:
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    return [dict(r) for r in cursor.fetchall()]


def _row_to_dict(conn, cursor):
    """Convert single cursor result to dict or None."""
    if _use_pg:
        row = cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    row = cursor.fetchone()
    return dict(row) if row else None


def create_session(session_id: str, title: str = "") -> None:
    db = _get_db()
    p = _ph()
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(
                f"INSERT INTO sessions (id, created_at, title) VALUES ({p}, {p}, {p}) ON CONFLICT DO NOTHING",
                (session_id, datetime.now().isoformat(), title),
            )
        db.commit()
    else:
        db.execute(
            f"INSERT OR IGNORE INTO sessions (id, created_at, title) VALUES ({p}, {p}, {p})",
            (session_id, datetime.now().isoformat(), title),
        )
        db.commit()
    _release_db(db)


def add_message(session_id: str, turn: int, role: str, content: str, sources: list | None = None) -> None:
    db = _get_db()
    p = _ph()
    sql = f"INSERT INTO messages (session_id, turn, role, content, sources, created_at) VALUES ({p}, {p}, {p}, {p}, {p}, {p})"
    args = (session_id, turn, role, content, json.dumps(sources or []), datetime.now().isoformat())
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, args)
        db.commit()
    else:
        db.execute(sql, args)
        db.commit()
    _release_db(db)


def add_messages_batch(session_id: str, messages: list[tuple[int, str, str, list | None]]) -> None:
    """Insert multiple messages in a single connection + transaction.

    Each message is a tuple of (turn, role, content, sources).
    """
    db = _get_db()
    p = _ph()
    sql = f"INSERT INTO messages (session_id, turn, role, content, sources, created_at) VALUES ({p}, {p}, {p}, {p}, {p}, {p})"
    now = datetime.now().isoformat()
    if _use_pg:
        with db.cursor() as cur:
            for turn, role, content, sources in messages:
                cur.execute(sql, (session_id, turn, role, content, json.dumps(sources or []), now))
        db.commit()
    else:
        for turn, role, content, sources in messages:
            db.execute(sql, (session_id, turn, role, content, json.dumps(sources or []), now))
        db.commit()
    _release_db(db)


def get_messages(session_id: str, limit: int = 100) -> list[dict]:
    db = _get_db()
    p = _ph()
    sql = f"SELECT role, content, turn, sources FROM messages WHERE session_id = {p} ORDER BY turn, id LIMIT {p}"
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, (session_id, limit))
            result = _rows_to_dicts(db, cur)
    else:
        rows = db.execute(sql, (session_id, limit)).fetchall()
        result = [dict(r) for r in rows]
    _release_db(db)
    return result


def get_turn_count(session_id: str) -> int:
    db = _get_db()
    p = _ph()
    sql = f"SELECT COALESCE(MAX(turn), 0) as max_turn FROM messages WHERE session_id = {p}"
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, (session_id,))
            row = cur.fetchone()
        _release_db(db)
        return row[0] if row else 0
    else:
        row = db.execute(sql, (session_id,)).fetchone()
        _release_db(db)
        return row["max_turn"] if row else 0


def save_summary(session_id: str, summary: str, covers_through_turn: int) -> None:
    db = _get_db()
    p = _ph()
    sql = f"INSERT INTO summaries (session_id, summary, covers_through_turn, created_at) VALUES ({p}, {p}, {p}, {p})"
    args = (session_id, summary, covers_through_turn, datetime.now().isoformat())
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, args)
        db.commit()
    else:
        db.execute(sql, args)
        db.commit()
    _release_db(db)


def get_latest_summary(session_id: str) -> dict | None:
    db = _get_db()
    p = _ph()
    sql = f"SELECT summary, covers_through_turn FROM summaries WHERE session_id = {p} ORDER BY covers_through_turn DESC LIMIT 1"
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, (session_id,))
            result = _row_to_dict(db, cur)
    else:
        row = db.execute(sql, (session_id,)).fetchone()
        result = dict(row) if row else None
    _release_db(db)
    return result


def upsert_summary(session_id: str, summary: str, covers_through_turn: int) -> None:
    db = _get_db()
    p = _ph()
    now = datetime.now().isoformat()
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(
                f"SELECT id FROM summaries WHERE session_id = {p} ORDER BY covers_through_turn DESC LIMIT 1",
                (session_id,),
            )
            existing = cur.fetchone()
            if existing:
                cur.execute(
                    f"UPDATE summaries SET summary = {p}, covers_through_turn = {p}, created_at = {p} WHERE id = {p}",
                    (summary, covers_through_turn, now, existing[0]),
                )
            else:
                cur.execute(
                    f"INSERT INTO summaries (session_id, summary, covers_through_turn, created_at) VALUES ({p}, {p}, {p}, {p})",
                    (session_id, summary, covers_through_turn, now),
                )
        db.commit()
    else:
        existing = db.execute(
            f"SELECT id FROM summaries WHERE session_id = {p} ORDER BY covers_through_turn DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if existing:
            db.execute(
                f"UPDATE summaries SET summary = {p}, covers_through_turn = {p}, created_at = {p} WHERE id = {p}",
                (summary, covers_through_turn, now, existing["id"]),
            )
        else:
            db.execute(
                f"INSERT INTO summaries (session_id, summary, covers_through_turn, created_at) VALUES ({p}, {p}, {p}, {p})",
                (session_id, summary, covers_through_turn, now),
            )
        db.commit()
    _release_db(db)


def get_sessions() -> list[dict]:
    db = _get_db()
    sql = """SELECT s.id, s.created_at, s.title,
           (SELECT content FROM messages WHERE session_id = s.id AND role = 'user' ORDER BY turn LIMIT 1) as first_message,
           (SELECT MAX(turn) FROM messages WHERE session_id = s.id) as turn_count
           FROM sessions s ORDER BY s.created_at DESC LIMIT 50"""
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql)
            result = _rows_to_dicts(db, cur)
    else:
        rows = db.execute(sql).fetchall()
        result = [dict(r) for r in rows]
    _release_db(db)
    return result


def update_session_title(session_id: str, title: str) -> None:
    db = _get_db()
    p = _ph()
    sql = f"UPDATE sessions SET title = {p} WHERE id = {p}"
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(sql, (title, session_id))
        db.commit()
    else:
        db.execute(sql, (title, session_id))
        db.commit()
    _release_db(db)
