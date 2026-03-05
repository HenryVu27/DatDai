"""SQLite database for conversation persistence."""
import json
import os
import sqlite3
from datetime import datetime

from app.config import DB_PATH

SCHEMA = """
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


def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def create_session(session_id: str, title: str = "") -> None:
    db = get_db()
    db.execute(
        "INSERT OR IGNORE INTO sessions (id, created_at, title) VALUES (?, ?, ?)",
        (session_id, datetime.now().isoformat(), title),
    )
    db.commit()
    db.close()


def add_message(session_id: str, turn: int, role: str, content: str, sources: list | None = None) -> None:
    db = get_db()
    db.execute(
        "INSERT INTO messages (session_id, turn, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, turn, role, content, json.dumps(sources or []), datetime.now().isoformat()),
    )
    db.commit()
    db.close()


def get_messages(session_id: str, limit: int = 100) -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT role, content, turn, sources FROM messages WHERE session_id = ? ORDER BY turn, id LIMIT ?",
        (session_id, limit),
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def get_turn_count(session_id: str) -> int:
    db = get_db()
    row = db.execute(
        "SELECT COALESCE(MAX(turn), 0) as max_turn FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    db.close()
    return row["max_turn"] if row else 0


def save_summary(session_id: str, summary: str, covers_through_turn: int) -> None:
    db = get_db()
    db.execute(
        "INSERT INTO summaries (session_id, summary, covers_through_turn, created_at) VALUES (?, ?, ?, ?)",
        (session_id, summary, covers_through_turn, datetime.now().isoformat()),
    )
    db.commit()
    db.close()


def get_latest_summary(session_id: str) -> dict | None:
    db = get_db()
    row = db.execute(
        "SELECT summary, covers_through_turn FROM summaries WHERE session_id = ? ORDER BY covers_through_turn DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    db.close()
    return dict(row) if row else None


def get_sessions() -> list[dict]:
    db = get_db()
    rows = db.execute(
        """SELECT s.id, s.created_at, s.title,
           (SELECT content FROM messages WHERE session_id = s.id AND role = 'user' ORDER BY turn LIMIT 1) as first_message,
           (SELECT MAX(turn) FROM messages WHERE session_id = s.id) as turn_count
           FROM sessions s ORDER BY s.created_at DESC LIMIT 50""",
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def update_session_title(session_id: str, title: str) -> None:
    db = get_db()
    db.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
    db.commit()
    db.close()
