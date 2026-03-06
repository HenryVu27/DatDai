# Auth Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add individual user authentication with admin management to the DatDai chatbot.

**Architecture:** Server-side session cookies backed by a DB table. Users and auth_sessions tables added to existing dual-backend DB (Postgres/SQLite). FastAPI middleware checks cookie on every request, redirects to login page if missing/expired. Admin page for user management.

**Tech Stack:** FastAPI, bcrypt, existing DB layer (psycopg2/sqlite3), vanilla HTML/CSS/JS for login and admin pages.

---

### Task 1: Add bcrypt dependency

**Files:**
- Modify: `requirements.txt`

**Step 1: Add bcrypt to requirements**

Add this line to `requirements.txt`:

```
bcrypt>=4.0.0
```

**Step 2: Install**

Run: `cd /Users/vuducdung/personal/DatDai && .venv/bin/pip install bcrypt>=4.0.0`

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat(auth): add bcrypt dependency"
```

---

### Task 2: Add auth config

**Files:**
- Modify: `app/config.py`

**Step 1: Add auth env vars to config**

Add at the end of `app/config.py`:

```python
# Auth
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
SESSION_EXPIRY_DAYS = 7
```

**Step 2: Commit**

```bash
git add app/config.py
git commit -m "feat(auth): add auth config vars"
```

---

### Task 3: Add users and auth_sessions tables + DB functions

**Files:**
- Modify: `app/db.py`

This is the largest task. Add schema for `users` and `auth_sessions` tables, plus CRUD functions.

**Step 1: Write tests for auth DB functions**

Create `tests/test_auth_db.py`:

```python
"""Tests for auth database functions."""
import os
os.environ["SUPABASE_DB_URL"] = ""  # Force SQLite

import pytest
from app import db


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Use a fresh SQLite DB for each test."""
    monkeypatch.setattr("app.config.DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr("app.db.DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr("app.db._use_pg", False)
    yield


def test_create_user_and_get():
    user_id = db.create_user("testuser", "hashed_pw", is_admin=False)
    assert user_id is not None
    user = db.get_user_by_username("testuser")
    assert user is not None
    assert user["username"] == "testuser"
    assert user["password_hash"] == "hashed_pw"
    assert user["is_admin"] is False or user["is_admin"] == 0


def test_create_duplicate_user():
    db.create_user("testuser", "hashed_pw")
    result = db.create_user("testuser", "other_pw")
    assert result is None


def test_get_nonexistent_user():
    user = db.get_user_by_username("ghost")
    assert user is None


def test_list_users():
    db.create_user("alice", "pw1")
    db.create_user("bob", "pw2", is_admin=True)
    users = db.list_users()
    assert len(users) == 2
    usernames = {u["username"] for u in users}
    assert usernames == {"alice", "bob"}


def test_delete_user():
    uid = db.create_user("doomed", "pw")
    db.delete_user(uid)
    assert db.get_user_by_username("doomed") is None


def test_create_and_validate_auth_session():
    uid = db.create_user("sessuser", "pw")
    token = db.create_auth_session(uid)
    assert token is not None
    user_id = db.validate_auth_session(token)
    assert user_id == uid


def test_validate_expired_session():
    uid = db.create_user("expuser", "pw")
    token = db.create_auth_session(uid, expiry_days=-1)
    user_id = db.validate_auth_session(token)
    assert user_id is None


def test_delete_auth_session():
    uid = db.create_user("logoutuser", "pw")
    token = db.create_auth_session(uid)
    db.delete_auth_session(token)
    assert db.validate_auth_session(token) is None
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_auth_db.py -v`
Expected: FAIL (functions don't exist yet)

**Step 3: Add schema to `app/db.py`**

Add to both `_PG_SCHEMA` and `_SQLITE_SCHEMA` strings, before the closing `"""`:

For `_PG_SCHEMA`, add:

```sql
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL
);
```

For `_SQLITE_SCHEMA`, add:

```sql
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

**Step 4: Add auth DB functions to `app/db.py`**

Add these imports at top of `db.py`:

```python
import secrets
```

Add these functions at the end of `db.py`:

```python
def create_user(username: str, password_hash: str, is_admin: bool = False) -> str | None:
    """Create a user. Returns user ID or None if username exists."""
    import uuid
    user_id = str(uuid.uuid4())
    conn = _get_db()
    p = _ph()
    try:
        sql = f"INSERT INTO users (id, username, password_hash, is_admin, created_at) VALUES ({p}, {p}, {p}, {p}, {p})"
        args = (user_id, username, password_hash, is_admin, datetime.now().isoformat())
        if _use_pg:
            with conn.cursor() as cur:
                cur.execute(sql, args)
            conn.commit()
        else:
            conn.execute(sql, args)
            conn.commit()
    except Exception:
        if _use_pg:
            conn.rollback()
        _release_db(conn)
        return None
    _release_db(conn)
    return user_id


def get_user_by_username(username: str) -> dict | None:
    conn = _get_db()
    p = _ph()
    sql = f"SELECT id, username, password_hash, is_admin, created_at FROM users WHERE username = {p}"
    if _use_pg:
        with conn.cursor() as cur:
            cur.execute(sql, (username,))
            result = _row_to_dict(conn, cur)
    else:
        row = conn.execute(sql, (username,)).fetchone()
        result = dict(row) if row else None
    _release_db(conn)
    return result


def get_user_by_id(user_id: str) -> dict | None:
    conn = _get_db()
    p = _ph()
    sql = f"SELECT id, username, is_admin, created_at FROM users WHERE id = {p}"
    if _use_pg:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            result = _row_to_dict(conn, cur)
    else:
        row = conn.execute(sql, (user_id,)).fetchone()
        result = dict(row) if row else None
    _release_db(conn)
    return result


def list_users() -> list[dict]:
    conn = _get_db()
    sql = "SELECT id, username, is_admin, created_at FROM users ORDER BY created_at"
    if _use_pg:
        with conn.cursor() as cur:
            cur.execute(sql)
            result = _rows_to_dicts(conn, cur)
    else:
        rows = conn.execute(sql).fetchall()
        result = [dict(r) for r in rows]
    _release_db(conn)
    return result


def delete_user(user_id: str) -> None:
    conn = _get_db()
    p = _ph()
    if _use_pg:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM auth_sessions WHERE user_id = {p}", (user_id,))
            cur.execute(f"DELETE FROM users WHERE id = {p}", (user_id,))
        conn.commit()
    else:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"DELETE FROM auth_sessions WHERE user_id = {p}", (user_id,))
        conn.execute(f"DELETE FROM users WHERE id = {p}", (user_id,))
        conn.commit()
    _release_db(conn)


def create_auth_session(user_id: str, expiry_days: int = 7) -> str:
    token = secrets.token_hex(32)
    from datetime import timedelta
    expires_at = (datetime.now() + timedelta(days=expiry_days)).isoformat()
    conn = _get_db()
    p = _ph()
    sql = f"INSERT INTO auth_sessions (token, user_id, expires_at) VALUES ({p}, {p}, {p})"
    if _use_pg:
        with conn.cursor() as cur:
            cur.execute(sql, (token, user_id, expires_at))
        conn.commit()
    else:
        conn.execute(sql, (token, user_id, expires_at))
        conn.commit()
    _release_db(conn)
    return token


def validate_auth_session(token: str) -> str | None:
    """Return user_id if session is valid and not expired, else None."""
    conn = _get_db()
    p = _ph()
    sql = f"SELECT user_id, expires_at FROM auth_sessions WHERE token = {p}"
    if _use_pg:
        with conn.cursor() as cur:
            cur.execute(sql, (token,))
            row = _row_to_dict(conn, cur)
    else:
        r = conn.execute(sql, (token,)).fetchone()
        row = dict(r) if r else None
    _release_db(conn)
    if not row:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.now():
        # Expired -- clean up
        delete_auth_session(token)
        return None
    return row["user_id"]


def delete_auth_session(token: str) -> None:
    conn = _get_db()
    p = _ph()
    sql = f"DELETE FROM auth_sessions WHERE token = {p}"
    if _use_pg:
        with conn.cursor() as cur:
            cur.execute(sql, (token,))
        conn.commit()
    else:
        conn.execute(sql, (token,))
        conn.commit()
    _release_db(conn)
```

**Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_auth_db.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add app/db.py tests/test_auth_db.py
git commit -m "feat(auth): add users and auth_sessions tables with CRUD"
```

---

### Task 4: Add auth middleware and login/logout endpoints

**Files:**
- Create: `app/auth.py`
- Modify: `app/main.py`

**Step 1: Create `app/auth.py`**

```python
"""Authentication middleware and helpers."""
import logging

import bcrypt
from fastapi import Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from app import db
from app.config import SESSION_SECRET, SESSION_EXPIRY_DAYS

logger = logging.getLogger(__name__)

COOKIE_NAME = "datdai_session"
PUBLIC_PATHS = {"/login", "/auth/login", "/health", "/auth/check"}


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def get_current_user(request: Request) -> dict | None:
    """Extract and validate user from session cookie. Returns user dict or None."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    user_id = db.validate_auth_session(token)
    if not user_id:
        return None
    return db.get_user_by_id(user_id)


async def auth_middleware(request: Request, call_next):
    """Middleware that checks auth for non-public paths."""
    path = request.url.path

    # Allow public paths and static files
    if path in PUBLIC_PATHS or path.startswith("/static"):
        return await call_next(request)

    user = get_current_user(request)
    if not user:
        # API calls get 401, browser requests get redirect
        if path.startswith("/api") or path.startswith("/chat") or path.startswith("/sessions") or path.startswith("/feedback"):
            return JSONResponse(status_code=401, content={"detail": "Chua dang nhap"})
        return RedirectResponse(url="/login", status_code=302)

    # Attach user to request state for use in endpoints
    request.state.user = user
    return await call_next(request)


def ensure_admin_user(admin_password: str) -> None:
    """Create admin user on first run if it doesn't exist."""
    if not admin_password:
        return
    existing = db.get_user_by_username("admin")
    if existing:
        return
    pw_hash = hash_password(admin_password)
    db.create_user("admin", pw_hash, is_admin=True)
    logger.info("Created initial admin user")
```

**Step 2: Add auth routes and middleware to `app/main.py`**

Add these imports at the top of `main.py`:

```python
from app.auth import auth_middleware, get_current_user, hash_password, verify_password, ensure_admin_user, COOKIE_NAME
from app.config import ADMIN_PASSWORD, SESSION_EXPIRY_DAYS
```

Add middleware registration right after the `app = FastAPI(...)` line:

```python
app.middleware("http")(auth_middleware)
```

Add `ensure_admin_user(ADMIN_PASSWORD)` inside the `lifespan` function, after the DB warmup block (after `logger.info("Startup warmup: DB ready")`).

Add these new endpoints before the static files section:

```python
@app.get("/login")
def login_page():
    return FileResponse(os.path.join(static_dir, "login.html"))


@app.post("/auth/login")
async def login(request: Request):
    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Vui long nhap ten dang nhap va mat khau.")

    user = db.get_user_by_username(username)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Ten dang nhap hoac mat khau khong dung.")

    token = db.create_auth_session(user["id"], expiry_days=SESSION_EXPIRY_DAYS)
    response = JSONResponse(content={"ok": True, "is_admin": bool(user["is_admin"])})
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=SESSION_EXPIRY_DAYS * 86400,
        httponly=True,
        secure=True,
        samesite="strict",
    )
    return response


@app.post("/auth/logout")
async def logout(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        db.delete_auth_session(token)
    response = JSONResponse(content={"ok": True})
    response.delete_cookie(COOKIE_NAME)
    return response


@app.get("/auth/check")
async def auth_check(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"authenticated": False})
    return {"authenticated": True, "username": user["username"], "is_admin": bool(user["is_admin"])}
```

**Step 3: Update existing endpoints to use user context**

Modify `get_sessions` to filter by user:

The `get_sessions` function in `db.py` needs a `user_id` parameter. Add a new function `get_sessions_for_user`:

```python
# In db.py
def get_sessions_for_user(user_id: str) -> list[dict]:
    conn = _get_db()
    p = _ph()
    sql = f"""SELECT s.id, s.created_at, s.title,
           (SELECT content FROM messages WHERE session_id = s.id AND role = 'user' ORDER BY turn LIMIT 1) as first_message,
           (SELECT MAX(turn) FROM messages WHERE session_id = s.id) as turn_count
           FROM sessions s WHERE s.user_id = {p} ORDER BY s.created_at DESC LIMIT 50"""
    if _use_pg:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            result = _rows_to_dicts(conn, cur)
    else:
        rows = conn.execute(sql, (user_id,)).fetchall()
        result = [dict(r) for r in rows]
    _release_db(conn)
    return result
```

Add `user_id` column to the `sessions` table schema (both PG and SQLite):

```sql
-- Add to sessions table in both schemas:
-- After title column:
user_id TEXT DEFAULT ''
```

Update `create_session` in `db.py` to accept `user_id`:

```python
def create_session(session_id: str, title: str = "", user_id: str = "") -> None:
```

And include `user_id` in the INSERT statement.

Update `main.py` endpoints:

```python
@app.get("/sessions")
def list_sessions(request: Request) -> list[SessionResponse]:
    user = request.state.user
    sessions = db.get_sessions_for_user(user["id"])
    return [SessionResponse(**s) for s in sessions]
```

Update `chat_endpoint` to pass `user_id` through to session creation. In `app/chat.py`, update `create_new_session` to accept `user_id` and pass it to `db.create_session`.

**Step 4: Commit**

```bash
git add app/auth.py app/main.py app/db.py app/chat.py
git commit -m "feat(auth): add middleware, login/logout endpoints, user-scoped sessions"
```

---

### Task 5: Create login page

**Files:**
- Create: `app/static/login.html`

**Step 1: Create login page**

Build `app/static/login.html` -- a simple login form that matches the existing design system (same fonts, colors, `--accent`, `--bg`, etc. from `index.html`). The page should:

- Show a centered card with username/password fields and a submit button
- POST to `/auth/login` via fetch
- On success, redirect to `/`
- On error, show the error message inline
- Match the existing warm/serif aesthetic (Lora headings, Be Vietnam Pro body, `var(--accent)` = `#b44a2f`)

**Step 2: Commit**

```bash
git add app/static/login.html
git commit -m "feat(auth): add login page"
```

---

### Task 6: Create admin page

**Files:**
- Create: `app/static/admin.html`
- Modify: `app/main.py` (add `/admin` route)

**Step 1: Add admin route to `main.py`**

```python
@app.get("/admin")
def admin_page(request: Request):
    user = request.state.user
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Khong co quyen truy cap.")
    return FileResponse(os.path.join(static_dir, "admin.html"))
```

Add admin API endpoints:

```python
@app.get("/admin/users")
def admin_list_users(request: Request):
    user = request.state.user
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Khong co quyen truy cap.")
    return db.list_users()


@app.post("/admin/users")
async def admin_create_user(request: Request):
    user = request.state.user
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Khong co quyen truy cap.")
    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Vui long nhap ten va mat khau.")
    pw_hash = hash_password(password)
    user_id = db.create_user(username, pw_hash, is_admin=False)
    if not user_id:
        raise HTTPException(status_code=409, detail="Ten dang nhap da ton tai.")
    return {"id": user_id, "username": username}


@app.delete("/admin/users/{user_id}")
def admin_delete_user(user_id: str, request: Request):
    user = request.state.user
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Khong co quyen truy cap.")
    if user_id == user["id"]:
        raise HTTPException(status_code=400, detail="Khong the xoa chinh minh.")
    db.delete_user(user_id)
    return {"ok": True}
```

**Step 2: Create `app/static/admin.html`**

A simple admin page matching existing design that:
- Lists all users in a table (username, admin status, created date)
- Has a form to add new user (username + password)
- Has a delete button per user (with confirmation)
- Calls the admin API endpoints via fetch
- Shows a link back to the chat

**Step 3: Commit**

```bash
git add app/static/admin.html app/main.py
git commit -m "feat(auth): add admin page for user management"
```

---

### Task 7: Add logout button and admin link to main UI

**Files:**
- Modify: `app/static/index.html`

**Step 1: Update sidebar footer in `index.html`**

Replace the `.sidebar-footer` div with a version that shows the current user, a logout button, and (if admin) a link to `/admin`. On page load, call `/auth/check` to get the current user info.

Add to the JS section:

```javascript
async function initAuth() {
  try {
    const res = await fetch('/auth/check');
    if (!res.ok) { window.location.href = '/login'; return; }
    const data = await res.json();
    const footer = document.querySelector('.sidebar-footer');
    let html = `<span>${data.username}</span>`;
    if (data.is_admin) html += ` | <a href="/admin" style="color:var(--accent)">Quan ly</a>`;
    html += ` | <a href="#" onclick="doLogout()" style="color:var(--text-muted)">Dang xuat</a>`;
    footer.innerHTML = html;
  } catch(e) {}
}

async function doLogout() {
  await fetch('/auth/logout', { method: 'POST' });
  window.location.href = '/login';
}
```

Call `initAuth()` in the init section.

**Step 2: Commit**

```bash
git add app/static/index.html
git commit -m "feat(auth): add logout button and admin link to sidebar"
```

---

### Task 8: Add ADMIN_PASSWORD to deployment config

**Files:**
- Modify: `.github/workflows/deploy.yml` (if applicable)
- Modify: `.env.example` or docs

**Step 1: Add env vars to deployment**

Add `ADMIN_PASSWORD` and `SESSION_SECRET` to Cloud Run environment or secrets. Update `.env.example` if it exists.

**Step 2: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "feat(auth): add auth env vars to deployment config"
```

---

### Task 9: Migration for existing sessions

**Files:**
- Modify: `app/db.py`

**Step 1: Handle existing sessions without user_id**

The schema change adds `user_id TEXT DEFAULT ''` to sessions. Existing sessions will have empty user_id. The `get_sessions_for_user` query already filters by user_id, so old sessions become invisible to normal users. Admin can still see them via direct DB access if needed.

No migration script needed -- the `DEFAULT ''` handles it. Just verify the schema change works with existing data.

**Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All pass

**Step 3: Commit if any fixes needed**

---

### Task 10: End-to-end manual test

**Step 1: Start the app locally**

```bash
ADMIN_PASSWORD=test123 .venv/bin/python -m uvicorn app.main:app --reload
```

**Step 2: Verify**

1. Visit `http://localhost:8000/` -- should redirect to `/login`
2. Log in as `admin` / `test123`
3. Should redirect to chat, sidebar shows "admin | Quan ly | Dang xuat"
4. Click "Quan ly" -- admin page loads
5. Add a new user "dad" with a password
6. Log out, log in as "dad"
7. "dad" should not see admin link
8. "dad" should not be able to access `/admin` directly
9. Chat works, sessions are scoped to user
