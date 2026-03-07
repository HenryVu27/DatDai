# UX Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix two critical bugs (streaming timeout, error retry) and three medium polish items (citation emoji, markdown tables, session deletion).

**Architecture:** Changes across three layers: `app/rag/citation_check.py` (1-line fix), `app/db.py` + `app/main.py` (delete endpoint), and `app/static/index.html` (JS/CSS). All tasks are independent.

**Tech Stack:** FastAPI, SQLite/PostgreSQL via psycopg2, vanilla JS/CSS in single `index.html`.

---

## Task 1: Fix streaming idle timeout (Critical F)

**The bug:** `handleSubmit` sets a 120s `setTimeout` on an `AbortController`, but `clearTimeout()` is called immediately after `fetch()` resolves (which happens when HTTP *headers* arrive, not when the body finishes). The streaming read loop has zero timeout protection -- a hanging Pro query will wait forever.

**Fix:** Replace the single upfront timeout with a rolling idle timer that resets on each received chunk.

**Files:**
- Modify: `app/static/index.html` — `handleSubmit` function (~lines 978-1061)

**Step 1: Locate the fetch setup block**

Find this block (~line 979):
```javascript
currentController = new AbortController();
const timeout = setTimeout(() => currentController.abort(), 120000);
const res = await fetch('/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question, session_id: currentSessionId }),
  signal: currentController.signal,
});
clearTimeout(timeout);
```

**Step 2: Replace with connection timeout + idle timer**

Replace the block above with:
```javascript
currentController = new AbortController();
// 20s to establish TCP connection / get first byte from server
const connectTimeout = setTimeout(() => currentController.abort(), 20000);
const res = await fetch('/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question, session_id: currentSessionId }),
  signal: currentController.signal,
});
clearTimeout(connectTimeout);
```

**Step 3: Add idle timer after `let buffer = '';`**

After `const reader = res.body.getReader();` and `const decoder = new TextDecoder();` and `let buffer = '';`, add:
```javascript
// Idle timeout: abort if no SSE data arrives for 60s (e.g., Pro model hangs mid-generation)
let idleTimer = null;
const IDLE_TIMEOUT_MS = 60000;
function resetIdleTimer() {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    if (!userAborted) currentController.abort();
  }, IDLE_TIMEOUT_MS);
}
resetIdleTimer();
```

**Step 4: Reset the timer on each received chunk**

Inside the `while (true)` loop, immediately after:
```javascript
buffer += decoder.decode(value, { stream: true });
```
Add:
```javascript
resetIdleTimer();
```

**Step 5: Clear the timer in the catch block**

At the very start of `catch (err) {`, add:
```javascript
if (idleTimer) clearTimeout(idleTimer);
```

Also add it after the while loop exits (before `finalizeBubble`):
```javascript
if (idleTimer) clearTimeout(idleTimer);
finalizeBubble(msgDiv, fullText, streamSources, streamTraceId);
```

**Step 6: Verify the timeout message still appears correctly**

The `AbortError` path in the catch block already shows "Xin lỗi, yêu cầu đã hết thời gian chờ." -- the idle timer fires `currentController.abort()` with `userAborted = false`, so this message will show correctly.

**Step 7: Commit**
```bash
git add app/static/index.html
git commit -m "fix: replace upfront stream timeout with per-chunk idle timer (60s)"
```

---

## Task 2: Error recovery UX -- retry button (Critical G)

**The bug:** On error, `handleSubmit` shows inline error text but no action button. `finalizeBubble` (which adds the regenerate button) is never called in the error path. Users must retype their question manually after a failed 60s query.

**Fix:** In the error handler's else branch, render the error message with a retry button using existing `regenerateResponse()` and CSS classes.

**Files:**
- Modify: `app/static/index.html` — catch block in `handleSubmit` (~lines 1034-1054)

**Step 1: Find the non-abort error path**

In the `catch (err)` block, find the `else` branch:
```javascript
} else {
  const errorMsg = err.name === 'AbortError'
    ? 'Xin lỗi, yêu cầu đã hết thời gian chờ. Vui lòng thử lại.'
    : `Xin lỗi, đã xảy ra lỗi: ${err.message}`;
  if (fullText) {
    fullText += '\n\n**Lỗi:** ' + errorMsg;
    finalizeBubble(msgDiv, fullText, streamSources, null);
  } else {
    msgDiv.querySelector('.bubble').innerHTML = formatContent(errorMsg);
  }
}
```

**Step 2: Replace the empty-bubble branch to include a retry button**

```javascript
} else {
  const errorMsg = err.name === 'AbortError'
    ? 'Xin lỗi, yêu cầu đã hết thời gian chờ. Vui lòng thử lại.'
    : `Xin lỗi, đã xảy ra lỗi: ${err.message}`;
  if (fullText) {
    fullText += '\n\n**Lỗi:** ' + errorMsg;
    finalizeBubble(msgDiv, fullText, streamSources, null);
  } else {
    // Remove any stale regenerate buttons from prior messages
    chatArea.querySelectorAll('.feedback-btn.regenerate').forEach(btn => btn.remove());
    msgDiv.querySelector('.bubble').innerHTML =
      formatContent(errorMsg) +
      '<div class="feedback"><button type="button" class="feedback-btn regenerate" onclick="regenerateResponse()" title="Thử lại">&#x21BB; Thử lại</button></div>';
  }
}
```

**Step 3: Verify `regenerateResponse` works on error path**

`regenerateResponse()` checks `if (!lastUserQuestion || isLoading) return;`. The `isLoading = false` line runs at the end of `handleSubmit` (after the catch block), so by the time the user clicks Retry, `isLoading` is `false` and `lastUserQuestion` still holds the failed question. No changes needed to `regenerateResponse`.

**Step 4: Commit**
```bash
git add app/static/index.html
git commit -m "fix: show retry button on error instead of dead-end error message"
```

---

## Task 3: Remove emoji from citation disclaimer (Medium H)

**Problem:** `_DISCLAIMER` in `citation_check.py` starts with `⚠️`. One-line fix.

**Files:**
- Modify: `app/rag/citation_check.py:26-30`

**Step 1: Edit the constant**

Change:
```python
_DISCLAIMER = (
    "\n\n⚠️ *Lưu ý: Một số trích dẫn trong câu trả lời "
    "chưa được xác minh từ văn bản gốc. "
    "Vui lòng kiểm tra lại các điều khoản cụ thể.*"
)
```

To:
```python
_DISCLAIMER = (
    "\n\n*Lưu ý: Một số trích dẫn trong câu trả lời "
    "chưa được xác minh từ văn bản gốc. "
    "Vui lòng kiểm tra lại các điều khoản cụ thể.*"
)
```

**Step 2: Quick smoke test**
```bash
cd /Users/vuducdung/personal/DatDai
.venv/bin/python -c "
from app.rag.citation_check import verify_citations
r, m = verify_citations('Theo Dieu 999 quy dinh...', [])
assert '⚠️' not in r, 'Emoji still present'
assert 'Lưu ý' in r, 'Disclaimer missing'
print('OK:', r[-80:])
"
```
Expected: `OK:` followed by disclaimer text without emoji.

**Step 3: Commit**
```bash
git add app/rag/citation_check.py
git commit -m "fix: remove emoji from citation disclaimer"
```

---

## Task 4: Add markdown table rendering (Medium C)

**Problem:** `formatContent()` in `index.html` uses custom regex for bold/headers/lists but has no table support. Gemini Pro produces pipe tables (e.g., price method comparisons) which render as raw `|` characters.

**Fix:** Add table regex to `formatContent` before paragraph splitting, plus CSS for table display.

**Files:**
- Modify: `app/static/index.html` — `<style>` block and `formatContent` function

**Step 1: Add table CSS to the `<style>` block**

Add after the `.source-tag` rule group (after the `}` that closes `.source-tag`, around line 278):
```css
/* Markdown tables */
.bubble table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.6em 0;
  font-size: 0.85rem;
  overflow-x: auto;
  display: block;
}
.bubble th, .bubble td {
  border: 1px solid var(--border);
  padding: 0.35rem 0.65rem;
  text-align: left;
  vertical-align: top;
}
.bubble th {
  background: var(--bg-warm);
  font-weight: 600;
}
.bubble tr:nth-child(even) td {
  background: var(--surface-hover);
}
```

**Step 2: Add table parsing to `formatContent`**

The current `formatContent` function (line 783):
```javascript
function formatContent(text) {
  let h = esc(text);
  // Bold
  h = h.replace(...);
  // Headers
  h = h.replace(...);
  // Lists
  h = h.replace(...);
  // Paragraphs
  h = h.split('\n\n').map(p => {
    ...
    if (p.startsWith('<h') || p.startsWith('<ul')) return p;
    ...
  }).join('');
  return h;
}
```

Add the table replacement block **before** the `// Paragraphs` section (i.e., after the list replacement, before `h = h.split('\n\n')`):
```javascript
// Tables: matches header row | sep row | data rows...
h = h.replace(
  /^(\|.+\|[ \t]*)(\n\|[-:| \t]+\|[ \t]*)(\n\|.+\|[ \t]*)+/gm,
  (match) => {
    const lines = match.trim().split('\n');
    if (lines.length < 3) return match;
    const headerCells = lines[0].split('|').slice(1, -1)
      .map(c => `<th>${c.trim()}</th>`);
    // lines[1] is the separator row -- skip it
    const bodyRows = lines.slice(2).map(line => {
      const cells = line.split('|').slice(1, -1)
        .map(c => `<td>${c.trim()}</td>`);
      return `<tr>${cells.join('')}</tr>`;
    });
    return `<table><thead><tr>${headerCells.join('')}</tr></thead>`
      + `<tbody>${bodyRows.join('')}</tbody></table>`;
  }
);
```

Also add `'<table'` to the block-element check in the paragraph splitting map:
```javascript
if (p.startsWith('<h') || p.startsWith('<ul') || p.startsWith('<table')) return p;
```

**Step 3: Test in browser console**

Open the app, open browser console (F12), paste:
```javascript
formatContent('| Phuong phap | Nguyen tac |\n|---|---|\n| So sanh | Dua vao gia thi truong |\n| Thu nhap | Von hoa thu nhap |')
```
Expected: returns a string containing `<table>`, `<thead>`, `<th>`, `<tbody>`, `<tr>`, `<td>` tags.

**Step 4: Commit**
```bash
git add app/static/index.html
git commit -m "feat: add markdown table rendering to chat bubbles"
```

---

## Task 5: Session deletion (Medium D)

**Problem:** Users can't delete sessions. The sidebar accumulates up to 50 entries with no cleanup. Three sub-parts: DB function, API endpoint, frontend button.

**Files:**
- Modify: `app/db.py` — add `delete_session()` after `update_session_title` (line 331)
- Modify: `app/main.py` — add `DELETE /sessions/{session_id}` after `POST /sessions` (~line 179)
- Modify: `app/static/index.html` — session item wrapper + delete button + CSS + `deleteSession()` function

### Part A: Backend

**Step 1: Add `delete_session` to `app/db.py`**

At the end of `app/db.py` (after `update_session_title`), add:
```python
def delete_session(session_id: str) -> None:
    """Delete a session and all its messages and summaries (cascade)."""
    db = _get_db()
    p = _ph()
    if _use_pg:
        with db.cursor() as cur:
            cur.execute(f"DELETE FROM summaries WHERE session_id = {p}", (session_id,))
            cur.execute(f"DELETE FROM messages WHERE session_id = {p}", (session_id,))
            cur.execute(f"DELETE FROM sessions WHERE id = {p}", (session_id,))
        db.commit()
    else:
        db.execute(f"DELETE FROM summaries WHERE session_id = {p}", (session_id,))
        db.execute(f"DELETE FROM messages WHERE session_id = {p}", (session_id,))
        db.execute(f"DELETE FROM sessions WHERE id = {p}", (session_id,))
        db.commit()
    _release_db(db)
```

Note: deletes in dependency order (summaries → messages → sessions) to satisfy FK constraints in PostgreSQL. SQLite doesn't enforce them by default but the order is still correct.

**Step 2: Add `DELETE /sessions/{session_id}` to `app/main.py`**

After the `POST /sessions` route (after line ~179):
```python
@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    db.delete_session(session_id)
    return {"status": "ok"}
```

**Step 3: Smoke test the backend**
```bash
cd /Users/vuducdung/personal/DatDai
.venv/bin/python -c "
from app import db
db.create_session('_test_delete_')
db.add_message('_test_delete_', 1, 'user', 'hello')
db.delete_session('_test_delete_')
sessions = db.get_sessions()
ids = [s['id'] for s in sessions]
assert '_test_delete_' not in ids, 'Session not deleted'
print('delete_session: OK')
"
```
Expected: `delete_session: OK`

**Step 4: Commit backend**
```bash
git add app/db.py app/main.py
git commit -m "feat: add DELETE /sessions/{id} with cascade delete"
```

### Part B: Frontend

**Step 5: Add delete button CSS**

Add to the `<style>` block near the `.session-item` rules:
```css
.session-item-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}
.session-item-wrapper .session-item {
  padding-right: 2rem; /* prevent text from sliding under delete button */
}
.session-delete-btn {
  position: absolute;
  right: 4px;
  top: 50%;
  transform: translateY(-50%);
  opacity: 0;
  background: none;
  border: none;
  cursor: pointer;
  padding: 4px 6px;
  border-radius: 4px;
  color: var(--text-muted);
  font-size: 0.75rem;
  line-height: 1;
  transition: opacity 0.15s, color 0.15s, background 0.15s;
  flex-shrink: 0;
}
.session-item-wrapper:hover .session-delete-btn,
.session-item-wrapper:focus-within .session-delete-btn {
  opacity: 1;
}
.session-delete-btn:hover {
  color: var(--accent);
  background: var(--accent-soft);
}
```

**Step 6: Update `loadSessions()` to wrap each item**

In `loadSessions()`, replace this block:
```javascript
const div = document.createElement('button');
div.type = 'button';
div.setAttribute('role', 'listitem');
div.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
div.dataset.id = s.id;
div.textContent = s.title || s.first_message || 'Cuộc hội thoại mới';
div.onclick = () => switchSession(s.id, s.title || s.first_message || 'Cuộc hội thoại mới');
sessionsList.appendChild(div);
```

With:
```javascript
const wrapper = document.createElement('div');
wrapper.className = 'session-item-wrapper';
wrapper.setAttribute('role', 'listitem');

const sessionTitle = s.title || s.first_message || 'Cuộc hội thoại mới';

const btn = document.createElement('button');
btn.type = 'button';
btn.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
btn.dataset.id = s.id;
btn.textContent = sessionTitle;
btn.onclick = () => switchSession(s.id, sessionTitle);

const delBtn = document.createElement('button');
delBtn.type = 'button';
delBtn.className = 'session-delete-btn';
delBtn.title = 'Xoa cuoc hoi thoai';
delBtn.setAttribute('aria-label', 'Xoa cuoc hoi thoai');
delBtn.innerHTML = '&#x2715;';
delBtn.onclick = (e) => { e.stopPropagation(); deleteSession(s.id); };

wrapper.appendChild(btn);
wrapper.appendChild(delBtn);
sessionsList.appendChild(wrapper);
```

**Step 7: Update active-highlight selector in `switchSession`**

The existing line in `switchSession`:
```javascript
const active = document.querySelector(`.session-item[data-id="${id}"]`);
```
This still works because the `data-id` attribute is on the `btn` element which has class `session-item`. No change needed.

**Step 8: Add `deleteSession()` function**

Add near `newChat()`:
```javascript
async function deleteSession(id) {
  try {
    await fetch(`/sessions/${id}`, { method: 'DELETE' });
  } catch(e) {
    // Silently ignore -- UI will still update
  }
  if (id === currentSessionId) {
    newChat(); // switches to new chat and reloads session list
  } else {
    loadSessions();
  }
}
```

**Step 9: Verify in browser**

- Hover over a session: X button appears on the right
- Click X on a non-active session: session disappears, active session unchanged
- Click X on the active session: switches to "Cuoc hoi thoai moi", session removed from list
- Refresh page: deleted sessions do not reappear

**Step 10: Commit frontend**
```bash
git add app/static/index.html
git commit -m "feat: add session delete button with hover reveal"
```

---

## Summary

| # | Issue | Files touched | Commit message |
|---|---|---|---|
| 1 | Stream idle timeout | `index.html` | fix: replace upfront stream timeout with per-chunk idle timer |
| 2 | Error retry button | `index.html` | fix: show retry button on error |
| 3 | Citation emoji | `citation_check.py` | fix: remove emoji from citation disclaimer |
| 4 | Markdown tables | `index.html` | feat: add markdown table rendering |
| 5 | Session deletion | `db.py`, `main.py`, `index.html` | feat: add session delete |

All tasks are independent. Implement in order 1-5 for clean commit history.
