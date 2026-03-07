# UX Fixes Design - Issues #9-13

**Date:** 2026-03-06
**Scope:** Frontend-only changes in `app/static/index.html`
**Risk:** Zero backend changes, zero Cloud Run impact

## Fix #9: Abort/Cancel Button

Transform send button into stop button during streaming.

- Move `AbortController` to module scope (`let currentController = null`)
- When `isLoading=true`: swap SVG to stop icon (square), update title/aria-label
- On click while loading: call `currentController.abort()`
- Existing `AbortError` catch block handles partial text display
- After abort: reset button back to send icon

**Files:** `app/static/index.html` (JS: handleSubmit, HTML: send-btn)

## Fix #10: Regenerate Button

Add regenerate button in feedback row alongside thumbs up/down.

- Store `lastUserQuestion` variable, updated on each submit
- In `finalizeBubble`: add regenerate button (circular arrow icon) next to feedback buttons
- On click: remove last assistant message, re-submit `lastUserQuestion`
- Only show on most recent assistant message (remove from previous on new response)
- Style: same `.feedback-btn` class, with aria-label

**Files:** `app/static/index.html` (JS: finalizeBubble, sendFeedback area, CSS: feedback styles)

## Fix #11: Accessibility (Essential WCAG 2.1 A)

- `aria-label` on all icon-only buttons: send-btn, toggle-sidebar, new-chat-btn, feedback buttons
- Session items: `div` -> `button` elements
- `sessions-list`: add `role="list"`
- Add `aria-live="polite"` region for streaming status announcements
- Feedback buttons: add text labels alongside emoji for screen readers

**Files:** `app/static/index.html` (HTML: sidebar, topbar, feedback; JS: loadSessions, addMessage)

## Fix #12: Mobile Usability

- Add `env(safe-area-inset-*)` padding on body and input-area for notched phones
- Increase feedback button min tap target to 44x44px in mobile breakpoint
- Add `clamp()` responsive font scaling for body text
- Ensure suggestion buttons have adequate padding (min 44px height) on mobile

**Files:** `app/static/index.html` (CSS: mobile media query, root styles)

## Fix #13: Dynamic Suggestions

Static pool of ~16 questions, randomly pick 4 on each new chat.

- Define `SUGGESTION_POOL` array with 16 diverse questions across legal topics
- `getRandomSuggestions()` function: Fisher-Yates shuffle, pick first 4
- Called in initial render and `newChat()`
- Replace hardcoded suggestion buttons with dynamic generation

**Files:** `app/static/index.html` (JS: new functions + newChat, HTML: welcome section)
