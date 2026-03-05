# Orchestrator Prompt & Pipeline Tuning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the orchestrator so it always retrieves for legal questions, improve query crafting with few-shot examples, add a code-level guardrail, and improve the generator's output quality.

**Architecture:** Prompt-only changes to orchestrator and generator, plus a regex-based legal keyword guardrail in `handle_message` that overrides `direct_response` when the user asks about law.

**Tech Stack:** Python 3.12, Gemini 3.x, existing `app/chat.py` prompts.

---

### Task 1: Rewrite ORCHESTRATOR_TOOLS_SECTION with hard forcing rule

**Files:**
- Modify: `app/chat.py:54-78`

**Step 1: Replace ORCHESTRATOR_TOOLS_SECTION**

Replace the entire `ORCHESTRATOR_TOOLS_SECTION` constant (lines 54-78) with:

```python
ORCHESTRATOR_TOOLS_SECTION = """
<tools>
Ban co the goi cac tool sau de tra cuu thong tin. Tra ve JSON hop le.

1. search_legal_docs: Tim kiem van ban phap luat bang ngon ngu tu nhien.
   Params: {"query": "cau truy van", "filters": {"doc_ids": ["nd102"], "dieu": "Dieu 15"}}
   - query: viet ro rang, day du ngu canh, khong dung dai tu
   - filters: tuy chon, chi dinh khi biet chinh xac van ban/dieu

2. lookup_amendment: Tra cuu cac sua doi giua cac nghi dinh.
   Params: {"target_doc": "nd102", "source_doc": "nd49", "dieu": "Dieu 15"}
   - target_doc: bat buoc - van ban bi sua doi
   - source_doc: tuy chon - van ban sua doi (neu biet)
   - dieu: tuy chon - so dieu cu the

3. lookup_specific_dieu: Lay toan bo noi dung mot dieu cu the.
   Params: {"doc_id": "ldd2024", "dieu": "Dieu 79"}
   - Dung khi biet chinh xac dieu va van ban can tra cuu

Chon tool phu hop:
- Biet chinh xac dieu + van ban -> lookup_specific_dieu
- Hoi ve sua doi giua cac ND -> lookup_amendment (+ search_legal_docs neu can them context)
- Cau hoi chung, khong biet dieu cu the -> search_legal_docs
- Cau hoi phuc tap, nhieu van ban -> nhieu tool cung luc
</tools>

<retrieval-policy>
QUY TAC BAT BUOC:
- BAT KY cau hoi lien quan den phap luat, dieu khoan, thu tuc, quyen, nghia vu, dat dai -> PHAI goi it nhat mot tool
- KHONG BAO GIO tu tra loi cau hoi phap luat tu kien thuc cua ban - LUON tra cuu truoc
- direct_response CHI dung cho: chao hoi, cam on, tam biet, noi chuyen xa giao, hoac hoi lai de lam ro cau hoi mo ho
- Khi nguoi dung hoi tiep ve noi dung vua tra loi (lam ro, giai thich them) -> van PHAI goi tool de dam bao chinh xac
- Khi khong chac co can tra cuu khong -> GOI TOOL (an toan hon la tu tra loi sai)
</retrieval-policy>"""
```

**Step 2: Verify import**

Run: `.venv/bin/python -c "from app.chat import ORCHESTRATOR_TOOLS_SECTION; print('<retrieval-policy>' in ORCHESTRATOR_TOOLS_SECTION)"`
Expected: `True`

**Step 3: Commit**

```bash
git add app/chat.py
git commit -m "orchestrator: add hard retrieval-policy forcing rule, tool selection guidance"
```

---

### Task 2: Rewrite ORCHESTRATOR_INSTRUCTIONS with few-shot examples

**Files:**
- Modify: `app/chat.py:80-101`

**Step 1: Replace ORCHESTRATOR_INSTRUCTIONS**

Replace the entire `ORCHESTRATOR_INSTRUCTIONS` constant (lines 80-101) with:

```python
ORCHESTRATOR_INSTRUCTIONS = """
<instructions>
Phan tich tin nhan cua nguoi dung va quyet dinh hanh dong.
Tra ve CHINH XAC mot JSON object (khong markdown, khong giai thich) voi format:
{
  "reasoning": "suy nghi ngan gon",
  "actions": [{"tool": "ten_tool", ...params}],
  "complexity": "simple" hoac "complex",
  "summary_update": "tom tat cap nhat" hoac null,
  "direct_response": "cau tra loi truc tiep" hoac null
}

Quy tac:
- actions va direct_response khong dong thoi co gia tri. Chon mot trong hai.
- Cau hoi phap luat -> actions (KHONG DUOC dung direct_response)
- Chao hoi, cam on, xa giao, hoi lai -> direct_response (KHONG CAN actions)
- complexity: "complex" khi so sanh nhieu van ban, phan tich tinh huong phuc tap, cau hoi lien quan den sua doi. "simple" cho con lai
- summary_update: chi khi cuoc hoi thoai da co 4+ luot trao doi. Tom tat PHAI bao gom summary truoc do va bo sung noi dung moi
- Khi viet query cho search_legal_docs: viet cau truy van day du ngu canh, khong dung dai tu (no, do, nay), khong viet tat
</instructions>

<examples>
INPUT: "Xin chao"
OUTPUT: {"reasoning": "Nguoi dung chao hoi, khong can tra cuu", "actions": [], "complexity": "simple", "summary_update": null, "direct_response": "Xin chao! Toi la chuyen gia tu van Luat Dat Dai. Ban can ho tro gi?"}

INPUT: "Dieu 79 Luat Dat Dai 2024 quy dinh gi?"
OUTPUT: {"reasoning": "Hoi noi dung cu the Dieu 79 LDD 2024, dung lookup_specific_dieu", "actions": [{"tool": "lookup_specific_dieu", "doc_id": "ldd2024", "dieu": "Dieu 79"}], "complexity": "simple", "summary_update": null, "direct_response": null}

INPUT: "Quyen cua nguoi su dung dat la gi?"
OUTPUT: {"reasoning": "Cau hoi chung ve quyen su dung dat, can search", "actions": [{"tool": "search_legal_docs", "query": "quyen cua nguoi su dung dat theo Luat Dat Dai 2024", "filters": {"doc_ids": ["ldd2024"]}}], "complexity": "simple", "summary_update": null, "direct_response": null}

INPUT: "ND 49 sua doi gi cua ND 102?"
OUTPUT: {"reasoning": "Hoi ve sua doi giua 2 ND, can lookup_amendment va search", "actions": [{"tool": "lookup_amendment", "target_doc": "nd102", "source_doc": "nd49"}, {"tool": "search_legal_docs", "query": "Nghi dinh 49/2026 sua doi bo sung Nghi dinh 102/2024 chi tiet thi hanh Luat Dat Dai", "filters": {"doc_ids": ["nd49"]}}], "complexity": "complex", "summary_update": null, "direct_response": null}

INPUT: "Cam on ban"
OUTPUT: {"reasoning": "Cam on, xa giao", "actions": [], "complexity": "simple", "summary_update": null, "direct_response": "Khong co gi! Neu ban co them cau hoi ve Luat Dat Dai, hay hoi bat cu luc nao."}
</examples>"""
```

**Step 2: Verify import**

Run: `.venv/bin/python -c "from app.chat import ORCHESTRATOR_INSTRUCTIONS; print('<examples>' in ORCHESTRATOR_INSTRUCTIONS)"`
Expected: `True`

**Step 3: Commit**

```bash
git add app/chat.py
git commit -m "orchestrator: add few-shot examples and tighten decision rules"
```

---

### Task 3: Add legal keyword guardrail in handle_message

**Files:**
- Modify: `app/chat.py` (imports at top, and in `handle_message`)

**Step 1: Add `re` import and LEGAL_KEYWORDS constant**

Add `import re` to the imports at the top of `chat.py` (line 1-22 area). Then add the `LEGAL_KEYWORDS` regex after the prompt constants (after `ORCHESTRATOR_INSTRUCTIONS`, before `_assemble_conversation_context`):

```python
# -- Legal keyword guardrail --

LEGAL_KEYWORDS = re.compile(
    r"(?:điều|dieu|đ\.\s*\d|luật|luat|nghị định|nghi dinh|nđ|nd\s*\d+"
    r"|quyền|quyen|thu hồi|thu hoi|bồi thường|boi thuong"
    r"|giấy chứng nhận|giay chung nhan|sử dụng đất|su dung dat"
    r"|tiền thuê|tien thue|chuyển nhượng|chuyen nhuong"
    r"|quy hoạch|quy hoach|giải phóng mặt bằng|giai phong mat bang"
    r"|đất đai|dat dai|thửa đất|thua dat|cấp đất|cap dat|giao đất|giao dat"
    r"|nghị quyết|nghi quyet|khiếu nại|khieu nai|tranh chấp|tranh chap"
    r"|đăng ký|dang ky|sổ đỏ|so do|sổ hồng|so hong)",
    re.IGNORECASE,
)


def _should_force_retrieval(user_message: str, decision: dict) -> bool:
    """Check if a direct_response should be overridden with retrieval."""
    if not decision.get("direct_response"):
        return False
    if decision.get("actions"):
        return False
    return bool(LEGAL_KEYWORDS.search(user_message))
```

**Step 2: Add guardrail logic in handle_message**

In `handle_message`, after the line `decision = _parse_orchestrator_response(orch_raw)` and its log line, add:

```python
    # Guardrail: force retrieval if orchestrator tried to answer a legal question directly
    if _should_force_retrieval(user_message, decision):
        filters = extract_filters(user_message)
        logger.warning(
            "[%s] Guardrail triggered: forcing retrieval for legal question answered directly",
            session_id[:8],
        )
        decision = {
            "reasoning": "Guardrail: legal question requires retrieval",
            "actions": [{"tool": "search_legal_docs", "query": user_message, "filters": {
                "doc_ids": filters.get("doc_ids"),
                "dieu": filters.get("dieu"),
            }}],
            "complexity": decision.get("complexity", "simple"),
            "summary_update": decision.get("summary_update"),
            "direct_response": None,
        }
```

This goes right after the orchestrator log line and before the `if decision["summary_update"]` line.

**Step 3: Verify import**

Run: `.venv/bin/python -c "from app.chat import _should_force_retrieval, LEGAL_KEYWORDS; print(bool(LEGAL_KEYWORDS.search('Dieu 79 luat dat dai')))"`
Expected: `True`

Run: `.venv/bin/python -c "from app.chat import _should_force_retrieval; print(_should_force_retrieval('Xin chao', {'direct_response': 'Hi', 'actions': []}))"`
Expected: `False`

**Step 4: Commit**

```bash
git add app/chat.py
git commit -m "chat: add legal keyword guardrail to force retrieval when orchestrator bypasses"
```

---

### Task 4: Improve generator prompts

**Files:**
- Modify: `app/chat.py` (`_build_generator_prompt` function)

**Step 1: Replace the prompt strings in _build_generator_prompt**

Replace the `if context:` and `else:` branches in `_build_generator_prompt`:

```python
    if context:
        prompt = f"""<retrieved-context>
{context}
</retrieved-context>

<user-question>
{user_message}
</user-question>

Hay tra loi cau hoi dua tren tai lieu tham khao o tren.
Yeu cau:
- Trich dan cu the so dieu, khoan, diem va ten van ban (vd: "Theo Dieu 79 Luat Dat Dai 2024...")
- Neu co nhieu van ban lien quan, giai thich moi quan he (luat goc -> nghi dinh huong dan -> nghi dinh sua doi)
- Neu thong tin trong tai lieu khong du de tra loi day du, noi ro phan nao chua tim thay
- Tra loi co cau truc, dung bullet points khi liet ke nhieu muc
- Khong tu them thong tin ngoai tai lieu duoc cung cap"""
    else:
        prompt = f"""<user-question>
{user_message}
</user-question>

Toi khong tim thay tai lieu lien quan trong co so du lieu cho cau hoi nay.
Hay noi ro voi nguoi dung rang ban khong tim thay thong tin cu the trong cac van ban hien co.
Goi y ho cach hoi cu the hon, vi du: chi dinh so dieu, ten van ban (Luat Dat Dai, ND 102...), hoac mo ta tinh huong cu the."""
```

**Step 2: Verify import**

Run: `.venv/bin/python -c "from app.chat import _build_generator_prompt; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/chat.py
git commit -m "generator: improve output guidance and honest no-context fallback"
```

---

### Task 5: Update tests for new prompt constants and guardrail

**Files:**
- Modify: `tests/test_orchestrator.py`

**Step 1: Update existing tests and add guardrail tests**

Replace the entire file:

```python
"""Tests for the orchestrator pipeline logic (no LLM calls)."""
import json
import pytest
from app.chat import (
    _parse_orchestrator_response,
    _should_force_retrieval,
    SYSTEM_BASE,
    ORCHESTRATOR_TOOLS_SECTION,
    ORCHESTRATOR_INSTRUCTIONS,
    LEGAL_KEYWORDS,
)


class TestParseOrchestratorResponse:
    def test_valid_json(self):
        raw = json.dumps({
            "reasoning": "User asks about Dieu 15",
            "actions": [{"tool": "lookup_specific_dieu", "doc_id": "ldd2024", "dieu": "Dieu 15"}],
            "complexity": "simple",
            "summary_update": None,
            "direct_response": None,
        })
        result = _parse_orchestrator_response(raw)
        assert result["complexity"] == "simple"
        assert len(result["actions"]) == 1
        assert result["actions"][0]["tool"] == "lookup_specific_dieu"

    def test_json_in_markdown_fences(self):
        raw = '```json\n{"reasoning": "test", "actions": [], "complexity": "simple", "direct_response": "Hi"}\n```'
        result = _parse_orchestrator_response(raw)
        assert result["direct_response"] == "Hi"
        assert result["actions"] == []

    def test_invalid_json_becomes_direct_response(self):
        raw = "Xin chao, toi co the giup gi cho ban?"
        result = _parse_orchestrator_response(raw)
        assert result["direct_response"] == raw
        assert result["actions"] == []

    def test_missing_keys_get_defaults(self):
        raw = json.dumps({"reasoning": "test"})
        result = _parse_orchestrator_response(raw)
        assert result["actions"] == []
        assert result["complexity"] == "simple"
        assert result["summary_update"] is None
        assert result["direct_response"] is None

    def test_multiple_actions(self):
        raw = json.dumps({
            "reasoning": "Need search and amendment check",
            "actions": [
                {"tool": "search_legal_docs", "query": "gia dat", "filters": {"doc_ids": ["nd71"]}},
                {"tool": "lookup_amendment", "target_doc": "nd71"},
            ],
            "complexity": "complex",
            "summary_update": None,
            "direct_response": None,
        })
        result = _parse_orchestrator_response(raw)
        assert len(result["actions"]) == 2
        assert result["complexity"] == "complex"


class TestLegalKeywordGuardrail:
    def test_legal_keyword_triggers_on_dieu(self):
        assert LEGAL_KEYWORDS.search("Dieu 79 Luat Dat Dai")

    def test_legal_keyword_triggers_on_nghi_dinh(self):
        assert LEGAL_KEYWORDS.search("ND 102 quy dinh gi")

    def test_legal_keyword_triggers_on_quyen(self):
        assert LEGAL_KEYWORDS.search("Quyen su dung dat la gi")

    def test_legal_keyword_triggers_on_thu_hoi(self):
        assert LEGAL_KEYWORDS.search("Thu hoi dat de lam duong")

    def test_legal_keyword_triggers_on_so_do(self):
        assert LEGAL_KEYWORDS.search("Lam so do can gi")

    def test_no_trigger_on_greeting(self):
        assert not LEGAL_KEYWORDS.search("Xin chao")

    def test_no_trigger_on_thanks(self):
        assert not LEGAL_KEYWORDS.search("Cam on ban nhieu")

    def test_should_force_retrieval_legal_direct(self):
        decision = {"direct_response": "Day la cau tra loi", "actions": []}
        assert _should_force_retrieval("Dieu 79 luat dat dai quy dinh gi", decision)

    def test_should_not_force_retrieval_greeting(self):
        decision = {"direct_response": "Xin chao!", "actions": []}
        assert not _should_force_retrieval("Xin chao", decision)

    def test_should_not_force_when_actions_present(self):
        decision = {"direct_response": None, "actions": [{"tool": "search"}]}
        assert not _should_force_retrieval("Dieu 79", decision)

    def test_should_not_force_when_no_direct_response(self):
        decision = {"direct_response": None, "actions": []}
        assert not _should_force_retrieval("Dieu 79", decision)


class TestSystemPromptStructure:
    def test_base_has_xml_tags(self):
        assert "<identity>" in SYSTEM_BASE
        assert "</identity>" in SYSTEM_BASE
        assert "<boundaries>" in SYSTEM_BASE
        assert "<legal-hierarchy>" in SYSTEM_BASE

    def test_tools_section_has_all_tools(self):
        assert "search_legal_docs" in ORCHESTRATOR_TOOLS_SECTION
        assert "lookup_amendment" in ORCHESTRATOR_TOOLS_SECTION
        assert "lookup_specific_dieu" in ORCHESTRATOR_TOOLS_SECTION

    def test_tools_section_has_retrieval_policy(self):
        assert "<retrieval-policy>" in ORCHESTRATOR_TOOLS_SECTION
        assert "BAT KY cau hoi lien quan den phap luat" in ORCHESTRATOR_TOOLS_SECTION

    def test_instructions_has_examples(self):
        assert "<examples>" in ORCHESTRATOR_INSTRUCTIONS
        assert "lookup_specific_dieu" in ORCHESTRATOR_INSTRUCTIONS

    def test_static_prefix_ordering(self):
        id_pos = SYSTEM_BASE.index("<identity>")
        bound_pos = SYSTEM_BASE.index("<boundaries>")
        hier_pos = SYSTEM_BASE.index("<legal-hierarchy>")
        assert id_pos < bound_pos < hier_pos
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_orchestrator.py -v`
Expected: All tests PASS (should be ~20 tests)

**Step 3: Commit**

```bash
git add tests/test_orchestrator.py
git commit -m "tests: add legal keyword guardrail tests, update prompt structure tests"
```

---

### Task 6: Smoke test with live LLM

**Step 1: Start the server**

```bash
cd /Users/vuducdung/personal/DatDai && .venv/bin/uvicorn app.main:app --reload --port 8000
```

**Step 2: Test scenarios**

Test 1 -- Greeting (should use direct_response, no retrieval):
```bash
curl -s http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "Xin chao"}' | python3 -m json.tool
```
Expected: response with empty sources.

Test 2 -- Specific dieu (MUST trigger retrieval now):
```bash
curl -s http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "Dieu 79 Luat Dat Dai 2024 quy dinh gi?", "session_id": "tune-test-1"}' | python3 -m json.tool
```
Expected: response with sources citing Dieu 79 LDD 2024. Previously returned 0 sources.

Test 3 -- Comparison (MUST trigger retrieval now):
```bash
curl -s http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "So sanh quy dinh ve gia dat giua ND 71 va ND 49"}' | python3 -m json.tool
```
Expected: response with sources from ND 71 and/or ND 49. Previously returned 0 sources.

Test 4 -- Follow-up (should trigger retrieval):
```bash
curl -s http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "Con dieu 80 thi sao?", "session_id": "tune-test-1"}' | python3 -m json.tool
```
Expected: response about Dieu 80 with sources.

**Step 3: Check server logs**

Look for:
- No `Guardrail triggered` warnings on greetings
- `Guardrail triggered` OR orchestrator correctly calling tools for legal questions
- `tools returned N chunks` with N > 0 for legal questions

**Step 4: Commit if fixes needed**

```bash
git add -A
git commit -m "fix: smoke test adjustments for orchestrator tuning"
```

---

## Summary of Changes

| Task | File | Action |
|------|------|--------|
| 1 | app/chat.py | Rewrite ORCHESTRATOR_TOOLS_SECTION with retrieval-policy |
| 2 | app/chat.py | Rewrite ORCHESTRATOR_INSTRUCTIONS with few-shot examples |
| 3 | app/chat.py | Add LEGAL_KEYWORDS guardrail + _should_force_retrieval |
| 4 | app/chat.py | Improve generator prompts (with-context + no-context) |
| 5 | tests/test_orchestrator.py | Add guardrail tests, update prompt structure tests |
| 6 | Manual | Smoke test: verify Dieu 79 + comparison now return sources |
