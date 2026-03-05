# Orchestrator Prompt & Pipeline Tuning Design

**Problem:** The orchestrator (Gemini Flash) bypasses retrieval for legal questions, answering from its own knowledge. This produces responses without source citations. Observed on "Dieu 79 LDD 2024" and "So sanh ND 71 va ND 49" queries -- both returned 0 sources.

**Root cause:** The `direct_response` instruction is too broad ("Neu co the tra loi truc tiep"), and the model interprets "co the" as "if I know the answer" rather than "if this is non-legal chitchat."

**Approach:** Prompt tuning + code guardrail (Approach B from brainstorming).

---

## Change 1: Orchestrator prompt -- hard forcing rule + few-shot examples

Rewrite `ORCHESTRATOR_TOOLS_SECTION` and `ORCHESTRATOR_INSTRUCTIONS` in `app/chat.py`.

**Tools section:** Replace the "KHONG goi tool khi" block with an explicit forcing rule that makes retrieval the default for any legal question. Only greetings, thanks, and clarification requests use `direct_response`.

**Instructions:** Add 5 few-shot examples covering: greeting, specific dieu lookup, general legal question, comparison/amendment question, and follow-up clarification. These anchor the model's behavior for the most common query types.

## Change 2: Code guardrail -- legal keyword override

Add a regex-based legal keyword detector in `handle_message`. After parsing the orchestrator response, if `direct_response` is set but the user message contains legal keywords (dieu, luat, nghi dinh, quyen, thu hoi, boi thuong, etc.), override to force a `search_legal_docs` call using `extract_filters` on the original message.

This is a safety net -- if prompts work correctly, it never triggers. Logged as a warning so we can track drift.

## Change 3: Generator prompt improvements

**With context:** Add structured output guidance -- cite specific dieu/khoan/van ban, explain relationships between documents, note when information is incomplete, use bullet points for lists.

**Without context (0 chunks):** Replace the current "Tra loi dua tren noi dung da thao luan" (which encourages hallucination) with an honest fallback that tells the user no documents were found and suggests how to refine their question.

## Verification

- Smoke test: "Dieu 79 LDD 2024" must return sources (currently returns 0)
- Smoke test: "So sanh ND 71 va ND 49" must return sources (currently returns 0)
- Smoke test: "Xin chao" must NOT trigger retrieval (currently works)
- Smoke test: "Con dieu 80 thi sao?" follow-up must return sources
- Unit tests: update test_orchestrator.py for new prompt constants
