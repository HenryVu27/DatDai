# Query Rewriting Design

## Problem

1. `query_rewriter.py` only extracts regex filters -- no actual query rewriting
2. No pronoun resolution from conversation history ("no" / "cai do" unresolved)
3. No legal term expansion ("so do" never becomes "giay chung nhan quyen su dung dat")
4. No multi-query generation for better recall
5. Memory/CLAUDE.md claims HyDE and pronoun resolution exist -- they don't

## Solution

Single LLM call (utility model, flash-lite) that runs before the orchestrator, doing:

1. **Decontextualization** -- resolve pronouns/references from conversation history
2. **Legal term expansion** -- expand colloquial terms to formal legal terms
3. **Multi-query generation** -- 2 search variant phrasings for multi-query retrieval
4. **Filter extraction** -- LLM-based replacement for regex (with regex as fallback)

## Architecture

```
User message + summary + last 5 turns
    |
    v
Query Rewriter (utility model, ~0.3-0.5s)
    |
    v
{standalone_query, search_queries[2], filters{doc_ids, dieu}}
    |
    v
Orchestrator (receives standalone_query instead of raw message)
    |
    v
Tools execute (search uses search_queries for multi-query + RRF merge)
    |
    v
Generator
```

## Input/Output Contract

### Input
- `user_message`: raw current user message
- `summary`: conversation summary (from existing summary system, may be None)
- `recent_messages`: last 5 turns of conversation history

### Output (JSON)
```json
{
  "standalone_query": "decontextualized, expanded query",
  "search_queries": ["variant phrasing 1", "variant phrasing 2"],
  "filters": {
    "doc_ids": ["nd102"] or null,
    "dieu": "Dieu 15" or null
  }
}
```

## Prompt Design

The rewriter prompt instructs the LLM to:
- Replace pronouns (no, do, cai nay, van ban nay) with actual entities from history
- Expand colloquial Vietnamese legal terms to formal terms (keeping original in parentheses)
- Generate 2 alternative search phrasings
- Extract doc_id and dieu filters from context (known doc_id values provided in prompt)
- Return ONLY JSON, no explanation

## Integration Points

### chat.py changes
- `handle_message` and `handle_message_stream`: call query rewriter before orchestrator
- Pass `standalone_query` to orchestrator instead of raw `user_message`
- Raw `user_message` still saved to DB and shown to user

### retriever.py changes
- `search_legal_docs` accepts optional `search_queries` parameter
- When provided, searches all queries in parallel, merges with deduplication
- Reranker sees merged candidate set

### query_rewriter.py changes
- Rename current `extract_filters` to `extract_filters_regex` (kept as fallback)
- New async `rewrite_query()` function that calls LLM
- Output validation: check doc_ids against known list, validate dieu format
- On LLM failure: fall back to regex extraction + raw query passthrough

## Robustness

- LLM returns bad JSON -> fall back to regex + raw query
- LLM hallucinates doc_ids -> validate against known list, strip invalid ones
- LLM hallucinates dieu -> validate format "Dieu \d+", strip if invalid
- First-turn simple query -> rewriter still runs (cheap enough), but decontextualization is a no-op
- Timeout -> 5s timeout on utility model, fall back to raw query

## Token Budget

- Input: ~200-400 tokens (system prompt + summary + 5 turns + current message)
- Output: ~100-150 tokens (JSON)
- Model: utility (flash-lite), cheapest available
- Added latency: ~0.3-0.5s per turn

## What This Replaces

- `extract_filters()` regex-only function (kept as fallback, no longer primary)
- Orchestrator's implicit query rewriting (orchestrator now receives clean input)

## What This Does NOT Do

- HyDE (not implementing -- industry evidence suggests query rewriting is higher ROI)
- Retrieval feedback loop (future work)
- Custom embedding fine-tuning (requires training infrastructure)
