"""Batch RAG evaluation using Langfuse traces + Gemini as judge.

Usage:
    .venv/bin/python scripts/evaluate_rag.py [--limit 50] [--days 7]

Pulls recent traces from Langfuse, evaluates answer quality with Gemini,
and posts scores back to Langfuse.
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EVAL_PROMPT = """Ban la chuyen gia danh gia chat luong tra loi phap luat. Danh gia cau tra loi sau dua tren tai lieu tham khao.

<retrieved-context>
{context}
</retrieved-context>

<question>
{question}
</question>

<answer>
{answer}
</answer>

Danh gia 3 tieu chi, moi tieu chi cho diem 0-1 (0=kem, 0.5=trung binh, 1=tot):
1. groundedness: Cau tra loi co dua tren tai lieu khong? Co bua dat thong tin khong?
2. relevance: Cau tra loi co dung trong tam cau hoi khong?
3. completeness: Cau tra loi co day du, khong thieu thong tin quan trong khong?

Tra ve CHINH XAC JSON (khong markdown):
{{"groundedness": 0.0, "relevance": 0.0, "completeness": 0.0, "reasoning": "giai thich ngan"}}"""


def get_traces(lf: Langfuse, limit: int, days: int) -> list:
    """Fetch recent traces that haven't been evaluated yet."""
    traces = lf.fetch_traces(
        name="handle_message",
        limit=limit,
        order_by="timestamp",
        order="DESC",
    )
    cutoff = datetime.now() - timedelta(days=days)
    result = []
    for t in traces.data:
        if t.timestamp and t.timestamp.replace(tzinfo=None) < cutoff:
            continue
        # Skip traces that already have eval scores
        existing_scores = {s.name for s in (t.scores or [])}
        if "groundedness" in existing_scores:
            continue
        result.append(t)
    return result


def extract_trace_data(lf: Langfuse, trace_id: str) -> dict | None:
    """Extract question, context, and answer from a trace's observations."""
    trace = lf.fetch_trace(trace_id)
    if not trace:
        return None

    question = ""
    context = ""
    answer = ""

    # The root observation input contains the user_message
    if trace.input:
        question = trace.input.get("user_message", "") if isinstance(trace.input, dict) else str(trace.input)

    # The root observation output contains the answer
    if trace.output:
        answer = trace.output.get("answer", "") if isinstance(trace.output, dict) else str(trace.output)

    # Look through observations for retrieval context
    observations = lf.fetch_observations(trace_id=trace_id)
    for obs in observations.data:
        if obs.name == "search_legal_docs" and obs.output:
            # The output is the list of chunks
            chunks = obs.output if isinstance(obs.output, list) else []
            context_parts = []
            for c in chunks[:8]:
                if isinstance(c, dict):
                    text = c.get("text", c.get("content", ""))
                    source = c.get("metadata_str", "")
                    context_parts.append(f"[{source}] {text}")
            context = "\n---\n".join(context_parts)

    if not question or not answer:
        return None

    return {"question": question, "context": context, "answer": answer}


def evaluate_with_gemini(client: genai.Client, data: dict) -> dict | None:
    """Run Gemini as judge on a single trace."""
    prompt = EVAL_PROMPT.format(**data)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=500),
        )
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
    except Exception as e:
        log.error("Gemini eval failed: %s", e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate RAG quality via Langfuse + Gemini")
    parser.add_argument("--limit", type=int, default=50, help="Max traces to evaluate")
    parser.add_argument("--days", type=int, default=7, help="Look back N days")
    args = parser.parse_args()

    lf = Langfuse()
    gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    log.info("Fetching recent traces (limit=%d, days=%d)...", args.limit, args.days)
    traces = get_traces(lf, args.limit, args.days)
    log.info("Found %d unevaluated traces", len(traces))

    evaluated = 0
    for trace in traces:
        data = extract_trace_data(lf, trace.id)
        if not data:
            log.warning("Skipping trace %s: could not extract data", trace.id)
            continue

        scores = evaluate_with_gemini(gemini, data)
        if not scores:
            continue

        # Post scores back to Langfuse
        for metric in ("groundedness", "relevance", "completeness"):
            if metric in scores:
                lf.score(
                    trace_id=trace.id,
                    name=metric,
                    value=scores[metric],
                    data_type="NUMERIC",
                    comment=scores.get("reasoning", ""),
                )

        evaluated += 1
        log.info(
            "Trace %s: groundedness=%.1f relevance=%.1f completeness=%.1f",
            trace.id[:8],
            scores.get("groundedness", 0),
            scores.get("relevance", 0),
            scores.get("completeness", 0),
        )

        # Rate limit: ~2 requests/sec
        time.sleep(0.5)

    lf.flush()
    log.info("Done. Evaluated %d/%d traces.", evaluated, len(traces))


if __name__ == "__main__":
    main()
