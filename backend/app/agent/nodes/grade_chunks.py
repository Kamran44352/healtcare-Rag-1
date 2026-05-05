"""grade_chunks — assess whether retrieved chunks actually cover the question."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI
from langchain_core.runnables import RunnableConfig

from app.agent.prompts import GRADE_CHUNKS_PROMPT
from app.agent.state import AgentState
from app.config import settings

log = logging.getLogger("clintel.agent.grade_chunks")
_openai = AsyncOpenAI(api_key=settings.openai_api_key)


def _build_chunk_summary(chunks: list[Any]) -> str:
    """Compact chunk summaries for the grader."""
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks[:10], 1):
        title = (chunk.doc_metadata or {}).get("guideline_title") or chunk.filename
        blocks.append(
            f"[SOURCE {idx}] {title}\n"
            f"Section: {chunk.section_path}\n"
            f"Snippet: {chunk.snippet[:300]}"
        )
    return "\n\n".join(blocks)


async def grade_chunks(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Grade how well retrieved chunks cover the user's question facets."""
    q = config.get("configurable", {}).get("event_queue")
    if q:
        await q.put(("step_started", {"node": "grade_chunks"}))
        
    question = state["question"]
    chunks = state.get("chunks", [])
    retries = state.get("retries", 0)
    started = time.perf_counter()

    # If no chunks at all, skip grading
    if not chunks:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        trace = {
            "node": "grade_chunks",
            "status": "done",
            "ms": elapsed_ms,
            "data": {"coverage_score": 0.0, "reason": "No chunks retrieved"},
        }
        return {
            "coverage_score": 0.0,
            "coverage_reasoning": "No chunks were retrieved.",
            "uncovered_facets": [],
            "retries": retries,
            "trace_events": state.get("trace_events", []) + [trace],
        }

    chunk_summary = _build_chunk_summary(chunks)

    try:
        response = await _openai.chat.completions.create(
            model=settings.background_model,
            temperature=0.0,
            max_completion_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": GRADE_CHUNKS_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User question: {question}\n\n"
                        f"Retrieved chunks:\n{chunk_summary}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        coverage_score = float(parsed.get("coverage_score", 0.5))
        coverage_reasoning = str(parsed.get("reasoning", ""))
        uncovered_facets = list(parsed.get("uncovered_facets", []))
    except Exception as exc:
        log.warning("grade_chunks failed, assuming adequate coverage: %s", exc)
        coverage_score = 0.7
        coverage_reasoning = "Grading failed, proceeding with available chunks."
        uncovered_facets = []

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    trace = {
        "node": "grade_chunks",
        "status": "done",
        "ms": elapsed_ms,
        "data": {
            "coverage_score": round(coverage_score, 2),
            "reason": coverage_reasoning[:120],
            "uncovered_facets": uncovered_facets[:3],
            "retry_number": retries,
        },
    }

    log.info(
        "Coverage graded: %.2f in %dms (retries=%d). Reasoning: %s",
        coverage_score, elapsed_ms, retries, coverage_reasoning[:100],
    )

    return {
        "coverage_score": coverage_score,
        "coverage_reasoning": coverage_reasoning,
        "uncovered_facets": uncovered_facets,
        "retries": retries,
        "trace_events": state.get("trace_events", []) + [trace],
    }
