"""extract_filters — pull structured metadata filters from the question and rewrite for retrieval."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI
from langchain_core.runnables import RunnableConfig

from app.agent.prompts import EXTRACT_FILTERS_PROMPT
from app.agent.state import AgentState
from app.config import settings

log = logging.getLogger("clintel.agent.extract_filters")
_openai = AsyncOpenAI(api_key=settings.openai_api_key)


def _build_history_snippet(history: list[dict[str, Any]]) -> str:
    """Compact last 6 messages for the LLM."""
    if not history:
        return "No prior conversation."
    lines: list[str] = []
    for msg in history[-6:]:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", ""))[:500].strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def _merge_filters(user: dict[str, Any] | None, extracted: dict[str, Any]) -> dict[str, Any]:
    """User-explicit filters take priority; extracted fills the gaps."""
    merged = dict(extracted)
    if user:
        for key, value in user.items():
            if value is not None:
                merged[key] = value
    # Strip null and False values so RetrievalFilters doesn't get confused
    # False usually implies "don't care", but passing False to Qdrant enforces an exact match
    return {k: v for k, v in merged.items() if v is not None and v is not False and v != []}


def _extract_last_diagnosis(history: list[dict[str, Any]]) -> str | None:
    """Pull the first ~200 chars of the last assistant answer as diagnosis context."""
    for msg in reversed(history):
        if str(msg.get("role", "")).lower() == "assistant":
            content = str(msg.get("content", "")).strip()
            if content and len(content) > 20:
                # Take the first sentence or 200 chars — usually contains the diagnosis
                first_chunk = content[:250].split("\n")[0]
                return first_chunk
    return None


async def extract_filters(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Extract structured filters and rewrite the query for standalone retrieval."""
    q = config.get("configurable", {}).get("event_queue")
    if q:
        await q.put(("step_started", {"node": "extract_filters"}))
        
    question = state["question"]
    history = state.get("history", [])
    user_filters = state.get("user_filters")
    started = time.perf_counter()

    history_snippet = _build_history_snippet(history)

    try:
        response = await _openai.chat.completions.create(
            model=settings.background_model,
            temperature=0.0,
            max_completion_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACT_FILTERS_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{history_snippet}\n\n"
                        f"Current question: {question}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        extracted = parsed.get("filters") or {}
        retrieval_query = str(parsed.get("retrieval_query", question)).strip() or question
    except Exception as exc:
        log.warning("extract_filters failed, using original question: %s", exc)
        extracted = {}
        retrieval_query = question

    # ── Deterministic fallback: if LLM didn't enrich a follow-up query ─────
    # If there's conversation history and the retrieval_query is still very
    # short or identical to the raw question, prepend last diagnosis context
    if history and (
        retrieval_query == question
        or len(retrieval_query) < 40
    ):
        last_diag = _extract_last_diagnosis(history)
        if last_diag and last_diag.lower() not in retrieval_query.lower():
            retrieval_query = f"{last_diag} — {retrieval_query}"
            log.info("Fallback: prepended last diagnosis to retrieval query")

    merged = _merge_filters(user_filters, extracted)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Build a human-readable summary of extracted filters
    filter_summary = []
    if merged.get("conditions_codes"):
        filter_summary.append(f"conditions: {', '.join(merged['conditions_codes'])}")
    if merged.get("specialties"):
        filter_summary.append(f"specialties: {', '.join(merged['specialties'])}")
    if merged.get("section_type"):
        filter_summary.append(f"section: {', '.join(merged['section_type'])}")
    if merged.get("care_setting"):
        filter_summary.append(f"setting: {', '.join(merged['care_setting'])}")
    if merged.get("has_red_flags"):
        filter_summary.append("red flags")
    if merged.get("has_dosing_tables"):
        filter_summary.append("dosing tables")

    trace = {
        "node": "extract_filters",
        "status": "done",
        "ms": elapsed_ms,
        "data": {
            "retrieval_query": retrieval_query,
            "filter_count": len(merged),
            "filters": "; ".join(filter_summary) if filter_summary else "none",
        },
    }

    log.info(
        "Filters extracted in %dms: query='%s', filters=%s",
        elapsed_ms, retrieval_query[:80], merged,
    )

    return {
        "extracted_filters": extracted,
        "merged_filters": merged,
        "retrieval_query": retrieval_query,
        "trace_events": state.get("trace_events", []) + [trace],
    }
