"""extract_filters — rewrite the query for standalone retrieval (no auto filter extraction).

Mirrors Phase 2 _build_retrieval_query(): the LLM's only job is to inject clinical
context from conversation history into the search query. No metadata filters are
guessed from the question text — any hard Qdrant field filters come exclusively from
user-supplied values via the API, preventing silent chunk exclusions.
"""

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
    """Compact last 6 messages for the LLM.

    Assistant messages get a higher char limit because the diagnosis is often
    buried inside a longer prior answer — truncating at 500 chars cuts it off.
    """
    if not history:
        return "No prior conversation."
    lines: list[str] = []
    for msg in history[-6:]:
        role = str(msg.get("role", "user")).upper()
        limit = 1000 if role == "ASSISTANT" else 400
        content = str(msg.get("content", ""))[:limit].strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def _user_filters_to_merged(user: dict[str, Any] | None) -> dict[str, Any]:
    """Pass user-supplied filters through, stripping nulls/falses."""
    if not user:
        return {}
    return {k: v for k, v in user.items() if v is not None and v is not False and v != []}


async def extract_filters(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Rewrite the query for standalone retrieval. Never applies auto-extracted filters."""
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
            max_completion_tokens=200,
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
        retrieval_queries = parsed.get("retrieval_queries", [])
        if not retrieval_queries:
            retrieval_queries = [question]
        retrieval_query = retrieval_queries[0]
    except Exception as exc:
        log.warning("Query rewriting failed, using original question: %s", exc)
        retrieval_query = question
        retrieval_queries = [question]

    # Only user-supplied filters reach Qdrant — no LLM guesses
    merged = _user_filters_to_merged(user_filters)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    trace = {
        "node": "extract_filters",
        "status": "done",
        "ms": elapsed_ms,
        "data": {
            "retrieval_query": retrieval_query,
            "filter_count": len(merged),
            "filters": "none" if not merged else "; ".join(f"{k}={v}" for k, v in merged.items()),
        },
    }

    log.info("Query rewritten in %dms: '%s'", elapsed_ms, retrieval_query[:80])

    return {
        "extracted_filters": {},
        "merged_filters": merged,
        "retrieval_query": retrieval_query,
        "retrieval_queries": retrieval_queries[:3],
        "trace_events": state.get("trace_events", []) + [trace],
    }
