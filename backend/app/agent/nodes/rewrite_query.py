"""rewrite_query — generate an alternative retrieval query targeting uncovered facets."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.llm import chat_completion
from langchain_core.runnables import RunnableConfig

from app.observability import traces_under_node

from app.agent.prompts import REWRITE_QUERY_PROMPT
from app.agent.state import AgentState
from app.config import settings

log = logging.getLogger("clintel.agent.rewrite_query")


@traces_under_node
async def rewrite_query(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Rewrite the retrieval query to target uncovered facets."""
    q = config.get("configurable", {}).get("event_queue")
    if q:
        await q.put(("step_started", {"node": "rewrite_query"}))
        
    original_query = state.get("retrieval_query", state["question"])
    coverage_reasoning = state.get("coverage_reasoning", "")
    uncovered_facets = state.get("uncovered_facets", [])
    retries = state.get("retries", 0)
    started = time.perf_counter()

    try:
        response = await chat_completion(
            model=settings.background_model,
            temperature=0.2,
            max_completion_tokens=150,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": REWRITE_QUERY_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Original question: {state['question']}\n"
                        f"Previous retrieval query: {original_query}\n"
                        f"Coverage reasoning: {coverage_reasoning}\n"
                        f"Uncovered facets: {', '.join(uncovered_facets)}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        rewritten = str(parsed.get("rewritten_query", original_query)).strip()
    except Exception as exc:
        log.warning("rewrite_query failed, using original query: %s", exc)
        rewritten = original_query

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    trace = {
        "node": "rewrite_query",
        "status": "done",
        "ms": elapsed_ms,
        "data": {
            "rewritten_query": rewritten[:120],
            "retry_number": retries + 1,
            "targeting_facets": uncovered_facets[:3],
        },
    }

    log.info("Query rewritten for retry %d in %dms: '%s'", retries + 1, elapsed_ms, rewritten[:80])

    return {
        "rewritten_query": rewritten,
        "retries": retries + 1,
        "trace_events": state.get("trace_events", []) + [trace],
    }
