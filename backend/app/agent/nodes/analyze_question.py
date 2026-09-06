"""analyze_question — run intent classification and query rewriting concurrently.

These were two sequential nodes costing ~2-3s each, back to back, before any
retrieval could start. They are fully independent: `extract_filters` reads only
`question`, `history`, and `user_filters` — nothing that `classify_intent`
produces. Running them serially was ~2.5s of pure dead time on every query.

Why a combined node rather than a LangGraph fan-out from START: `AgentState.
trace_events` is a plain `list[dict]` with no merge reducer, so two nodes writing
it in the same superstep would clobber each other and one node's trace would
silently vanish. Gathering inside one node keeps both prompts, both trace events,
and the existing routing exactly as they were.

If the classifier short-circuits (greeting / out_of_scope), the rewrite result is
simply discarded — a wasted cheap call, but it costs no wall-clock time because
it ran in parallel, and short-circuits are rare next to real questions.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agent.nodes.classify_intent import classify_intent
from app.agent.nodes.extract_filters import extract_filters
from app.agent.state import AgentState
from app.observability import trace_span, traces_under_node

log = logging.getLogger("clintel.agent.analyze_question")

# Keys classify_intent sets when it short-circuits. If present, the run ends here
# and anything extract_filters produced is irrelevant.
_SHORT_CIRCUIT_KEY = "final_answer"


def _new_traces(result: dict[str, Any], before: int) -> list[dict[str, Any]]:
    """Pull just the trace events this node appended.

    Each node returns the FULL accumulated list (`state[...] + [trace]`), so
    merging two of them naively would duplicate history. Slice off the tail.
    """
    return (result.get("trace_events") or [])[before:]


async def _spanned(name: str, coro_fn, state, config):
    """Run one sub-step inside its own named LangSmith span.

    Without this the two calls collapse into two anonymous `ChatOpenAI` runs
    under `analyze_question`, and you lose the ability to tell classification
    from query rewriting in a trace.
    """
    with trace_span(name):
        return await coro_fn(state, config)


@traces_under_node
async def analyze_question(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    before = len(state.get("trace_events", []))

    intent_result, filters_result = await asyncio.gather(
        _spanned("classify_intent", classify_intent, state, config),
        _spanned("extract_filters", extract_filters, state, config),
        return_exceptions=True,
    )

    # Either node failing must not take down the run — both already fail soft
    # internally, so an exception here means something unexpected.
    if isinstance(intent_result, BaseException):
        log.error("classify_intent failed: %s", intent_result, exc_info=intent_result)
        intent_result = {"intent": "clinical_query", "query_intent": "general",
                         "intent_confidence": 0.5, "trace_events": state.get("trace_events", [])}
    if isinstance(filters_result, BaseException):
        log.error("extract_filters failed: %s", filters_result, exc_info=filters_result)
        filters_result = {"trace_events": state.get("trace_events", [])}

    merged: dict[str, Any] = {}
    merged.update(filters_result)
    # Intent wins on any key collision: it owns routing and the short-circuit
    # payload, and must not be overwritten by the rewrite result.
    merged.update(intent_result)

    # Preserve chronological order: classification, then rewriting.
    merged["trace_events"] = (
        state.get("trace_events", [])
        + _new_traces(intent_result, before)
        + _new_traces(filters_result, before)
    )

    if merged.get(_SHORT_CIRCUIT_KEY):
        log.info("Short-circuiting on intent=%s — discarding the parallel query rewrite",
                 merged.get("intent"))

    return merged
