"""LangGraph agent graph — the state machine that drives Phase 3."""

from __future__ import annotations
import logging
from typing import Any
from uuid import UUID

from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes.classify_intent import classify_intent
from app.agent.nodes.extract_filters import extract_filters
from app.agent.nodes.retrieve import retrieve_node
from app.agent.nodes.grade_chunks import grade_chunks
from app.agent.nodes.rewrite_query import rewrite_query
from app.agent.nodes.generate_answer import generate_answer
from app.agent.nodes.verify_grounding import verify_grounding
from app.config import settings

log = logging.getLogger("clintel.agent.graph")


# ── Routing functions ──────────────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """Route after classify_intent: short-circuit or proceed to extraction."""
    intent = state.get("intent", "clinical_query")
    if intent in ("greeting", "out_of_scope", "needs_clarification"):
        return "short_circuit"
    return "extract_filters"


def route_after_grading(state: AgentState) -> str:
    """Route after grade_chunks: proceed to answer or retry retrieval."""
    coverage = state.get("coverage_score", 0.5)
    retries = state.get("retries", 0)
    threshold = settings.agent_coverage_threshold

    if coverage >= threshold:
        return "generate_answer"
    if retries < settings.agent_max_retries:
        return "rewrite_query"
    # Max retries exhausted — best-effort answer
    log.info("Max retries (%d) reached with coverage %.2f, proceeding to answer", retries, coverage)
    return "generate_answer"


# ── Build the graph ────────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """Construct and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("extract_filters", extract_filters)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_chunks", grade_chunks)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_grounding", verify_grounding)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Edges
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "extract_filters": "extract_filters",
            "short_circuit": END,
        },
    )
    graph.add_edge("extract_filters", "retrieve")
    graph.add_edge("retrieve", "grade_chunks")
    graph.add_conditional_edges(
        "grade_chunks",
        route_after_grading,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", "verify_grounding")
    graph.add_edge("verify_grounding", END)

    return graph.compile()


# Singleton compiled graph
agent = build_agent_graph()


# ── Entry points ───────────────────────────────────────────────────────────

async def run_agent(
    *,
    question: str,
    session_id: UUID,
    tenant_id: UUID,
    history: list[dict[str, Any]],
    user_filters: dict[str, Any] | None = None,
    top_k: int = 8,
) -> AgentState:
    """Run the agent graph synchronously (non-streaming). Returns final state."""
    initial_state: AgentState = {
        "question": question,
        "session_id": session_id,
        "tenant_id": tenant_id,
        "history": history,
        "user_filters": user_filters,
        "top_k": top_k,
        "retries": 0,
        "trace_events": [],
    }

    try:
        result = await agent.ainvoke(initial_state)
        return result
    except Exception as exc:
        log.error("Agent graph failed: %s", exc, exc_info=True)
        # Return a minimal error state
        return {
            **initial_state,
            "final_answer": "I encountered an internal error. Please try again.",
            "final_abstained": True,
            "final_abstain_reason": f"Agent error: {exc}",
            "final_confidence": 0.0,
            "chunks": [],
            "follow_up_questions": [],
            "used_sources": [],
            "trace_events": initial_state.get("trace_events", []) + [
                {"node": "agent", "status": "error", "ms": 0, "data": {"error": str(exc)[:200]}}
            ],
        }


async def stream_agent_to_queue(
    queue: "asyncio.Queue[tuple[str, Any]]",
    *,
    question: str,
    session_id: UUID,
    tenant_id: UUID,
    history: list[dict[str, Any]],
    user_filters: dict[str, Any] | None = None,
    top_k: int = 8,
) -> None:
    """Run the agent graph and push trace events into an asyncio.Queue.

    This runs as a background task. The SSE endpoint reads from the queue.
    Pushes tuples of (event_type, data_dict):
      - ("trace", {...})  — after each node completes
      - ("final", state)  — the complete final state
      - ("done", None)    — sentinel to signal end of stream
    """
    import asyncio

    initial_state: AgentState = {
        "question": question,
        "session_id": session_id,
        "tenant_id": tenant_id,
        "history": history,
        "user_filters": user_filters,
        "top_k": top_k,
        "retries": 0,
        "trace_events": [],
    }

    last_trace_count = 0
    final_state = None

    try:
        async for state_snapshot in agent.astream(
            initial_state,
            stream_mode="values",
            config={"configurable": {"event_queue": queue}}
        ):
            final_state = state_snapshot
            all_traces = state_snapshot.get("trace_events", [])
            new_traces = all_traces[last_trace_count:]
            last_trace_count = len(all_traces)
            for trace in new_traces:
                await queue.put(("trace", trace))
    except Exception as exc:
        log.error("Agent stream failed: %s", exc, exc_info=True)
        final_state = {
            **initial_state,
            "final_answer": "I encountered an internal error. Please try again.",
            "final_abstained": True,
            "final_abstain_reason": f"Agent error: {exc}",
            "final_confidence": 0.0,
            "chunks": [],
            "follow_up_questions": [],
            "used_sources": [],
        }
        await queue.put(("trace", {"node": "agent", "status": "error", "ms": 0, "data": {"error": str(exc)[:200]}}))

    if final_state:
        await queue.put(("final", final_state))

    await queue.put(("done", None))
