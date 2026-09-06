"""retrieve — call Phase 2 retrieval pipeline as a tool (no LLM)."""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID
from langchain_core.runnables import RunnableConfig

from app.observability import traces_under_node

from app.agent.state import AgentState
from app.retrieval import RetrievalFilters
from app.retrieval.retriever import retrieve, retrieve_multi

log = logging.getLogger("clintel.agent.retrieve")


@traces_under_node
async def retrieve_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Run the Phase 2 hybrid retrieval pipeline."""
    q = config.get("configurable", {}).get("event_queue")
    if q:
        await q.put(("step_started", {"node": "retrieve"}))
        
    query_intent = state.get("query_intent", "general")
    tenant_id = state["tenant_id"]
    merged_filters = state.get("merged_filters", {})
    top_k = state.get("top_k", 8)
    started = time.perf_counter()

    # Build RetrievalFilters from the merged dict
    filter_kwargs = {}
    for field in RetrievalFilters.model_fields:
        if field in merged_filters:
            filter_kwargs[field] = merged_filters[field]
    filters = RetrievalFilters(**filter_kwargs) if filter_kwargs else None
    
    # Use rewritten query (from retry) if available
    rewritten = state.get("rewritten_query")
    if rewritten:
        queries = [rewritten]
        rerank_query = rewritten
    else:
        queries = state.get("retrieval_queries") or [state.get("retrieval_query", state["question"])]
        rerank_query = state["question"]

    try:
        result = await retrieve_multi(
            queries,
            rerank_query=rerank_query,
            tenant_id=UUID(str(tenant_id)),
            query_intent=query_intent,
            filters=filters,
            top_k=top_k,
            expand_parents=True,
        )
        chunks = result.chunks
        retrieval_ms = result.query_ms
    except Exception as exc:
        log.error("Retrieval failed: %s", exc)
        chunks = []
        retrieval_ms = int((time.perf_counter() - started) * 1000)

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    trace = {
        "node": "retrieve",
        "status": "done",
        "ms": elapsed_ms,
        "data": {
            "queries": [str(q)[:120] for q in queries],
            "query_intent": query_intent,
            "chunk_count": len(chunks),
            "retrieval_ms": retrieval_ms,
        },
    }

    log.info("Retrieved %d chunks in %dms for queries: %s", len(chunks), elapsed_ms, queries)

    return {
        "chunks": chunks,
        "retrieval_ms": retrieval_ms,
        "chunk_count": len(chunks),
        # Clear rewritten_query so next retry doesn't reuse a stale one
        "rewritten_query": None,
        "trace_events": state.get("trace_events", []) + [trace],
    }
