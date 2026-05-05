"""AgentState — the typed-dict that flows through every LangGraph node."""

from __future__ import annotations

from typing import Any, TypedDict
from uuid import UUID


class AgentState(TypedDict, total=False):
    # ── Input (set once by the router) ──────────────────────────────────
    question: str
    session_id: UUID
    tenant_id: UUID
    history: list[dict[str, Any]]
    user_filters: dict[str, Any] | None
    top_k: int

    # ── classify_intent ─────────────────────────────────────────────────
    intent: str                        # "clinical_query" | "greeting" | "out_of_scope" | "clarification"
    intent_confidence: float

    # ── extract_filters ─────────────────────────────────────────────────
    extracted_filters: dict[str, Any]
    merged_filters: dict[str, Any]
    retrieval_query: str               # standalone retrieval query

    # ── retrieve ────────────────────────────────────────────────────────
    chunks: list[Any]                  # list[RetrievedChunk] — Any to avoid circular import
    retrieval_ms: int
    chunk_count: int

    # ── grade_chunks ────────────────────────────────────────────────────
    coverage_score: float
    coverage_reasoning: str
    uncovered_facets: list[str]
    retries: int

    # ── rewrite_query ───────────────────────────────────────────────────
    rewritten_query: str | None

    # ── generate_answer ─────────────────────────────────────────────────
    raw_answer: str
    abstained: bool
    abstain_reason: str | None
    confidence: float | None
    used_sources: list[int]
    follow_up_questions: list[str]

    # ── verify_grounding ────────────────────────────────────────────────
    grounding_verdict: str             # "pass" | "partial" | "fail"
    ungrounded_claims: list[str]
    final_answer: str
    final_abstained: bool
    final_abstain_reason: str | None
    final_confidence: float | None

    # ── Trace events (accumulated for SSE) ──────────────────────────────
    trace_events: list[dict[str, Any]]
