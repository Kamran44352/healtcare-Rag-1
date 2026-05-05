from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.db import get_db
from app.retrieval import RetrievalFilters, retrieve
from app.schemas.api import (
    ChatHistoryResponse,
    ChatMessageRecord,
    ChatQueryRequest,
    ChatResponse,
    ChatSessionResponse,
    Citation,
    FeedbackRequest,
)

router = APIRouter()
log = logging.getLogger("clintel.chat")
_openai = AsyncOpenAI(api_key=settings.openai_api_key)

_SOURCE_TAG_RE = re.compile(r"\[SOURCE\s+(\d+)\]", re.IGNORECASE)

# ── Phase 2 system prompt (kept for fallback) ──────────────────────────────
_SYSTEM_PROMPT = """You are ClinTel, a clinical decision support assistant trained on healthcare guidelines.

## Grounding
- Answer ONLY from the provided retrieved sources. Do not use outside medical knowledge.
- Combine evidence across sources when relevant — do not rely on a single passage.
- When information is absent from the sources, say "not covered in the available guidelines" — never fill gaps by inference.
- If sources are insufficient to answer safely, set abstained=true.

## How to Answer
Write like a senior clinician giving a colleague a concise, confident summary — not like a research paper or a checklist.

- Lead with what matters most: the most likely diagnosis or the most dangerous condition that must be ruled out first.
- Weave safety considerations (urgency, red flags, referral triggers) naturally into the answer where clinically relevant. Only break them out as a separate section if the question is specifically about risk or triage.
- Use decisive language: "most consistent with…", "requires urgent referral", "the guideline recommends…"
- Avoid: "could be", "might be", "the source states", "as per the document", "it is important to note"
- Cite every factual claim with [SOURCE n]. Only use numbers from the provided source list.
- If evidence is weak or indirect, say so in one phrase — do not over-hedge the entire answer.
- Use markdown formatting where it improves clarity: **bold** for key terms, diagnoses, and warnings; bullet lists for multiple items or criteria; `>` blockquotes for direct guideline recommendations. Do not strip formatting to plain prose.
- Only avoid headings when the answer is short enough not to need them. For multi-part answers, headings improve readability.
- Be concise. No repetition, no padding.

## Terminology
- Always spell out acronyms and abbreviations on first use, e.g. "Intraocular Pressure (IOP)", then use the short form thereafter.
- This applies to all medical acronyms including: IOP, FBC, ESR, CRP, RAPD, OCT, VA, VF, AC, C/D, FFA, HVF, AION, GCA, TAB, PACG, PDS, APAC, LFTs, U+E, CXR, ANCA, ACE, IGRA, etc.
- Do NOT assume the reader knows what an acronym stands for.

## Follow-up Questions
Generate 3 follow-up questions the user would naturally want to ask next, based on the content of your answer.
- Written from the user's perspective — questions they would type into this system to go deeper.
- Directly relevant to what was just answered (not generic).
- Examples of good follow-ups: "What are the first-line treatment options for this?", "How do I differentiate this from X?", "At what point does this require emergency care?"
- Examples of bad follow-ups (never generate these): "Is it painful?", "Is it unilateral?" — those are history-taking questions, not user queries to a guideline system.

Return JSON exactly in this shape:
{
  "answer": "markdown answer with [SOURCE n] citations",
  "abstained": false,
  "abstain_reason": null,
  "confidence": 0.0,
  "used_sources": [1, 2],
  "follow_up_questions": ["user-pov question 1", "user-pov question 2", "user-pov question 3"]
}
"""

_QUERY_REWRITE_PROMPT = (
    "You are a clinical search-query rewriter. "
    "Given a conversation between a clinician and an assistant, rewrite the user's "
    "latest follow-up question as a STANDALONE clinical search query that includes "
    "the relevant diagnosis, condition name, and key clinical findings from the "
    "conversation so that a vector search engine can retrieve the correct guideline "
    "sections without needing the conversation history.\n\n"
    "Rules:\n"
    "- Include the specific diagnosis or condition name from the prior assistant answer.\n"
    "- Include key clinical findings (e.g. IOP value, gonioscopy findings) if they are "
    "relevant to the follow-up question.\n"
    "- Keep the rewritten query concise — one or two sentences maximum.\n"
    "- If the question is already self-contained with a specific clinical topic, "
    "return it unchanged.\n"
    "- Return ONLY the rewritten query text, nothing else.\n"
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalize_citation_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": str(raw["chunk_id"]),
        "source_index": int(raw["source_index"]),
        "filename": str(raw["filename"]),
        "section_path": str(raw["section_path"]),
        "snippet": str(raw["snippet"]),
        "full_snippet": raw.get("full_snippet"),
        "doc_metadata": raw.get("doc_metadata") or {},
    }


def _extract_source_order(answer: str) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for match in _SOURCE_TAG_RE.finditer(answer or ""):
        value = int(match.group(1))
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _remap_answer_citations(answer: str, used_sources: list[int]) -> tuple[str, dict[int, int]]:
    mapping = {source: index for index, source in enumerate(used_sources, start=1)}

    def replace(match: re.Match[str]) -> str:
        original = int(match.group(1))
        mapped = mapping.get(original)
        return f"[SOURCE {mapped}]" if mapped is not None else match.group(0)

    return _SOURCE_TAG_RE.sub(replace, answer or ""), mapping


def _build_retrieval_citations(result_chunks: list[Any]) -> tuple[list[Citation], dict[int, Citation]]:
    citations: list[Citation] = []
    citation_map: dict[int, Citation] = {}
    for index, chunk in enumerate(result_chunks, start=1):
        citation = Citation(
            chunk_id=chunk.chunk_id,
            source_index=index,
            filename=chunk.filename,
            section_path=chunk.section_path,
            snippet=chunk.snippet,
            full_snippet=chunk.full_snippet,
            doc_metadata=chunk.doc_metadata,
        )
        citations.append(citation)
        citation_map[index] = citation
    return citations, citation_map


def _build_source_context(chunks: list[Any]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        doc_meta = chunk.doc_metadata or {}
        guideline_title = doc_meta.get("guideline_title") or chunk.filename
        reference = doc_meta.get("reference")
        header = f"{guideline_title} ({reference})" if reference else str(guideline_title)
        context = chunk.parent_text or chunk.full_snippet or chunk.snippet
        blocks.append(
            "\n".join([
                f"[SOURCE {index}]", f"Title: {header}",
                f"Section path: {chunk.section_path}",
                f"Section type: {chunk.section_type or 'unknown'}",
                f"Recommendation strength: {chunk.recommendation_strength or 'unknown'}",
                f"Snippet: {chunk.snippet}", "Context:", str(context),
            ])
        )
    return "\n\n".join(blocks)


def _build_history_context(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "No prior conversation."
    lines: list[str] = []
    for message in messages[-6:]:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


async def _build_retrieval_query(question: str, history: list[dict[str, Any]]) -> str:
    if not history:
        return question
    recent = [m for m in history[-6:] if m.get("content")]
    assistant_msgs = [m for m in recent if m.get("role") == "assistant"]
    if not assistant_msgs:
        return question
    history_text = "\n".join(
        f"{str(m.get('role', 'user')).upper()}: {str(m['content'])[:500]}" for m in recent
    )
    try:
        response = await _openai.chat.completions.create(
            model=settings.background_model, temperature=0.0, max_completion_tokens=150,
            messages=[
                {"role": "system", "content": _QUERY_REWRITE_PROMPT},
                {"role": "user", "content": f"Conversation so far:\n{history_text}\n\nFollow-up question: {question}"},
            ],
        )
        rewritten = (response.choices[0].message.content or "").strip()
        if rewritten:
            log.info("Query rewritten: '%s' -> '%s'", question, rewritten)
            return rewritten
    except Exception as exc:
        log.warning("Query rewriting failed: %s", exc)
    return question


def _sanitize_follow_ups(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    cleaned: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= 3:
            break
    return cleaned


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
async def _generate_grounded_answer(*, question: str, history_context: str, source_context: str) -> dict[str, Any]:
    response = await _openai.chat.completions.create(
        model=settings.foreground_model,
        response_format={"type": "json_object"},
        temperature=0.1, max_completion_tokens=1400,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation so far:\n{history_context}\n\nCurrent user question:\n{question}\n\nRetrieved sources:\n{source_context}"},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Model did not return a JSON object.")
    return parsed


# ── DB helpers ─────────────────────────────────────────────────────────────

async def _create_session(tenant_id: UUID) -> dict[str, Any]:
    db = await get_db()
    session_id = uuid4()
    created_at = datetime.utcnow().isoformat()
    payload = {"session_id": str(session_id), "tenant_id": str(tenant_id), "metadata": {}, "created_at": created_at}
    await db.table("chat_sessions").insert(payload).execute()
    payload["expires_at"] = None
    return payload


async def _get_or_create_session(session_id: UUID | None, tenant_id: UUID) -> dict[str, Any]:
    if session_id is None:
        return await _create_session(tenant_id)
    db = await get_db()
    result = (
        await db.table("chat_sessions").select("*")
        .eq("session_id", str(session_id)).eq("tenant_id", str(tenant_id))
        .single().execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return result.data


async def _get_session_history(session_id: UUID, tenant_id: UUID) -> list[dict[str, Any]]:
    db = await get_db()
    result = (
        await db.table("chat_messages").select("*")
        .eq("session_id", str(session_id)).eq("tenant_id", str(tenant_id))
        .order("created_at").execute()
    )
    return result.data or []


async def _insert_chat_message(
    *, session_id: UUID, tenant_id: UUID, role: str, content: str,
    citations: list[Citation] | None = None, abstained: bool = False,
    confidence: float | None = None,
) -> UUID:
    db = await get_db()
    message_id = uuid4()
    await db.table("chat_messages").insert({
        "message_id": str(message_id), "session_id": str(session_id),
        "role": role, "content": content,
        "citations": [_normalize_citation_payload(c.model_dump(mode="json")) for c in (citations or [])],
        "abstained": abstained, "confidence": confidence, "tenant_id": str(tenant_id),
    }).execute()
    return message_id


# ── Agent-powered response builder ────────────────────────────────────────

def _build_agent_response(
    state: dict[str, Any], chunks: list[Any], session_id: UUID, message_id: UUID,
    include_debug: bool = False,
) -> ChatResponse:
    """Convert agent final state into a ChatResponse."""
    answer = state.get("final_answer", "")
    abstained = state.get("final_abstained", False)
    abstain_reason = state.get("final_abstain_reason")
    confidence = state.get("final_confidence")
    used_sources = state.get("used_sources", [])
    follow_ups = state.get("follow_up_questions", [])

    if not chunks:
        return ChatResponse(
            answer=answer or "I could not find relevant guideline evidence.",
            abstained=True, abstain_reason=abstain_reason or "No sources retrieved.",
            confidence=0.0, citations=[], session_id=session_id, message_id=message_id,
            retrieval_debug=None, follow_up_questions=follow_ups,
        )

    all_citations, citation_map = _build_retrieval_citations(chunks)

    # Validate used_sources against available chunks
    valid_sources = [s for s in used_sources if s in citation_map]
    if not valid_sources:
        valid_sources = [v for v in _extract_source_order(answer) if v in citation_map]

    answer, remapped = _remap_answer_citations(answer, valid_sources)
    selected = [
        citation_map[idx].model_copy(update={"source_index": remapped[idx]})
        for idx in valid_sources
    ]

    debug_info = None
    if include_debug:
        debug_info = {"trace_events": state.get("trace_events", [])}

    return ChatResponse(
        answer=answer, abstained=abstained, abstain_reason=abstain_reason,
        confidence=confidence, citations=selected, session_id=session_id,
        message_id=message_id, retrieval_debug=debug_info,
        follow_up_questions=follow_ups,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/chat/sessions", response_model=ChatSessionResponse)
async def create_session():
    payload = await _create_session(settings.default_tenant_id)
    return ChatSessionResponse(
        session_id=UUID(payload["session_id"]),
        metadata=payload.get("metadata") or {},
        created_at=_parse_dt(payload.get("created_at")) or datetime.utcnow(),
        expires_at=_parse_dt(payload.get("expires_at")),
    )


@router.get("/chat/sessions/{session_id}/messages", response_model=ChatHistoryResponse)
async def get_session_messages(session_id: UUID):
    rows = await _get_session_history(session_id, settings.default_tenant_id)
    messages = [
        ChatMessageRecord(
            message_id=UUID(row["message_id"]), session_id=UUID(row["session_id"]),
            role=str(row["role"]), content=str(row["content"]),
            citations=[Citation.model_validate(item) for item in (row.get("citations") or [])],
            abstained=bool(row.get("abstained")),
            confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
            created_at=_parse_dt(row.get("created_at")) or datetime.utcnow(),
        )
        for row in rows
    ]
    return ChatHistoryResponse(session_id=session_id, messages=messages)


@router.post("/chat/query", response_model=ChatResponse)
async def chat_query(payload: ChatQueryRequest):
    tenant_id = settings.default_tenant_id
    session = await _get_or_create_session(payload.session_id, tenant_id)
    session_id = UUID(session["session_id"])
    prior_history = await _get_session_history(session_id, tenant_id)
    await _insert_chat_message(session_id=session_id, tenant_id=tenant_id, role="user", content=payload.question)

    # ── Phase 3 Agent path ──
    if settings.agent_enabled:
        from app.agent.graph import run_agent
        state = await run_agent(
            question=payload.question, session_id=session_id, tenant_id=tenant_id,
            history=prior_history,
            user_filters=payload.filters.model_dump() if payload.filters else None,
            top_k=payload.top_k,
        )
        chunks = state.get("chunks", [])
        answer = state.get("final_answer", "")
        abstained = state.get("final_abstained", False)
        confidence = state.get("final_confidence")
        follow_ups = state.get("follow_up_questions", [])
        used_sources = state.get("used_sources", [])

        # Build citations from chunks
        if chunks:
            all_cit, cit_map = _build_retrieval_citations(chunks)
            valid_src = [s for s in used_sources if s in cit_map]
            if not valid_src:
                valid_src = [v for v in _extract_source_order(answer) if v in cit_map]
            answer, remapped = _remap_answer_citations(answer, valid_src)
            selected = [
                cit_map[idx].model_copy(update={"source_index": remapped[idx]})
                for idx in valid_src
            ]
        else:
            selected = []

        msg_id = await _insert_chat_message(
            session_id=session_id, tenant_id=tenant_id, role="assistant",
            content=answer, abstained=abstained, confidence=confidence,
            citations=selected,
        )
        return ChatResponse(
            answer=answer, abstained=abstained,
            abstain_reason=state.get("final_abstain_reason"),
            confidence=confidence, citations=selected,
            session_id=session_id, message_id=msg_id,
            retrieval_debug={"trace_events": state.get("trace_events", [])} if payload.include_debug else None,
            follow_up_questions=follow_ups,
        )

    # ── Phase 2 fallback path ──
    retrieval_query = await _build_retrieval_query(payload.question, prior_history)
    retrieval_result = await retrieve(
        retrieval_query, tenant_id=tenant_id, filters=payload.filters,
        profile="specific", top_k=payload.top_k, expand_parents=True,
    )

    if not retrieval_result.chunks:
        answer = "I could not find enough relevant guideline evidence to answer confidently from the indexed corpus."
        msg_id = await _insert_chat_message(
            session_id=session_id, tenant_id=tenant_id, role="assistant",
            content=answer, citations=[], abstained=True, confidence=0.0,
        )
        return ChatResponse(
            answer=answer, abstained=True, abstain_reason="No relevant sources were retrieved.",
            confidence=0.0, citations=[], session_id=session_id, message_id=msg_id,
            retrieval_debug=retrieval_result.model_dump(mode="json") if payload.include_debug else None,
            follow_up_questions=[],
        )

    all_citations, citation_map = _build_retrieval_citations(retrieval_result.chunks)
    source_context = _build_source_context(retrieval_result.chunks)
    history_context = _build_history_context(prior_history)
    model_output = await _generate_grounded_answer(
        question=payload.question, history_context=history_context, source_context=source_context,
    )

    answer = str(model_output.get("answer") or "").strip()
    abstained = bool(model_output.get("abstained", False))
    abstain_reason = str(model_output.get("abstain_reason")).strip() if model_output.get("abstain_reason") not in (None, "") else None
    conf_raw = model_output.get("confidence")
    try:
        confidence = float(conf_raw) if conf_raw is not None else None
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))

    used_sources_raw = model_output.get("used_sources")
    used_sources: list[int] = []
    if isinstance(used_sources_raw, list):
        seen: set[int] = set()
        for item in used_sources_raw:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value in citation_map and value not in seen:
                seen.add(value); used_sources.append(value)
    if not used_sources:
        used_sources = [v for v in _extract_source_order(answer) if v in citation_map]

    answer, remapped_indices = _remap_answer_citations(answer, used_sources)
    selected_citations = [
        citation_map[idx].model_copy(update={"source_index": remapped_indices[idx]})
        for idx in used_sources
    ]
    follow_up_questions = _sanitize_follow_ups(model_output.get("follow_up_questions"))

    if not answer:
        answer = "I could not produce a grounded answer from the retrieved guideline evidence."
        abstained = True
        abstain_reason = abstain_reason or "The model did not return a usable grounded answer."
        confidence = 0.0 if confidence is None else confidence

    msg_id = await _insert_chat_message(
        session_id=session_id, tenant_id=tenant_id, role="assistant",
        content=answer, citations=selected_citations, abstained=abstained, confidence=confidence,
    )
    return ChatResponse(
        answer=answer, abstained=abstained, abstain_reason=abstain_reason,
        confidence=confidence, citations=selected_citations, session_id=session_id,
        message_id=msg_id,
        retrieval_debug=retrieval_result.model_dump(mode="json") if payload.include_debug else None,
        follow_up_questions=follow_up_questions,
    )


# ── SSE streaming endpoint (Phase 3) ──────────────────────────────────────

# Human-readable labels for trace nodes
_NODE_LABELS = {
    "classify_intent": ("Analyzing question...", "Classifying intent"),
    "extract_filters": ("Extracting clinical filters...", "Identifying conditions, specialties & context"),
    "retrieve": ("Searching guidelines...", "Querying knowledge base"),
    "grade_chunks": ("Evaluating relevance...", "Checking if retrieved evidence covers the question"),
    "rewrite_query": ("Refining search...", "Adjusting query for better coverage"),
    "generate_answer": ("Generating answer...", "Synthesizing clinical response from evidence"),
    "verify_grounding": ("Verifying citations...", "Checking every claim is supported by sources"),
}


@router.post("/chat/query/stream")
async def chat_query_stream(payload: ChatQueryRequest):
    import asyncio
    from sse_starlette.sse import EventSourceResponse, ServerSentEvent

    if not settings.agent_enabled:
        return JSONResponse(status_code=501, content={"detail": "Agent not enabled. Use /chat/query instead."})

    tenant_id = settings.default_tenant_id
    session = await _get_or_create_session(payload.session_id, tenant_id)
    session_id = UUID(session["session_id"])
    prior_history = await _get_session_history(session_id, tenant_id)
    await _insert_chat_message(session_id=session_id, tenant_id=tenant_id, role="user", content=payload.question)

    # Queue for real-time event passing from agent background task → SSE generator
    event_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def event_generator():
        from app.agent.graph import stream_agent_to_queue

        # Launch the agent as a background task — it pushes events into the queue
        agent_task = asyncio.create_task(
            stream_agent_to_queue(
                event_queue,
                question=payload.question,
                session_id=session_id,
                tenant_id=tenant_id,
                history=prior_history,
                user_filters=payload.filters.model_dump() if payload.filters else None,
                top_k=payload.top_k,
            )
        )

        final_state = None

        try:
            while True:
                # Read from the queue — wakes up immediately when agent pushes an event
                event_type, data = await event_queue.get()

                if event_type == "done":
                    break

                if event_type == "step_started":
                    node = data.get("node", "")
                    label, description = _NODE_LABELS.get(node, (node, ""))
                    yield ServerSentEvent(
                        data=json.dumps({
                            "stage": node,
                            "label": label,
                            "description": description,
                            "status": "running",
                        }),
                        event="step_started",
                    )

                elif event_type == "trace":
                    node = data.get("node", "")
                    label, description = _NODE_LABELS.get(node, (node, ""))
                    trace_data = {
                        "stage": node,
                        "label": label,
                        "description": description,
                        "status": data.get("status", "done"),
                        "ms": data.get("ms", 0),
                        **data.get("data", {}),
                    }
                    yield ServerSentEvent(data=json.dumps(trace_data), event="stage")

                elif event_type == "final":
                    final_state = data

        except Exception as exc:
            log.error("SSE generator error: %s", exc)
            yield ServerSentEvent(data=json.dumps({"message": str(exc)[:200]}), event="error")
            return
        finally:
            if not agent_task.done():
                agent_task.cancel()

        if not final_state:
            yield ServerSentEvent(data=json.dumps({"message": "Agent produced no final state"}), event="error")
            return

        chunks = final_state.get("chunks", [])
        answer = final_state.get("final_answer", "")
        abstained = final_state.get("final_abstained", False)
        confidence = final_state.get("final_confidence")
        follow_ups = final_state.get("follow_up_questions", [])
        used_sources = final_state.get("used_sources", [])

        # Stream answer in chunks for the typing effect
        chunk_size = 12
        for i in range(0, len(answer), chunk_size):
            delta = answer[i:i + chunk_size]
            yield ServerSentEvent(data=json.dumps({"delta": delta}), event="answer_delta")
            await asyncio.sleep(0.015)

        # Build citations
        if chunks:
            all_cit, cit_map = _build_retrieval_citations(chunks)
            valid_src = [s for s in used_sources if s in cit_map]
            if not valid_src:
                valid_src = [v for v in _extract_source_order(answer) if v in cit_map]
            _, remapped = _remap_answer_citations(answer, valid_src)
            selected = [
                cit_map[idx].model_copy(update={"source_index": remapped[idx]})
                for idx in valid_src
            ]
            yield ServerSentEvent(data=json.dumps({"citations": [c.model_dump(mode="json") for c in selected]}), event="citations_ready")
        else:
            selected = []
            yield ServerSentEvent(data=json.dumps({"citations": []}), event="citations_ready")

        if follow_ups:
            yield ServerSentEvent(data=json.dumps({"follow_up_questions": follow_ups}), event="follow_up_questions_ready")

        # Save to DB
        msg_id = await _insert_chat_message(
            session_id=session_id, tenant_id=tenant_id, role="assistant",
            content=answer, citations=selected, abstained=abstained, confidence=confidence,
        )

        final_resp = ChatResponse(
            answer=answer, abstained=abstained,
            abstain_reason=final_state.get("final_abstain_reason"),
            confidence=confidence, citations=selected,
            session_id=session_id, message_id=msg_id,
            retrieval_debug={"trace_events": final_state.get("trace_events", [])} if payload.include_debug else None,
            follow_up_questions=follow_ups,
        )
        yield ServerSentEvent(data=json.dumps(final_resp.model_dump(mode="json")), event="final_response")
        yield ServerSentEvent(data="{}", event="done")

    return EventSourceResponse(
        event_generator(),
        headers={
            "X-Accel-Buffering": "no",
        },
        ping=300,  # keepalive every 5 min (effectively off for short queries)
    )


@router.patch("/chat/messages/{message_id}/feedback")
async def submit_feedback(message_id: UUID, payload: FeedbackRequest):
    rating = payload.rating_int()
    if rating not in (1, -1, None):
        raise HTTPException(status_code=400, detail="Rating must be 'up', 'down', 1, -1, or null.")
    db = await get_db()
    update_result = (
        await db.table("chat_messages").update({"feedback": rating})
        .eq("message_id", str(message_id)).eq("tenant_id", str(settings.default_tenant_id))
        .execute()
    )
    if not update_result.data:
        raise HTTPException(status_code=404, detail="Chat message not found.")
    return {"message_id": str(message_id), "rating": payload.rating}
