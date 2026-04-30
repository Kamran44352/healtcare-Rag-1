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

_SYSTEM_PROMPT = """You are ClinTel, a healthcare guideline assistant.

Answer only from the provided retrieved sources. Do not use outside medical knowledge.
If the sources are insufficient, abstain.

Requirements:
- Write a concise, clinician-friendly answer in markdown.
- Support factual claims with inline citations in the exact form [SOURCE n].
- Only cite source numbers that were provided in the source list.
- If evidence is weak or indirect, say so plainly.
- Do not mention internal system details.

Return JSON exactly in this shape:
{
  "answer": "markdown answer with [SOURCE n] citations",
  "abstained": true,
  "abstain_reason": "short reason or null",
  "confidence": 0.0,
  "used_sources": [1, 2],
  "follow_up_questions": ["...", "...", "..."]
}
"""

_SOURCE_TAG_RE = re.compile(r"\[SOURCE\s+(\d+)\]", re.IGNORECASE)


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
            "\n".join(
                [
                    f"[SOURCE {index}]",
                    f"Title: {header}",
                    f"Section path: {chunk.section_path}",
                    f"Section type: {chunk.section_type or 'unknown'}",
                    f"Recommendation strength: {chunk.recommendation_strength or 'unknown'}",
                    f"Snippet: {chunk.snippet}",
                    "Context:",
                    str(context),
                ]
            )
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
async def _generate_grounded_answer(
    *,
    question: str,
    history_context: str,
    source_context: str,
) -> dict[str, Any]:
    response = await _openai.chat.completions.create(
        model=settings.foreground_model,
        response_format={"type": "json_object"},
        temperature=0.1,
        max_completion_tokens=1400,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation so far:\n{history_context}\n\n"
                    f"Current user question:\n{question}\n\n"
                    f"Retrieved sources:\n{source_context}"
                ),
            },
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Model did not return a JSON object.")
    return parsed


async def _create_session(tenant_id: UUID) -> dict[str, Any]:
    db = await get_db()
    session_id = uuid4()
    created_at = datetime.utcnow().isoformat()
    payload = {
        "session_id": str(session_id),
        "tenant_id": str(tenant_id),
        "metadata": {},
        "created_at": created_at,
    }
    await db.table("chat_sessions").insert(payload).execute()
    payload["expires_at"] = None
    return payload


async def _get_or_create_session(session_id: UUID | None, tenant_id: UUID) -> dict[str, Any]:
    if session_id is None:
        return await _create_session(tenant_id)

    db = await get_db()
    result = (
        await db.table("chat_sessions")
        .select("*")
        .eq("session_id", str(session_id))
        .eq("tenant_id", str(tenant_id))
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return result.data


async def _get_session_history(session_id: UUID, tenant_id: UUID) -> list[dict[str, Any]]:
    db = await get_db()
    result = (
        await db.table("chat_messages")
        .select("*")
        .eq("session_id", str(session_id))
        .eq("tenant_id", str(tenant_id))
        .order("created_at")
        .execute()
    )
    return result.data or []


async def _insert_chat_message(
    *,
    session_id: UUID,
    tenant_id: UUID,
    role: str,
    content: str,
    citations: list[Citation] | None = None,
    abstained: bool = False,
    confidence: float | None = None,
) -> UUID:
    db = await get_db()
    message_id = uuid4()
    await db.table("chat_messages").insert(
        {
            "message_id": str(message_id),
            "session_id": str(session_id),
            "role": role,
            "content": content,
            "citations": [
                _normalize_citation_payload(citation.model_dump(mode="json"))
                for citation in (citations or [])
            ],
            "abstained": abstained,
            "confidence": confidence,
            "tenant_id": str(tenant_id),
        }
    ).execute()
    return message_id


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
            message_id=UUID(row["message_id"]),
            session_id=UUID(row["session_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
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
    await _insert_chat_message(
        session_id=session_id,
        tenant_id=tenant_id,
        role="user",
        content=payload.question,
    )

    retrieval_result = await retrieve(
        payload.question,
        tenant_id=tenant_id,
        filters=payload.filters,
        profile="specific",
        top_k=payload.top_k,
        expand_parents=True,
    )

    if not retrieval_result.chunks:
        answer = (
            "I could not find enough relevant guideline evidence to answer confidently from the indexed corpus."
        )
        assistant_message_id = await _insert_chat_message(
            session_id=session_id,
            tenant_id=tenant_id,
            role="assistant",
            content=answer,
            citations=[],
            abstained=True,
            confidence=0.0,
        )
        return ChatResponse(
            answer=answer,
            abstained=True,
            abstain_reason="No relevant sources were retrieved.",
            confidence=0.0,
            citations=[],
            session_id=session_id,
            message_id=assistant_message_id,
            retrieval_debug=(
                retrieval_result.model_dump(mode="json") if payload.include_debug else None
            ),
            follow_up_questions=[],
        )

    all_citations, citation_map = _build_retrieval_citations(retrieval_result.chunks)
    source_context = _build_source_context(retrieval_result.chunks)
    history_context = _build_history_context(prior_history)

    model_output = await _generate_grounded_answer(
        question=payload.question,
        history_context=history_context,
        source_context=source_context,
    )

    answer = str(model_output.get("answer") or "").strip()
    abstained = bool(model_output.get("abstained", False))
    abstain_reason = (
        str(model_output.get("abstain_reason")).strip()
        if model_output.get("abstain_reason") not in (None, "")
        else None
    )
    confidence_raw = model_output.get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else None
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
                seen.add(value)
                used_sources.append(value)

    if not used_sources:
        used_sources = [value for value in _extract_source_order(answer) if value in citation_map]

    answer, remapped_indices = _remap_answer_citations(answer, used_sources)
    selected_citations = [
        citation_map[index].model_copy(update={"source_index": remapped_indices[index]})
        for index in used_sources
    ]
    follow_up_questions = _sanitize_follow_ups(model_output.get("follow_up_questions"))

    if not answer:
        answer = (
            "I could not produce a grounded answer from the retrieved guideline evidence."
        )
        abstained = True
        abstain_reason = abstain_reason or "The model did not return a usable grounded answer."
        confidence = 0.0 if confidence is None else confidence

    assistant_message_id = await _insert_chat_message(
        session_id=session_id,
        tenant_id=tenant_id,
        role="assistant",
        content=answer,
        citations=selected_citations,
        abstained=abstained,
        confidence=confidence,
    )

    return ChatResponse(
        answer=answer,
        abstained=abstained,
        abstain_reason=abstain_reason,
        confidence=confidence,
        citations=selected_citations,
        session_id=session_id,
        message_id=assistant_message_id,
        retrieval_debug=retrieval_result.model_dump(mode="json") if payload.include_debug else None,
        follow_up_questions=follow_up_questions,
    )


@router.post("/chat/query/stream")
async def chat_query_stream():
    return JSONResponse(
        status_code=501,
        content={"detail": "Streaming chat is not implemented yet. The frontend will fall back to non-streaming chat."},
    )


@router.patch("/chat/messages/{message_id}/feedback")
async def submit_feedback(message_id: UUID, payload: FeedbackRequest):
    rating = payload.rating_int()
    if rating not in (1, -1, None):
        raise HTTPException(status_code=400, detail="Rating must be 'up', 'down', 1, -1, or null.")

    db = await get_db()
    update_result = (
        await db.table("chat_messages")
        .update({"feedback": rating})
        .eq("message_id", str(message_id))
        .eq("tenant_id", str(settings.default_tenant_id))
        .execute()
    )
    if not update_result.data:
        raise HTTPException(status_code=404, detail="Chat message not found.")

    return {"message_id": str(message_id), "rating": payload.rating}
