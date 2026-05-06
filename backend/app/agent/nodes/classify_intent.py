"""classify_intent — decide what kind of question the user is asking."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI
from langchain_core.runnables import RunnableConfig

from app.agent.prompts import CLASSIFY_INTENT_PROMPT, GREETING_RESPONSE, OUT_OF_SCOPE_RESPONSE
from app.agent.state import AgentState
from app.config import settings

log = logging.getLogger("clintel.agent.classify_intent")
_openai = AsyncOpenAI(api_key=settings.openai_api_key)


def _build_classify_context(question: str, history: list[dict[str, Any]]) -> str:
    """Build a context string for the classifier that includes recent history."""
    if not history:
        return question
    # Include the last 4 messages so the classifier understands follow-up context
    lines: list[str] = []
    for msg in history[-4:]:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", ""))[:300].strip()
        if content:
            lines.append(f"{role}: {content}")
    if not lines:
        return question
    return (
        f"Conversation so far:\n"
        + "\n".join(lines)
        + f"\n\nCurrent question: {question}"
    )


async def classify_intent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Classify the user's question intent."""
    q = config.get("configurable", {}).get("event_queue")
    if q:
        await q.put(("step_started", {"node": "classify_intent"}))
        
    question = state["question"]
    history = state.get("history", [])
    started = time.perf_counter()

    # Build context that includes conversation history so follow-up questions
    # (e.g. "What medications for this patient?") are not classified as vague
    classify_input = _build_classify_context(question, history)

    try:
        response = await _openai.chat.completions.create(
            model=settings.background_model,
            temperature=0.0,
            max_completion_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CLASSIFY_INTENT_PROMPT},
                {"role": "user", "content": classify_input},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        intent = str(parsed.get("intent", "clinical_query"))
        query_intent = str(parsed.get("query_intent", "general"))
        confidence = float(parsed.get("confidence", 0.8))
        clarification_question = parsed.get("clarification_question")
    except Exception as exc:
        log.warning("classify_intent failed, defaulting to clinical_query: %s", exc)
        intent = "clinical_query"
        query_intent = "general"
        confidence = 0.5
        clarification_question = None

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Build trace event
    trace = {
        "node": "classify_intent",
        "status": "done",
        "ms": elapsed_ms,
        "data": {"intent": intent, "confidence": confidence},
    }

    log.info("Intent classified: %s (%.2f) in %dms", intent, confidence, elapsed_ms)

    result: dict[str, Any] = {
        "intent": intent,
        "query_intent": query_intent,
        "intent_confidence": confidence,
        "trace_events": state.get("trace_events", []) + [trace],
    }

    # Short-circuit for non-clinical intents
    if intent == "greeting":
        result.update(
            final_answer=GREETING_RESPONSE,
            final_abstained=False,
            final_abstain_reason=None,
            final_confidence=1.0,
            chunks=[],
            used_sources=[],
            grounding_verdict="skip",
            ungrounded_claims=[],
            follow_up_questions=[
                "What are the NICE red flag symptoms for headaches?",
                "How should I manage a new diagnosis of type 2 diabetes?",
                "What are the referral criteria for suspected cancer?",
            ],
        )
    elif intent == "out_of_scope":
        result.update(
            final_answer=OUT_OF_SCOPE_RESPONSE,
            final_abstained=True,
            final_abstain_reason="Question is outside clinical scope.",
            final_confidence=1.0,
            chunks=[],
            used_sources=[],
            grounding_verdict="skip",
            ungrounded_claims=[],
            follow_up_questions=[],
        )
    elif intent == "needs_clarification":
        clarification_msg = str(clarification_question or "").strip()
        if not clarification_msg:
            clarification_msg = (
                "Could you provide more details about your clinical question? "
                "For example, which condition, symptom, or clinical scenario "
                "are you asking about?"
            )
        result.update(
            final_answer=(
                f"I'd like to help, but I need a bit more detail to find the right guideline evidence.\n\n"
                f"**{clarification_msg}**\n\n"
                f"The more specific your question, the better I can match it to NICE guidelines."
            ),
            final_abstained=False,
            final_abstain_reason=None,
            final_confidence=1.0,
            chunks=[],
            used_sources=[],
            grounding_verdict="skip",
            ungrounded_claims=[],
            follow_up_questions=[],
        )

    return result
