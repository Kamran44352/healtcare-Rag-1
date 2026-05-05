"""verify_grounding — check every [SOURCE n] claim is actually supported."""

from __future__ import annotations
import json, logging, time, re
from typing import Any
from openai import AsyncOpenAI
from langchain_core.runnables import RunnableConfig
from app.agent.prompts import VERIFY_GROUNDING_PROMPT
from app.agent.state import AgentState
from app.config import settings

log = logging.getLogger("clintel.agent.verify_grounding")
_openai = AsyncOpenAI(api_key=settings.openai_api_key)
_SOURCE_RE = re.compile(r"\[SOURCE\s+(\d+)\]", re.IGNORECASE)


def _build_source_ref(chunks: list[Any]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, 1):
        dm = c.doc_metadata or {}
        title = dm.get("guideline_title") or c.filename
        ctx = c.parent_text or c.full_snippet or c.snippet
        blocks.append(f"[SOURCE {i}] {title}\n{ctx[:600]}")
    return "\n\n".join(blocks)


async def verify_grounding(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    q = config.get("configurable", {}).get("event_queue")
    if q:
        await q.put(("step_started", {"node": "verify_grounding"}))
        
    raw_answer = state.get("raw_answer", "")
    chunks = state.get("chunks", [])
    abstained = state.get("abstained", False)
    abstain_reason = state.get("abstain_reason")
    confidence = state.get("confidence")
    started = time.perf_counter()

    # If already abstained, or no answer, or no chunks, skip verification
    # We pass through the LLM's raw_answer and abstained state
    if abstained or not raw_answer.strip() or not chunks:
        ms = int((time.perf_counter() - started) * 1000)
        return {
            "grounding_verdict": "skip",
            "ungrounded_claims": [],
            "final_answer": raw_answer,
            "final_abstained": abstained,
            "final_abstain_reason": abstain_reason,
            "final_confidence": confidence or 0.0,
            "trace_events": state.get("trace_events", []) + [
                {"node": "verify_grounding", "status": "done", "ms": ms,
                 "data": {"verdict": "skip", "reason": "Already abstained"}}],
        }

    # Check if there are any SOURCE citations to verify
    source_refs = _SOURCE_RE.findall(raw_answer)
    if not source_refs:
        ms = int((time.perf_counter() - started) * 1000)
        return {
            "grounding_verdict": "pass",
            "ungrounded_claims": [],
            "final_answer": raw_answer,
            "final_abstained": abstained,
            "final_abstain_reason": abstain_reason,
            "final_confidence": confidence,
            "trace_events": state.get("trace_events", []) + [
                {"node": "verify_grounding", "status": "done", "ms": ms,
                 "data": {"verdict": "pass", "reason": "No citations to verify"}}],
        }

    source_ref_text = _build_source_ref(chunks)

    try:
        resp = await _openai.chat.completions.create(
            model=settings.background_model,
            temperature=0.0, max_completion_tokens=500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": VERIFY_GROUNDING_PROMPT},
                {"role": "user", "content": f"Answer to verify:\n{raw_answer}\n\nSource texts:\n{source_ref_text}"},
            ],
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        verdict = str(parsed.get("verdict", "pass"))
        ungrounded = list(parsed.get("ungrounded_claims", []))
        summary = str(parsed.get("summary", ""))
    except Exception as exc:
        log.warning("verify_grounding failed, passing through: %s", exc)
        verdict = "pass"
        ungrounded = []
        summary = "Verification failed, passing through."

    ms = int((time.perf_counter() - started) * 1000)

    # Decide final output based on verdict
    if verdict == "fail":
        final_answer = (
            "I could not produce a reliably grounded answer from the retrieved guidelines. "
            "The evidence retrieved does not adequately support a confident response."
        )
        final_abstained = True
        final_reason = f"Grounding check failed: {len(ungrounded)} claims unsupported. {summary}"
        final_conf = 0.0
    elif verdict == "partial":
        final_answer = raw_answer
        final_abstained = False
        final_reason = abstain_reason
        # Reduce confidence for partial grounding
        final_conf = min((confidence or 0.5) * 0.8, 0.6)
    else:  # pass
        final_answer = raw_answer
        final_abstained = abstained
        final_reason = abstain_reason
        final_conf = confidence

    trace = {
        "node": "verify_grounding",
        "status": "done",
        "ms": ms,
        "data": {
            "verdict": verdict,
            "ungrounded_count": len(ungrounded),
            "summary": summary[:120],
        },
    }

    log.info("Grounding verdict: %s in %dms (%d ungrounded)", verdict, ms, len(ungrounded))

    return {
        "grounding_verdict": verdict,
        "ungrounded_claims": ungrounded,
        "final_answer": final_answer,
        "final_abstained": final_abstained,
        "final_abstain_reason": final_reason,
        "final_confidence": final_conf,
        "trace_events": state.get("trace_events", []) + [trace],
    }
