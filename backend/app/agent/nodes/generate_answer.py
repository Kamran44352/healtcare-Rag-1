"""generate_answer — produce grounded clinical answer with citation discipline."""

from __future__ import annotations
import json, logging, re, time
from typing import Any
from openai import AsyncOpenAI
from langchain_core.runnables import RunnableConfig
from app.agent.prompts import GENERATE_ANSWER_PROMPT
from app.agent.state import AgentState
from app.config import settings

log = logging.getLogger("clintel.agent.generate_answer")
_openai = AsyncOpenAI(api_key=settings.openai_api_key)


def _build_source_context(chunks: list[Any]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, 1):
        dm = c.doc_metadata or {}
        title = dm.get("guideline_title") or c.filename
        ref = dm.get("reference")
        header = f"{title} ({ref})" if ref else str(title)
        ctx = c.parent_text or c.full_snippet or c.snippet
        blocks.append("\n".join([
            f"[SOURCE {i}]", f"Title: {header}", f"Section path: {c.section_path}",
            f"Section type: {c.section_type or 'unknown'}",
            f"Recommendation strength: {c.recommendation_strength or 'unknown'}",
            f"Snippet: {c.snippet}", "Context:", str(ctx),
        ]))
    return "\n\n".join(blocks)


def _build_history_context(msgs: list[dict[str, Any]]) -> str:
    if not msgs:
        return "No prior conversation."
    lines = []
    for m in msgs[-6:]:
        r = str(m.get("role", "user")).upper()
        ct = str(m.get("content", "")).strip()
        if ct:
            lines.append(f"{r}: {ct}")
    return "\n".join(lines) if lines else "No prior conversation."


def _sanitize_follow_ups(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        t = str(item).strip()
        if t and t not in out:
            out.append(t)
        if len(out) >= 3:
            break
    return out


# ── Deterministic acronym expansion (catches what the LLM misses) ─────────

_ACRONYM_MAP: dict[str, str] = {
    "IOP":  "Intraocular Pressure",
    "FBC":  "Full Blood Count",
    "ESR":  "Erythrocyte Sedimentation Rate",
    "CRP":  "C-Reactive Protein",
    "RAPD": "Relative Afferent Pupillary Defect",
    "OCT":  "Optical Coherence Tomography",
    "VA":   "Visual Acuity",
    "VF":   "Visual Field",
    "AC":   "Anterior Chamber",
    "FFA":  "Fundus Fluorescein Angiography",
    "ICG":  "Indocyanine Green Angiography",
    "HVF":  "Humphrey Visual Field",
    "AION": "Anterior Ischaemic Optic Neuropathy",
    "GCA":  "Giant Cell Arteritis",
    "TAB":  "Temporal Artery Biopsy",
    "PACG": "Primary Angle-Closure Glaucoma",
    "PDS":  "Pigment Dispersion Syndrome",
    "APAC": "Acute Primary Angle Closure",
    "POAG": "Primary Open-Angle Glaucoma",
    "NTG":  "Normal-Tension Glaucoma",
    "LFTs": "Liver Function Tests",
    "CXR":  "Chest X-Ray",
    "ANCA": "Antineutrophil Cytoplasmic Antibody",
    "ACE":  "Angiotensin-Converting Enzyme",
    "ANA":  "Antinuclear Antibody",
    "IGRA": "Interferon-Gamma Release Assay",
    "MRI":  "Magnetic Resonance Imaging",
    "CT":   "Computed Tomography",
    "LP":   "Lumbar Puncture",
    "PCR":  "Polymerase Chain Reaction",
    "ELISA":"Enzyme-Linked Immunosorbent Assay",
    "CMO":  "Cystoid Macular Oedema",
    "PDR":  "Proliferative Diabetic Retinopathy",
    "NPDR": "Non-Proliferative Diabetic Retinopathy",
    "AMD":  "Age-Related Macular Degeneration",
    "CRVO": "Central Retinal Vein Occlusion",
    "CRAO": "Central Retinal Artery Occlusion",
    "BRVO": "Branch Retinal Vein Occlusion",
    "ALT":  "Argon Laser Trabeculoplasty",
    "SLT":  "Selective Laser Trabeculoplasty",
    "PI":   "Peripheral Iridotomy",
    "VKH":  "Vogt-Koyanagi-Harada",
    "JIA":  "Juvenile Idiopathic Arthritis",
    "GPA":  "Granulomatosis with Polyangiitis",
    "TINU": "Tubulointerstitial Nephritis and Uveitis",
    "HIV":  "Human Immunodeficiency Virus",
    "TB":   "Tuberculosis",
    "PCO":  "Posterior Capsule Opacification",
    "NVD":  "Neovascularisation of the Disc",
}

# Pre-compile a regex for each acronym — match word-boundary isolated acronyms
_ACRONYM_PATTERNS: list[tuple[re.Pattern[str], str, str]] = []
for _acr, _full in _ACRONYM_MAP.items():
    _ACRONYM_PATTERNS.append((
        re.compile(r"(?<!\()\b" + re.escape(_acr) + r"\b(?!\))"),
        _acr,
        _full,
    ))


def _expand_acronyms(text: str) -> str:
    """Ensure every medical acronym is spelled out on first use.

    Scans the answer for known acronyms. If an acronym appears without its
    expansion already present nearby, replaces the FIRST occurrence with
    'Full Name (ACRONYM)' and leaves subsequent uses as-is.
    """
    for pattern, acr, full in _ACRONYM_PATTERNS:
        # Skip if the answer already contains the expansion
        if full.lower() in text.lower():
            continue
        # Skip if the acronym isn't in the text at all
        if acr not in text:
            continue
        # Skip if it already appears as "(ACRONYM)" — means it was expanded
        if f"({acr})" in text:
            continue
        # Replace the first isolated occurrence
        text = pattern.sub(f"{full} ({acr})", text, count=1)
    return text


async def generate_answer(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    q = config.get("configurable", {}).get("event_queue")
    if q:
        await q.put(("step_started", {"node": "generate_answer"}))
        
    question = state["question"]
    history = state.get("history", [])
    chunks = state.get("chunks", [])
    started = time.perf_counter()

    if not chunks:
        # Instead of skipping the LLM, we just pass an empty source context so the LLM 
        # can follow its instruction to ask for clarification when sources are insufficient.
        src_ctx = "No relevant guidelines retrieved from the database for this specific query."
    else:
        src_ctx = _build_source_context(chunks)
    hist_ctx = _build_history_context(history)

    try:
        resp = await _openai.chat.completions.create(
            model=settings.foreground_model,
            response_format={"type": "json_object"},
            temperature=0.1, max_completion_tokens=1400,
            messages=[
                {"role": "system", "content": GENERATE_ANSWER_PROMPT},
                {"role": "user", "content": f"Conversation so far:\n{hist_ctx}\n\nCurrent user question:\n{question}\n\nRetrieved sources:\n{src_ctx}"},
            ],
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("Not a JSON object")
    except Exception as exc:
        log.error("generate_answer failed: %s", exc)
        ms = int((time.perf_counter() - started) * 1000)
        return {
            "raw_answer": "I encountered an error generating the answer. Please try again.",
            "abstained": True, "abstain_reason": f"Generation failed: {exc}",
            "confidence": 0.0, "used_sources": [], "follow_up_questions": [],
            "trace_events": state.get("trace_events", []) + [
                {"node": "generate_answer", "status": "error", "ms": ms,
                 "data": {"error": str(exc)[:100]}}],
        }

    answer = _expand_acronyms(str(parsed.get("answer", "")).strip())
    abstained = bool(parsed.get("abstained", False))
    abstain_reason = str(parsed["abstain_reason"]).strip() if parsed.get("abstain_reason") else None
    conf_raw = parsed.get("confidence")
    try:
        conf = max(0.0, min(1.0, float(conf_raw))) if conf_raw is not None else None
    except (TypeError, ValueError):
        conf = None

    used: list[int] = []
    if isinstance(parsed.get("used_sources"), list):
        seen: set[int] = set()
        for s in parsed["used_sources"]:
            try:
                v = int(s)
            except (TypeError, ValueError):
                continue
            if v not in seen and 1 <= v <= len(chunks):
                seen.add(v); used.append(v)

    fups = _sanitize_follow_ups(parsed.get("follow_up_questions"))
    ms = int((time.perf_counter() - started) * 1000)

    log.info("Answer generated in %dms: conf=%.2f, sources=%d", ms, conf or 0, len(used))

    return {
        "raw_answer": answer, "abstained": abstained, "abstain_reason": abstain_reason,
        "confidence": conf, "used_sources": used, "follow_up_questions": fups,
        "trace_events": state.get("trace_events", []) + [
            {"node": "generate_answer", "status": "done", "ms": ms,
             "data": {"confidence": round(conf or 0, 2), "abstained": abstained,
                      "sources_used": len(used), "answer_length": len(answer)}}],
    }
