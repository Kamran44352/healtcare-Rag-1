from __future__ import annotations
import asyncio
import json
import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.schemas.healthcare_metadata import DocumentMetadata

if TYPE_CHECKING:
    from app.pipeline.chunking import ChildChunk

log = logging.getLogger("clintel.pipeline")
_client = AsyncOpenAI(api_key=settings.openai_api_key)
_LLM_BATCH = 20

_SECTION_TYPES = (
    "recommendation", "evidence", "background", "dosing-table",
    "red-flag", "contraindication", "monitoring", "referral", "other"
)
_STRENGTHS = ("strong", "conditional", "informative")

_SYSTEM_PROMPT = """You are a healthcare NLP assistant. For each chunk of clinical text, return:
- section_type: one of {section_types}
- recommendation_strength: one of {strengths} when section_type is "recommendation", else null
- entities: medical entities found in the text, with codes where confident
  - drugs: list of {{label, code (ATC code or null), system ("ATC" or "free-text")}}
  - conditions: list of {{label, code (ICD-10 or SNOMED-CT or null), system ("ICD-10" or "SNOMED-CT" or "free-text")}}
  - procedures: list of {{label, code (SNOMED-CT or null), system ("SNOMED-CT" or "free-text")}}
  Only include entities actually mentioned in the text. Use null for code when unsure — never guess codes.

Return JSON exactly as:
{{"results": [{{"section_type": "...", "recommendation_strength": "...", "entities": {{"drugs": [], "conditions": [], "procedures": []}}}}, ...]}}
One object per input chunk, in the same order. No markdown fences.""".format(
    section_types=str(_SECTION_TYPES), strengths=str(_STRENGTHS)
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
async def _enrich_batch(
    batch: list["ChildChunk"],
    doc_type: str,
) -> list[dict]:
    items = [
        {"index": i, "section": c.section_path, "text": c.text[:600]}
        for i, c in enumerate(batch)
    ]
    response = await _client.chat.completions.create(
        model=settings.background_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Document type: {doc_type}\n\n"
                    f"Chunks:\n{json.dumps(items, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=0,
        max_tokens=2048,
    )
    raw = json.loads(response.choices[0].message.content)
    if isinstance(raw, dict):
        if "results" in raw and isinstance(raw["results"], list):
            return raw["results"]
        for v in raw.values():
            if isinstance(v, list):
                return v
    return raw if isinstance(raw, list) else []


async def enrich_chunks(
    children: list["ChildChunk"],
    doc_metadata: DocumentMetadata,
) -> None:
    """Stage F: LLM-based classification + entity extraction in a single batched call."""
    doc_type = doc_metadata.document_type
    batches = [children[i : i + _LLM_BATCH] for i in range(0, len(children), _LLM_BATCH)]
    tasks = [_enrich_batch(batch, doc_type) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for batch, result in zip(batches, results):
        if isinstance(result, Exception):
            log.warning("Chunk enrichment batch failed: %s", result)
            continue
        for child, item in zip(batch, result):
            if not isinstance(item, dict):
                continue

            st = item.get("section_type")
            if st in _SECTION_TYPES:
                child.section_type = st

            rs = item.get("recommendation_strength")
            if rs in _STRENGTHS:
                child.recommendation_strength = rs

            ents = item.get("entities")
            if isinstance(ents, dict):
                child.entities = {
                    "drugs": ents.get("drugs") or [],
                    "conditions": ents.get("conditions") or [],
                    "procedures": ents.get("procedures") or [],
                }
