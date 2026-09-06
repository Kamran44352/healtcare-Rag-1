from __future__ import annotations
import asyncio
import json
from typing import TYPE_CHECKING

from app.llm import chat_completion
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.schemas.healthcare_metadata import DocumentMetadata

if TYPE_CHECKING:
    from app.pipeline.chunking import ParentChunk

_BATCH = 20  # parents per LLM call

_SYSTEM_PROMPT = """Generate a brief contextual prefix (≤60 tokens) for each document chunk.
The prefix should describe WHERE this chunk sits in the document and WHAT it covers.
Format: "From [title] ([ref]) > [section]: [one sentence summary]"

Return JSON exactly as: {"prefixes": ["prefix for chunk 0", "prefix for chunk 1", ...]}
One string per input chunk, in the same order. No markdown fences."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
async def _generate_prefixes(
    batch: list["ParentChunk"],
    title: str,
    reference_id: str | None,
) -> list[str]:
    ref = f"({reference_id})" if reference_id else ""
    items = [
        {
            "index": i,
            "section": p.section_path,
            "preview": p.text[:300],
        }
        for i, p in enumerate(batch)
    ]
    response = await chat_completion(
        model=settings.background_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Document: {title} {ref}\n\n"
                    f"Chunks:\n{json.dumps(items, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=0,
        max_tokens=2048,
        timeout=settings.llm_call_timeout_seconds,
    )
    raw = json.loads(response.choices[0].message.content)
    # Handle {"prefixes": [...]} or any dict containing a list, or bare list
    if isinstance(raw, dict):
        # Try "prefixes" key first, then any list value
        if "prefixes" in raw and isinstance(raw["prefixes"], list):
            return [str(x) for x in raw["prefixes"]]
        for v in raw.values():
            if isinstance(v, list):
                return [str(x) for x in v]
        # Fallback: dict with string values keyed by index
        if all(isinstance(v, str) for v in raw.values()):
            return [raw[str(i)] for i in range(len(raw)) if str(i) in raw]
    return [str(x) for x in raw] if isinstance(raw, list) else []


async def add_context_prefixes(
    parents: list["ParentChunk"],
    doc_metadata: DocumentMetadata,
) -> None:
    """Stage G: generates a contextual prefix per parent, prepends to each child."""
    title = doc_metadata.title
    ref = doc_metadata.reference_id

    batches = [parents[i : i + _BATCH] for i in range(0, len(parents), _BATCH)]
    tasks = [_generate_prefixes(batch, title, ref) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for batch, prefixes in zip(batches, results):
        if isinstance(prefixes, Exception):
            prefixes = []
        for parent, prefix in zip(batch, prefixes + [""] * len(batch)):
            for child in parent.children:
                child.contextualized_text = f"{prefix}\n\n{child.text}".strip()
