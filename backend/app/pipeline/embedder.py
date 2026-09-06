from __future__ import annotations
from typing import TYPE_CHECKING

from app.llm import openai_client
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

if TYPE_CHECKING:
    from app.pipeline.chunking import ChildChunk

_client = openai_client
_BATCH_SIZE = 96


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    response = await _client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        dimensions=settings.embedding_dimensions,
    )
    # Response items are ordered by index
    items = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in items]


async def embed_children(children: list["ChildChunk"]) -> list[list[float]]:
    """
    Stage H: Embed all child chunks using OpenAI text-embedding-3-large (1536d).
    Returns vectors in the same order as the input list.
    """
    texts = [c.contextualized_text or c.text for c in children]
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        vecs = await _embed_batch(batch)
        all_vectors.extend(vecs)

    return all_vectors
