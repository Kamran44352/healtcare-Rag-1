from __future__ import annotations

import asyncio

from fastembed import SparseTextEmbedding
from app.llm import openai_client
from qdrant_client.models import SparseVector
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.observability import traceable
from app.retrieval.cache import get_embedding_cache

_openai = openai_client
_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
async def _create_dense_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a batch in ONE API call.

    The embeddings endpoint accepts a list, and one batched call costs about the
    same as one single-text call. Issuing N concurrent calls instead meant the
    caller waited on the SLOWEST of N — measured at 2.2s, 2.4s and 5.7s for the
    same three query variants, so the tail dominated retrieval latency.
    """
    response = await _openai.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        dimensions=settings.embedding_dimensions,
    )
    # The API may return items out of order; `index` is authoritative.
    ordered = sorted(response.data, key=lambda d: d.index)
    return [d.embedding for d in ordered]


async def _create_dense_embedding(text: str) -> list[float]:
    return (await _create_dense_embeddings([text]))[0]


def _build_sparse_embedding(text: str) -> SparseVector:
    result = next(_sparse_model.embed([text]))
    return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())


@traceable(run_type="tool", name="dense_embed")
async def query_embed_dense(text: str) -> list[float]:
    cache = get_embedding_cache()
    cached = await cache.get_dense(text)
    if cached is not None:
        return cached

    embedding = await _create_dense_embedding(text)
    await cache.set_dense(text, embedding)
    return embedding


@traceable(run_type="tool", name="sparse_embed")
async def query_embed_sparse(text: str) -> SparseVector:
    cache = get_embedding_cache()
    cached = await cache.get_sparse(text)
    if cached is not None:
        return SparseVector(
            indices=[int(item) for item in cached["indices"]],
            values=[float(item) for item in cached["values"]],
        )

    loop = asyncio.get_running_loop()
    sparse = await loop.run_in_executor(None, _build_sparse_embedding, text)
    await cache.set_sparse(
        text,
        {"indices": list(sparse.indices), "values": list(sparse.values)},
    )
    return sparse


@traceable(run_type="tool", name="dense_embed_batch")
async def query_embed_dense_many(texts: list[str]) -> list[list[float]]:
    """Cache-aware batched dense embedding, preserving input order.

    Cached entries are served locally; only the misses go to OpenAI, and they go
    as a single request rather than one concurrent request per text.
    """
    if not texts:
        return []

    cache = get_embedding_cache()
    results: list[list[float] | None] = await asyncio.gather(*(cache.get_dense(t) for t in texts))

    missing_idx = [i for i, r in enumerate(results) if r is None]
    if missing_idx:
        fresh = await _create_dense_embeddings([texts[i] for i in missing_idx])
        for i, emb in zip(missing_idx, fresh):
            results[i] = emb
        await asyncio.gather(*(cache.set_dense(texts[i], results[i]) for i in missing_idx))  # type: ignore[arg-type]

    return [r for r in results if r is not None]
