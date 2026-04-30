from __future__ import annotations

import asyncio

from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI
from qdrant_client.models import SparseVector
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.retrieval.cache import get_embedding_cache

_openai = AsyncOpenAI(api_key=settings.openai_api_key)
_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
async def _create_dense_embedding(text: str) -> list[float]:
    response = await _openai.embeddings.create(
        model=settings.embedding_model,
        input=[text],
        dimensions=settings.embedding_dimensions,
    )
    return response.data[0].embedding


def _build_sparse_embedding(text: str) -> SparseVector:
    result = next(_sparse_model.embed([text]))
    return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())


async def query_embed_dense(text: str) -> list[float]:
    cache = get_embedding_cache()
    cached = await cache.get_dense(text)
    if cached is not None:
        return cached

    embedding = await _create_dense_embedding(text)
    await cache.set_dense(text, embedding)
    return embedding


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
