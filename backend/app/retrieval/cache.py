from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from uuid import UUID

from cachetools import TTLCache

from app.config import settings
from app.redis_client import close_redis_client, get_redis_client
from app.retrieval.models import RetrievalFilters, RetrievalProfile, RetrievalResult

log = logging.getLogger("clintel.retrieval")

_EMBED_DENSE_LOCAL: TTLCache[str, list[float]] = TTLCache(maxsize=1000, ttl=3600)
_EMBED_SPARSE_LOCAL: TTLCache[str, dict[str, list[int] | list[float]]] = TTLCache(maxsize=1000, ttl=3600)
_RESULT_LOCAL: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=200, ttl=900)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_key(namespace: str, payload: Any) -> str:
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"{settings.redis_key_prefix}:retrieval:{namespace}:{digest}"


class _RedisStore:
    """Thin get/set wrapper over the shared Redis client (app.redis_client).

    Unlike the crawl queue, this store is allowed to silently degrade to the local
    TTLCache — any failure to obtain a client or execute a command just means
    "no shared cache this time", never an error surfaced to the caller.
    """

    async def get(self, key: str) -> str | None:
        client = await get_redis_client()
        if client is None:
            return None
        try:
            return await client.get(key)
        except Exception as exc:
            log.warning("Redis get failed for %s: %s", key, exc)
            return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        client = await get_redis_client()
        if client is None:
            return
        try:
            await client.set(key, value, ex=ttl_seconds)
        except Exception as exc:
            log.warning("Redis set failed for %s: %s", key, exc)

    async def close(self) -> None:
        await close_redis_client()


_redis_store = _RedisStore()


class EmbeddingCache:
    def _dense_key(self, query_text: str) -> str:
        return _hash_key(
            "embed:dense",
            {
                "model": settings.embedding_model,
                "dimensions": settings.embedding_dimensions,
                "query": query_text,
            },
        )

    def _sparse_key(self, query_text: str) -> str:
        return _hash_key(
            "embed:sparse",
            {
                "model": "Qdrant/bm25",
                "query": query_text,
            },
        )

    async def get_dense(self, query_text: str) -> list[float] | None:
        key = self._dense_key(query_text)
        cached = _EMBED_DENSE_LOCAL.get(key)
        if cached is not None:
            return cached

        raw = await _redis_store.get(key)
        if raw is None:
            return None

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                dense = [float(item) for item in parsed]
                _EMBED_DENSE_LOCAL[key] = dense
                return dense
        except Exception as exc:
            log.warning("Dense embedding cache decode failed: %s", exc)
        return None

    async def set_dense(self, query_text: str, embedding: list[float]) -> None:
        key = self._dense_key(query_text)
        dense = [float(item) for item in embedding]
        _EMBED_DENSE_LOCAL[key] = dense
        await _redis_store.set(key, json.dumps(dense), ttl_seconds=3600)

    async def get_sparse(self, query_text: str) -> dict[str, list[int] | list[float]] | None:
        key = self._sparse_key(query_text)
        cached = _EMBED_SPARSE_LOCAL.get(key)
        if cached is not None:
            return cached

        raw = await _redis_store.get(key)
        if raw is None:
            return None

        try:
            parsed = json.loads(raw)
            indices = [int(item) for item in parsed.get("indices", [])]
            values = [float(item) for item in parsed.get("values", [])]
            sparse = {"indices": indices, "values": values}
            _EMBED_SPARSE_LOCAL[key] = sparse
            return sparse
        except Exception as exc:
            log.warning("Sparse embedding cache decode failed: %s", exc)
        return None

    async def set_sparse(self, query_text: str, sparse: dict[str, list[int] | list[float]]) -> None:
        key = self._sparse_key(query_text)
        payload = {
            "indices": [int(item) for item in sparse["indices"]],
            "values": [float(item) for item in sparse["values"]],
        }
        _EMBED_SPARSE_LOCAL[key] = payload
        await _redis_store.set(key, json.dumps(payload), ttl_seconds=3600)


class RetrievalCache:
    def build_key(
        self,
        *,
        tenant_id: UUID,
        query_text: str,
        filters: RetrievalFilters | None,
        profile: RetrievalProfile,
        top_k: int,
        expand_parents: bool,
        corpus_version: int,
    ) -> str:
        return _hash_key(
            "result",
            {
                "tenant_id": str(tenant_id),
                "query": query_text,
                "filters": filters.model_dump(exclude_none=True) if filters else None,
                "profile": profile,
                "top_k": top_k,
                "expand_parents": expand_parents,
                "corpus_version": corpus_version,
            },
        )

    async def get(self, cache_key: str) -> RetrievalResult | None:
        cached = _RESULT_LOCAL.get(cache_key)
        if cached is not None:
            return RetrievalResult.model_validate(cached)

        raw = await _redis_store.get(cache_key)
        if raw is None:
            return None

        try:
            parsed = json.loads(raw)
            _RESULT_LOCAL[cache_key] = parsed
            return RetrievalResult.model_validate(parsed)
        except Exception as exc:
            log.warning("Retrieval cache decode failed: %s", exc)
            return None

    async def set(self, cache_key: str, result: RetrievalResult) -> None:
        payload = result.model_dump(mode="json")
        _RESULT_LOCAL[cache_key] = payload
        await _redis_store.set(cache_key, json.dumps(payload), ttl_seconds=900)


_embedding_cache = EmbeddingCache()
_retrieval_cache = RetrievalCache()


def get_embedding_cache() -> EmbeddingCache:
    return _embedding_cache


def get_retrieval_cache() -> RetrievalCache:
    return _retrieval_cache


async def close_retrieval_cache() -> None:
    await _redis_store.close()
