from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import UUID

from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models.models import ScoredPoint
from qdrant_client.models import Fusion, FusionQuery, Prefetch

from app.db import get_db
from app.qdrant import get_qdrant
from app.config import settings
from app.retrieval.cache import get_retrieval_cache
from app.retrieval.embedder import query_embed_dense, query_embed_sparse
from app.retrieval.filters import build_qdrant_filter
from app.retrieval.models import RetrievedChunk, RetrievalFilters, RetrievalProfile, RetrievalResult
from app.retrieval.reranker import cohere_rerank, reranker_enabled

log = logging.getLogger("clintel.retrieval")

_PROFILE_CONFIG: dict[str, dict[str, float | int]] = {
    # Default profiles
    "specific": {"candidates": 50, "rerank_limit": 10, "min_rerank_score": 0.25},
    "narrow": {"candidates": 30, "rerank_limit": 8, "min_rerank_score": 0.35},
    "broad": {"candidates": 100, "rerank_limit": 15, "min_rerank_score": 0.15},
    
    # Intent-based profiles (Adaptive rerank threshold)
    "definition": {"candidates": 100, "rerank_limit": 12, "min_rerank_score": 0.18},
    "mechanism": {"candidates": 100, "rerank_limit": 12, "min_rerank_score": 0.20},
    "differential": {"candidates": 100, "rerank_limit": 12, "min_rerank_score": 0.25},
    "treatment": {"candidates": 100, "rerank_limit": 10, "min_rerank_score": 0.30},
    "prognosis": {"candidates": 100, "rerank_limit": 10, "min_rerank_score": 0.20},
    "investigations": {"candidates": 100, "rerank_limit": 10, "min_rerank_score": 0.25},
    "general": {"candidates": 100, "rerank_limit": 15, "min_rerank_score": 0.20},
}

_DOC_METADATA_KEYS = {
    "doc_type",
    "specialties",
    "conditions_codes",
    "geographic_scope",
    "care_setting",
    "has_dosing_tables",
    "has_red_flags",
    "has_referral_criteria",
    "issuing_body",
    "category",
    "reference",
    "guideline_title",
    "pdf_url",
}


class RetrievalUnavailableError(Exception):
    pass


async def _get_corpus_version(tenant_id: UUID) -> int:
    db = await get_db()
    try:
        result = (
            await db.table("corpus_config")
            .select("corpus_version")
            .eq("tenant_id", str(tenant_id))
            .single()
            .execute()
        )
        return int((result.data or {}).get("corpus_version", 1))
    except Exception as exc:
        log.warning("Falling back to corpus_version=1: %s", exc)
        return 1


def _extract_doc_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in _DOC_METADATA_KEYS if key in payload}


def _dedupe_best_child_per_parent(points: list[ScoredPoint]) -> list[ScoredPoint]:
    best_by_parent: dict[str, ScoredPoint] = {}

    for point in points:
        payload = point.payload or {}
        parent_chunk_id = str(payload.get("parent_chunk_id") or "")
        if not parent_chunk_id:
            continue

        current = best_by_parent.get(parent_chunk_id)
        if current is None or float(point.score or 0.0) > float(current.score or 0.0):
            best_by_parent[parent_chunk_id] = point

    return sorted(best_by_parent.values(), key=lambda item: float(item.score or 0.0), reverse=True)


async def _fetch_parent_texts(parent_ids: list[str]) -> dict[str, str]:
    if not parent_ids:
        return {}

    db = await get_db()
    result = await db.table("parent_chunks").select("chunk_id,text").in_("chunk_id", parent_ids).execute()
    return {
        str(row["chunk_id"]): str(row["text"])
        for row in (result.data or [])
        if row.get("chunk_id") and row.get("text")
    }


def _build_retrieved_chunk(
    point: ScoredPoint,
    *,
    parent_map: dict[str, str],
    rerank_score: float | None,
) -> RetrievedChunk:
    payload = point.payload or {}
    chunk_id = str(point.id)
    parent_chunk_id = str(payload["parent_chunk_id"])
    full_snippet = str(payload.get("full_snippet") or payload.get("snippet") or "")
    snippet = str(payload.get("snippet") or full_snippet[:300])

    return RetrievedChunk(
        chunk_id=UUID(chunk_id),
        parent_chunk_id=UUID(parent_chunk_id),
        document_id=UUID(str(payload["document_id"])),
        filename=str(payload.get("filename") or ""),
        section_path=str(payload.get("section_path") or ""),
        section_type=payload.get("section_type"),
        recommendation_strength=payload.get("recommendation_strength"),
        entities=payload.get("entities") or {},
        snippet=snippet,
        full_snippet=full_snippet,
        parent_text=parent_map.get(parent_chunk_id),
        doc_metadata=_extract_doc_metadata(payload),
        dense_score=None,
        fused_score=float(point.score or 0.0),
        rerank_score=rerank_score,
    )


async def retrieve(
    query: str,
    *,
    tenant_id: UUID,
    filters: RetrievalFilters | None = None,
    profile: RetrievalProfile = "specific",
    top_k: int = 10,
    expand_parents: bool = True,
) -> RetrievalResult:
    started_at = time.perf_counter()
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Query must not be empty.")
    profile_key = profile if profile in _PROFILE_CONFIG else "specific"
    config = _PROFILE_CONFIG[profile_key]
    rerank_limit = int(config["rerank_limit"])
    candidates = int(config["candidates"])
    min_rerank_score = float(config["min_rerank_score"])
    final_limit = min(max(1, top_k), rerank_limit)

    corpus_version = await _get_corpus_version(tenant_id)

    retrieval_cache = get_retrieval_cache()
    cache_key = retrieval_cache.build_key(
        tenant_id=tenant_id,
        query_text=normalized_query,
        filters=filters,
        profile=profile,
        top_k=final_limit,
        expand_parents=expand_parents,
        corpus_version=corpus_version,
    )
    cached = await retrieval_cache.get(cache_key)
    if cached is not None:
        return cached.model_copy(
            update={
                "cache_hit": True,
                "query_ms": int((time.perf_counter() - started_at) * 1000),
            }
        )

    dense_vector, sparse_vector = await asyncio.gather(
        query_embed_dense(normalized_query),
        query_embed_sparse(normalized_query),
    )

    qdrant = get_qdrant()
    qdrant_filter = build_qdrant_filter(tenant_id, filters)

    try:
        hybrid_response = await qdrant.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=candidates),
                Prefetch(query=sparse_vector, using="sparse", limit=candidates),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=candidates,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
            timeout=settings.qdrant_retrieval_timeout_seconds,
        )
        deduped_points = _dedupe_best_child_per_parent(list(hybrid_response.points))
    except ResponseHandlingException as exc:
        raise RetrievalUnavailableError(
            "Hybrid Qdrant search timed out. Increase Qdrant timeout or improve Qdrant query latency."
        ) from exc

    reranked_points: list[tuple[ScoredPoint, float | None]]
    rerank_succeeded = False
    if reranker_enabled() and deduped_points:
        try:
            reranked = await cohere_rerank(
                normalized_query,
                deduped_points,
                top_n=min(len(deduped_points), rerank_limit),
            )
            rerank_succeeded = True
            reranked_points = [
                (point, score)
                for point, score in reranked
                if score >= min_rerank_score
            ]
        except Exception as exc:
            log.warning("Cohere rerank failed, falling back to fused ranking: %s", exc)
            reranked_points = []
    else:
        reranked_points = []

    if reranked_points:
        selected_points = reranked_points[:final_limit]
    elif rerank_succeeded:
        selected_points = []
    else:
        selected_points = [(point, None) for point in deduped_points[:final_limit]]

    parent_map: dict[str, str] = {}
    if expand_parents:
        parent_ids = list(
            {
                str((point.payload or {}).get("parent_chunk_id"))
                for point, _ in selected_points
                if (point.payload or {}).get("parent_chunk_id")
            }
        )
        parent_map = await _fetch_parent_texts(parent_ids)

    chunks = [
        _build_retrieved_chunk(
            point,
            parent_map=parent_map,
            rerank_score=score,
        )
        for point, score in selected_points
    ]

    result = RetrievalResult(
        chunks=chunks,
        query_ms=int((time.perf_counter() - started_at) * 1000),
        corpus_version=corpus_version,
        cache_hit=False,
    )
    await retrieval_cache.set(cache_key, result)
    return result


async def retrieve_multi(
    queries: list[str],
    rerank_query: str,
    *,
    tenant_id: UUID,
    query_intent: str = "general",
    filters: RetrievalFilters | None = None,
    top_k: int = 10,
    expand_parents: bool = True,
) -> RetrievalResult:
    """Run parallel retrieval for multiple query variants and rerank the merged results."""
    started_at = time.perf_counter()
    
    # Use intent-based profile, fallback to general
    profile_key = query_intent if query_intent in _PROFILE_CONFIG else "general"
    config = _PROFILE_CONFIG[profile_key]
    rerank_limit = int(config["rerank_limit"])
    candidates = int(config["candidates"])
    min_rerank_score = float(config["min_rerank_score"])
    final_limit = min(max(1, top_k), rerank_limit)

    corpus_version = await _get_corpus_version(tenant_id)

    # Fast path if only 1 query
    if len(queries) == 1:
        return await retrieve(
            queries[0],
            tenant_id=tenant_id,
            filters=filters,
            profile=profile_key,  # type: ignore
            top_k=top_k,
            expand_parents=expand_parents,
        )

    # 1. Parallel dense/sparse embedding for all queries
    embed_tasks = []
    for q in queries:
        norm_q = q.strip()
        if norm_q:
            embed_tasks.append(query_embed_dense(norm_q))
            embed_tasks.append(query_embed_sparse(norm_q))
            
    if not embed_tasks:
        return RetrievalResult(chunks=[], query_ms=0, corpus_version=corpus_version, cache_hit=False)

    embed_results = await asyncio.gather(*embed_tasks)
    
    # 2. Parallel Qdrant queries
    qdrant = get_qdrant()
    qdrant_filter = build_qdrant_filter(tenant_id, filters)
    
    qdrant_tasks = []
    # embed_results alternates [dense0, sparse0, dense1, sparse1, ...]
    for i in range(0, len(embed_results), 2):
        dense_vec = embed_results[i]
        sparse_vec = embed_results[i+1]
        
        task = qdrant.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=candidates),
                Prefetch(query=sparse_vec, using="sparse", limit=candidates),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=candidates,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
            timeout=settings.qdrant_retrieval_timeout_seconds,
        )
        qdrant_tasks.append(task)
        
    try:
        qdrant_results = await asyncio.gather(*qdrant_tasks)
    except ResponseHandlingException as exc:
        raise RetrievalUnavailableError(
            "Hybrid Qdrant search timed out. Increase Qdrant timeout or improve Qdrant query latency."
        ) from exc

    # 3. Merge and dedupe
    all_points = []
    for res in qdrant_results:
        all_points.extend(res.points)
        
    deduped_points = _dedupe_best_child_per_parent(all_points)

    # 4. Rerank the merged set
    reranked_points: list[tuple[ScoredPoint, float | None]]
    rerank_succeeded = False
    if reranker_enabled() and deduped_points:
        try:
            reranked = await cohere_rerank(
                rerank_query,
                deduped_points,
                top_n=min(len(deduped_points), rerank_limit),
            )
            rerank_succeeded = True
            reranked_points = [
                (point, score)
                for point, score in reranked
                if score >= min_rerank_score
            ]
        except Exception as exc:
            log.warning("Cohere rerank failed, falling back to fused ranking: %s", exc)
            reranked_points = []
    else:
        reranked_points = []

    # 5. Trim and fetch parent texts
    if reranked_points:
        selected_points = reranked_points[:final_limit]
    elif rerank_succeeded:
        selected_points = []
    else:
        selected_points = [(point, None) for point in deduped_points[:final_limit]]

    parent_map: dict[str, str] = {}
    if expand_parents:
        parent_ids = list(
            {
                str((point.payload or {}).get("parent_chunk_id"))
                for point, _ in selected_points
                if (point.payload or {}).get("parent_chunk_id")
            }
        )
        parent_map = await _fetch_parent_texts(parent_ids)

    chunks = [
        _build_retrieved_chunk(
            point,
            parent_map=parent_map,
            rerank_score=score,
        )
        for point, score in selected_points
    ]

    return RetrievalResult(
        chunks=chunks,
        query_ms=int((time.perf_counter() - started_at) * 1000),
        corpus_version=corpus_version,
        cache_hit=False,
    )
