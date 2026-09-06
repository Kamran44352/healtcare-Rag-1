from __future__ import annotations

from typing import Any

import httpx
from qdrant_client.http.models.models import ScoredPoint

from app.config import settings
from app.observability import traceable

_timeout = httpx.Timeout(timeout=12.0, connect=5.0)
_client = httpx.AsyncClient(base_url="https://api.cohere.com", timeout=_timeout)


def reranker_enabled() -> bool:
    return bool(settings.cohere_api_key)


@traceable(run_type="tool", name="cohere_rerank")
async def cohere_rerank(
    query: str,
    points: list[ScoredPoint],
    top_n: int,
) -> list[tuple[ScoredPoint, float]]:
    if not points or not reranker_enabled():
        return []

    documents = [
        str((point.payload or {}).get("full_snippet") or (point.payload or {}).get("snippet") or "")
        for point in points
    ]

    response = await _client.post(
        "/v2/rerank",
        headers={
            "Authorization": f"Bearer {settings.cohere_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.cohere_rerank_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        },
    )
    response.raise_for_status()

    payload: dict[str, Any] = response.json()
    reranked: list[tuple[ScoredPoint, float]] = []
    for item in payload.get("results", []):
        index = int(item["index"])
        if 0 <= index < len(points):
            reranked.append((points[index], float(item.get("relevance_score", 0.0))))
    return reranked


async def close_reranker() -> None:
    await _client.aclose()
