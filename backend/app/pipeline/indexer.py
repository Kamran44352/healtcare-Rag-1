from __future__ import annotations
import asyncio
import logging
from uuid import UUID

import httpx
from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import PointStruct, SparseVector
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.db import get_db
from app.observability import traceable
from app.pipeline.chunking import ChildChunk, ParentChunk
from app.schemas.healthcare_metadata import DocumentMetadata

log = logging.getLogger("clintel.pipeline")

_QDRANT_BATCH = 100
_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# Cap concurrent Qdrant writes across all in-flight pipelines. Many pipelines
# upserting at once over one shared HTTP/2 connection triggers stream resets
# (PROTOCOL_ERROR) and read timeouts on Qdrant Cloud. Serialising to a small
# number keeps each request healthy while still allowing parallel ingestion.
_qdrant_write_sem = asyncio.Semaphore(settings.qdrant_write_concurrency)


def _is_transient_qdrant_error(exc: BaseException) -> bool:
    """Retry network-level Qdrant failures (timeouts, HTTP/2 stream resets,
    connection drops) and 5xx responses — but never client (4xx) errors."""
    if isinstance(exc, (ResponseHandlingException, httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code >= 500
    return False


@retry(
    retry=retry_if_exception(_is_transient_qdrant_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
@traceable(run_type="tool", name="qdrant_upsert")
async def _upsert_batch(qdrant: AsyncQdrantClient, batch: list[PointStruct]) -> None:
    async with _qdrant_write_sem:
        await qdrant.upsert(
            collection_name=settings.qdrant_collection,
            points=batch,
            wait=True,
        )


def _build_sparse(text: str) -> SparseVector:
    result = list(_sparse_model.embed([text]))[0]
    return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())


def _build_payload(
    child: ChildChunk,
    filename: str,
    doc_metadata: DocumentMetadata,
    user_metadata: dict,
    tenant_id: UUID,
) -> dict:
    return {
        "tenant_id": str(tenant_id),
        "document_id": str(child.document_id),
        "parent_chunk_id": str(child.parent_chunk_id),
        "filename": filename,
        "section_path": child.section_path,
        "section_type": child.section_type,
        "recommendation_strength": child.recommendation_strength,
        "entities": child.entities,
        # Indexed payload fields for filtered retrieval
        "doc_type": doc_metadata.document_type,
        "specialties": doc_metadata.specialties,
        "conditions_codes": [c.code for c in doc_metadata.conditions if c.code],
        "geographic_scope": doc_metadata.geographic_scope,
        "has_dosing_tables": doc_metadata.has_dosing_tables,
        "has_red_flags": doc_metadata.has_red_flags,
        "has_referral_criteria": doc_metadata.has_referral_criteria,
        "issuing_body": doc_metadata.issuing_body,
        "care_setting": doc_metadata.care_setting,
        # User-supplied metadata
        "category": user_metadata.get("category"),
        "reference": user_metadata.get("reference"),
        "guideline_title": user_metadata.get("guideline_title"),
        "pdf_url": user_metadata.get("pdf_url"),
        # For citation rendering
        "snippet": child.text[:300],
        "full_snippet": child.text,
        "corpus_version": 1,
    }


async def index_document(
    qdrant: AsyncQdrantClient,
    document_id: UUID,
    filename: str,
    parents: list[ParentChunk],
    dense_vectors: list[list[float]],
    doc_metadata: DocumentMetadata,
    user_metadata: dict,
    tenant_id: UUID,
) -> dict:
    """Stages I+J: BM25 sparse embed + Qdrant upsert + Postgres bulk inserts."""
    db = await get_db()

    all_children = [child for p in parents for child in p.children]
    assert len(all_children) == len(dense_vectors), "Vector count mismatch"

    # ── Stage I: BM25 sparse vectors ─────────────────────────────────────
    sparse_vectors = [
        _build_sparse(c.contextualized_text or c.text) for c in all_children
    ]

    # ── Stage J.1: Qdrant upsert ──────────────────────────────────────────
    points = [
        PointStruct(
            id=str(child.chunk_id),
            vector={"dense": dense, "sparse": sparse},
            payload=_build_payload(child, filename, doc_metadata, user_metadata, tenant_id),
        )
        for child, dense, sparse in zip(all_children, dense_vectors, sparse_vectors)
    ]

    for i in range(0, len(points), _QDRANT_BATCH):
        await _upsert_batch(qdrant, points[i : i + _QDRANT_BATCH])

    # ── Stage J.2: Postgres parent_chunks ────────────────────────────────
    parent_rows = [
        {
            "chunk_id": str(p.chunk_id),
            "document_id": str(p.document_id),
            "chunk_index": p.chunk_index,
            "text": p.text,
            "section_path": p.section_path,
            "token_count": p.token_count,
            "tenant_id": str(tenant_id),
        }
        for p in parents
    ]
    await db.table("parent_chunks").insert(parent_rows).execute()

    # ── Stage J.3: Postgres child_chunks ─────────────────────────────────
    child_rows = [
        {
            "chunk_id": str(c.chunk_id),
            "parent_chunk_id": str(c.parent_chunk_id),
            "document_id": str(c.document_id),
            "chunk_index": c.chunk_index,
            "text": c.text,
            "contextualized_text": c.contextualized_text,
            "section_type": c.section_type,
            "recommendation_strength": c.recommendation_strength,
            "entities": c.entities,
            "qdrant_point_id": str(c.qdrant_point_id) if c.qdrant_point_id else None,
            "token_count": c.token_count,
            "tenant_id": str(tenant_id),
        }
        for c in all_children
    ]
    await db.table("child_chunks").insert(child_rows).execute()

    return {
        "parent_chunks": len(parents),
        "child_chunks": len(all_children),
        "qdrant_points": len(points),
    }
