"""Rebuild Qdrant vectors directly from Postgres, without re-scraping/re-parsing.

Recovery tool for when Qdrant Cloud's free-tier cluster wipes all vector data
(e.g. after ~30 days of inactivity) while Supabase — the source of truth for
chunk text, classification, and contextualization — is untouched. Every field
Qdrant needs is already sitting in `documents` / `parent_chunks` / `child_chunks`;
the only thing that actually needs regenerating is the embedding itself. This
skips Firecrawl, LlamaParse, and every GPT metadata/enrichment/contextualization
call the original ingestion made — only OpenAI embedding calls are re-run.

Usage:
    python reindex_qdrant_from_postgres.py            # reindex every document
    python reindex_qdrant_from_postgres.py <doc_id> [<doc_id> ...]   # just these
"""
from __future__ import annotations

import asyncio
import sys
import time
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from app.config import settings
from app.db import get_db
from app.pipeline.chunking import ChildChunk
from app.pipeline.embedder import embed_children
from app.pipeline.indexer import _QDRANT_BATCH, _build_payload, _build_sparse, _upsert_batch
from app.schemas.healthcare_metadata import DocumentMetadata

_PAGE_SIZE = 1000  # PostgREST's default max-rows cap — must paginate past it explicitly


async def _fetch_all(db, table: str, columns: str, document_id: str, order_by: str | None = None) -> list[dict]:
    """Fetch every row for one document_id, paginating past PostgREST's default
    1000-row response cap. A plain .execute() with no .range() silently truncates
    at 1000 rows with no error — this bit two of the largest documents on the
    first run of this script (2142 and 1809 child_chunks respectively)."""
    rows: list[dict] = []
    offset = 0
    while True:
        query = db.table(table).select(columns).eq("document_id", document_id)
        if order_by:
            query = query.order(order_by)
        batch = (await query.range(offset, offset + _PAGE_SIZE - 1).execute()).data or []
        rows.extend(batch)
        if len(batch) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


async def reindex_document(db, qdrant: AsyncQdrantClient, doc: dict) -> dict:
    document_id = doc["document_id"]
    tenant_id = UUID(doc["tenant_id"])
    filename = doc["filename"]
    user_metadata = doc.get("metadata") or {}

    try:
        doc_metadata = DocumentMetadata.model_validate(doc.get("doc_metadata") or {})
    except Exception as exc:
        return {"document_id": document_id, "filename": filename, "skipped": True, "reason": f"bad doc_metadata: {exc}"}

    parents = await _fetch_all(db, "parent_chunks", "chunk_id, section_path", document_id)
    section_path_by_parent = {p["chunk_id"]: p["section_path"] for p in parents}

    child_rows = await _fetch_all(db, "child_chunks", "*", document_id, order_by="chunk_index")
    if not child_rows:
        return {"document_id": document_id, "filename": filename, "skipped": True, "reason": "no chunks in Postgres"}

    children = [
        ChildChunk(
            chunk_id=UUID(c["chunk_id"]),
            parent_chunk_id=UUID(c["parent_chunk_id"]),
            document_id=UUID(c["document_id"]),
            chunk_index=c["chunk_index"],
            text=c["text"],
            section_path=section_path_by_parent.get(c["parent_chunk_id"], ""),
            token_count=c["token_count"] or 0,
            contextualized_text=c["contextualized_text"] or "",
            section_type=c["section_type"],
            recommendation_strength=c["recommendation_strength"],
            entities=c["entities"] or {},
            qdrant_point_id=UUID(c["qdrant_point_id"]) if c.get("qdrant_point_id") else UUID(c["chunk_id"]),
        )
        for c in child_rows
    ]

    dense_vectors = await embed_children(children)
    sparse_vectors = [_build_sparse(c.contextualized_text or c.text) for c in children]

    points = [
        PointStruct(
            id=str(c.chunk_id),
            vector={"dense": dense, "sparse": sparse},
            payload=_build_payload(c, filename, doc_metadata, user_metadata, tenant_id),
        )
        for c, dense, sparse in zip(children, dense_vectors, sparse_vectors)
    ]

    for i in range(0, len(points), _QDRANT_BATCH):
        await _upsert_batch(qdrant, points[i : i + _QDRANT_BATCH])

    return {"document_id": document_id, "filename": filename, "points": len(points)}


async def main() -> None:
    only_ids = set(sys.argv[1:]) or None

    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=settings.qdrant_request_timeout_seconds
    )
    db = await get_db()

    query = db.table("documents").select("document_id, tenant_id, filename, metadata, doc_metadata")
    docs = (await query.execute()).data or []
    if only_ids:
        docs = [d for d in docs if d["document_id"] in only_ids]

    print(f"Reindexing {len(docs)} document(s) into Qdrant collection '{settings.qdrant_collection}'...\n")

    total_points = 0
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    t0 = time.monotonic()

    for i, doc in enumerate(docs, 1):
        label = f"[{i}/{len(docs)}] {doc['filename'][:60]!r}"
        try:
            result = await reindex_document(db, qdrant, doc)
        except Exception as exc:
            print(f"{label} FAILED: {exc}")
            failed.append((doc["document_id"], str(exc)))
            continue

        if result.get("skipped"):
            print(f"{label} SKIPPED ({result['reason']})")
            skipped.append((doc["document_id"], result["reason"]))
        else:
            total_points += result["points"]
            print(f"{label} -> {result['points']} points")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s. total_points_indexed={total_points}  skipped={len(skipped)}  failed={len(failed)}")
    if skipped:
        print("\nSkipped:")
        for doc_id, reason in skipped:
            print(f"  {doc_id}: {reason}")
    if failed:
        print("\nFailed:")
        for doc_id, reason in failed:
            print(f"  {doc_id}: {reason}")

    await qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())
