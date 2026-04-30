from __future__ import annotations
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import settings
from app.db import get_db
from app.qdrant import get_qdrant
from app.schemas.api import DocumentRecord
from app.storage import delete_file

log = logging.getLogger("clintel.pipeline")
router = APIRouter()


@router.get("/documents")
async def list_documents(limit: int = 50, offset: int = 0):
    db = await get_db()
    tenant_id = settings.default_tenant_id
    result = (
        await db.table("documents_with_status")
        .select("*")
        .eq("tenant_id", str(tenant_id))
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    items = [DocumentRecord.model_validate(r).model_dump(mode="json") for r in (result.data or [])]
    return {"items": items, "limit": limit, "offset": offset}


@router.delete("/documents/{document_id}")
async def delete_document(document_id: UUID):
    db = await get_db()
    qdrant = get_qdrant()
    tenant_id = settings.default_tenant_id

    # 1. Verify document exists and belongs to this tenant
    result = (
        await db.table("documents")
        .select("storage_path, filename")
        .eq("document_id", str(document_id))
        .eq("tenant_id", str(tenant_id))
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    storage_path = result.data["storage_path"]
    filename = result.data["filename"]

    # 2. Delete Qdrant points (filter by both document_id + tenant_id)
    await qdrant.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
                FieldCondition(key="tenant_id", match=MatchValue(value=str(tenant_id))),
            ]
        ),
        wait=True,
    )
    log.info("Deleted Qdrant points for document_id=%s", document_id)

    # 3. Delete PDF from Supabase Storage
    try:
        await delete_file(storage_path)
        log.info("Deleted storage file: %s", storage_path)
    except Exception as exc:
        log.warning("Storage delete failed (non-fatal): %s", exc)

    # 4. Delete ingestion rows for this document (schema is ON DELETE SET NULL,
    #    so they would survive as orphans — delete them explicitly for a clean wipe)
    await db.table("ingestions").delete().eq("document_id", str(document_id)).execute()

    # 5. Delete document row — cascades to parent_chunks and child_chunks
    await db.table("documents").delete().eq("document_id", str(document_id)).execute()

    # 6. Bump corpus version so retrieval caches know the corpus changed
    await db.rpc("increment_corpus_version", {"p_tenant_id": str(tenant_id)}).execute()
    log.info("Corpus version bumped after deleting document_id=%s", document_id)

    return {"document_id": str(document_id), "filename": filename, "deleted": True}
