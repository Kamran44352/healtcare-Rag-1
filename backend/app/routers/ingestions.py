from __future__ import annotations
import json
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.db import get_db
from app.pipeline.orchestrator import create_ingestion_job
from app.queue import get_queue
from app.schemas.api import IngestionCreated, IngestionStatus, IngestionMetadata

router = APIRouter()


@router.post("/ingestions", response_model=IngestionCreated)
async def submit_ingestion(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    force_reingest: bool = Form(False),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        meta_dict = json.loads(metadata)
        ing_meta = IngestionMetadata.model_validate(meta_dict)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid metadata JSON")

    tenant_id = settings.default_tenant_id
    ingestion_id, document_id, reused = await create_ingestion_job(
        pdf_bytes=pdf_bytes,
        filename=file.filename,
        metadata=ing_meta,
        force_reingest=force_reingest,
        tenant_id=tenant_id,
    )

    if not reused:
        await get_queue().enqueue(
            ingestion_id=ingestion_id,
            document_id=document_id,
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            user_metadata=ing_meta.model_dump(),
            tenant_id=tenant_id,
        )

    return IngestionCreated(
        ingestion_id=ingestion_id,
        document_id=document_id,
        status="completed" if reused else "pending",
        stage="deduped" if reused else "queued",
        reused_existing_document=reused,
    )


@router.get("/ingestions/{ingestion_id}", response_model=IngestionStatus)
async def get_ingestion(ingestion_id: UUID):
    db = await get_db()
    result = (
        await db.table("ingestions")
        .select("*")
        .eq("ingestion_id", str(ingestion_id))
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Ingestion not found")
    return IngestionStatus.model_validate(result.data)


@router.get("/ingestions")
async def list_ingestions(limit: int = 50, offset: int = 0):
    db = await get_db()
    tenant_id = settings.default_tenant_id
    result = (
        await db.table("ingestions")
        .select("*")
        .eq("tenant_id", str(tenant_id))
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    items = [IngestionStatus.model_validate(r).model_dump(mode="json") for r in (result.data or [])]
    return {"items": items, "limit": limit, "offset": offset}
