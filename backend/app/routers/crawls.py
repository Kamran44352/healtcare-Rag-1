from __future__ import annotations
import logging

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.db import get_db
from app.pipeline.orchestrator import create_url_ingestion_job
from app.pipeline.scraper import ScrapeError, scrape_url
from app.queue import get_queue
from app.schemas.api import CrawlRequest, DocumentRecord, IngestionCreated

log = logging.getLogger("clintel.pipeline")
router = APIRouter()


@router.post("/crawls", response_model=IngestionCreated)
async def submit_crawl(req: CrawlRequest):
    """Scrape a single web page with Firecrawl and ingest it like a document.

    Re-submitting the same URL re-scrapes it: unchanged content is skipped,
    changed content replaces the existing document's chunks in place.
    """
    if not settings.firecrawl_api_key:
        raise HTTPException(status_code=503, detail="Firecrawl is not configured (set FIRECRAWL_API_KEY)")

    try:
        scrape = await scrape_url(req.url)
    except ScrapeError as exc:
        status = 400 if exc.code == "INVALID_URL" else 502
        raise HTTPException(status_code=status, detail=f"[{exc.code}] {exc.message}")

    # Default the citation link to the scraped page when the caller didn't supply one.
    metadata = req.metadata.model_copy()
    if not metadata.pdf_url:
        metadata.pdf_url = scrape.source_url

    interval = (
        req.rescrape_interval_hours
        if req.rescrape_interval_hours is not None
        else settings.rescrape_default_interval_hours
    )

    tenant_id = settings.default_tenant_id
    ingestion_id, document_id, reused, _is_update = await create_url_ingestion_job(
        scrape=scrape,
        metadata=metadata,
        force_rescrape=req.force_rescrape,
        tenant_id=tenant_id,
        rescrape_interval_hours=interval,
    )

    if not reused:
        await get_queue().enqueue_url(
            ingestion_id=ingestion_id,
            document_id=document_id,
            markdown=scrape.markdown,
            filename=scrape.title or scrape.source_url,
            user_metadata=metadata.model_dump(),
            tenant_id=tenant_id,
        )

    return IngestionCreated(
        ingestion_id=ingestion_id,
        document_id=document_id,
        status="completed" if reused else "pending",
        stage="deduped" if reused else "queued",
        reused_existing_document=reused,
    )


@router.get("/crawls")
async def list_crawls(limit: int = 100, offset: int = 0):
    """List web sources (documents ingested via crawling)."""
    db = await get_db()
    tenant_id = settings.default_tenant_id
    result = (
        await db.table("documents_with_status")
        .select("*")
        .eq("tenant_id", str(tenant_id))
        .eq("source_type", "url")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    items = [DocumentRecord.model_validate(r).model_dump(mode="json") for r in (result.data or [])]
    return {"items": items, "limit": limit, "offset": offset}
