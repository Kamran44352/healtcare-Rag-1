from __future__ import annotations
import datetime
import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.db import get_db
from app.pipeline.crawl_worker import dispatch_item
from app.pipeline.orchestrator import create_url_ingestion_job
from app.pipeline.scraper import ScrapeError, normalize_url, scrape_url
from app.queue import get_queue
from app.schemas.api import (
    BulkCrawlCreated,
    BulkCrawlRequest,
    CrawlBatchStatus,
    CrawlBatchSummary,
    CrawlRequest,
    DocumentRecord,
    IngestionCreated,
)

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


# ── Bulk crawl: durable multi-URL submission ─────────────────────────────────
#
# Unlike POST /crawls, this endpoint never scrapes inline. It durably persists
# the entire URL list in Postgres (crawl_batches / crawl_batch_items) and wakes
# the background crawl worker (app/pipeline/crawl_worker.py) via a Redis stream.
# The request returns as soon as the rows are written — the caller can close
# their browser immediately afterward and every URL still gets processed.

@router.post("/crawls/bulk", response_model=BulkCrawlCreated)
async def submit_bulk_crawl(req: BulkCrawlRequest):
    if not settings.firecrawl_api_key:
        raise HTTPException(status_code=503, detail="Firecrawl is not configured (set FIRECRAWL_API_KEY)")
    if len(req.urls) > settings.crawl_batch_max_urls:
        raise HTTPException(
            status_code=422,
            detail=f"Too many URLs in one submission (max {settings.crawl_batch_max_urls}, got {len(req.urls)})",
        )

    db = await get_db()
    tenant_id = settings.default_tenant_id
    interval = (
        req.rescrape_interval_hours
        if req.rescrape_interval_hours is not None
        else settings.rescrape_default_interval_hours
    )

    seen: set[str] = set()
    rows: list[dict] = []
    duplicate_count = 0

    for raw_url in req.urls:
        raw_url = raw_url.strip()
        if not raw_url:
            continue
        try:
            normalized = normalize_url(raw_url)
        except ScrapeError as exc:
            # Malformed URLs are still recorded — visible immediately as failed,
            # never silently dropped — but never touch Redis/the worker.
            rows.append({
                "tenant_id": str(tenant_id),
                "url": raw_url,
                "normalized_url": f"invalid:{uuid4()}",  # never needs dedup
                "status": "failed",
                "error_code": exc.code,
                "error_message": exc.message,
                "max_attempts": settings.crawl_batch_max_attempts,
                "finished_at": datetime.datetime.utcnow().isoformat(),
            })
            continue

        if normalized in seen:
            duplicate_count += 1
            continue
        seen.add(normalized)
        rows.append({
            "tenant_id": str(tenant_id),
            "url": raw_url,
            "normalized_url": normalized,
            "status": "queued",
            "max_attempts": settings.crawl_batch_max_attempts,
        })

    if not rows:
        raise HTTPException(status_code=422, detail="No valid URLs submitted")

    batch = (
        await db.table("crawl_batches")
        .insert({
            "tenant_id": str(tenant_id),
            "metadata": req.metadata.model_dump(),
            "rescrape_interval_hours": interval,
            "total_count": len(rows),
            "duplicate_count": duplicate_count,
        })
        .execute()
    ).data[0]
    batch_id = batch["batch_id"]

    for row in rows:
        row["batch_id"] = batch_id

    inserted = (await db.table("crawl_batch_items").insert(rows).execute()).data or []

    queued_items = [r for r in inserted if r["status"] == "queued"]
    for item in queued_items:
        await dispatch_item(UUID(item["item_id"]))

    log.info(
        "[BULK] Batch created  batch_id=%s  accepted=%d  duplicates=%d  submitted=%d",
        batch_id, len(queued_items), duplicate_count, len(req.urls),
    )

    return BulkCrawlCreated(
        batch_id=UUID(batch_id),
        accepted_count=len(queued_items),
        duplicate_count=duplicate_count,
        total_submitted=len(req.urls),
    )


@router.get("/crawls/bulk", response_model=list[CrawlBatchSummary])
async def list_bulk_crawl_batches(limit: int = 50, offset: int = 0):
    """Batch history — past and in-flight bulk submissions, most recent first."""
    db = await get_db()
    tenant_id = settings.default_tenant_id
    result = (
        await db.table("crawl_batches_with_stats")
        .select("*")
        .eq("tenant_id", str(tenant_id))
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return [
        CrawlBatchSummary.model_validate({**r, "is_finished": r["finished_at"] is not None})
        for r in (result.data or [])
    ]


@router.get("/crawls/bulk/{batch_id}", response_model=CrawlBatchStatus)
async def get_bulk_crawl_status(batch_id: UUID):
    db = await get_db()
    batch = (
        await db.table("crawl_batches_with_stats")
        .select("*")
        .eq("batch_id", str(batch_id))
        .single()
        .execute()
    ).data
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    items = (
        await db.table("crawl_batch_items")
        .select("*")
        .eq("batch_id", str(batch_id))
        .order("created_at")
        .execute()
    ).data or []

    return CrawlBatchStatus.model_validate({
        **batch,
        "is_finished": batch["finished_at"] is not None,
        "items": items,
    })
