from __future__ import annotations

import asyncio
import datetime
import logging
from uuid import UUID

from app.config import settings
from app.db import get_db
from app.observability import traceable
from app.pipeline.orchestrator import create_url_ingestion_job
from app.pipeline.scraper import ScrapeError, scrape_url
from app.queue import get_queue
from app.request_context import new_context_id
from app.schemas.api import IngestionMetadata

log = logging.getLogger("clintel.scheduler")


async def _claim_due_documents() -> list[dict]:
    """Fetch web sources whose re-scrape is due and immediately push their
    next_rescrape_at forward so a concurrent worker won't pick them up again."""
    db = await get_db()
    now = datetime.datetime.utcnow().isoformat()
    tenant_id = settings.default_tenant_id

    due = (
        await db.table("documents")
        .select("document_id, source_url, metadata, rescrape_interval_hours")
        .eq("tenant_id", str(tenant_id))
        .eq("source_type", "url")
        .gt("rescrape_interval_hours", 0)
        .lte("next_rescrape_at", now)
        .execute()
    ).data or []

    claimed: list[dict] = []
    for doc in due:
        interval = doc.get("rescrape_interval_hours") or 0
        next_due = (
            datetime.datetime.utcnow() + datetime.timedelta(hours=interval)
        ).isoformat()
        # Claim: move the due time forward before doing slow work.
        await db.table("documents").update({"next_rescrape_at": next_due}).eq(
            "document_id", doc["document_id"]
        ).execute()
        claimed.append(doc)
    return claimed


@traceable(run_type="chain", name="rescrape_document")
async def _rescrape_one(doc: dict) -> None:
    new_context_id("rescrape")
    url = doc["source_url"]
    interval = doc.get("rescrape_interval_hours") or settings.rescrape_default_interval_hours
    try:
        metadata = IngestionMetadata.model_validate(doc.get("metadata") or {})
    except Exception:
        metadata = IngestionMetadata()

    try:
        scrape = await scrape_url(url)
    except ScrapeError as exc:
        log.warning("[RESCRAPE] Skipped %s — scrape failed [%s] %s", url, exc.code, exc.message)
        return

    if not metadata.pdf_url:
        metadata.pdf_url = scrape.source_url

    ingestion_id, document_id, reused, is_update = await create_url_ingestion_job(
        scrape=scrape,
        metadata=metadata,
        force_rescrape=False,  # change-detection decides whether to re-index
        tenant_id=settings.default_tenant_id,
        rescrape_interval_hours=interval,
    )

    if reused:
        log.info("[RESCRAPE] %s unchanged — no re-index", url)
        return

    await get_queue().enqueue_url(
        ingestion_id=ingestion_id,
        document_id=document_id,
        markdown=scrape.markdown,
        filename=scrape.title or scrape.source_url,
        user_metadata=metadata.model_dump(),
        tenant_id=settings.default_tenant_id,
    )
    log.info("[RESCRAPE] %s changed — re-indexing (document_id=%s)", url, document_id)


async def _tick() -> None:
    try:
        due = await _claim_due_documents()
    except Exception as exc:
        log.warning("[RESCRAPE] Failed to query due documents: %s", exc)
        return

    if not due:
        return
    log.info("[RESCRAPE] %d web source(s) due for re-scrape", len(due))
    for doc in due:
        try:
            await _rescrape_one(doc)
        except Exception as exc:
            log.warning("[RESCRAPE] Error re-scraping %s: %s", doc.get("source_url"), exc)


async def rescrape_loop() -> None:
    """Background loop: periodically re-scrape web sources whose interval has elapsed."""
    interval = settings.rescrape_check_interval_seconds
    log.info("[RESCRAPE] Auto re-scrape loop started (every %ds)", interval)
    while True:
        await _tick()
        await asyncio.sleep(interval)


_task: asyncio.Task[None] | None = None


def start_scheduler() -> None:
    global _task
    if not settings.rescrape_enabled:
        log.info("[RESCRAPE] Auto re-scrape disabled (RESCRAPE_ENABLED=false)")
        return
    if not settings.firecrawl_api_key:
        log.info("[RESCRAPE] Auto re-scrape inactive — FIRECRAWL_API_KEY not set")
        return
    if _task is None or _task.done():
        _task = asyncio.create_task(rescrape_loop())


async def stop_scheduler() -> None:
    global _task
    if _task is not None and not _task.done():
        _task.cancel()
        try:
            await _task
        except asyncio.CancelledError:
            pass
    _task = None
