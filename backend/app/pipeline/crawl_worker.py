from __future__ import annotations

import asyncio
import datetime
import json
import logging
import random
from uuid import UUID

from app.config import settings
from app.db import get_db
from app.observability import (
    TRACE_HEADER_FIELD,
    current_trace_headers,
    trace_context_from_headers,
    traceable,
)
from app.pipeline.orchestrator import IngestionError, create_url_ingestion_job
from app.pipeline.scraper import ScrapeError, scrape_url
from app.queue import get_queue
from app.redis_client import get_queue_redis_client
from app.request_context import new_context_id
from app.schemas.api import IngestionMetadata

log = logging.getLogger("clintel.crawl_worker")

# Error codes that re-scraping/re-processing the same URL cannot fix — these go
# straight to 'failed' rather than being retried.
_PERMANENT_ERROR_CODES = {"INVALID_URL", "NO_API_KEY", "EMPTY_CONTENT", "LOW_TEXT_DENSITY"}


def stream_key() -> str:
    return f"{settings.redis_key_prefix}:{settings.crawl_stream_name}"


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


# ── Enqueue helper (used by the router and by this module's own recovery paths) ──

async def enqueue_item(item_id: UUID) -> bool:
    """Push one crawl_batch_items id onto the Redis stream to wake a worker.

    Returns False (never raises) if Redis is unavailable — a failed enqueue is
    never fatal to the batch: the reconciliation sweep re-enqueues any 'queued'
    item that doesn't get picked up in time, so durability never depends on this
    call succeeding.
    """
    client = await get_queue_redis_client()
    if client is None:
        return False
    try:
        fields: dict[str, str] = {"item_id": str(item_id)}
        # Carry the caller's trace context through the stream so the worker can
        # re-parent to it — otherwise the trace ends at XADD and the item's
        # ingestion shows up as an unrelated root run.
        headers = current_trace_headers()
        if headers:
            fields[TRACE_HEADER_FIELD] = json.dumps(headers)
        await client.xadd(
            stream_key(),
            fields,
            maxlen=settings.crawl_stream_maxlen,
            approximate=True,
        )
        return True
    except Exception as exc:
        log.warning("[BULK] Failed to enqueue item %s onto stream: %s", item_id, exc)
        return False


async def _ensure_group(client) -> None:
    try:
        await client.xgroup_create(
            name=stream_key(),
            groupname=settings.crawl_stream_consumer_group,
            id="0",
            mkstream=True,
        )
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            log.warning("[BULK] xgroup_create failed: %s", exc)


# ── Error classification ─────────────────────────────────────────────────────

def _classify_error(exc: Exception) -> tuple[str, str, bool]:
    """Returns (error_code, error_message, retryable)."""
    if isinstance(exc, (ScrapeError, IngestionError)):
        retryable = exc.code not in _PERMANENT_ERROR_CODES
        return exc.code, exc.message, retryable
    return "INTERNAL_ERROR", str(exc) or exc.__class__.__name__, True


async def _delayed_requeue(item_id: UUID, delay_seconds: float) -> None:
    await asyncio.sleep(delay_seconds)
    await dispatch_item(item_id)


async def dispatch_item(item_id: UUID) -> None:
    """Hand an item off for processing: Redis Streams is the fast path, but if
    Redis is unavailable this falls back to processing the item directly (as a
    background task) instead of doing nothing. This is what makes the durability
    guarantee hold even through a full Redis outage, not just lost stream state
    — Postgres status is what's ever relied on for correctness; Redis is purely
    an accelerator. Used both by the bulk-submit router (initial dispatch) and
    by the reconciliation sweep (recovery dispatch)."""
    ok = await enqueue_item(item_id)
    if not ok:
        # Same bounded path the dispatcher uses, so a Redis outage can't spawn
        # an unbounded number of concurrent pipelines (a 500-URL batch would
        # otherwise fan out 500 tasks at once).
        _spawn_item(item_id, "reconciler-direct", None)


# ── Per-item processing ───────────────────────────────────────────────────────

@traceable(run_type="chain", name="crawl_item")
async def _process_item(item_id: UUID, consumer_name: str) -> None:
    # Give every item its own correlation id so all the log lines it produces
    # (scrape, parse, chunk, embed, index) can be grepped as one unit of work.
    new_context_id("crawl")
    db = await get_db()

    # Atomic conditional claim — race-safe against duplicate delivery (a retry
    # racing the original delivery, or the reconciliation sweep racing a live
    # worker): only one caller can flip 'queued' -> 'processing' for this row.
    claimed = (
        await db.table("crawl_batch_items")
        .update({"status": "processing", "started_at": _now_iso(), "claimed_by": consumer_name})
        .eq("item_id", str(item_id))
        .eq("status", "queued")
        .execute()
    )
    if not claimed.data:
        # Lost the claim race (duplicate delivery, or the reconciliation sweep
        # racing a live worker). A no-op by design, but log it at debug — it was
        # previously a completely silent return, which made duplicate-delivery
        # behaviour impossible to observe.
        log.debug("[BULK] %s: item %s already claimed elsewhere", consumer_name, item_id)
        return

    item = claimed.data[0]
    url = item["url"]
    batch_id = item["batch_id"]
    tenant_id = UUID(item["tenant_id"])
    attempt = int(item["attempt_count"]) + 1
    max_attempts = int(item["max_attempts"])

    try:
        batch_row = (
            await db.table("crawl_batches")
            .select("metadata, rescrape_interval_hours")
            .eq("batch_id", batch_id)
            .single()
            .execute()
        ).data or {}
        try:
            metadata = IngestionMetadata.model_validate(batch_row.get("metadata") or {})
        except Exception:
            metadata = IngestionMetadata()

        scrape = await scrape_url(url)
        if not metadata.pdf_url:
            metadata.pdf_url = scrape.source_url

        ingestion_id, document_id, reused, _is_update = await create_url_ingestion_job(
            scrape=scrape,
            metadata=metadata,
            force_rescrape=False,
            tenant_id=tenant_id,
            rescrape_interval_hours=batch_row.get("rescrape_interval_hours"),
        )

        await db.table("crawl_batch_items").update({
            "ingestion_id": str(ingestion_id),
            "document_id": str(document_id),
        }).eq("item_id", str(item_id)).execute()

        if not reused:
            await get_queue().run_url_pipeline_bounded(
                ingestion_id=ingestion_id,
                document_id=document_id,
                markdown=scrape.markdown,
                filename=scrape.title or scrape.source_url,
                user_metadata=metadata.model_dump(),
                tenant_id=tenant_id,
            )
            # run_url_pipeline swallows IngestionError internally (it records the
            # failure on the `ingestions` row but does not raise) — check the
            # outcome explicitly so a quality-gate/pipeline failure still surfaces
            # here and gets classified/retried like any other error.
            ingestion_row = (
                await db.table("ingestions")
                .select("status, error_code, error_message")
                .eq("ingestion_id", str(ingestion_id))
                .single()
                .execute()
            ).data or {}
            if ingestion_row.get("status") == "failed":
                raise IngestionError(
                    ingestion_row.get("error_code") or "INTERNAL_ERROR",
                    ingestion_row.get("error_message") or "Pipeline failed",
                )

        await db.table("crawl_batch_items").update({
            "status": "completed",
            "attempt_count": attempt,
            "finished_at": _now_iso(),
        }).eq("item_id", str(item_id)).execute()
        log.info("[BULK] Item completed  item_id=%s  url=%s", item_id, url)

    except Exception as exc:
        code, message, retryable = _classify_error(exc)
        if retryable and attempt < max_attempts:
            delay = min(
                settings.crawl_batch_retry_backoff_base_seconds * (2 ** (attempt - 1)),
                settings.crawl_batch_retry_backoff_max_seconds,
            )
            next_attempt_at = (
                datetime.datetime.utcnow() + datetime.timedelta(seconds=delay)
            ).isoformat()
            await db.table("crawl_batch_items").update({
                "status": "queued",
                "attempt_count": attempt,
                "error_code": code,
                "error_message": message,
                "next_attempt_at": next_attempt_at,
                "claimed_by": None,
            }).eq("item_id", str(item_id)).execute()
            asyncio.create_task(_delayed_requeue(item_id, delay))
            log.warning(
                "[BULK] Item failed, retry %d/%d in %.0fs  item_id=%s  url=%s  [%s] %s",
                attempt, max_attempts, delay, item_id, url, code, message,
            )
        else:
            await db.table("crawl_batch_items").update({
                "status": "failed",
                "attempt_count": attempt,
                "error_code": code,
                "error_message": message,
                "finished_at": _now_iso(),
            }).eq("item_id", str(item_id)).execute()
            log.error(
                "[BULK] Item permanently failed  item_id=%s  url=%s  [%s] %s",
                item_id, url, code, message,
            )
        # Never re-raise — the consumer loop must move on to the next item
        # unconditionally, regardless of what happened to this one.


# ── Dispatcher ────────────────────────────────────────────────────────────────
#
# ONE task reads the stream; processing concurrency is enforced by a semaphore
# instead of by the number of readers. The previous design ran N blocking
# consumers, which meant N Upstash connections were permanently parked inside a
# blocking XREADGROUP — starving the pool that the retrieval cache also used, and
# producing a continuous timeout/warning loop even with an empty stream.

# Bounds how many items are processed at once. Deliberately the same knob as
# before (`url_ingestion_concurrency`), so the effective concurrency is unchanged
# — only the number of held Redis connections drops, from N to 1.
_worker_sem: asyncio.Semaphore | None = None
_inflight: set[asyncio.Task[None]] = set()


def _get_worker_sem() -> asyncio.Semaphore:
    """The concurrency gate, created on first use.

    Lazily built because `dispatch_item` can run from a request handler before
    (or without) `start_crawl_worker` — e.g. when Redis is down and the router
    falls back to direct processing.
    """
    global _worker_sem
    if _worker_sem is None:
        _worker_sem = asyncio.Semaphore(max(1, settings.url_ingestion_concurrency))
    return _worker_sem


async def _run_item(item_id: UUID, consumer_name: str, headers: dict[str, str] | None) -> None:
    """Process one item under the concurrency semaphore, re-parented to the
    trace that enqueued it."""
    async with _get_worker_sem():
        try:
            with trace_context_from_headers(headers):
                await _process_item(item_id, consumer_name)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[BULK] Unhandled error processing item %s", item_id)


def _spawn_item(item_id: UUID, consumer_name: str, headers: dict[str, str] | None) -> None:
    task = asyncio.create_task(_run_item(item_id, consumer_name, headers))
    # Hold a strong reference: asyncio only keeps weak refs to running tasks, so
    # an unreferenced task can be garbage-collected mid-flight.
    _inflight.add(task)
    task.add_done_callback(_inflight.discard)


async def _dispatcher(stop_event: asyncio.Event) -> None:
    client = await get_queue_redis_client()
    if client is None:
        log.warning("[BULK] Dispatcher exiting — Redis unavailable (Postgres reconciliation still active)")
        return

    try:
        from redis.exceptions import TimeoutError as RedisTimeoutError
    except Exception:  # pragma: no cover
        RedisTimeoutError = ()  # type: ignore[assignment]

    await _ensure_group(client)
    consumer_name = "dispatcher-0"
    log.info(
        "[BULK] Dispatcher started (block=%dms, max concurrent items=%d)",
        settings.crawl_stream_block_ms,
        settings.url_ingestion_concurrency,
    )

    backoff = settings.crawl_stream_error_backoff_base_seconds
    consecutive_errors = 0

    while not stop_event.is_set():
        try:
            resp = await client.xreadgroup(
                groupname=settings.crawl_stream_consumer_group,
                consumername=consumer_name,
                streams={stream_key(): ">"},
                # Drain up to a full concurrency window per read instead of one
                # message per round trip.
                count=max(1, settings.url_ingestion_concurrency),
                block=settings.crawl_stream_block_ms,
            )
        except asyncio.CancelledError:
            raise
        except RedisTimeoutError:
            # EXPECTED, not an error: a blocking read that reaches its window
            # with no messages is the normal idle path. Treating this as a
            # failure (warn + sleep) is what produced the ~7s warning loop.
            log.debug("[BULK] xreadgroup idle timeout — no messages")
            continue
        except Exception as exc:
            consecutive_errors += 1
            # Log the first failure and then only on backoff escalation, so a
            # sustained outage cannot flood the log the way it did before.
            if consecutive_errors == 1 or backoff >= settings.crawl_stream_error_backoff_max_seconds:
                log.warning(
                    "[BULK] xreadgroup failed (%d consecutive): %s — retrying in %.1fs",
                    consecutive_errors, exc, backoff,
                )
            # Jitter so multiple replicas don't reconnect in lockstep.
            await asyncio.sleep(backoff * (0.5 + random.random()))
            backoff = min(backoff * 2, settings.crawl_stream_error_backoff_max_seconds)
            continue

        if consecutive_errors:
            log.info("[BULK] Stream read recovered after %d consecutive errors", consecutive_errors)
            consecutive_errors = 0
            backoff = settings.crawl_stream_error_backoff_base_seconds

        if not resp:
            continue

        for _stream_name, messages in resp:
            for message_id, fields in messages:
                # Ack immediately on receipt. Correctness lives in Postgres, not
                # the Redis Pending Entries List: a message being "unacked" here
                # would only matter if we relied on PEL/XAUTOCLAIM for crash
                # recovery, and we deliberately don't — the reconciliation sweep
                # (below) requeues anything that doesn't reach a terminal status
                # in Postgres, so acking early can never lose a URL.
                try:
                    await client.xack(stream_key(), settings.crawl_stream_consumer_group, message_id)
                except Exception as exc:
                    log.debug("[BULK] xack failed for %s: %s", message_id, exc)

                raw_item_id = fields.get("item_id")
                if not raw_item_id:
                    continue
                try:
                    item_id = UUID(raw_item_id)
                except ValueError:
                    log.warning("[BULK] Skipping message %s — malformed item_id %r", message_id, raw_item_id)
                    continue

                headers: dict[str, str] | None = None
                raw_headers = fields.get(TRACE_HEADER_FIELD)
                if raw_headers:
                    try:
                        headers = json.loads(raw_headers)
                    except Exception:
                        headers = None

                # Hand off without awaiting, so the dispatcher goes straight back
                # to reading. The semaphore inside `_run_item` is what caps
                # actual concurrency.
                _spawn_item(item_id, consumer_name, headers)

    log.info("[BULK] Dispatcher stopped")


# ── Reconciliation sweep (Postgres-only safety net) ──────────────────────────

@traceable(run_type="chain", name="crawl_reconcile")
async def _reconcile_once() -> None:
    new_context_id("reconcile")
    db = await get_db()
    now = datetime.datetime.utcnow()
    queued_cutoff = (now - datetime.timedelta(seconds=settings.crawl_batch_queued_grace_seconds)).isoformat()
    stale_cutoff = (now - datetime.timedelta(seconds=settings.crawl_batch_stale_processing_seconds)).isoformat()
    now_iso = now.isoformat()

    # 1. 'queued' items whose original XADD may never have landed (e.g. the
    #    request crashed between the Postgres insert and the XADD).
    stuck_queued = (
        await db.table("crawl_batch_items")
        .select("item_id")
        .eq("status", "queued")
        .lte("created_at", queued_cutoff)
        .is_("next_attempt_at", "null")
        .limit(200)
        .execute()
    ).data or []
    for row in stuck_queued:
        await dispatch_item(UUID(row["item_id"]))

    # 1b. Backoff-scheduled retries whose delay has elapsed — covers the in-process
    #     _delayed_requeue task itself dying with the process (e.g. Railway restart
    #     mid-backoff-sleep).
    due_retries = (
        await db.table("crawl_batch_items")
        .select("item_id")
        .eq("status", "queued")
        .lte("next_attempt_at", now_iso)
        .limit(200)
        .execute()
    ).data or []
    for row in due_retries:
        await dispatch_item(UUID(row["item_id"]))

    # 2. 'processing' items stuck past the staleness threshold — the worker that
    #    claimed them is presumed dead. Reset to 'queued' and re-enqueue.
    stale_processing = (
        await db.table("crawl_batch_items")
        .select("item_id")
        .eq("status", "processing")
        .lte("started_at", stale_cutoff)
        .limit(200)
        .execute()
    ).data or []
    for row in stale_processing:
        item_id = UUID(row["item_id"])
        reset = (
            await db.table("crawl_batch_items")
            .update({
                "status": "queued",
                "claimed_by": None,
                "error_code": "WORKER_TIMEOUT",
                "error_message": "Processing did not finish in time — requeued.",
            })
            .eq("item_id", str(item_id))
            .eq("status", "processing")
            .execute()
        )
        if reset.data:
            await dispatch_item(item_id)
            log.warning("[BULK] Reclaimed stale processing item %s", item_id)

    # 3. Close out batches whose items are all terminal.
    open_batches = (
        await db.table("crawl_batches")
        .select("batch_id")
        .is_("finished_at", "null")
        .limit(200)
        .execute()
    ).data or []
    for row in open_batches:
        batch_id = row["batch_id"]
        remaining = (
            await db.table("crawl_batch_items")
            .select("item_id", count="exact")
            .eq("batch_id", batch_id)
            .in_("status", ["queued", "processing"])
            .limit(1)
            .execute()
        )
        if (remaining.count or 0) == 0:
            await db.table("crawl_batches").update({"finished_at": _now_iso()}).eq(
                "batch_id", batch_id
            ).execute()


async def _reconcile_loop(stop_event: asyncio.Event) -> None:
    interval = settings.crawl_batch_reconciliation_interval_seconds
    log.info("[BULK] Reconciliation loop started (every %ds)", interval)
    while not stop_event.is_set():
        try:
            await _reconcile_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[BULK] Reconciliation sweep failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


# ── Lifespan wiring ───────────────────────────────────────────────────────────

_tasks: list[asyncio.Task[None]] = []
_stop_event: asyncio.Event | None = None


def start_crawl_worker() -> None:
    global _stop_event
    if not settings.firecrawl_api_key:
        log.info("[BULK] Crawl worker inactive — FIRECRAWL_API_KEY not set")
        return
    if not settings.redis_url:
        log.info("[BULK] Crawl worker inactive — REDIS_URL not set")
        return
    if _tasks:
        return

    _stop_event = asyncio.Event()
    _get_worker_sem()
    _tasks.append(asyncio.create_task(_dispatcher(_stop_event)))
    _tasks.append(asyncio.create_task(_reconcile_loop(_stop_event)))
    log.info("[BULK] Crawl worker started")


async def stop_crawl_worker() -> None:
    global _tasks, _stop_event
    if _stop_event is not None:
        _stop_event.set()

    # Let in-flight items finish rather than cancelling them mid-pipeline — an
    # item cancelled while 'processing' would sit there until the staleness
    # sweep reclaims it (up to crawl_batch_stale_processing_seconds later).
    if _inflight:
        log.info("[BULK] Waiting for %d in-flight item(s) to finish", len(_inflight))
        done, pending = await asyncio.wait(set(_inflight), timeout=30)
        if pending:
            log.warning("[BULK] %d item(s) still running at shutdown — cancelling", len(pending))
            for task in pending:
                task.cancel()

    for task in _tasks:
        task.cancel()
    for task in _tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("[BULK] Error while stopping crawl worker task")
    _tasks = []
    _stop_event = None
