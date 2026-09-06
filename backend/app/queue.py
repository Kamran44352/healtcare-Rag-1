from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from app.config import settings

log = logging.getLogger("clintel.queue")


class IngestionQueue:
    def __init__(self):
        self._sem = asyncio.Semaphore(settings.ingestion_concurrency)
        self._url_sem = asyncio.Semaphore(settings.url_ingestion_concurrency)
        self._tasks: set[asyncio.Task[None]] = set()

    async def enqueue(
        self,
        ingestion_id: UUID,
        document_id: UUID,
        pdf_bytes: bytes,
        filename: str,
        user_metadata: dict,
        tenant_id: UUID,
    ) -> None:
        task = asyncio.create_task(
            self._run(ingestion_id, document_id, pdf_bytes, filename, user_metadata, tenant_id)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._consume_task_result)

    async def _run(
        self,
        ingestion_id: UUID,
        document_id: UUID,
        pdf_bytes: bytes,
        filename: str,
        user_metadata: dict,
        tenant_id: UUID,
    ) -> None:
        from app.pipeline.orchestrator import run_pipeline

        async with self._sem:
            await run_pipeline(
                ingestion_id=ingestion_id,
                document_id=document_id,
                pdf_bytes=pdf_bytes,
                filename=filename,
                user_metadata=user_metadata,
                tenant_id=tenant_id,
            )

    async def enqueue_url(
        self,
        ingestion_id: UUID,
        document_id: UUID,
        markdown: str,
        filename: str,
        user_metadata: dict,
        tenant_id: UUID,
    ) -> None:
        task = asyncio.create_task(
            self._run_url(ingestion_id, document_id, markdown, filename, user_metadata, tenant_id)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._consume_task_result)

    async def _run_url(
        self,
        ingestion_id: UUID,
        document_id: UUID,
        markdown: str,
        filename: str,
        user_metadata: dict,
        tenant_id: UUID,
    ) -> None:
        from app.pipeline.orchestrator import run_url_pipeline

        async with self._url_sem:
            await run_url_pipeline(
                ingestion_id=ingestion_id,
                document_id=document_id,
                markdown=markdown,
                filename=filename,
                user_metadata=user_metadata,
                tenant_id=tenant_id,
            )

    async def run_url_pipeline_bounded(
        self,
        ingestion_id: UUID,
        document_id: UUID,
        markdown: str,
        filename: str,
        user_metadata: dict,
        tenant_id: UUID,
    ) -> None:
        """Like enqueue_url, but awaits completion under the shared _url_sem instead
        of fire-and-forget. Used by the bulk crawl worker (app/pipeline/crawl_worker.py),
        which needs to know when the pipeline actually finishes so it can record the
        outcome on the batch item — and shares this semaphore so bulk crawling never
        exceeds the same total URL-pipeline concurrency as manual crawls/re-scrape."""
        from app.pipeline.orchestrator import run_url_pipeline

        async with self._url_sem:
            await run_url_pipeline(
                ingestion_id=ingestion_id,
                document_id=document_id,
                markdown=markdown,
                filename=filename,
                user_metadata=user_metadata,
                tenant_id=tenant_id,
            )

    @staticmethod
    def _consume_task_result(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            log.info("Ingestion task cancelled")
        except Exception:
            # run_pipeline already records the failure and logs the traceback
            pass


_queue: IngestionQueue | None = None


def get_queue() -> IngestionQueue:
    global _queue
    if _queue is None:
        _queue = IngestionQueue()
    return _queue
