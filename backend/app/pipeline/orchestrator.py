from __future__ import annotations
import datetime
import hashlib
import logging
import time
from uuid import UUID, uuid4

from app.config import settings
from app.db import get_db
from app.qdrant import get_qdrant
from app.pipeline import (
    chunking,
    contextualize,
    embedder,
    indexer,
    metadata_chunk,
    metadata_doc,
    parser,
    pdf_splitter,
)
from app.schemas.api import IngestionMetadata
from app.storage import upload_pdf

log = logging.getLogger("clintel.pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


class IngestionError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


async def _update_stage(ingestion_id: UUID, stage: str, status: str = "processing") -> None:
    db = await get_db()
    await db.table("ingestions").update({"stage": stage, "status": status}).eq(
        "ingestion_id", str(ingestion_id)
    ).execute()


async def _fail(ingestion_id: UUID, code: str, message: str) -> None:
    db = await get_db()
    await db.table("ingestions").update({
        "status": "failed",
        "error_code": code,
        "error_message": message,
        "finished_at": datetime.datetime.utcnow().isoformat(),
    }).eq("ingestion_id", str(ingestion_id)).execute()


# ── Stage A ───────────────────────────────────────────────────────────────

async def create_ingestion_job(
    pdf_bytes: bytes,
    filename: str,
    metadata: IngestionMetadata,
    force_reingest: bool,
    tenant_id: UUID,
) -> tuple[UUID, UUID, bool]:
    db = await get_db()
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()

    log.info("[A] Checking dedup  file=%s  sha256=%s…", filename, sha256[:12])

    existing = (
        await db.table("documents").select("document_id").eq("sha256_hash", sha256).execute()
    ).data

    if existing and not force_reingest:
        doc_id = UUID(existing[0]["document_id"])
        ing_id = uuid4()
        log.info("[A] Duplicate detected — reusing document_id=%s", doc_id)
        await db.table("ingestions").insert({
            "ingestion_id": str(ing_id),
            "document_id": str(doc_id),
            "filename": filename,
            "status": "completed",
            "stage": "deduped",
            "tenant_id": str(tenant_id),
        }).execute()
        return ing_id, doc_id, True

    doc_id = uuid4()
    log.info("[A] New document  document_id=%s  size=%.1f KB", doc_id, len(pdf_bytes) / 1024)
    storage_path = await upload_pdf(tenant_id, doc_id, filename, pdf_bytes)
    if settings.store_pdf_in_bucket:
        log.info("[A] Uploaded to storage  path=%s", storage_path)
    else:
        log.info("[A] Skipped storage upload (STORE_PDF_IN_BUCKET is false)")

    await db.table("documents").insert({
        "document_id": str(doc_id),
        "filename": filename,
        "sha256_hash": sha256,
        "metadata": metadata.model_dump(),
        "storage_path": storage_path,
        "tenant_id": str(tenant_id),
    }).execute()

    ing_id = uuid4()
    await db.table("ingestions").insert({
        "ingestion_id": str(ing_id),
        "document_id": str(doc_id),
        "filename": filename,
        "status": "pending",
        "stage": "queued",
        "tenant_id": str(tenant_id),
    }).execute()

    return ing_id, doc_id, False


# ── Stages B→K ────────────────────────────────────────────────────────────

async def run_pipeline(
    ingestion_id: UUID,
    document_id: UUID,
    pdf_bytes: bytes,
    filename: str,
    user_metadata: dict,
    tenant_id: UUID,
) -> None:
    db = await get_db()
    t0 = time.monotonic()

    log.info("=" * 60)
    log.info("PIPELINE START  file=%s  ingestion=%s", filename, str(ingestion_id)[:8])
    log.info("=" * 60)

    await db.table("ingestions").update({
        "status": "processing",
        "started_at": datetime.datetime.utcnow().isoformat(),
    }).eq("ingestion_id", str(ingestion_id)).execute()

    try:
        # ── Split PDF into page-range batches ───────────────────────
        batches = pdf_splitter.split_pdf_bytes(pdf_bytes)
        log.info(
            "[BATCH] %d batch(es) of ≤200 pages  total_size=%.1f KB",
            len(batches),
            len(pdf_bytes) / 1024,
        )

        # ── Stage B: Parse first batch (for quality check + metadata) ──
        t_stage = time.monotonic()
        log.info("[B] Parsing batch 1/%d with LlamaParse…", len(batches))
        await _update_stage(ingestion_id, "parsing")
        batch0_result = await parser.parse(batches[0], filename)
        log.info(
            "[B] Batch 1 parsed  provider=%s  pages=%d  chars=%d  (%.1fs)",
            batch0_result.provider,
            batch0_result.page_count,
            len(batch0_result.markdown),
            time.monotonic() - t_stage,
        )

        # ── Stage C: Quality gate (based on first batch) ────────────
        await _update_stage(ingestion_id, "quality_check")
        chars_per_page_0 = len(batch0_result.markdown) / max(batch0_result.page_count, 1)
        log.info("[C] Quality check  chars_per_page=%.0f", chars_per_page_0)
        if chars_per_page_0 < 100 or len(batch0_result.markdown) < 200:
            raise IngestionError(
                "LOW_TEXT_DENSITY",
                f"Insufficient extractable text ({chars_per_page_0:.0f} chars/page). "
                "Document may be scanned-only.",
            )
        log.info("[C] Quality OK")

        # ── Stage D: Document metadata (from first batch only) ──────
        t_stage = time.monotonic()
        log.info("[D] Extracting document metadata with %s…", settings.foreground_model)
        await _update_stage(ingestion_id, "extracting_metadata")
        doc_meta = await metadata_doc.extract_doc_metadata(batch0_result.markdown)
        log.info(
            "[D] Metadata extracted  title=%r  type=%s  issuing_body=%s  "
            "conditions=%d  drugs=%d  (%.1fs)",
            doc_meta.title,
            doc_meta.document_type,
            doc_meta.issuing_body,
            len(doc_meta.conditions),
            len(doc_meta.drugs),
            time.monotonic() - t_stage,
        )

        # ── Stages E-J: Process each batch ──────────────────────────
        accumulated = {"qdrant_points": 0, "parent_chunks": 0, "child_chunks": 0}
        total_page_count = 0
        total_chars = 0
        all_warnings: list[str] = []
        qdrant = get_qdrant()

        for batch_idx, batch_bytes in enumerate(batches):
            label = f"batch {batch_idx + 1}/{len(batches)}"
            log.info("[BATCH] Starting E→J  %s", label)

            if batch_idx == 0:
                batch_result = batch0_result
            else:
                t_stage = time.monotonic()
                log.info("[B] Parsing %s…", label)
                await _update_stage(ingestion_id, "parsing")
                batch_result = await parser.parse(batch_bytes, filename)
                log.info(
                    "[B] Parsed OK  %s  pages=%d  chars=%d  (%.1fs)",
                    label,
                    batch_result.page_count,
                    len(batch_result.markdown),
                    time.monotonic() - t_stage,
                )

            total_page_count += batch_result.page_count
            total_chars += len(batch_result.markdown)
            all_warnings.extend(batch_result.warnings)

            # Stage E: Chunking
            t_stage = time.monotonic()
            await _update_stage(ingestion_id, "chunking")
            parents = chunking.chunk_document(batch_result.markdown, document_id)
            all_children = [c for p in parents for c in p.children]
            log.info(
                "[E] Chunked  %s  parents=%d  children=%d  (%.1fs)",
                label, len(parents), len(all_children), time.monotonic() - t_stage,
            )

            # Stage F: Chunk enrichment
            t_stage = time.monotonic()
            await _update_stage(ingestion_id, "enriching_chunks")
            await metadata_chunk.enrich_chunks(all_children, doc_meta)
            classified = sum(1 for c in all_children if c.section_type)
            log.info(
                "[F] Enriched  %s  classified=%d/%d  (%.1fs)",
                label, classified, len(all_children), time.monotonic() - t_stage,
            )

            # Stage G: Contextual prefixes
            t_stage = time.monotonic()
            await _update_stage(ingestion_id, "contextualizing")
            await contextualize.add_context_prefixes(parents, doc_meta)
            log.info("[G] Context prefixes done  %s  (%.1fs)", label, time.monotonic() - t_stage)

            # Stage H: Embed
            t_stage = time.monotonic()
            await _update_stage(ingestion_id, "embedding")
            dense_vectors = await embedder.embed_children(all_children)
            log.info(
                "[H] Embedded  %s  vectors=%d  (%.1fs)",
                label, len(dense_vectors), time.monotonic() - t_stage,
            )

            # Stages I+J: Index
            t_stage = time.monotonic()
            await _update_stage(ingestion_id, "indexing")
            stats = await indexer.index_document(
                qdrant=qdrant,
                document_id=document_id,
                filename=filename,
                parents=parents,
                dense_vectors=dense_vectors,
                doc_metadata=doc_meta,
                user_metadata=user_metadata,
                tenant_id=tenant_id,
            )
            log.info(
                "[I/J] Indexed  %s  qdrant_points=%d  parent_chunks=%d  child_chunks=%d  (%.1fs)",
                label,
                stats["qdrant_points"],
                stats["parent_chunks"],
                stats["child_chunks"],
                time.monotonic() - t_stage,
            )

            accumulated["qdrant_points"] += stats["qdrant_points"]
            accumulated["parent_chunks"] += stats["parent_chunks"]
            accumulated["child_chunks"] += stats["child_chunks"]

            # Release this batch's markdown from memory before the next batch
            del batch_result, parents, all_children, dense_vectors

        # ── Update documents table with full accumulated data ────────
        await db.table("documents").update({
            "page_count": total_page_count,
            "parser_provider": batch0_result.provider,
            "parser_warnings": all_warnings,
            "doc_metadata": doc_meta.model_dump(mode="json"),
        }).eq("document_id", str(document_id)).execute()

        # ── Stage K: Corpus version bump ────────────────────────────
        await _update_stage(ingestion_id, "finalising")
        await db.rpc("increment_corpus_version", {"p_tenant_id": str(tenant_id)}).execute()
        log.info("[K] Corpus version bumped")

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        quality_report = {
            "chars_per_page": round(total_chars / max(total_page_count, 1), 1),
            "total_chars": total_chars,
            "parser": batch0_result.provider,
            "warnings": all_warnings,
            "batches": len(batches),
        }
        await db.table("ingestions").update({
            "status": "completed",
            "stage": "completed",
            "stats": {**accumulated, "elapsed_ms": elapsed_ms},
            "quality_report": quality_report,
            "finished_at": datetime.datetime.utcnow().isoformat(),
        }).eq("ingestion_id", str(ingestion_id)).execute()

        log.info("=" * 60)
        log.info(
            "PIPELINE COMPLETE  file=%s  total=%.1fs  batches=%d  chunks=%d",
            filename,
            elapsed_ms / 1000,
            len(batches),
            accumulated["child_chunks"],
        )
        log.info("=" * 60)

    except IngestionError as exc:
        log.error("PIPELINE FAILED [%s]  %s", exc.code, exc.message)
        await _fail(ingestion_id, exc.code, exc.message)
    except Exception as exc:
        log.exception("PIPELINE UNEXPECTED ERROR  %s", exc)
        await _fail(ingestion_id, "INTERNAL_ERROR", str(exc))
        raise
