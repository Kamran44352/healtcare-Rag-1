from __future__ import annotations
from datetime import datetime
from typing import Any, Literal
from uuid import UUID
from pydantic import BaseModel, Field

from app.retrieval.models import RetrievalFilters, RetrievalResult


# ── Ingestion ──────────────────────────────────────────────────────────────

class IngestionMetadata(BaseModel):
    category: str | None = None
    reference: str | None = None
    guideline_title: str | None = None
    pdf_url: str | None = None


class IngestionCreated(BaseModel):
    ingestion_id: UUID
    document_id: UUID
    status: str
    stage: str | None
    reused_existing_document: bool


# ── Web crawl (Firecrawl single-page scrape) ────────────────────────────────

class CrawlRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    metadata: IngestionMetadata = Field(default_factory=IngestionMetadata)
    force_rescrape: bool = False
    # Auto re-scrape cadence in hours; null/0 = no scheduled re-scrape.
    rescrape_interval_hours: int | None = Field(default=None, ge=0, le=8760)


class BulkCrawlRequest(BaseModel):
    # Upper bound enforced at the router against settings.crawl_batch_max_urls
    # (kept out of this schema so the cap has one source of truth: config.py).
    urls: list[str] = Field(min_length=1)
    metadata: IngestionMetadata = Field(default_factory=IngestionMetadata)
    rescrape_interval_hours: int | None = Field(default=None, ge=0, le=8760)


class BulkCrawlCreated(BaseModel):
    batch_id: UUID
    accepted_count: int
    duplicate_count: int
    total_submitted: int


class CrawlBatchItemStatus(BaseModel):
    item_id: UUID
    url: str
    normalized_url: str
    status: str
    ingestion_id: UUID | None
    document_id: UUID | None
    attempt_count: int
    max_attempts: int
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class CrawlBatchSummary(BaseModel):
    batch_id: UUID
    total_count: int
    duplicate_count: int
    queued_count: int
    processing_count: int
    completed_count: int
    failed_count: int
    is_finished: bool
    created_at: datetime
    finished_at: datetime | None


class CrawlBatchStatus(CrawlBatchSummary):
    items: list[CrawlBatchItemStatus]


class IngestionStatus(BaseModel):
    ingestion_id: UUID
    document_id: UUID | None
    filename: str
    status: str
    stage: str | None
    error_code: str | None
    error_message: str | None
    stats: dict[str, Any]
    quality_report: dict[str, Any] | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


# ── Documents ─────────────────────────────────────────────────────────────

class DocumentRecord(BaseModel):
    document_id: UUID
    filename: str
    metadata: dict[str, Any]
    doc_metadata: dict[str, Any] | None = None
    page_count: int | None
    parser_provider: str | None
    parser_warnings: list[Any]
    storage_path: str
    source_type: str = "pdf"
    source_url: str | None = None
    rescrape_interval_hours: int | None = None
    last_rescrape_at: datetime | None = None
    next_rescrape_at: datetime | None = None
    created_at: datetime
    latest_ingestion_id: UUID | None
    latest_status: str | None
    latest_stage: str | None
    latest_error_code: str | None
    latest_error_message: str | None
    latest_created_at: datetime | None
    latest_finished_at: datetime | None


# ── Chat ──────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    chunk_id: UUID
    source_index: int
    filename: str
    section_path: str
    snippet: str
    full_snippet: str | None = None
    doc_metadata: dict[str, Any] = {}


class ChatSessionResponse(BaseModel):
    session_id: UUID
    metadata: dict[str, Any]
    created_at: datetime
    expires_at: datetime | None


class ChatMessageRecord(BaseModel):
    message_id: UUID
    session_id: UUID
    role: str
    content: str
    citations: list[Citation]
    abstained: bool
    confidence: float | None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    session_id: UUID
    messages: list[ChatMessageRecord]


class ChatQueryRequest(BaseModel):
    session_id: UUID | None = None
    question: str = Field(min_length=1, max_length=4000)
    filters: RetrievalFilters | None = None
    top_k: int = Field(default=6, ge=1, le=15)
    include_debug: bool = False


class ChatResponse(BaseModel):
    answer: str
    abstained: bool
    abstain_reason: str | None
    confidence: float | None
    citations: list[Citation]
    session_id: UUID
    message_id: UUID
    retrieval_debug: dict[str, Any] | None = None
    follow_up_questions: list[str] | None = None


class FeedbackRequest(BaseModel):
    rating: int | str  # 1/-1 or "up"/"down"

    def rating_int(self) -> int | None:
        if self.rating == "up":
            return 1
        if self.rating == "down":
            return -1
        if self.rating in (1, -1):
            return int(self.rating)
        return None


class RetrievalSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    filters: RetrievalFilters | None = None
    profile: Literal["specific", "narrow", "broad"] = "specific"
    top_k: int = Field(default=10, ge=1, le=20)
    expand_parents: bool = True


class RetrievalSearchResponse(RetrievalResult):
    pass
