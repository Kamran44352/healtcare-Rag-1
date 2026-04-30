from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

RetrievalProfile = Literal["specific", "narrow", "broad"]


class RetrievalFilters(BaseModel):
    doc_type: list[str] | None = None
    section_type: list[str] | None = None
    specialties: list[str] | None = None
    conditions_codes: list[str] | None = None
    geographic_scope: list[str] | None = None
    care_setting: list[str] | None = None
    has_dosing_tables: bool | None = None
    has_red_flags: bool | None = None


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    parent_chunk_id: UUID
    document_id: UUID
    filename: str
    section_path: str
    section_type: str | None = None
    recommendation_strength: str | None = None
    entities: dict[str, Any] = Field(default_factory=dict)
    snippet: str
    full_snippet: str
    parent_text: str | None = None
    doc_metadata: dict[str, Any] = Field(default_factory=dict)
    dense_score: float | None = None
    fused_score: float
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    query_ms: int
    corpus_version: int
    cache_hit: bool = False
