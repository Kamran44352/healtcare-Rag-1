from __future__ import annotations

from uuid import UUID

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from app.retrieval.models import RetrievalFilters


def _append_match_any(must: list[FieldCondition], key: str, values: list[str] | None) -> None:
    if values:
        must.append(FieldCondition(key=key, match=MatchAny(any=values)))


def _append_match_bool(must: list[FieldCondition], key: str, value: bool | None) -> None:
    if value is not None:
        must.append(FieldCondition(key=key, match=MatchValue(value=value)))


def build_qdrant_filter(tenant_id: UUID, filters: RetrievalFilters | None) -> Filter:
    must: list[FieldCondition] = [
        FieldCondition(key="tenant_id", match=MatchValue(value=str(tenant_id)))
    ]

    if filters is None:
        return Filter(must=must)

    _append_match_any(must, "doc_type", filters.doc_type)
    _append_match_any(must, "section_type", filters.section_type)
    _append_match_any(must, "specialties", filters.specialties)
    _append_match_any(must, "conditions_codes", filters.conditions_codes)
    _append_match_any(must, "geographic_scope", filters.geographic_scope)
    _append_match_any(must, "care_setting", filters.care_setting)
    _append_match_bool(must, "has_dosing_tables", filters.has_dosing_tables)
    _append_match_bool(must, "has_red_flags", filters.has_red_flags)

    return Filter(must=must)
