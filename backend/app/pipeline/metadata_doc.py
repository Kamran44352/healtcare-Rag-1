from __future__ import annotations
import json
import re

import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.schemas.healthcare_metadata import DocumentMetadata

_ENCODER = tiktoken.get_encoding("cl100k_base")
_MAX_INPUT_TOKENS = 8_000
_client = AsyncOpenAI(api_key=settings.openai_api_key)

_SYSTEM_PROMPT = """You are a clinical informatics specialist. Extract structured metadata from the healthcare document below.

Rules:
- Use SNOMED CT IDs for conditions/procedures, ATC codes for drugs — only when highly confident.
- If unsure about a code, set code=null and system="free-text". NEVER invent codes.
- geographic_scope: ISO-3166 alpha-2 codes (e.g. "GB", "US") or ["GLOBAL"].
- summary: ≤280 characters describing what this document covers.
- Return ONLY valid JSON matching the schema. No markdown fences.

Schema:
{
  "title": str,
  "document_type": "clinical_guideline"|"protocol"|"drug_monograph"|"research_paper"|"sop"|"patient_leaflet"|"review"|"other",
  "issuing_body": str|null,
  "issuing_body_authority": "national"|"international"|"regional"|"local"|"academic"|"unknown",
  "reference_id": str|null,
  "publication_date": "YYYY-MM-DD"|null,
  "last_updated": "YYYY-MM-DD"|null,
  "version": str|null,
  "geographic_scope": [str],
  "language": str,
  "specialties": [str],
  "conditions": [{"label": str, "code": str|null, "system": str}],
  "drugs": [{"label": str, "code": str|null, "system": str}],
  "procedures": [{"label": str, "code": str|null, "system": str}],
  "populations": {
    "age_min_years": float|null,
    "age_max_years": float|null,
    "pregnancy": "applicable"|"excluded"|"specific"|"unspecified",
    "comorbidities_focus": [str]
  },
  "care_setting": ["primary"|"secondary"|"tertiary"|"community"|"emergency"|"inpatient"|"outpatient"],
  "evidence_grading_system": str|null,
  "has_dosing_tables": bool,
  "has_red_flags": bool,
  "has_referral_criteria": bool,
  "summary": str
}"""


def _truncate(markdown: str) -> str:
    tokens = _ENCODER.encode(markdown)[:_MAX_INPUT_TOKENS]
    return _ENCODER.decode(tokens)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
async def _call_llm(text: str) -> dict:
    response = await _client.chat.completions.create(
        model=settings.foreground_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Document:\n\n{text}"},
        ],
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


async def extract_doc_metadata(markdown: str) -> DocumentMetadata:
    """Stage D: Extract universal healthcare metadata from document text."""
    truncated = _truncate(markdown)
    raw = await _call_llm(truncated)
    return DocumentMetadata.model_validate(raw)
