from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.observability import traceable

_LLAMA_FILES_URL = "https://api.cloud.llamaindex.ai/api/v1/files"
_LLAMA_PARSE_V2_URL = "https://api.cloud.llamaindex.ai/api/v2/parse"


@dataclass
class ParseResult:
    markdown: str
    page_count: int
    provider: str
    warnings: list[str] = field(default_factory=list)


def _should_retry_llama(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.TimeoutException)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


def _llama_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.llama_cloud_api_key}"}


def _llama_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=30, read=300, write=300, pool=60)


def _llama_ocr_languages() -> list[str]:
    values = [part.strip() for part in settings.llama_parse_ocr_languages.split(",")]
    languages = [part for part in values if part]
    return languages or ["en"]


def _build_v2_parse_request(file_id: str) -> dict[str, Any]:
    request: dict[str, Any] = {
        "file_id": file_id,
        "tier": settings.llama_parse_tier,
        "version": settings.llama_parse_version,
        "processing_options": {
            "ocr_parameters": {"languages": _llama_ocr_languages()},
        },
        "output_options": {
            "markdown": {
                "tables": {
                    "merge_continued_tables": settings.llama_parse_merge_continued_tables,
                    "output_tables_as_markdown": True,
                }
            }
        },
        "processing_control": {
            "timeouts": {
                "base_in_seconds": settings.llama_parse_timeout_base_seconds,
                "extra_time_per_page_in_seconds": settings.llama_parse_timeout_extra_per_page_seconds,
            },
            "job_failure_conditions": {
                "allowed_page_failure_ratio": settings.llama_parse_allowed_page_failure_ratio,
            },
        },
    }

    if settings.llama_parse_cost_optimizer and settings.llama_parse_tier in {"agentic", "agentic_plus"}:
        request["processing_options"]["cost_optimizer"] = {"enable": True}

    return request


def _format_llama_job_error(job: dict[str, Any]) -> str:
    for key in ("error_message", "error", "message", "detail", "reason"):
        value = job.get(key)
        if value:
            return str(value)

    summary = {
        key: job.get(key)
        for key in ("id", "status", "name", "error_code", "error_message")
        if job.get(key) is not None
    }
    if summary:
        return json.dumps(summary, ensure_ascii=True)

    return "unknown LlamaParse job error"


def _parse_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    job = payload.get("job")
    return job if isinstance(job, dict) else payload


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=10),
    retry=retry_if_exception(_should_retry_llama),
    reraise=True,
)
@traceable(run_type="tool", name="llamaparse")
async def _llamaparse(pdf_bytes: bytes, filename: str) -> ParseResult:
    async with httpx.AsyncClient(timeout=_llama_timeout()) as client:
        upload_resp = await client.post(
            _LLAMA_FILES_URL,
            headers=_llama_headers(),
            files={"upload_file": (filename, pdf_bytes, "application/pdf")},
        )
        upload_resp.raise_for_status()
        file_id = upload_resp.json()["id"]

        job_resp = await client.post(
            _LLAMA_PARSE_V2_URL,
            headers=_llama_headers(),
            json=_build_v2_parse_request(file_id),
        )
        job_resp.raise_for_status()
        job_id = job_resp.json()["id"]

        waited_seconds = 0
        while waited_seconds < settings.llama_parse_max_poll_seconds:
            await asyncio.sleep(settings.llama_parse_poll_interval_seconds)
            waited_seconds += settings.llama_parse_poll_interval_seconds

            status_resp = await client.get(
                f"{_LLAMA_PARSE_V2_URL}/{job_id}",
                headers=_llama_headers(),
            )
            status_resp.raise_for_status()
            job = _parse_job_payload(status_resp.json())
            status = job["status"]

            if status == "COMPLETED":
                break
            if status in {"FAILED", "CANCELLED"}:
                raise RuntimeError(f"LlamaParse job failed: {_format_llama_job_error(job)}")
        else:
            raise TimeoutError("LlamaParse job polling timed out")

        result_resp = await client.get(
            f"{_LLAMA_PARSE_V2_URL}/{job_id}",
            headers=_llama_headers(),
            params={"expand": "markdown"},
        )
        result_resp.raise_for_status()
        result = result_resp.json()
        job = _parse_job_payload(result)

        if job["status"] != "COMPLETED":
            raise RuntimeError(f"LlamaParse job was not completed: {_format_llama_job_error(job)}")

        markdown_pages = ((result.get("markdown") or {}).get("pages") or [])
        warnings: list[str] = []
        page_markdown: list[str] = []

        for page in markdown_pages:
            if not page.get("success", True) or "error" in page:
                warnings.append(
                    f"Page {page.get('page_number', '?')} failed in LlamaParse: "
                    f"{page.get('error', 'unknown page error')}"
                )
                continue
            page_markdown.append(page.get("markdown", ""))

        markdown = "\n\n".join(part for part in page_markdown if part)
        page_count = len(markdown_pages) or max(1, len(markdown) // 3000)
        provider = f"llamaparse_v2:{settings.llama_parse_tier}"

    return ParseResult(markdown=markdown, page_count=page_count, provider=provider, warnings=warnings)


async def parse(pdf_bytes: bytes, filename: str) -> ParseResult:
    """Parse a PDF to markdown using LlamaParse cloud."""
    return await _llamaparse(pdf_bytes, filename)
