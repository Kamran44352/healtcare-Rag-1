from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.observability import traceable

_FIRECRAWL_SCRAPE_PATH = "/v1/scrape"


class ScrapeError(Exception):
    """Raised when Firecrawl fails to return usable markdown for a single page."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass
class ScrapeResult:
    markdown: str
    title: str | None
    source_url: str
    provider: str = "firecrawl"


def normalize_url(url: str) -> str:
    """Canonicalize a URL for dedup keying: lowercase scheme/host, drop fragment,
    strip a trailing slash on the path. Query string is preserved (it can change content)."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise ScrapeError("INVALID_URL", f"Unsupported URL scheme: {parts.scheme or '(none)'}")
    if not parts.netloc:
        raise ScrapeError("INVALID_URL", "URL is missing a host")
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # 429 is retried here too — most rate-limit blips clear within this
        # function's own exponential backoff, well before it ever costs a bulk
        # batch item one of its (Postgres-tracked) retry attempts.
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return False


@retry(
    retry=retry_if_exception(_should_retry),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
async def _scrape_request(url: str) -> dict:
    if not settings.firecrawl_api_key:
        raise ScrapeError("NO_API_KEY", "FIRECRAWL_API_KEY is not configured")

    endpoint = settings.firecrawl_api_url.rstrip("/") + _FIRECRAWL_SCRAPE_PATH
    headers = {
        "Authorization": f"Bearer {settings.firecrawl_api_key}",
        "Content-Type": "application/json",
    }
    # Single-page scrape only — Firecrawl /scrape never follows links (unlike /crawl).
    body = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
    }
    timeout = httpx.Timeout(
        connect=30,
        read=settings.firecrawl_timeout_seconds,
        write=60,
        pool=60,
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(endpoint, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


@traceable(run_type="tool", name="firecrawl_scrape")
async def scrape_url(url: str) -> ScrapeResult:
    """Scrape a single web page with Firecrawl and return its markdown.

    Raises ScrapeError on invalid input, auth/config problems, or empty results.
    """
    normalized = normalize_url(url)

    try:
        payload = await _scrape_request(normalized)
    except ScrapeError:
        raise
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:300] if exc.response is not None else str(exc)
        raise ScrapeError("SCRAPE_FAILED", f"Firecrawl returned {exc.response.status_code}: {detail}")
    except httpx.HTTPError as exc:
        raise ScrapeError("SCRAPE_FAILED", f"Firecrawl request failed: {exc}")

    if not payload.get("success", True):
        raise ScrapeError("SCRAPE_FAILED", str(payload.get("error") or "Firecrawl reported failure"))

    data = payload.get("data") or {}
    markdown = (data.get("markdown") or "").strip()
    if not markdown:
        raise ScrapeError(
            "EMPTY_CONTENT",
            "Firecrawl returned no markdown content for this page (it may be blocked, "
            "JavaScript-gated, or behind a login).",
        )

    meta = data.get("metadata") or {}
    title = meta.get("title") or meta.get("ogTitle") or None

    # Dedup keys on the normalized URL the caller submitted so manual + scheduled
    # re-scrapes of the same input always resolve to the same document.
    return ScrapeResult(
        markdown=markdown,
        title=title.strip() if isinstance(title, str) else None,
        source_url=normalized,
    )
