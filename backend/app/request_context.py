"""Per-request correlation id, propagated through async tasks.

`ContextVar` is inherited by every `asyncio.Task` spawned from the current
context, so background work kicked off by a request (the ingestion pipeline, a
crawl item dispatched inline) keeps the originating request id for free — no
threading of an extra argument through every call site.

The same id is attached to LangSmith run metadata (see `observability.py`), so a
log line and a trace can be joined on it.
"""
from __future__ import annotations

from contextvars import ContextVar
from uuid import uuid4

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    return _request_id.get()


def set_request_id(value: str | None = None) -> str:
    """Set (or mint) the current correlation id. Returns the value set."""
    rid = (value or "").strip() or uuid4().hex[:16]
    _request_id.set(rid)
    return rid


def new_context_id(prefix: str) -> str:
    """Mint a correlation id for work with no HTTP request behind it.

    Used by the crawl worker and the rescrape scheduler so their log lines are
    still groupable, e.g. `crawl-3f2a1b9c`.
    """
    return set_request_id(f"{prefix}-{uuid4().hex[:8]}")
