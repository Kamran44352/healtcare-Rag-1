"""Explicit logging setup for the whole backend.

This replaces a `logging.basicConfig(...)` that lived as an import side effect in
`app/pipeline/orchestrator.py` — it only worked because `main.py` happened to
import that module transitively, and any change to the import graph would have
silently switched logging off.

Two substantive changes over what it configured:
  * the timestamp carries a date (it was `%H:%M:%S` only, so lines could not be
    correlated across days);
  * every `clintel.*` line carries the request id, which is the join key against
    LangSmith traces.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.request_context import get_request_id

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  [%(request_id)s]  %(name)s  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


class _RequestIdFilter(logging.Filter):
    """Inject the current correlation id into every record.

    Attached to the root handler rather than to `clintel` alone, because the
    formatter applies to third-party records (httpx, uvicorn) too and would
    raise KeyError on a missing `request_id`.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id()
        return True


def configure_logging() -> None:
    """Idempotently configure root logging. Call once, first thing at startup."""
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    # `force`-style reset: drop anything basicConfig or uvicorn installed first,
    # otherwise records get emitted twice with two different formats.
    for existing in root.handlers[:]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    logging.getLogger("clintel").setLevel(level)
    # httpx logs every request at INFO; useful, but it is the single noisiest
    # source in the log and drowns pipeline stages during a bulk crawl.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _configured = True
