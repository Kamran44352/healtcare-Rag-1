"""LangSmith tracing — the single instrumentation surface for the whole backend.

Everything here degrades to a no-op when tracing is off or the `langsmith`
package is missing, so no other module ever needs to guard its use of
`@traceable`. That matters: `@traceable` is applied at import time on hot
functions across the pipeline, retrieval, and the agent, and none of them should
grow a `try/except ImportError`.

Three things are provided:

1. `traceable` — the decorator (real, or a transparent passthrough).
2. `traced_openai()` — one shared, instrumented `AsyncOpenAI` client (see `app/llm.py`).
3. Trace-context propagation across the Redis stream boundary, so a URL submitted
   to `POST /v1/crawls/bulk` and the ingestion it eventually triggers on a worker
   land in LangSmith as ONE trace instead of two unrelated roots.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Iterator, TypeVar

from app.config import settings, tracing_enabled
from app.request_context import get_request_id

log = logging.getLogger("clintel.observability")

F = TypeVar("F", bound=Callable[..., Any])

# Redis stream field carrying the serialized parent run context. Namespaced so it
# can never collide with a real payload field like `item_id`.
TRACE_HEADER_FIELD = "_ls_trace"

_LANGSMITH_AVAILABLE = False
_traceable: Any = None
_get_current_run_tree: Any = None
_tracing_context: Any = None

try:  # pragma: no cover - import-shape dependent
    from langsmith import traceable as _traceable  # type: ignore[no-redef]
    from langsmith.run_helpers import (  # type: ignore[no-redef]
        get_current_run_tree as _get_current_run_tree,
        tracing_context as _tracing_context,
    )

    _LANGSMITH_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    log.info("LangSmith SDK unavailable, tracing disabled: %s", exc)


def _noop_traceable(*d_args: Any, **d_kwargs: Any) -> Any:
    """Passthrough stand-in for `@traceable` in both call shapes.

    Supports bare `@traceable` and parameterised `@traceable(run_type=..., name=...)`,
    returning the undecorated function in either case.
    """
    if len(d_args) == 1 and not d_kwargs and callable(d_args[0]):
        return d_args[0]

    def _decorator(func: F) -> F:
        return func

    return _decorator


def _resolve_traceable() -> Any:
    """Pick the real decorator or the no-op, once, at import time.

    Bound at import rather than per-call because `@traceable` decorates functions
    when their module is first imported — flipping tracing at runtime would not
    retro-apply anyway, and a per-call check would cost on every hot path.
    """
    if _LANGSMITH_AVAILABLE and tracing_enabled():
        return _traceable
    return _noop_traceable


traceable: Any = _resolve_traceable()


# ── OpenAI client instrumentation ──────────────────────────────────────────

def traced_openai() -> Any:
    """Build the shared AsyncOpenAI client, wrapped for tracing when enabled.

    Wrapping makes every `chat.completions.create` / `embeddings.create` show up
    as an LLM child run with token counts and latency, under whatever run is
    currently active — no per-call-site changes needed.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    if not (_LANGSMITH_AVAILABLE and tracing_enabled()):
        return client
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception as exc:  # pragma: no cover
        log.warning("wrap_openai failed, LLM calls will not be traced: %s", exc)
        return client


# ── Run metadata ───────────────────────────────────────────────────────────

def run_metadata(**extra: Any) -> dict[str, Any]:
    """Standard metadata attached to every root run.

    `request_id` is the join key between LangSmith traces and `clintel.*` log
    lines — see `app/logging_config.py`.
    """
    meta: dict[str, Any] = {"request_id": get_request_id()}
    meta.update({k: v for k, v in extra.items() if v is not None})
    return meta


# ── Distributed tracing across the Redis stream boundary ───────────────────

def current_trace_headers() -> dict[str, str]:
    """Serialize the active run context so a consumer can re-parent to it.

    Returns `{}` when there is no active run or tracing is off, which callers
    treat as "nothing to propagate".
    """
    if not (_LANGSMITH_AVAILABLE and tracing_enabled()):
        return {}
    try:
        tree = _get_current_run_tree()
        if tree is None:
            return {}
        return {k: v for k, v in tree.to_headers().items() if v}
    except Exception as exc:
        log.debug("current_trace_headers failed: %s", exc)
        return {}


@contextmanager
def trace_span(name: str, **metadata: Any) -> Iterator[None]:
    """Open a root span around a block of code.

    The decorator form (`@traceable`) is unsafe on FastAPI route handlers — it
    injects a `config` keyword-only parameter that FastAPI publishes as a query
    parameter. This gives routes a span without touching their signature.
    """
    if not (_LANGSMITH_AVAILABLE and tracing_enabled()):
        yield
        return
    try:
        from langsmith.run_helpers import trace as _ls_trace

        with _ls_trace(name=name, run_type="chain", metadata=run_metadata(**metadata)):
            yield
    except Exception as exc:
        log.debug("trace_span(%s) failed: %s", name, exc)
        yield


# ── Bridging LangGraph's tracing context to LangSmith's ────────────────────

@contextmanager
def langchain_parent(config: Any) -> Iterator[None]:
    """Install the current LangGraph node as the LangSmith parent run.

    LangGraph/LangChain track runs through the callback manager carried on
    `RunnableConfig`; `@traceable` and `wrap_openai` track them through a
    langsmith contextvar. LangGraph does NOT populate that contextvar, so
    without this bridge every LLM call, retriever span, and rerank inside a node
    is emitted as its OWN root run — you get a correct 7-node agent tree plus a
    dozen orphaned sibling runs, and no way to see which query produced which
    LLM call.

    `RunTree.from_runnable_config(config)` resolves the node's run from the
    callback manager; installing it as `parent` makes everything downstream nest
    correctly. Verified empirically: `get_current_run_tree()` returns None inside
    a node, while `from_runnable_config(config)` returns the node's run.
    """
    if not (_LANGSMITH_AVAILABLE and tracing_enabled()) or not config:
        yield
        return
    try:
        from langsmith.run_trees import RunTree

        parent = RunTree.from_runnable_config(config)
        if parent is None:
            yield
            return
        with _tracing_context(parent=parent):
            yield
    except Exception as exc:
        log.debug("langchain_parent bridge failed: %s", exc)
        yield


def traces_under_node(func: F) -> F:
    """Decorator form of `langchain_parent` for LangGraph nodes.

    Every node has the signature `(state, config)`, so this wraps the body in the
    bridge without touching the node's own code.
    """
    import functools

    @functools.wraps(func)
    async def wrapper(state: Any, config: Any = None, *args: Any, **kwargs: Any) -> Any:
        with langchain_parent(config):
            return await func(state, config, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


@contextmanager
def trace_context_from_headers(headers: dict[str, str] | None) -> Iterator[None]:
    """Re-attach to a parent run serialized by `current_trace_headers()`.

    Used by the crawl worker: the dispatcher enters this before processing an
    item so the item's ingestion nests under the batch submission that queued it.
    Falls back to a plain no-op context on any failure — a broken trace context
    must never stop an item from being processed.
    """
    if not headers or not (_LANGSMITH_AVAILABLE and tracing_enabled()):
        with nullcontext():
            yield
        return
    try:
        with _tracing_context(parent=headers):
            yield
    except Exception as exc:
        log.debug("trace_context_from_headers failed, tracing as a new root: %s", exc)
        with nullcontext():
            yield


# ── Lifecycle ──────────────────────────────────────────────────────────────

def log_tracing_status() -> None:
    if not settings.langsmith_tracing:
        log.info("LangSmith tracing disabled (LANGSMITH_TRACING not set)")
    elif not settings.langsmith_api_key:
        log.warning("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is unset — tracing disabled")
    elif not _LANGSMITH_AVAILABLE:
        log.warning("LANGSMITH_TRACING is on but the langsmith package is unavailable — tracing disabled")
    else:
        log.info(
            "LangSmith tracing enabled (project=%s endpoint=%s)",
            settings.langsmith_project,
            settings.langsmith_endpoint,
        )


async def flush_tracers() -> None:
    """Flush buffered runs on shutdown.

    Necessary because the crawl worker and rescrape scheduler are long-lived
    background tasks; without an explicit flush their spans are dropped when
    Railway sends SIGTERM.
    """
    if not (_LANGSMITH_AVAILABLE and tracing_enabled()):
        return
    try:
        # `get_cached_client` returns the same Client the tracing decorators
        # buffer into, so this actually drains their queue (a freshly
        # constructed Client would have an empty one).
        from langsmith.run_trees import get_cached_client

        client = get_cached_client()
        if client is not None:
            client.flush()
            log.info("LangSmith tracers flushed")
    except Exception as exc:  # pragma: no cover
        log.warning("LangSmith flush failed: %s", exc)
