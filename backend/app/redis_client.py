"""Redis connection management — two independent pools, one per workload.

The cache and the crawl queue want opposite things from a connection. The
retrieval cache does short request/response GETs and must never block a chat
query. The crawl worker holds a *blocking* XREADGROUP open for seconds at a
time. Sharing one pool meant the worker's long-lived blocked connections sat in
front of every cache lookup, so cache GETs intermittently got a reaped socket
and failed. They get separate pools here.

Every connection is built with the resilience settings Upstash actually needs:

  * `health_check_interval` — Upstash reaps idle TLS connections server-side. A
    pooled socket the server already closed is otherwise handed straight to the
    next command, which fails. This PINGs first.
  * `socket_timeout` — must exceed the longest blocking command's window
    (`crawl_stream_block_ms`), or a blocking read trips the client deadline every
    single cycle. This is what produced the ~7s `Timeout reading from ...`
    warning loop.
  * `socket_keepalive` — holds NAT/proxy state open across idle periods.
  * `retry` — redis-py will not retry a TimeoutError unless told to.
  * `max_connections` — bounds each pool independently against the plan's limit.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.config import settings

log = logging.getLogger("clintel.redis")

# Two singletons. `_cache_client` serves the retrieval cache; `_queue_client`
# serves the crawl worker's stream reads.
_cache_client: Any = None
_queue_client: Any = None

# A failed init disables Redis only until this timestamp, rather than latching
# off for the life of the process — a transient DNS/TLS blip at boot should not
# permanently disable the queue fast path.
_disabled_until: float = 0.0


def _build_client(max_connections: int) -> Any | None:
    """Construct one configured client, or None if Redis is unusable."""
    global _disabled_until

    if not settings.redis_url:
        return None
    if time.monotonic() < _disabled_until:
        return None

    try:
        import redis.asyncio as redis
        from redis.asyncio.retry import Retry
        from redis.backoff import ExponentialBackoff
        from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError
    except Exception as exc:
        log.warning("Redis package unavailable: %s", exc)
        _disabled_until = time.monotonic() + settings.redis_disabled_cooldown_seconds
        return None

    try:
        kwargs: dict[str, Any] = {
            "encoding": "utf-8",
            "decode_responses": True,
            "socket_timeout": settings.redis_socket_timeout_seconds,
            "socket_connect_timeout": settings.redis_socket_connect_timeout_seconds,
            "socket_keepalive": True,
            "health_check_interval": settings.redis_health_check_interval_seconds,
            "max_connections": max_connections,
            "retry": Retry(ExponentialBackoff(), settings.redis_retry_attempts),
            "retry_on_error": [RedisConnectionError, RedisTimeoutError],
        }
        if settings.redis_url.startswith("rediss://"):
            # redis-py's default SSL context falls back to the OS certificate
            # store, which on some machines (seen on Windows here) carries a
            # stale/incomplete root chain and rejects an otherwise-valid
            # Let's Encrypt cert (e.g. Upstash) as "expired". Pinning certifi's
            # actively-maintained bundle avoids depending on the host's store.
            import certifi

            kwargs["ssl_ca_certs"] = certifi.where()
        return redis.from_url(settings.redis_url, **kwargs)
    except Exception as exc:
        log.warning("Redis client init failed: %s", exc)
        _disabled_until = time.monotonic() + settings.redis_disabled_cooldown_seconds
        return None


async def get_redis_client() -> Any | None:
    """The retrieval cache's client. Returns None (never raises) when unavailable.

    Callers that just want a cache treat None as "fall back to the local cache".
    """
    global _cache_client
    if _cache_client is not None:
        return _cache_client
    _cache_client = _build_client(settings.redis_max_connections)
    return _cache_client


async def get_queue_redis_client() -> Any | None:
    """The crawl worker's client — its own pool, isolated from the cache.

    Returns None when unavailable; the crawl worker treats that as "queue
    temporarily unavailable" and falls back to its Postgres-backed recovery path
    rather than assuming delivery.
    """
    global _queue_client
    if _queue_client is not None:
        return _queue_client
    # The queue needs far fewer connections than the cache now that a single
    # dispatcher does the blocking read: one for the blocked XREADGROUP, plus
    # headroom for concurrent XADD/XACK.
    _queue_client = _build_client(max(4, settings.url_ingestion_concurrency // 2))
    return _queue_client


async def close_redis_client() -> None:
    """Close both pools and clear the disable cooldown."""
    global _cache_client, _queue_client, _disabled_until
    for name in ("_cache_client", "_queue_client"):
        client = globals()[name]
        if client is not None:
            try:
                await client.aclose()
            except Exception as exc:
                log.warning("Error closing Redis client (%s): %s", name, exc)
            finally:
                globals()[name] = None
    _disabled_until = 0.0
