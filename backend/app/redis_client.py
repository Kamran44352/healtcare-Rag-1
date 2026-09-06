from __future__ import annotations

import logging
from typing import Any

from app.config import settings

log = logging.getLogger("clintel.redis")

_client: Any = None
_disabled = False


async def get_redis_client() -> Any | None:
    """Lazy singleton over REDIS_URL, shared by the retrieval cache and the bulk
    crawl queue. Returns None (never raises) if REDIS_URL is unset or the redis
    package can't be imported/connected.

    Callers that just want a cache (retrieval cache) treat None as "fall back to
    local cache". Callers that require Redis for correctness (the crawl worker)
    must treat None as "queue temporarily unavailable" and rely on their own
    Postgres-backed recovery path instead of assuming delivery.
    """
    global _client, _disabled
    if _disabled or not settings.redis_url:
        return None
    if _client is not None:
        return _client

    try:
        import redis.asyncio as redis
    except Exception as exc:
        log.warning("Redis package unavailable: %s", exc)
        _disabled = True
        return None

    try:
        kwargs: dict[str, Any] = {"encoding": "utf-8", "decode_responses": True}
        if settings.redis_url.startswith("rediss://"):
            # redis-py's default SSL context falls back to the OS certificate
            # store, which on some machines (seen on Windows here) carries a
            # stale/incomplete root chain and rejects an otherwise-valid
            # Let's Encrypt cert (e.g. Upstash) as "expired". Pinning certifi's
            # actively-maintained bundle avoids depending on the host's store.
            import certifi

            kwargs["ssl_ca_certs"] = certifi.where()
        _client = redis.from_url(settings.redis_url, **kwargs)
    except Exception as exc:
        log.warning("Redis client init failed: %s", exc)
        _disabled = True
        return None

    return _client


async def close_redis_client() -> None:
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        finally:
            _client = None
