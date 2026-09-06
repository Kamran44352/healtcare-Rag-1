"""The single shared OpenAI client, plus a model-capability-tolerant call wrapper.

Previously 13 modules each constructed their own module-level
`AsyncOpenAI(api_key=settings.openai_api_key)`, which meant 13 independent httpx
connection pools and — once tracing landed — 13 separate places to wrap. One
shared client fixes both: connections are pooled across the whole process, and
instrumentation is applied exactly once here.

`chat_completion()` exists because models are configured from `.env` and newer
OpenAI models reject parameters older ones require:

  * reasoning-style models accept only the default `temperature` (1) and return
    400 `unsupported_value` for anything else;
  * they also reject `max_tokens` in favour of `max_completion_tokens`.

Hardcoding a list of which model accepts what would go stale the moment someone
edits FOREGROUND_MODEL. Instead this learns from the API's own error response,
caches the result per model, and retries once — so the first call after a model
swap pays one extra round trip and every call after it is clean.
"""
from __future__ import annotations

import logging
from typing import Any

from app.observability import traced_openai

log = logging.getLogger("clintel.llm")

openai_client: Any = traced_openai()

# model -> set of parameter names that model rejected. Populated at runtime from
# 400 responses; never persisted, so a model's capabilities are re-learned on
# each boot (cheap: one extra round trip, once).
_unsupported_params: dict[str, set[str]] = {}


def _extract_bad_param(exc: Exception) -> str | None:
    """Pull the offending parameter name out of an OpenAI 400.

    Prefers the structured `param` field and falls back to scanning the message,
    since the exact error shape has changed between API versions.
    """
    param = getattr(exc, "param", None)
    if isinstance(param, str) and param:
        return param

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("param"), str) and err["param"]:
            return err["param"]

    message = str(exc)
    for candidate in ("temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"):
        if f"'{candidate}'" in message and (
            "Unsupported value" in message or "Unsupported parameter" in message
        ):
            return candidate
    return None


def _strip(kwargs: dict[str, Any], model: str) -> dict[str, Any]:
    """Remove params this model has already rejected, translating where possible."""
    bad = _unsupported_params.get(model)
    if not bad:
        return kwargs
    out = dict(kwargs)
    for param in bad:
        if param not in out:
            continue
        value = out.pop(param)
        # `max_tokens` has a direct successor; everything else just drops to the
        # model's default.
        if param == "max_tokens" and "max_completion_tokens" not in out:
            out["max_completion_tokens"] = value
    return out


async def chat_completion(**kwargs: Any) -> Any:
    """`client.chat.completions.create`, tolerant of model parameter differences.

    On a 400 naming an unsupported parameter, the parameter is recorded against
    the model and the call is retried once without it. Any other error, and any
    second failure, propagates unchanged — callers still see real failures.
    """
    model = kwargs.get("model") or ""
    attempted = _strip(kwargs, model)

    try:
        return await openai_client.chat.completions.create(**attempted)
    except Exception as exc:
        if getattr(exc, "status_code", None) != 400:
            raise
        param = _extract_bad_param(exc)
        if not param or param not in attempted:
            raise

        _unsupported_params.setdefault(model, set()).add(param)
        retry_kwargs = _strip(attempted, model)
        if retry_kwargs == attempted:
            raise
        log.warning(
            "Model %r rejected %r — retrying without it and skipping it for "
            "subsequent calls this process. (%s)",
            model, param, str(exc)[:160],
        )
        return await openai_client.chat.completions.create(**retry_kwargs)
