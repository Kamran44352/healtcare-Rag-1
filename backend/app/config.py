from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from uuid import UUID


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str

    # Firecrawl (website crawling — single page scrape)
    firecrawl_api_key: str | None = None
    firecrawl_api_url: str = "https://api.firecrawl.dev"
    firecrawl_timeout_seconds: int = 120

    # LlamaParse
    llama_cloud_api_key: str
    llama_parse_tier: str = "agentic"
    llama_parse_version: str = "latest"
    llama_parse_ocr_languages: str = "en"
    llama_parse_cost_optimizer: bool = True
    llama_parse_merge_continued_tables: bool = True
    llama_parse_timeout_base_seconds: int = 1800
    llama_parse_timeout_extra_per_page_seconds: int = 30
    llama_parse_allowed_page_failure_ratio: float = 0.05
    llama_parse_poll_interval_seconds: int = 4
    llama_parse_max_poll_seconds: int = 14400

    # Qdrant Cloud
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "healthcare_chunks"
    qdrant_request_timeout_seconds: int = 60
    qdrant_retrieval_timeout_seconds: int = 20
    qdrant_write_concurrency: int = 4  # max simultaneous Qdrant upserts across all pipelines
    # corpus_version gates the retrieval cache key. It only changes when a
    # document is (re)indexed, but was being SELECTed from Postgres on every
    # single query (~900ms). Cache it briefly in-process.
    corpus_version_ttl_seconds: int = 30

    # Supabase
    supabase_url: str
    supabase_service_role_key: str
    supabase_storage_bucket: str = "documents"
    store_pdf_in_bucket: bool = True

    # Redis
    redis_url: str | None = None
    redis_key_prefix: str = "clintel"
    # Connection resilience. Upstash terminates idle TLS connections, so a pooled
    # socket can be handed out already dead — health_check_interval PINGs it first.
    # socket_timeout MUST exceed crawl_stream_block_ms/1000, or a blocking
    # XREADGROUP is guaranteed to trip the client read deadline every cycle.
    redis_socket_timeout_seconds: float = 30.0
    redis_socket_connect_timeout_seconds: float = 10.0
    redis_health_check_interval_seconds: int = 30
    redis_max_connections: int = 16          # per pool; cache and queue have separate pools
    redis_retry_attempts: int = 3            # redis-py internal retries on timeout/conn errors
    redis_disabled_cooldown_seconds: int = 60  # retry a failed client init after this

    # App
    default_tenant_id: UUID = UUID("00000000-0000-0000-0000-000000000000")
    ingestion_concurrency: int = 3          # PDF pipeline slots
    url_ingestion_concurrency: int = 8      # URL pipeline slots (separate, lighter jobs)
    llm_call_timeout_seconds: int = 60      # per-call timeout for OpenAI stage calls
    cors_origins: str = "*"

    # Scheduled auto re-scrape (web sources only)
    rescrape_enabled: bool = True
    rescrape_check_interval_seconds: int = 900  # how often the loop wakes to look for due URLs
    rescrape_default_interval_hours: int = 24  # default cadence applied to new crawls (0 = off)

    # Bulk crawl (durable multi-URL submission via Postgres + Redis Streams)
    crawl_batch_max_urls: int = 500                       # hard cap per bulk submission
    crawl_batch_max_attempts: int = 5                     # per-URL attempts before permanent failure
    crawl_batch_retry_backoff_base_seconds: int = 30      # first retry delay
    crawl_batch_retry_backoff_max_seconds: int = 900      # backoff ceiling
    crawl_batch_stale_processing_seconds: int = 600       # worker presumed dead past this — item requeued
    crawl_batch_queued_grace_seconds: int = 30            # grace before reconciliation re-enqueues a 'queued' row
    crawl_batch_reconciliation_interval_seconds: int = 90  # how often the Postgres safety-net sweep runs
    crawl_stream_name: str = "crawl:url_queue"            # Redis stream key (prefixed with redis_key_prefix)
    crawl_stream_consumer_group: str = "crawl_workers"
    crawl_stream_maxlen: int = 100_000                    # approximate XADD MAXLEN trim
    crawl_stream_block_ms: int = 15_000                   # XREADGROUP block window (< redis_socket_timeout)
    crawl_stream_error_backoff_base_seconds: float = 1.0  # backoff after a real (non-timeout) stream error
    crawl_stream_error_backoff_max_seconds: float = 60.0

    # Models
    foreground_model: str = "gpt-5.4-mini"
    background_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1536
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"

    # Per-step model overrides. Each falls back to background_model when unset, so
    # existing deployments behave identically until a value is set. These exist so
    # the cheap, high-volume steps can be tuned independently — they are short,
    # structured, JSON-output tasks where a smaller/faster model is often enough,
    # and they sit directly in the user-visible latency path.
    classify_model: str | None = None    # intent classification (analyze_question)
    rewrite_model: str | None = None     # query rewriting (analyze_question)
    grading_model: str | None = None     # chunk coverage grading
    grounding_model: str | None = None   # citation grounding verification

    # Agent (Phase 3)
    agent_enabled: bool = True
    agent_coverage_threshold: float = 0.6
    agent_max_retries: int = 1
    agent_grounding_threshold: float = 0.5
    # Refusing a question as out_of_scope is high-stakes and irreversible for the
    # user (no retrieval is attempted at all), so only do it when the classifier is
    # very sure. Below this, the question falls through to retrieval and the
    # evidence decides — a wrong refusal is far worse than a wasted retrieval.
    agent_out_of_scope_confidence: float = 0.9
    # When the reranker already returned this many chunks, route_after_grading
    # goes to generate_answer regardless of coverage score — so paying for the
    # grading LLM call cannot change the outcome. Skip it and save the round trip.
    agent_grading_skip_chunk_count: int = 4

    # Observability (LangSmith)
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "clintel"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _default_step_models(self):
        """Resolve unset per-step models to background_model.

        Done here rather than at each call site so the call sites stay a plain
        `settings.classify_model` and there is exactly one place defining the
        fallback.
        """
        for field in ("classify_model", "rewrite_model", "grading_model", "grounding_model"):
            # Falsy, not just None: a commented-out-style `CLASSIFY_MODEL=` in .env
            # arrives as an empty STRING, which would otherwise be sent to the API
            # verbatim as model="".
            if not (getattr(self, field) or "").strip():
                setattr(self, field, self.background_model)
        return self


settings = Settings()


def _export_langsmith_env() -> None:
    """Bridge LangSmith settings from `.env` into `os.environ`.

    The LangSmith SDK reads LANGSMITH_* straight from `os.environ`; pydantic-settings
    reading `.env` does NOT populate it. This module is imported before every agent
    and pipeline module, which makes here the only reliably-early hook. `setdefault`
    so a real environment variable (Railway, CI) always wins over `.env`.
    """
    import os

    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    else:
        # Explicitly off — stops the SDK picking up a stray ambient LANGSMITH_TRACING.
        os.environ["LANGSMITH_TRACING"] = "false"


_export_langsmith_env()


def tracing_enabled() -> bool:
    """True when LangSmith tracing is both requested and credentialed."""
    return bool(settings.langsmith_tracing and settings.langsmith_api_key)
