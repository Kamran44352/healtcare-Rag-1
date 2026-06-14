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

    # Supabase
    supabase_url: str
    supabase_service_role_key: str
    supabase_storage_bucket: str = "documents"
    store_pdf_in_bucket: bool = True

    # Redis
    redis_url: str | None = None
    redis_key_prefix: str = "clintel"

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

    # Models
    foreground_model: str = "gpt-5.4-mini"
    background_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1536
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"

    # Agent (Phase 3)
    agent_enabled: bool = True
    agent_coverage_threshold: float = 0.6
    agent_max_retries: int = 1
    agent_grounding_threshold: float = 0.5


settings = Settings()
