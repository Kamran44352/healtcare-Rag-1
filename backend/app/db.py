from supabase import AsyncClient, acreate_client
from app.config import settings

_client: AsyncClient | None = None


async def get_db() -> AsyncClient:
    global _client
    if _client is None:
        _client = await acreate_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
    return _client


async def close_db() -> None:
    global _client
    if _client is not None:
        postgrest = getattr(_client, "postgrest", None)
        storage = getattr(_client, "storage", None)

        if postgrest is not None and hasattr(postgrest, "aclose"):
            await postgrest.aclose()
        if storage is not None and hasattr(storage, "session") and hasattr(storage.session, "aclose"):
            await storage.session.aclose()

        _client = None
