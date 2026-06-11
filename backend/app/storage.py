from uuid import UUID
from app.config import settings
from app.db import get_db


async def upload_pdf(tenant_id: UUID, document_id: UUID, filename: str, data: bytes) -> str:
    """Upload PDF bytes to Supabase Storage if configured. Returns the storage path."""
    path = f"{tenant_id}/{document_id}/{filename}"
    
    if not settings.store_pdf_in_bucket:
        return path

    db = await get_db()
    await db.storage.from_(settings.supabase_storage_bucket).upload(
        path=path,
        file=data,
        file_options={"content-type": "application/pdf", "upsert": "true"},
    )
    return path


async def upload_markdown(tenant_id: UUID, document_id: UUID, markdown: str) -> str:
    """Upload scraped markdown to Supabase Storage if configured. Returns the storage path."""
    path = f"{tenant_id}/{document_id}/page.md"

    if not settings.store_pdf_in_bucket:
        return path

    db = await get_db()
    await db.storage.from_(settings.supabase_storage_bucket).upload(
        path=path,
        file=markdown.encode("utf-8"),
        file_options={"content-type": "text/markdown; charset=utf-8", "upsert": "true"},
    )
    return path


async def delete_file(storage_path: str) -> None:
    db = await get_db()
    await db.storage.from_(settings.supabase_storage_bucket).remove([storage_path])


async def get_signed_url(storage_path: str, expires_in: int = 3600) -> str:
    db = await get_db()
    result = await db.storage.from_(settings.supabase_storage_bucket).create_signed_url(
        storage_path, expires_in
    )
    return result["signedURL"]
