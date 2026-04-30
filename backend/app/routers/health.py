from fastapi import APIRouter
from app.db import get_db
from app.qdrant import get_qdrant
from app.config import settings

router = APIRouter()


@router.get("/health")
async def health():
    checks: dict = {
        "ok": True,
        "status": "healthy",
        "qdrant": "unknown",
        "supabase": "unknown",
        "qdrant_payload_indexes": [],
        "corpus_version": None,
    }
    try:
        qdrant = get_qdrant()
        collection = await qdrant.get_collection(settings.qdrant_collection)
        checks["qdrant"] = "ok"
        checks["qdrant_payload_indexes"] = sorted((collection.payload_schema or {}).keys())
    except Exception as exc:
        checks["qdrant"] = f"error: {exc}"
        checks["status"] = "degraded"
        checks["ok"] = False

    try:
        db = await get_db()
        result = await db.table("corpus_config").select("corpus_version").limit(1).execute()
        checks["supabase"] = "ok"
        rows = result.data or []
        if rows:
            checks["corpus_version"] = rows[0].get("corpus_version")
    except Exception as exc:
        checks["supabase"] = f"error: {exc}"
        checks["status"] = "degraded"
        checks["ok"] = False

    return checks
