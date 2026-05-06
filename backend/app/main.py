from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings
from app.db import close_db, get_db
from app.qdrant import set_qdrant, get_qdrant  # noqa: F401 — re-exported for convenience
from app.retrieval.cache import close_retrieval_cache
from app.retrieval.reranker import close_reranker
from app.routers import chat, documents, health, ingestions, retrieval

# Payload fields we filter on — must be indexed in Qdrant
_PAYLOAD_INDEXES = {
    "document_id": PayloadSchemaType.KEYWORD,
    "tenant_id": PayloadSchemaType.KEYWORD,
    "section_type": PayloadSchemaType.KEYWORD,
    "doc_type": PayloadSchemaType.KEYWORD,
    "specialties": PayloadSchemaType.KEYWORD,
    "conditions_codes": PayloadSchemaType.KEYWORD,
    "geographic_scope": PayloadSchemaType.KEYWORD,
    "care_setting": PayloadSchemaType.KEYWORD,
    "has_dosing_tables": PayloadSchemaType.BOOL,
    "has_red_flags": PayloadSchemaType.BOOL,
}


async def _ensure_qdrant_collection(client: AsyncQdrantClient) -> None:
    exists = await client.collection_exists(settings.qdrant_collection)
    if not exists:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": VectorParams(size=settings.embedding_dimensions, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )

    # Ensure payload indexes exist (idempotent — safe to call on existing collection)
    collection_info = await client.get_collection(settings.qdrant_collection)
    existing_indexes = set(collection_info.payload_schema.keys()) if collection_info.payload_schema else set()
    for field, field_schema in _PAYLOAD_INDEXES.items():
        if field not in existing_indexes:
            await client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=field_schema,
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_request_timeout_seconds,
    )
    set_qdrant(client)
    await _ensure_qdrant_collection(client)
    await get_db()
    
    # Warm up Qdrant connection pool
    try:
        await client.scroll(collection_name=settings.qdrant_collection, limit=1)
    except Exception:
        pass
        
    yield
    await close_reranker()
    await close_retrieval_cache()
    await client.close()
    await close_db()


app = FastAPI(title="Clintel Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingestions.router, prefix="/v1")
app.include_router(documents.router, prefix="/v1")
app.include_router(chat.router, prefix="/v1")
app.include_router(retrieval.router, prefix="/v1")
