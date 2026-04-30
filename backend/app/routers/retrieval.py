from fastapi import APIRouter, HTTPException

from app.config import settings
from app.retrieval import retrieve
from app.retrieval.retriever import RetrievalUnavailableError
from app.schemas.api import RetrievalSearchRequest, RetrievalSearchResponse

router = APIRouter()


@router.post("/retrieval/search", response_model=RetrievalSearchResponse)
async def search_retrieval(payload: RetrievalSearchRequest):
    try:
        return await retrieve(
            payload.query,
            tenant_id=settings.default_tenant_id,
            filters=payload.filters,
            profile=payload.profile,
            top_k=payload.top_k,
            expand_parents=payload.expand_parents,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RetrievalUnavailableError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
