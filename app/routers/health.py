from fastapi import APIRouter

from app.core.config import settings
from app.dependencies import get_llm_service, get_vector_store

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str | bool | int]:
    llm_service = get_llm_service()
    vector_store = get_vector_store()

    return {
        "status": "ok",
        "llm_configured": llm_service.is_available(),
        "llm_model": settings.gemini_model,
        "catalog_chunks": vector_store.count(),
    }
