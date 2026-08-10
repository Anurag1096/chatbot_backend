from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.services.chat import ChatService
from app.services.llm import LLMService
from app.services.llm_rate_limiter import LLMRateLimiter
from app.services.vector_store import VectorStoreService


@lru_cache
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()


@lru_cache
def get_llm_rate_limiter() -> LLMRateLimiter:
    return LLMRateLimiter()


def get_chat_service(
    vector_store: Annotated[VectorStoreService, Depends(get_vector_store)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    llm_rate_limiter: Annotated[LLMRateLimiter, Depends(get_llm_rate_limiter)],
) -> ChatService:
    return ChatService(
        vector_store=vector_store,
        llm_service=llm_service,
        llm_rate_limiter=llm_rate_limiter,
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
