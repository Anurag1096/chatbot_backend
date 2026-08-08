from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.services.chat import ChatService
from app.services.vector_store import VectorStoreService


@lru_cache
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()


def get_chat_service(
    vector_store: Annotated[VectorStoreService, Depends(get_vector_store)],
) -> ChatService:
    return ChatService(vector_store=vector_store)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
