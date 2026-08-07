from collections.abc import AsyncIterable
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.sse import EventSourceResponse

from app.dependencies import ChatServiceDep
from app.schemas.chat import ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat", response_class=EventSourceResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatServiceDep,
) -> AsyncIterable[dict[str, Any]]:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail={"message": "message is required"})

    async for chunk in chat_service.stream_response(message, request.history):
        yield chunk.model_dump(exclude_none=True)
