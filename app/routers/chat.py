from collections.abc import AsyncIterable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.sse import EventSourceResponse

from app.core.client import get_client_key
from app.dependencies import ChatServiceDep
from app.schemas.chat import ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat", response_class=EventSourceResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatServiceDep,
    http_request: Request,
) -> AsyncIterable[dict[str, Any]]:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail={"message": "message is required"})

    client_key = get_client_key(http_request)
    async for chunk in chat_service.stream_response(message, request.history, client_key):
        yield chunk.model_dump(exclude_none=True)
