from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=settings.chat_max_history_content_length)


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(min_length=1, max_length=settings.chat_max_message_length)
    conversation_id: str | None = Field(default=None, alias="conversationId")
    history: list[HistoryMessage] = Field(default_factory=list, max_length=settings.chat_max_history_turns)


class StreamChunk(BaseModel):
    delta: str | None = None
    error: str | None = None
    done: bool | None = None
