from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str
    conversation_id: str | None = Field(default=None, alias="conversationId")
    history: list[HistoryMessage] = []


class StreamChunk(BaseModel):
    delta: str | None = None
    error: str | None = None
    done: bool | None = None
