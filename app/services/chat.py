import asyncio
import re
from collections.abc import AsyncIterable

from app.core.config import settings
from app.schemas.chat import HistoryMessage, StreamChunk


class ChatService:
    def build_reply(self, message: str, history: list[HistoryMessage]) -> str:
        prior_turns = len(history)
        trimmed = message.strip()

        follow_up = re.match(
            r"^(what about that|tell me more|explain that|and\?)",
            trimmed,
            re.IGNORECASE,
        )

        if follow_up and prior_turns > 0:
            last_user = next(
                (item.content for item in reversed(history) if item.role == "user"),
                None,
            )
            if last_user:
                return (
                    f'Following up on your earlier question ("{last_user}"): '
                    f'here is a placeholder answer for "{trimmed}".'
                )

        return (
            f'Placeholder reply to "{trimmed}". '
            f"I received {prior_turns} prior message(s) as conversation history."
        )

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\S+\s*|\s+", text) or [text]

    async def stream_response(
        self,
        message: str,
        history: list[HistoryMessage],
    ) -> AsyncIterable[StreamChunk]:
        reply = self.build_reply(message, history)

        for token in self.tokenize(reply):
            yield StreamChunk(delta=token)
            await asyncio.sleep(settings.stream_token_delay_seconds)

        yield StreamChunk(done=True)
