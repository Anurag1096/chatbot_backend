import asyncio
import re
from collections.abc import AsyncIterable

from app.core.config import settings
from app.schemas.chat import HistoryMessage, StreamChunk
from app.services.vector_store import VectorStoreService


class ChatService:
    def __init__(self, vector_store: VectorStoreService | None = None) -> None:
        self.vector_store = vector_store

    async def build_reply(self, message: str, history: list[HistoryMessage]) -> str:
        if self.vector_store:
            matches = await asyncio.to_thread(self._search_catalog, message)
            if matches:
                return self._build_rag_reply(message, history, matches)

        return self._build_placeholder_reply(message, history)

    def _search_catalog(self, message: str) -> list[dict]:
        if self.vector_store is None or self.vector_store.count() == 0:
            return []
        return self.vector_store.search(message)

    def _build_rag_reply(
        self,
        message: str,
        history: list[HistoryMessage],
        matches: list[dict],
    ) -> str:
        prior_turns = len(history)
        lines = [
            f"Based on the bookstore catalog, here is what I found for: \"{message.strip()}\"",
            "",
        ]

        for index, match in enumerate(matches, start=1):
            metadata = match.get("metadata", {})
            title = metadata.get("book_title") or "Unknown book"
            price = metadata.get("price") or "Price unavailable"
            rating = metadata.get("rating") or "Rating unavailable"
            category = metadata.get("category") or "Uncategorized"

            lines.append(f"{index}. {title}")
            lines.append(f"   Category: {category}")
            lines.append(f"   Price: {price}")
            lines.append(f"   Rating: {rating}")

            description = self._extract_description(match.get("text", ""))
            if description:
                lines.append(f"   Summary: {description}")

            product_url = metadata.get("product_url")
            if product_url:
                lines.append(f"   URL: {product_url}")
            lines.append("")

        if prior_turns > 0:
            lines.append(f"(Using {prior_turns} prior turn(s) from this conversation.)")

        return "\n".join(lines).strip()

    def _build_placeholder_reply(self, message: str, history: list[HistoryMessage]) -> str:
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

    def _extract_description(self, text: str) -> str:
        match = re.search(r"Description:\s*(.+?)(?:\nProduct-URL:|\Z)", text, re.DOTALL)
        if not match:
            return ""

        description = re.sub(r"\s+", " ", match.group(1)).strip()
        if len(description) > 220:
            description = description[:217].rstrip() + "..."
        return description

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\S+\s*|\s+", text) or [text]

    async def stream_response(
        self,
        message: str,
        history: list[HistoryMessage],
    ) -> AsyncIterable[StreamChunk]:
        reply = await self.build_reply(message, history)

        for token in self.tokenize(reply):
            yield StreamChunk(delta=token)
            await asyncio.sleep(settings.stream_token_delay_seconds)

        yield StreamChunk(done=True)
