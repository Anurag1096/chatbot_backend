import asyncio
import re
from collections.abc import AsyncIterable

from app.core.config import settings
from app.schemas.chat import HistoryMessage, StreamChunk
from app.services.catalog_search import CatalogSearchResult
from app.services.llm import LLMError, LLMService
from app.services.llm_rate_limiter import LLMRateLimiter
from app.services.query_parser import ParsedQuery
from app.services.rag_prompt import build_rag_messages
from app.services.retrieval_context import build_retrieval_context, context_note
from app.services.vector_store import VectorStoreService


class ChatService:
    def __init__(
        self,
        vector_store: VectorStoreService | None = None,
        llm_service: LLMService | None = None,
        llm_rate_limiter: LLMRateLimiter | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.llm_rate_limiter = llm_rate_limiter or LLMRateLimiter()

    async def build_reply(
        self,
        message: str,
        history: list[HistoryMessage],
        client_key: str = "unknown",
    ) -> str:
        search_result = await self._run_search(message, history)
        if search_result is None:
            return self._build_placeholder_reply(message, history)

        if self._should_use_llm(search_result):
            if not self.llm_rate_limiter.is_allowed(client_key):
                notice = f"{settings.llm_rate_limit_message}\n\n"
                return notice + self._build_rag_reply(message, history, search_result)

            self.llm_rate_limiter.record(client_key)
            reply = await self._generate_llm_reply(message, history, search_result)
            if reply is not None:
                return reply

        return self._build_rag_reply(message, history, search_result)

    async def stream_response(
        self,
        message: str,
        history: list[HistoryMessage],
        client_key: str = "unknown",
    ) -> AsyncIterable[StreamChunk]:
        search_result = await self._run_search(message, history)
        if search_result is None:
            reply = self._build_placeholder_reply(message, history)
            async for chunk in self._stream_template_reply(reply):
                yield chunk
            return

        if self._should_use_llm(search_result):
            if not self.llm_rate_limiter.is_allowed(client_key):
                notice = f"{settings.llm_rate_limit_message}\n\n"
                reply = notice + self._build_rag_reply(message, history, search_result)
                async for chunk in self._stream_template_reply(reply):
                    yield chunk
                return

            try:
                self.llm_rate_limiter.record(client_key)
                messages = build_rag_messages(
                    message,
                    history,
                    search_result,
                    catalog_total=self._catalog_total(),
                )
                async for token in self.llm_service.stream_completion(messages):
                    yield StreamChunk(delta=token)
                yield StreamChunk(done=True)
                return
            except LLMError:
                notice = f"{settings.llm_error_fallback_message}\n\n"
                reply = notice + self._build_rag_reply(message, history, search_result)
                async for chunk in self._stream_template_reply(reply):
                    yield chunk
                return

        reply = self._build_rag_reply(message, history, search_result)
        async for chunk in self._stream_template_reply(reply):
            yield chunk

    async def _run_search(
        self,
        message: str,
        history: list[HistoryMessage],
    ) -> CatalogSearchResult | None:
        if not self.vector_store:
            return None

        search_result = await asyncio.to_thread(self._search_catalog, message, history)
        if (
            search_result.count is not None
            or search_result.matches
            or search_result.parsed.has_filters
        ):
            return search_result
        return None

    def _should_use_llm(self, search_result: CatalogSearchResult) -> bool:
        return self.llm_service is not None and self.llm_service.is_available()

    async def _generate_llm_reply(
        self,
        message: str,
        history: list[HistoryMessage],
        search_result: CatalogSearchResult,
    ) -> str | None:
        try:
            messages = build_rag_messages(
                message,
                history,
                search_result,
                catalog_total=self._catalog_total(),
            )
            parts: list[str] = []
            async for token in self.llm_service.stream_completion(messages):
                parts.append(token)
            return "".join(parts).strip() or None
        except LLMError:
            return None

    async def _stream_template_reply(self, reply: str) -> AsyncIterable[StreamChunk]:
        for token in self.tokenize(reply):
            yield StreamChunk(delta=token)
            await asyncio.sleep(settings.stream_token_delay_seconds)
        yield StreamChunk(done=True)

    def _search_catalog(self, message: str, history: list[HistoryMessage]) -> CatalogSearchResult:
        if self.vector_store is None or self.vector_store.count() == 0:
            context = build_retrieval_context(message, history)
            return CatalogSearchResult(matches=[], parsed=context.parsed)

        context = build_retrieval_context(message, history)
        parsed = context.parsed
        query = context.semantic_query or message
        follow_up_note = context_note(context)

        if parsed.intent == "count":
            count = self.vector_store.count_filtered(
                price_max=parsed.price_max,
                price_min=parsed.price_min,
                category=parsed.category,
                rating_min=parsed.rating_min,
            )
            examples, used_fallback = self.vector_store.search_filtered(
                query,
                price_max=parsed.price_max,
                price_min=parsed.price_min,
                category=parsed.category,
                rating_min=parsed.rating_min,
                top_k=settings.chroma_top_k,
                fallback_semantic=False,
            )
            return CatalogSearchResult(
                matches=examples,
                parsed=parsed,
                count=count,
                used_fallback=used_fallback,
                follow_up_note=follow_up_note,
            )

        if parsed.intent == "cheapest":
            matches, used_fallback = self.vector_store.search_filtered(
                query,
                price_max=parsed.price_max,
                price_min=parsed.price_min,
                category=parsed.category,
                rating_min=parsed.rating_min,
                top_k=settings.chroma_cheapest_top_k,
                sort_by_price="asc",
            )
            return CatalogSearchResult(
                matches=matches,
                parsed=parsed,
                used_fallback=used_fallback,
                follow_up_note=follow_up_note,
            )

        if parsed.has_filters:
            matches, used_fallback = self.vector_store.search_filtered(
                query,
                price_max=parsed.price_max,
                price_min=parsed.price_min,
                category=parsed.category,
                rating_min=parsed.rating_min,
                top_k=settings.chroma_filter_top_k,
                fallback_semantic=not self._strict_price_filter(parsed),
            )
            return CatalogSearchResult(
                matches=matches,
                parsed=parsed,
                used_fallback=used_fallback,
                follow_up_note=follow_up_note,
            )

        top_k = settings.chroma_top_k
        if context.is_follow_up and context.focused_book_title:
            top_k = 1

        matches = self.vector_store.search(query, top_k=top_k)
        return CatalogSearchResult(
            matches=matches,
            parsed=parsed,
            follow_up_note=follow_up_note,
        )

    def _build_rag_reply(
        self,
        message: str,
        history: list[HistoryMessage],
        search_result: CatalogSearchResult,
    ) -> str:
        prior_turns = len(history)
        parsed = search_result.parsed
        matches = search_result.matches
        lines: list[str] = []

        if search_result.follow_up_note:
            lines.append(search_result.follow_up_note)
            lines.append("")

        if parsed.intent == "count" and search_result.count is not None:
            lines.append(self._build_count_header(message, parsed, search_result.count))
            lines.append("")
            if search_result.count == 0:
                lines.append("No books in the catalog match those filters.")
            elif matches:
                lines.append("Examples:")
                lines.append("")
                lines.extend(self._format_match_lines(matches))
            return self._finalize_reply(lines, prior_turns)

        header = self._build_list_header(message, parsed, search_result.used_fallback)
        lines.append(header)
        lines.append("")

        if not matches:
            lines.append("No books in the catalog match those filters.")
            return self._finalize_reply(lines, prior_turns)

        lines.extend(self._format_match_lines(matches))
        return self._finalize_reply(lines, prior_turns)

    def _build_count_header(self, message: str, parsed: ParsedQuery, count: int) -> str:
        filter_text = self._describe_filters(parsed)
        if filter_text:
            return (
                f'For "{message.strip()}": there are {count} book(s) in the catalog '
                f"matching {filter_text} (from {self._catalog_total()} total)."
            )
        return (
            f'For "{message.strip()}": there are {count} book(s) in the catalog '
            f"(from {self._catalog_total()} total)."
        )

    def _build_list_header(
        self,
        message: str,
        parsed: ParsedQuery,
        used_fallback: bool,
    ) -> str:
        if parsed.intent == "cheapest":
            filter_text = self._describe_filters(parsed)
            prefix = f"Cheapest books{' matching ' + filter_text if filter_text else ''}"
            header = f'{prefix} for: "{message.strip()}"'
        else:
            filter_text = self._describe_filters(parsed)
            if filter_text:
                header = f'Books matching {filter_text} for: "{message.strip()}"'
            else:
                header = f'Based on the bookstore catalog, here is what I found for: "{message.strip()}"'

        if used_fallback:
            header += " (no exact filter matches; showing similar results instead)"
        return header

    def _describe_filters(self, parsed: ParsedQuery) -> str:
        parts: list[str] = []
        if parsed.category:
            parts.append(parsed.category)
        if parsed.price_min is not None and parsed.price_max is not None:
            parts.append(f"£{parsed.price_min:g}–£{parsed.price_max:g}")
        elif parsed.price_max is not None:
            parts.append(f"under £{parsed.price_max:g}")
        elif parsed.price_min is not None:
            parts.append(f"over £{parsed.price_min:g}")
        if parsed.rating_min is not None:
            parts.append(f"rating {parsed.rating_min}+ stars")
        return ", ".join(parts)

    def _format_match_lines(self, matches: list[dict]) -> list[str]:
        lines: list[str] = []
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
        return lines

    def _strict_price_filter(self, parsed: ParsedQuery) -> bool:
        return parsed.price_max is not None or parsed.price_min is not None

    def _catalog_total(self) -> int:
        if self.vector_store is None:
            return 0
        return self.vector_store.count()

    def _finalize_reply(self, lines: list[str], prior_turns: int) -> str:
        if prior_turns > 0:
            lines.append(f"(Using {prior_turns} prior turn(s) from this conversation.)")
        return "\n".join(lines).strip()

    def _build_placeholder_reply(self, message: str, history: list[HistoryMessage]) -> str:
        prior_turns = len(history)
        trimmed = message.strip()

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
