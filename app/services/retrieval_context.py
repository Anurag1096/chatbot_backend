import re
from dataclasses import dataclass

from app.core.config import settings
from app.schemas.chat import HistoryMessage
from app.services.query_parser import ParsedQuery, parse_query

CONTINUATION_PATTERN = re.compile(
    r"^(?:"
    r"tell me more(?: about (?:it|that|this|them))?"
    r"|more (?:details|info|information)(?: about (?:it|that|this|them))?"
    r"|explain (?:that|it|this|them)"
    r"|go on"
    r"|and\?"
    r"|what else(?: can you tell me)?"
    r"|anything else"
    r"|continue"
    r"|expand on that"
    r")(?:[.!?]|$)",
    re.IGNORECASE,
)
TOPIC_SHIFT_PATTERN = re.compile(
    r"^(?:what about|how about|any|show me|find me|do you have)\b",
    re.IGNORECASE,
)
BOOK_LINE_PATTERN = re.compile(r"^\d+\.\s+(.+?)\s*$", re.MULTILINE)
PRONOUN_REFERENCE_PATTERN = re.compile(
    r"\b(?:that one|this one|the book|that book|this book|it|them|those)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RetrievalContext:
    semantic_query: str
    parsed: ParsedQuery
    is_follow_up: bool = False
    prior_user_query: str | None = None
    focused_book_title: str | None = None
    inherited_filters: bool = False


def build_retrieval_context(message: str, history: list[HistoryMessage]) -> RetrievalContext:
    current_parsed = parse_query(message)
    prior_user_query = _last_user_message(history)
    is_continuation = bool(CONTINUATION_PATTERN.match(message.strip()))
    is_topic_shift = bool(TOPIC_SHIFT_PATTERN.match(message.strip()))
    is_follow_up = is_continuation or is_topic_shift or _has_pronoun_reference(message)

    if not history or not is_follow_up:
        semantic_query = _build_semantic_query_from_history(message, history, current_parsed)
        return RetrievalContext(
            semantic_query=semantic_query,
            parsed=current_parsed,
        )

    if is_continuation:
        focused_title = _focused_book_title(history)
        semantic_query = focused_title or prior_user_query or message
        return RetrievalContext(
            semantic_query=semantic_query,
            parsed=ParsedQuery(semantic_query=semantic_query, intent="semantic"),
            is_follow_up=True,
            prior_user_query=prior_user_query,
            focused_book_title=focused_title,
        )

    merged_parsed = _merge_with_prior_filters(current_parsed, prior_user_query)
    semantic_query = _build_semantic_query_from_history(message, history, merged_parsed)
    return RetrievalContext(
        semantic_query=semantic_query,
        parsed=merged_parsed,
        is_follow_up=True,
        prior_user_query=prior_user_query,
        inherited_filters=merged_parsed.has_filters and not current_parsed.has_filters,
    )


def context_note(context: RetrievalContext) -> str | None:
    if not context.is_follow_up:
        return None

    if context.focused_book_title:
        return f'Following up on "{context.focused_book_title}"'

    if context.prior_user_query and context.inherited_filters:
        return f'Following up on your earlier question ("{context.prior_user_query}")'

    if context.prior_user_query:
        return f'In context of your earlier question ("{context.prior_user_query}")'

    return "Following up on the previous answer"


def _build_semantic_query_from_history(
    message: str,
    history: list[HistoryMessage],
    parsed: ParsedQuery,
) -> str:
    if not history:
        return parsed.semantic_query or message.strip()

    recent_user_messages = _recent_user_messages(history, settings.retrieval_history_turns)
    if not recent_user_messages:
        return parsed.semantic_query or message.strip()

    current_semantic = parsed.semantic_query or message.strip()
    if current_semantic in recent_user_messages:
        parts = recent_user_messages
    else:
        parts = [*recent_user_messages, current_semantic]

    combined = ". ".join(part.strip() for part in parts if part.strip())
    return combined or current_semantic


def _merge_with_prior_filters(current: ParsedQuery, prior_user_query: str | None) -> ParsedQuery:
    if not prior_user_query:
        return current

    prior = parse_query(prior_user_query)
    return ParsedQuery(
        semantic_query=current.semantic_query,
        price_max=current.price_max if current.price_max is not None else prior.price_max,
        price_min=current.price_min if current.price_min is not None else prior.price_min,
        category=current.category or prior.category,
        rating_min=current.rating_min if current.rating_min is not None else prior.rating_min,
        intent=current.intent if current.intent != "semantic" else prior.intent,
    )


def _focused_book_title(history: list[HistoryMessage]) -> str | None:
    last_assistant = _last_assistant_message(history)
    if not last_assistant:
        return None

    titles = extract_book_titles(last_assistant)
    if not titles:
        return None
    return titles[0]


def extract_book_titles(text: str) -> list[str]:
    return [match.group(1).strip() for match in BOOK_LINE_PATTERN.finditer(text)]


def _recent_user_messages(history: list[HistoryMessage], max_turns: int) -> list[str]:
    messages: list[str] = []
    for item in reversed(history):
        if item.role != "user":
            continue
        content = item.content.strip()
        if content:
            messages.append(content)
        if len(messages) >= max_turns:
            break
    messages.reverse()
    return messages


def _last_user_message(history: list[HistoryMessage]) -> str | None:
    for item in reversed(history):
        if item.role == "user" and item.content.strip():
            return item.content.strip()
    return None


def _last_assistant_message(history: list[HistoryMessage]) -> str | None:
    for item in reversed(history):
        if item.role == "assistant" and item.content.strip():
            return item.content.strip()
    return None


def _has_pronoun_reference(message: str) -> bool:
    return bool(PRONOUN_REFERENCE_PATTERN.search(message.strip()))
