from app.schemas.chat import HistoryMessage
from app.services.query_parser import parse_query
from app.services.retrieval_context import build_retrieval_context


def test_parse_price_filter() -> None:
    parsed = parse_query("books under £20")
    assert parsed.price_max == 20
    assert parsed.intent == "list"


def test_parse_count_intent() -> None:
    parsed = parse_query("how many books under 20")
    assert parsed.intent == "count"
    assert parsed.price_max == 20


def test_follow_up_focuses_on_prior_book() -> None:
    history = [
        HistoryMessage(role="user", content="poetry books"),
        HistoryMessage(
            role="assistant",
            content=(
                "Books matching Poetry for: \"poetry books\"\n\n"
                "1. Moonlit Verses\n"
                "   Category: Poetry\n"
                "   Price: £14.00\n"
            ),
        ),
    ]
    context = build_retrieval_context("tell me more", history)
    assert context.is_follow_up is True
    assert context.focused_book_title == "Moonlit Verses"
