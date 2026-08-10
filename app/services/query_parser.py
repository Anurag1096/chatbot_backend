import re
from dataclasses import dataclass
from typing import Literal

from app.services.ingestion_utils import parse_price_value

QueryIntent = Literal["list", "count", "cheapest", "semantic"]

# Canonical category names from scraped catalog (longer aliases first for matching).
CATEGORY_ALIASES: dict[str, str] = {
    "historical fiction": "Historical Fiction",
    "science fiction": "Science Fiction",
    "christian fiction": "Christian Fiction",
    "womens fiction": "Womens Fiction",
    "adult fiction": "Adult Fiction",
    "new adult": "New Adult",
    "young adult": "Young Adult",
    "food and drink": "Food and Drink",
    "self help": "Self Help",
    "short stories": "Short Stories",
    "sequential art": "Sequential Art",
    "sports and games": "Sports and Games",
    "academic": "Academic",
    "autobiography": "Autobiography",
    "biography": "Biography",
    "business": "Business",
    "childrens": "Childrens",
    "children's": "Childrens",
    "christian": "Christian",
    "classics": "Classics",
    "contemporary": "Contemporary",
    "cultural": "Cultural",
    "erotica": "Erotica",
    "fantasy": "Fantasy",
    "fiction": "Fiction",
    "health": "Health",
    "historical": "Historical",
    "history": "History",
    "horror": "Horror",
    "humor": "Humor",
    "music": "Music",
    "mystery": "Mystery",
    "nonfiction": "Nonfiction",
    "novels": "Novels",
    "paranormal": "Paranormal",
    "parenting": "Parenting",
    "philosophy": "Philosophy",
    "poetry": "Poetry",
    "politics": "Politics",
    "psychology": "Psychology",
    "religion": "Religion",
    "romance": "Romance",
    "science": "Science",
    "spirituality": "Spirituality",
    "suspense": "Suspense",
    "thriller": "Thriller",
    "travel": "Travel",
    "art": "Art",
}

PRICE_MAX_PATTERN = re.compile(
    r"(?:under|below|less\s+than|cheaper\s+than|at\s+most|max(?:imum)?|up\s+to)"
    r"\s*(?:£|\$|gbp|pounds?|usd)?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PRICE_MIN_PATTERN = re.compile(
    r"(?:over|above|more\s+than|at\s+least|min(?:imum)?|from)"
    r"\s*(?:£|\$|gbp|pounds?|usd)?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PRICE_BETWEEN_PATTERN = re.compile(
    r"between\s*(?:£|\$|gbp|pounds?|usd)?\s*(\d+(?:\.\d+)?)"
    r"\s*(?:and|-)\s*(?:£|\$|gbp|pounds?|usd)?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
RATING_MIN_PATTERN = re.compile(
    r"(?:rated|rating|at\s+least|minimum)\s*(?:of\s*)?(\d+)\s*(?:\+|stars?|out\s+of\s+5)?",
    re.IGNORECASE,
)
COUNT_PATTERN = re.compile(
    r"\b(?:how\s+many|number\s+of|count(?:\s+of)?)\b",
    re.IGNORECASE,
)
CHEAPEST_PATTERN = re.compile(
    r"\b(?:cheapest|lowest\s+price|most\s+affordable|least\s+expensive)\b",
    re.IGNORECASE,
)

STRIP_PATTERNS = [
    PRICE_BETWEEN_PATTERN,
    PRICE_MAX_PATTERN,
    PRICE_MIN_PATTERN,
    RATING_MIN_PATTERN,
    COUNT_PATTERN,
    CHEAPEST_PATTERN,
    re.compile(r"\b(?:books?|bookstore|catalog(?:ue)?)\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class ParsedQuery:
    semantic_query: str
    price_max: float | None = None
    price_min: float | None = None
    category: str | None = None
    rating_min: int | None = None
    intent: QueryIntent = "semantic"

    @property
    def has_filters(self) -> bool:
        return any(
            value is not None
            for value in (self.price_max, self.price_min, self.category, self.rating_min)
        )


def parse_query(message: str) -> ParsedQuery:
    text = message.strip()
    lowered = text.lower()

    price_min: float | None = None
    price_max: float | None = None
    rating_min: int | None = None
    category: str | None = None

    between_match = PRICE_BETWEEN_PATTERN.search(text)
    if between_match:
        low = float(between_match.group(1))
        high = float(between_match.group(2))
        price_min = min(low, high)
        price_max = max(low, high)
    else:
        max_match = PRICE_MAX_PATTERN.search(text)
        if max_match:
            price_max = float(max_match.group(1))

        min_match = PRICE_MIN_PATTERN.search(text)
        if min_match:
            price_min = float(min_match.group(1))

    rating_match = RATING_MIN_PATTERN.search(text)
    if rating_match:
        rating_min = int(rating_match.group(1))

    category = _match_category(lowered)

    intent: QueryIntent = "semantic"
    if COUNT_PATTERN.search(text):
        intent = "count"
    elif CHEAPEST_PATTERN.search(text):
        intent = "cheapest"
    elif price_max is not None or price_min is not None or category or rating_min is not None:
        intent = "list"

    semantic_query = _build_semantic_query(text, category)

    return ParsedQuery(
        semantic_query=semantic_query,
        price_max=price_max,
        price_min=price_min,
        category=category,
        rating_min=rating_min,
        intent=intent,
    )


def _match_category(lowered: str) -> str | None:
    for alias, canonical in sorted(CATEGORY_ALIASES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return canonical
    return None


def _build_semantic_query(text: str, category: str | None) -> str:
    cleaned = text
    for pattern in STRIP_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)

    if category:
        for alias in CATEGORY_ALIASES:
            if CATEGORY_ALIASES[alias] == category:
                cleaned = re.sub(rf"\b{re.escape(alias)}\b", " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?")
    if cleaned:
        return cleaned

    if category:
        return category

    fallback = parse_price_value(text)
    if fallback is not None:
        return "books"

    return text.strip() or "books"
