from dataclasses import dataclass

from app.services.query_parser import ParsedQuery


@dataclass
class CatalogSearchResult:
    matches: list[dict]
    parsed: ParsedQuery
    count: int | None = None
    used_fallback: bool = False
    follow_up_note: str | None = None
