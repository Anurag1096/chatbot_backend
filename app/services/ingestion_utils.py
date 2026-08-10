import re

PRICE_VALUE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")
RATING_VALUE_PATTERN = re.compile(r"(\d+)\s*out\s*of\s*5", re.IGNORECASE)


def parse_price_value(price: str) -> float | None:
    if not price:
        return None
    match = PRICE_VALUE_PATTERN.search(price.replace(",", ""))
    if match is None:
        return None
    return float(match.group(1))


def parse_rating_value(rating: str) -> int | None:
    if not rating:
        return None
    match = RATING_VALUE_PATTERN.search(rating)
    if match is None:
        return None
    return int(match.group(1))


def dedupe_repeated_text(text: str) -> str:
    """Remove back-to-back repeated description text common on books.toscrape.com."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"\s*\.\.\.more\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    if len(cleaned) < 80:
        return cleaned

    # Site often repeats the same opening block twice; the second copy is usually complete.
    for length in range(min(120, len(cleaned) - 1), 49, -1):
        for start in range(min(20, len(cleaned) - length)):
            prefix = cleaned[start : start + length]
            first = cleaned.find(prefix)
            second = cleaned.find(prefix, first + len(prefix))
            if second == -1:
                continue

            first_copy = cleaned[first:second]
            second_copy = cleaned[second:]
            third = second_copy.find(prefix, len(prefix))
            if third != -1:
                second_copy = second_copy[:third]

            best = first_copy if len(first_copy) >= len(second_copy) else second_copy
            return best.strip()

    return cleaned
