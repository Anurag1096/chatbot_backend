from app.services.document_chunker import DocumentChunker
from app.services.vector_store import VectorStoreService


def test_chunk_site_creates_chunks(chunked_payload: dict) -> None:
    result = chunked_payload["_chunk_result"]
    assert result.chunk_count > 0
    assert len(chunked_payload["chunks"]) == result.chunk_count


def test_chunks_have_numeric_metadata(chunked_payload: dict) -> None:
    for chunk in chunked_payload["chunks"]:
        metadata = chunk["metadata"]
        assert "price_value" in metadata
        assert "rating_value" in metadata


def test_ingest_and_count(populated_store: VectorStoreService, chunked_payload: dict) -> None:
    assert populated_store.count() == len(chunked_payload["chunks"])


def test_search_finds_poetry_book(populated_store: VectorStoreService) -> None:
    results = populated_store.search("poetry", top_k=3)
    titles = [item["metadata"].get("book_title", "") for item in results]
    assert any("Moonlit Verses" in title for title in titles)


def test_count_filtered_under_twenty(populated_store: VectorStoreService) -> None:
    count = populated_store.count_filtered(price_max=20)
    assert count == 1
