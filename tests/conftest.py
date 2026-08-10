import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_llm_service, get_vector_store
from app.main import app
from app.services.document_chunker import DocumentChunker
from app.services.llm import LLMService
from app.services.vector_store import VectorStoreService

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "mini_site"


@pytest.fixture
def mini_scraped_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def tmp_chroma_dir(tmp_path: Path) -> Path:
    return tmp_path / "chroma"


@pytest.fixture
def chunk_output_dir(tmp_path: Path) -> Path:
    return tmp_path / "chunks"


@pytest.fixture
def chunked_payload(mini_scraped_dir: Path, chunk_output_dir: Path) -> dict:
    chunker = DocumentChunker(
        scraped_dir=mini_scraped_dir,
        output_dir=chunk_output_dir,
    )
    result = chunker.chunk_site()
    payload = json.loads(result.output_file.read_text(encoding="utf-8"))
    payload["_chunk_result"] = result
    return payload


@pytest.fixture
def populated_store(chunked_payload: dict, tmp_chroma_dir: Path) -> VectorStoreService:
    store = VectorStoreService(
        persist_dir=tmp_chroma_dir,
        collection_name="test_books",
    )
    store.ingest_chunks(chunked_payload["chunks"])
    return store


@pytest.fixture
def empty_store(tmp_chroma_dir: Path) -> VectorStoreService:
    return VectorStoreService(
        persist_dir=tmp_chroma_dir,
        collection_name="test_books_empty",
    )


@pytest.fixture
def test_client(populated_store: VectorStoreService) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_vector_store] = lambda: populated_store
    app.dependency_overrides[get_llm_service] = lambda: LLMService()

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def empty_client(empty_store: VectorStoreService) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_vector_store] = lambda: empty_store
    app.dependency_overrides[get_llm_service] = lambda: LLMService()

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    app.dependency_overrides.clear()


def parse_sse_body(body: str) -> list[dict]:
    chunks: list[dict] = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw:
            continue
        chunks.append(json.loads(raw))
    return chunks
