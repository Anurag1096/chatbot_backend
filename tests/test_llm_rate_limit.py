from collections.abc import AsyncIterable

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_llm_rate_limiter, get_llm_service, get_vector_store
from app.main import app
from app.services.llm import LLMError, LLMService
from app.services.llm_rate_limiter import LLMRateLimiter
from tests.conftest import parse_sse_body


class MockLLMService:
    def is_available(self) -> bool:
        return True

    async def stream_completion(self, messages: list[dict[str, str]]) -> AsyncIterable[str]:
        yield "Mock LLM reply."


class FailingLLMService:
    def is_available(self) -> bool:
        return True

    async def stream_completion(self, messages: list[dict[str, str]]) -> AsyncIterable[str]:
        raise LLMError("The AI service returned an error. Please try again.")
        yield ""  # pragma: no cover


def test_rate_limiter_allows_up_to_limit() -> None:
    limiter = LLMRateLimiter(limit=2, enabled=True)
    assert limiter.is_allowed("client-a") is True
    limiter.record("client-a")
    assert limiter.is_allowed("client-a") is True
    limiter.record("client-a")
    assert limiter.is_allowed("client-a") is False


def test_rate_limiter_is_per_client() -> None:
    limiter = LLMRateLimiter(limit=1, enabled=True)
    limiter.record("client-a")
    assert limiter.is_allowed("client-a") is False
    assert limiter.is_allowed("client-b") is True


def test_rate_limiter_disabled_always_allows() -> None:
    limiter = LLMRateLimiter(limit=1, enabled=False)
    limiter.record("client-a")
    assert limiter.is_allowed("client-a") is True


@pytest.fixture
def rate_limited_client(populated_store) -> TestClient:
    limiter = LLMRateLimiter(limit=1, enabled=True)
    app.dependency_overrides[get_vector_store] = lambda: populated_store
    app.dependency_overrides[get_llm_service] = lambda: MockLLMService()
    app.dependency_overrides[get_llm_rate_limiter] = lambda: limiter

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    app.dependency_overrides.clear()


def test_rate_limit_falls_back_to_template_with_notice(rate_limited_client) -> None:
    headers = {"Accept": "text/event-stream"}

    first = rate_limited_client.post(
        "/chat",
        json={"message": "poetry books", "history": []},
        headers=headers,
    )
    assert first.status_code == 200
    first_chunks = parse_sse_body(first.text)
    first_text = "".join(chunk.get("delta", "") for chunk in first_chunks if chunk.get("delta"))
    assert "Mock LLM reply." in first_text
    assert not any(chunk.get("error") for chunk in first_chunks)

    second = rate_limited_client.post(
        "/chat",
        json={"message": "poetry books", "history": []},
        headers=headers,
    )
    assert second.status_code == 200
    second_chunks = parse_sse_body(second.text)
    second_text = "".join(chunk.get("delta", "") for chunk in second_chunks if chunk.get("delta"))
    assert "AI rate limit reached" in second_text
    assert "Moonlit Verses" in second_text or "Poetry" in second_text
    assert not any(chunk.get("error") for chunk in second_chunks)


@pytest.fixture
def failing_llm_client(populated_store) -> TestClient:
    limiter = LLMRateLimiter(limit=100, enabled=True)
    app.dependency_overrides[get_vector_store] = lambda: populated_store
    app.dependency_overrides[get_llm_service] = lambda: FailingLLMService()
    app.dependency_overrides[get_llm_rate_limiter] = lambda: limiter

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    app.dependency_overrides.clear()


def test_llm_error_falls_back_to_template_with_notice(failing_llm_client) -> None:
    response = failing_llm_client.post(
        "/chat",
        json={"message": "poetry books", "history": []},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    chunks = parse_sse_body(response.text)
    text = "".join(chunk.get("delta", "") for chunk in chunks if chunk.get("delta"))
    assert "temporarily unavailable" in text
    assert "Moonlit Verses" in text or "Poetry" in text
    assert not any(chunk.get("error") for chunk in chunks)
