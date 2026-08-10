from collections.abc import AsyncIterable

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from app.core.config import settings


class LLMError(Exception):
    """User-safe LLM failure."""


class LLMService:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        if settings.llm_enabled:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
            )

    def is_available(self) -> bool:
        return self._client is not None

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncIterable[str]:
        if self._client is None:
            raise LLMError("LLM is not configured.")

        try:
            stream = await self._client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                max_tokens=settings.openai_max_tokens,
                temperature=settings.openai_temperature,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except RateLimitError as exc:
            raise LLMError("The AI service is rate limited. Please try again shortly.") from exc
        except APITimeoutError as exc:
            raise LLMError("The AI service timed out. Please try again.") from exc
        except APIConnectionError as exc:
            raise LLMError("Could not reach the AI service. Please try again.") from exc
        except APIError as exc:
            raise LLMError("The AI service returned an error. Please try again.") from exc
