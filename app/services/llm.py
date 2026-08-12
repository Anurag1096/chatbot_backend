from collections.abc import AsyncIterable

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import settings


class LLMError(Exception):
    """User-safe LLM failure."""


def _to_gemini_format(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[types.Content]]:
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for message in messages:
        role = message["role"]
        text = message["content"]
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=text)]))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


class LLMService:
    def __init__(self) -> None:
        self._client: genai.Client | None = None
        if settings.llm_enabled:
            self._client = genai.Client(api_key=settings.gemini_api_key)

    def is_available(self) -> bool:
        return self._client is not None

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncIterable[str]:
        if self._client is None:
            raise LLMError("LLM is not configured.")

        system_instruction, contents = _to_gemini_format(messages)

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=settings.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=settings.gemini_max_tokens,
                    temperature=settings.gemini_temperature,
                ),
            )

            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except genai_errors.ClientError as exc:
            if exc.code == 429:
                raise LLMError(
                    "The AI service is rate limited. Please try again shortly.",
                ) from exc
            raise LLMError("The AI service returned an error. Please try again.") from exc
        except genai_errors.ServerError as exc:
            raise LLMError("The AI service is temporarily unavailable.") from exc
        except genai_errors.APIError as exc:
            raise LLMError("Could not reach the AI service. Please try again.") from exc
