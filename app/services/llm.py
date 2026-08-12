import logging
from collections.abc import AsyncIterable

from google import genai
from google.genai import errors as genai_errors

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """User-safe LLM failure."""


def _to_interactions_input(
    messages: list[dict[str, str]],
) -> tuple[str | None, str | list[dict[str, object]]]:
    system_parts: list[str] = []
    turns: list[dict[str, object]] = []

    for message in messages:
        role = message["role"]
        text = message["content"]
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            turns.append(
                {
                    "type": "user_input",
                    "content": [{"type": "text", "text": text}],
                }
            )
        elif role == "assistant":
            turns.append(
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": text}],
                }
            )

    system_instruction = "\n\n".join(system_parts) if system_parts else None

    if not turns:
        raise LLMError("No valid messages were provided to the AI service.")

    if len(turns) == 1 and turns[0]["type"] == "user_input":
        content = turns[0]["content"]
        if isinstance(content, list) and content and isinstance(content[0], dict):
            return system_instruction, str(content[0]["text"])

    return system_instruction, turns


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

        system_instruction, interaction_input = _to_interactions_input(messages)

        try:
            stream = await self._client.aio.interactions.create(
                model=settings.gemini_model,
                input=interaction_input,
                system_instruction=system_instruction,
                stream=True,
                store=False,
                generation_config={
                    "max_output_tokens": settings.gemini_max_tokens,
                    "temperature": settings.gemini_temperature,
                },
            )

            async for event in stream:
                event_type = getattr(event, "event_type", None)

                if event_type == "error":
                    error = getattr(event, "error", None)
                    message = getattr(error, "message", None) if error else None
                    logger.warning("Gemini interaction error event: %s", message or event)
                    raise LLMError("The AI service returned an error. Please try again.")

                if event_type != "step.delta":
                    continue

                delta = getattr(event, "delta", None)
                if delta is None or getattr(delta, "type", None) != "text":
                    continue

                text = getattr(delta, "text", None)
                if text:
                    yield text
        except genai_errors.ClientError as exc:
            logger.warning(
                "Gemini client error (code=%s): %s",
                exc.code,
                exc.message,
            )
            if exc.code == 429:
                raise LLMError(
                    "The AI service is rate limited. Please try again shortly.",
                ) from exc
            raise LLMError("The AI service returned an error. Please try again.") from exc
        except genai_errors.ServerError as exc:
            logger.warning("Gemini server error: %s", exc)
            raise LLMError("The AI service is temporarily unavailable.") from exc
        except genai_errors.APIError as exc:
            logger.warning("Gemini API error: %s", exc)
            raise LLMError("Could not reach the AI service. Please try again.") from exc
        except LLMError:
            raise
        except Exception as exc:
            logger.exception("Unexpected Gemini error")
            raise LLMError("The AI service returned an error. Please try again.") from exc
