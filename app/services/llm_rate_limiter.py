import threading
import time
from collections import deque

from app.core.config import settings


class LLMRateLimiter:
    """In-memory sliding-window rate limiter for LLM calls."""

    def __init__(
        self,
        *,
        limit: int | None = None,
        window_seconds: float = 60.0,
        enabled: bool | None = None,
    ) -> None:
        self._limit = limit if limit is not None else settings.llm_rate_limit_per_minute
        self._window_seconds = window_seconds
        self._enabled = enabled if enabled is not None else settings.llm_rate_limit_enabled
        self._requests: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_key: str) -> bool:
        if not self._enabled:
            return True

        key = client_key or "unknown"
        now = time.monotonic()

        with self._lock:
            timestamps = self._requests.setdefault(key, deque())
            self._prune(timestamps, now)
            return len(timestamps) < self._limit

    def record(self, client_key: str) -> None:
        if not self._enabled:
            return

        key = client_key or "unknown"
        now = time.monotonic()

        with self._lock:
            timestamps = self._requests.setdefault(key, deque())
            self._prune(timestamps, now)
            timestamps.append(now)

    def _prune(self, timestamps: deque[float], now: float) -> None:
        cutoff = now - self._window_seconds
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
