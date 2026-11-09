"""Small process-local HTTP controls for the Flask modular monolith.

The limiter deliberately bounds one web process only.  A shared gateway or
durable rate-limit store is required before horizontally scaled deployment;
the response behavior and headers are still testable locally today.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import threading
import time
from uuid import uuid4


REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")


def request_id_from_header(value: object) -> str:
    """Use a safe client correlation value or create an opaque server value."""
    if isinstance(value, str) and REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return str(uuid4())


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int


class FixedWindowRateLimiter:
    """Thread-safe, bounded fixed-window limiter for one process."""

    def __init__(self, *, max_keys: int = 10_000) -> None:
        self.max_keys = max_keys
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[int, int]] = {}

    def check(self, key: str, *, limit: int, window_seconds: int, now: float | None = None) -> RateLimitDecision:
        if limit < 1 or window_seconds < 1:
            raise ValueError("Rate-limit limit and window must both be positive integers.")
        now = time.monotonic() if now is None else now
        window = int(now // window_seconds)
        elapsed = now - window * window_seconds
        retry_after = max(1, math.ceil(window_seconds - elapsed))
        with self._lock:
            previous_window, count = self._buckets.get(key, (window, 0))
            if previous_window != window:
                count = 0
            if count >= limit:
                self._buckets[key] = (window, count)
                return RateLimitDecision(False, limit, 0, retry_after)
            count += 1
            if key not in self._buckets and len(self._buckets) >= self.max_keys:
                # Bounded-memory safety: evict one old/arbitrary key.  This is
                # acceptable only for a local per-process limiter.
                self._buckets.pop(next(iter(self._buckets)))
            self._buckets[key] = (window, count)
            return RateLimitDecision(True, limit, max(0, limit - count), retry_after)

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()
