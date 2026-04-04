"""
Resilience — Retry and rate limiting (Python)

This sample demonstrates production-grade resilience patterns: retry with
exponential backoff and token-bucket rate limiting. It mirrors the Julia
vignette 11_resilience.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio
import random
import time
from collections.abc import Callable
from typing import Any

from agent_framework.ollama import OllamaChatClient


# ── Retry Configuration ──────────────────────────────────────────────────────


class RetryConfig:
    """Configuration for retry with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: float = 0.1,
        retryable_status_codes: list[int] | None = None,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.retryable_status_codes = retryable_status_codes or [429, 500, 502, 503, 504]

    def compute_delay(self, attempt: int) -> float:
        """Compute backoff delay with jitter for a given attempt."""
        base_delay = min(self.initial_delay * (self.multiplier ** attempt), self.max_delay)
        jitter_amount = base_delay * self.jitter * (2 * random.random() - 1)
        return max(0.0, base_delay + jitter_amount)

    def is_retryable(self, error: Exception) -> bool:
        """Check if an error should trigger a retry."""
        error_msg = str(error).lower()
        retryable_keywords = ["timeout", "rate limit", "connection", "temporary"]
        for code in self.retryable_status_codes:
            if str(code) in error_msg:
                return True
        return any(kw in error_msg for kw in retryable_keywords)


# ── Token Bucket Rate Limiter ────────────────────────────────────────────────


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, requests_per_second: float = 10.0, burst: int = 10):
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.refill_rate = requests_per_second
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    async def acquire(self, timeout: float = 30.0) -> bool:
        """Wait until a token is available or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            wait_time = min(1.0 / self.refill_rate, deadline - time.monotonic())
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        return False


# ── Resilient Agent Wrapper ──────────────────────────────────────────────────


async def run_with_retry(
    agent_func: Callable[..., Any],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Run an async agent function with retry logic."""
    config = config or RetryConfig()
    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await agent_func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt == config.max_retries or not config.is_retryable(e):
                raise
            delay = config.compute_delay(attempt)
            print(f"  Retry {attempt + 1}/{config.max_retries}: "
                  f"waiting {delay:.2f}s after {type(e).__name__}")
            await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]


async def main() -> None:
    client = OllamaChatClient(host="http://localhost:11434", model_id="qwen3:8b")
    agent = client.as_agent(
        name="ResilientAgent",
        instructions="You are a helpful assistant. Keep answers brief.",
    )

    # ── 1. Retry Configuration ───────────────────────────────────────────
    print("=== Retry Configuration ===")
    config = RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        multiplier=2.0,
        jitter=0.1,
    )
    for attempt in range(4):
        delay = config.compute_delay(attempt)
        print(f"  Attempt {attempt}: delay = {delay:.2f}s")

    # ── 2. Rate Limiter ──────────────────────────────────────────────────
    print("\n=== Rate Limiter ===")
    limiter = TokenBucketRateLimiter(requests_per_second=5.0, burst=10)
    acquired = await limiter.acquire(timeout=5.0)
    print(f"  Token acquired: {acquired}")

    # ── 3. Resilient Agent Call ──────────────────────────────────────────
    print("\n=== Resilient Agent Call ===")
    result = await run_with_retry(
        agent.run,
        "What is the speed of light?",
        config=config,
    )
    print(f"  Agent: {result}")


if __name__ == "__main__":
    asyncio.run(main())
