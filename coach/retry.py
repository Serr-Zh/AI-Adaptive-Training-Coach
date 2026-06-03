import asyncio
import logging

from openai import APIStatusError, RateLimitError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503}
MAX_RETRIES = 4
BASE_DELAY_S = 2.0


async def retry_with_backoff(coro_factory, *args, **kwargs):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await coro_factory(*args, **kwargs)
        except APIStatusError as exc:
            if exc.status_code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES:
                raise
            delay = BASE_DELAY_S * (2 ** attempt)
            logger.warning(
                "API error (status=%s), retry %d/%d через %.1fs",
                exc.status_code, attempt + 1, MAX_RETRIES, delay,
            )
            await asyncio.sleep(delay)
