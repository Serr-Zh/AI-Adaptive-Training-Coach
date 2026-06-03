import logging

from langfuse import get_client
from langfuse.openai import AsyncOpenAI

from coach.config import get_settings

logger = logging.getLogger(__name__)

_openai_client: AsyncOpenAI | None = None


def get_training_llm_client() -> AsyncOpenAI:
    global _openai_client

    if _openai_client is None:
        settings = get_settings()
        _openai_client = AsyncOpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )

    return _openai_client


def get_max_tokens() -> int:
    return get_settings().llm.max_tokens


def get_model_name() -> str:
    return get_settings().llm.model


def flush_langfuse() -> None:
    try:
        get_client().flush()
    except Exception as exc:
        logger.warning("Не удалось выполнить Langfuse flush: %s", exc)
