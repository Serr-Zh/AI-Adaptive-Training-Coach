import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class LLMConfig(BaseModel):
    api_key: str = Field(default="")
    base_url: str = Field(default="")
    model: str = Field(default="coach-model")
    max_tokens: int = Field(default=2048, gt=0)

    @model_validator(mode="after")
    def check_required_fields(self) -> "LLMConfig":
        if not self.api_key or not self.base_url:
            raise RuntimeError(
                "Не заданы LLM_API_KEY или LLM_BASE_URL — проверь .env файл"
            )
        return self

    @classmethod
    def from_env(cls) -> "LLMConfig":
        raw_max = os.getenv("LLM_MAX_TOKENS", "2048")
        try:
            max_tokens = int(raw_max)
        except ValueError as exc:
            raise RuntimeError(
                f"Некорректное значение LLM_MAX_TOKENS={raw_max!r}. "
                "Ожидалось целое число."
            ) from exc

        return cls(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", ""),
            model=os.getenv("LLM_MODEL", "coach-model"),
            max_tokens=max_tokens,
        )


class AppSettings(BaseModel):
    load_test_mode: bool = Field(default=False)
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(api_key="x", base_url="x"))

    @classmethod
    def from_env(cls) -> "AppSettings":
        raw = os.getenv("LOAD_TEST_MODE", "false").strip().lower()
        load_test_mode = raw in {"1", "true", "yes", "on"}
        return cls(load_test_mode=load_test_mode, llm=LLMConfig.from_env())


_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = AppSettings.from_env()
    return _settings
