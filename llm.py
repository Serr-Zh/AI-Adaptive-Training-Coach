import json
import os
import re
import logging

from openai import AsyncOpenAI, BadRequestError

from models import CoachResponse
from prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

_openai_client: AsyncOpenAI | None = None


def get_training_llm_client() -> AsyncOpenAI:
    """
    Возвращает общий клиент для обращения к LLM.

    В проекте клиент нужен один и тот же для всех запросов:
    он берет настройки из .env и работает через OpenAI-совместимый API
    (OpenRouter, OpenAI, Ollama, vLLM и т.д.).
    """
    global _openai_client

    if _openai_client is None:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")

        if not api_key or not base_url:
            raise RuntimeError(
                "Не заданы LLM_API_KEY или LLM_BASE_URL — проверь .env файл"
            )

        _openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    return _openai_client


def extract_json_from_model_answer(raw_answer: str) -> str:
    """
    Достает JSON из ответа модели.

    Некоторые модели оборачивают JSON в ```json ... ``` или добавляют
    лишний текст до и после объекта. Для API нам нужен только сам JSON.
    """
    raw_answer = re.sub(r"```(?:json)?\s*", "", raw_answer).strip()

    json_start = raw_answer.find("{")
    json_end = raw_answer.rfind("}")

    if json_start == -1 or json_end == -1:
        raise ValueError(f"JSON не найден в ответе модели: {raw_answer[:300]}")

    return raw_answer[json_start:json_end + 1]


def normalize_coach_response_shape(response_data: dict) -> dict:
    """
    Приводит ответ модели к ожидаемой схеме CoachResponse.

    На практике даже хорошая модель иногда возвращает:
    - mode в другом формате;
    - плоский ответ без блока next_session;
    - reasoning не внутри next_session, а рядом.

    Здесь мы мягко выравниваем такие ответы до структуры,
    которую уже ожидает Pydantic-модель и API.
    """
    if "mode" in response_data and isinstance(response_data["mode"], str):
        response_data["mode"] = response_data["mode"].strip().replace(" ", "_").lower()

    if "next_session" not in response_data:
        session_decision = (
            response_data.pop("decision", None)
            or response_data.pop("recommendation", None)
            or response_data.pop("plan_summary", None)
            or ""
        )

        decision_reasoning = (
            response_data.pop("reasoning", None)
            or response_data.pop("explanation", None)
            or response_data.pop("rationale", None)
            or ""
        )

        exercise_adjustments = (
            response_data.pop("exercise_changes", None)
            or response_data.pop("changes", [])
        )

        response_data["next_session"] = {
            "decision": session_decision,
            "exercise_changes": (
                exercise_adjustments if isinstance(exercise_adjustments, list) else []
            ),
            "reasoning": decision_reasoning,
        }

    next_session_block = response_data.get("next_session", {})

    if not next_session_block.get("reasoning") and response_data.get("reasoning"):
        next_session_block["reasoning"] = response_data.pop("reasoning")
        response_data["next_session"] = next_session_block

    return response_data


async def get_coach_response(request_data: dict) -> CoachResponse:
    """
    Главная точка входа для получения ответа от AI Adaptive Training Coach.

    Последовательность такая:
    1. Собираем клиент и настройки модели.
    2. Формируем сообщения для LLM.
    3. Получаем сырой текстовый ответ.
    4. Извлекаем JSON.
    5. Нормализуем форму ответа.
    6. Валидируем его через CoachResponse.
    """
    client = get_training_llm_client()

    model_name = os.getenv("LLM_MODEL")
    if not model_name:
        raise RuntimeError("Не задан LLM_MODEL — проверь .env файл")

    temperature = request_data.get("temperature", 0.3)

    dialog = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(request_data)},
    ]

    raw_answer = await _request_model_response(
        client=client,
        model_name=model_name,
        temperature=temperature,
        messages=dialog,
    )

    normalized_data = normalize_coach_response_shape(
        json.loads(extract_json_from_model_answer(raw_answer))
    )

    return CoachResponse(**normalized_data)


async def _request_model_response(
    client: AsyncOpenAI,
    model_name: str,
    temperature: float,
    messages: list[dict],
) -> str:
    """
    Пытается получить ответ модели в максимально строгом формате.

    Стратегия по убыванию надежности:
    1. json_schema — лучший вариант, если провайдер умеет strict structured output.
    2. json_object — запасной вариант.
    3. обычный текстовый ответ с явной инструкцией вернуть только JSON.
    """
    response_schema = CoachResponse.model_json_schema()

    try:
        response = await client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "CoachResponse",
                    "strict": True,
                    "schema": response_schema,
                },
            },
        )
        return response.choices[0].message.content

    except BadRequestError:
        logger.warning(
            "Провайдер не поддерживает structured output через json_schema. "
            "Пробую режим json_object."
        )

    try:
        response = await client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    except BadRequestError:
        logger.warning(
            "Провайдер не поддерживает json_object. "
            "Перехожу к обычному запросу с инструкцией вернуть только JSON."
        )

    plain_json_messages = messages[:-1] + [{
        "role": "user",
        "content": messages[-1]["content"] + "\n\nВерни только JSON, без markdown и без дополнительных пояснений.",
    }]

    response = await client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=plain_json_messages,
    )
    return response.choices[0].message.content