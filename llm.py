import json
import os
import re
import logging
from typing import Any

from openai import AsyncOpenAI, BadRequestError

from models import CoachResponse
from prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

_openai_client: AsyncOpenAI | None = None


def get_training_llm_client() -> AsyncOpenAI:
    """
    Возвращает общий клиент для обращения к LLM.
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
    Достаёт JSON из ответа модели.
    """
    raw_answer = re.sub(r"```(?:json)?\s*", "", raw_answer).strip()
    json_start = raw_answer.find("{")
    json_end = raw_answer.rfind("}")

    if json_start == -1 or json_end == -1:
        raise ValueError(f"JSON не найден в ответе модели: {raw_answer[:300]}")

    return raw_answer[json_start:json_end + 1]


def _stringify_value(value: Any) -> str | None:
    """
    Приводит произвольное значение к строке.
    Нужно для случаев, когда LLM возвращает dict/list вместо строки.
    """
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        return text or None

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _stringify_value(item)
            if text:
                parts.append(text)
        return "; ".join(parts) if parts else None

    if isinstance(value, dict):
        parts: list[str] = []
        for key, val in value.items():
            text = _stringify_value(val)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts) if parts else None

    return str(value)


def _normalize_warnings(value: Any) -> list[str]:
    """
    Приводит safety_warnings к list[str].
    """
    if value is None:
        return []

    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            text = _stringify_value(item)
            if text:
                result.append(text)
        return result

    text = _stringify_value(value)
    return [text] if text else []


def _normalize_exercise_changes(value: Any) -> list[dict[str, str]]:
    """
    Приводит exercise_changes к ожидаемому формату list[dict].
    """
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, str]] = []

    for item in value:
        if isinstance(item, dict):
            normalized.append(
                {
                    "exercise_name": _stringify_value(item.get("exercise_name") or item.get("name")) or "Не указано",
                    "change_type": _stringify_value(item.get("change_type") or item.get("type")) or "изменить",
                    "details": _stringify_value(item.get("details") or item.get("description")) or "Без деталей",
                }
            )
        else:
            text = _stringify_value(item) or "Без деталей"
            normalized.append(
                {
                    "exercise_name": "Не указано",
                    "change_type": "изменить",
                    "details": text,
                }
            )

    return normalized


def normalize_coach_response_shape(response_data: dict) -> dict:
    """
    Приводит ответ модели к ожидаемой схеме CoachResponse.
    """
    if "mode" in response_data and isinstance(response_data["mode"], str):
        response_data["mode"] = response_data["mode"].strip().replace(" ", "_").lower()

    if response_data.get("mode") not in {"initial_plan", "adaptation"}:
        if response_data.get("session_assessment") or response_data.get("current_session_assessment"):
            response_data["mode"] = "adaptation"
        else:
            response_data["mode"] = "initial_plan"

    if "session_assessment" in response_data:
        response_data["session_assessment"] = _stringify_value(response_data.get("session_assessment"))

    if "long_term_recommendation" in response_data:
        response_data["long_term_recommendation"] = _stringify_value(
            response_data.get("long_term_recommendation")
        )

    response_data["safety_warnings"] = _normalize_warnings(response_data.get("safety_warnings"))

    if isinstance(response_data.get("refused"), str):
        response_data["refused"] = response_data["refused"].strip().lower() in {
            "true",
            "1",
            "yes",
            "да",
        }
    else:
        response_data["refused"] = bool(response_data.get("refused", False))

    response_data["refuse_reason"] = _stringify_value(response_data.get("refuse_reason"))

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
            "decision": _stringify_value(session_decision) or "",
            "exercise_changes": _normalize_exercise_changes(exercise_adjustments),
            "reasoning": _stringify_value(decision_reasoning) or "",
        }

    next_session_block = response_data.get("next_session")

    if not isinstance(next_session_block, dict):
        next_session_block = {
            "decision": _stringify_value(next_session_block) or "",
            "exercise_changes": [],
            "reasoning": "",
        }

    if not next_session_block.get("reasoning") and response_data.get("reasoning"):
        next_session_block["reasoning"] = _stringify_value(response_data.pop("reasoning")) or ""

    next_session_block["decision"] = _stringify_value(next_session_block.get("decision")) or ""
    next_session_block["reasoning"] = _stringify_value(next_session_block.get("reasoning")) or ""
    next_session_block["exercise_changes"] = _normalize_exercise_changes(
        next_session_block.get("exercise_changes")
    )

    response_data["next_session"] = next_session_block

    return response_data


async def get_coach_response(request_data: dict) -> CoachResponse:
    """
    Главная точка входа для получения ответа от AI Adaptive Training Coach.
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

            plain_json_messages = messages[:-1] + [
                {
                    "role": "user",
                    "content": messages[-1]["content"]
                    + "\n\nВерни только JSON, без markdown и без дополнительных пояснений.",
                }
            ]

            response = await client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=plain_json_messages,
            )
            return response.choices[0].message.content