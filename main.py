import json
import os

from fastapi import FastAPI, HTTPException

from llm import get_coach_response
from models import CoachRequest, CoachResponse
from locust_models import InfoResponse, InputType, RunRequest, RunResponse
from locust_adapter import build_coach_request_from_locust

app = FastAPI(
    title="AI Adaptive Training Coach",
    description="AI-система для генерации и адаптации тренировочных программ",
    version="0.1.0",
)


def is_litellm_budget_exceeded_error(error: Exception) -> bool:
    """
    Проверяет, что ошибка пришла от LiteLLM из-за превышения бюджета.

    LiteLLM может вернуть это как 429, но для нашего API по заданию
    мы должны преобразовать такую ситуацию в HTTP 402 Payment Required.
    """
    error_text = str(error).lower()

    budget_error_markers = (
        "budget_exceeded",
        "budget has been exceeded",
        "max budget",
        "current cost",
        "requires more credits",
        "fewer max_tokens",
        "can only afford",
        "payment required",
        "openrouterexception",
        "code\":402",
        "code': '402'",
        "code': 402",
    )

    return any(marker in error_text for marker in budget_error_markers)

def is_load_test_mode_enabled() -> bool:
    return os.getenv("LOAD_TEST_MODE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def build_load_test_response(mode: str = "initial_plan") -> dict:
    safe_mode = "adaptation" if mode == "adaptation" else "initial_plan"

    return {
        "mode": safe_mode,
        "session_assessment": (
            "Тестовый режим нагрузочного тестирования. Внешняя LLM не вызывается."
            if safe_mode == "adaptation"
            else None
        ),
        "next_session": {
            "decision": (
                "maintain"
                if safe_mode == "adaptation"
                else "create_initial_plan"
            ),
            "exercise_changes": [],
            "reasoning": (
                "Ответ сформирован в режиме нагрузочного тестирования. "
                "Проверяется API-контракт, сериализация JSON, endpoint /run "
                "и устойчивость сервиса при параллельных запросах."
            ),
        },
        "long_term_recommendation": (
            "В рабочем режиме рекомендация формируется через LiteLLM и выбранную LLM."
        ),
        "safety_warnings": [],
        "refused": False,
        "refuse_reason": None,
    }

@app.get("/")
async def root():
    return {
        "service": "AI Adaptive Training Coach",
        "version": "0.1.0",
        "endpoints": {
            "POST /coach": "Получить тренировочную рекомендацию",
            "GET /health": "Проверка состояния сервиса",
            "GET /schema": "JSON Schema ответа системы",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/schema")
async def get_schema():
    """Возвращает JSON Schema структурированного ответа системы."""
    return CoachResponse.model_json_schema()


@app.post("/coach", response_model=CoachResponse)
async def coach(request: CoachRequest):
    """
    Основной endpoint.

    - Если session_history пустой и current_session не передан → генерирует стартовый план
    - Если есть история или current_session → адаптирует следующую тренировку
    """
    try:
        response = await get_coach_response(request.model_dump())
        return response

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"LLM вернул невалидный JSON: {str(e)}",
        )

    except Exception as e:
        if is_litellm_budget_exceeded_error(e):
            raise HTTPException(
                status_code=402,
                detail=(
                    "Дневной бюджет LiteLLM превышен. "
                    f"Исходная ошибка: {str(e)}"
                ),
            )

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обращении к LLM: {str(e)}",
        )
    
@app.get("/info", response_model=InfoResponse)
async def info():
    return InfoResponse(
        input_type=InputType.TEXT,
        input_schema={
            "type": "object",
            "properties": {
                "scenario": {
                    "type": "string",
                    "description": "Сценарий тестового запроса",
                    "default": "initial_plan",
                },
                "temperature": {
                    "type": "number",
                    "description": "Температура генерации",
                    "default": 0.3,
                },
            },
        },
        output_schema=CoachResponse.model_json_schema(),
    )

@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest):
    try:
        if not isinstance(request.content, str):
            return RunResponse(
                status="error",
                error="Сервис поддерживает только текстовый content.",
            )

        scenario = request.extra_body.get("scenario", "initial_plan")

        if is_load_test_mode_enabled():
            return RunResponse(
                status="success",
                result=build_load_test_response(mode=scenario),
                error=None,
            )

        coach_request = build_coach_request_from_locust(
            content=request.content,
            extra_body=request.extra_body,
        )

        response = await get_coach_response(coach_request.model_dump())

        return RunResponse(
            status="success",
            result=response.model_dump(),
            error=None,
        )

    except json.JSONDecodeError as e:
        return RunResponse(
            status="error",
            result=None,
            error=f"LLM вернул невалидный JSON: {str(e)}",
        )

    except Exception as e:
        if is_litellm_budget_exceeded_error(e):
            raise HTTPException(
                status_code=402,
                detail=f"Дневной бюджет LiteLLM превышен. Исходная ошибка: {str(e)}",
            )

        return RunResponse(
            status="error",
            result=None,
            error=f"Ошибка при выполнении /run: {str(e)}",
        )