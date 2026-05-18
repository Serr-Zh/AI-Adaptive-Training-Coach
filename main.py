import json

from fastapi import FastAPI, HTTPException

from llm import get_coach_response
from models import CoachRequest, CoachResponse


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
    )

    return any(marker in error_text for marker in budget_error_markers)


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