from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from models import CoachRequest, CoachResponse
from llm import get_coach_response
import json

app = FastAPI(
    title="AI Adaptive Training Coach",
    description="AI-система для генерации и адаптации тренировочных программ",
    version="0.1.0",
)


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
    """Возвращает JSON Schema структурированного ответа системы"""
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
            detail=f"LLM вернул невалидный JSON: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обращении к LLM: {str(e)}"
        )
