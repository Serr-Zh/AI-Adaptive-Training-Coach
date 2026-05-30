import asyncio
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from coach.llm import get_coach_response, get_sgr_response
from coach.models import CoachResponse, CoachSGRResponse


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def to_plain_dict(value):
    if is_dataclass(value):
        return asdict(value)

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if isinstance(value, dict):
        return value

    if hasattr(value, "__dict__"):
        return dict(value.__dict__)

    raise TypeError(f"Не удалось преобразовать результат типа {type(value)} к словарю")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[OK] {message}")


def build_initial_plan_case() -> dict:
    return {
        "user_profile": {
            "goal": "hypertrophy",
            "experience_level": "beginner",
            "equipment": ["гантели", "турник"],
            "restrictions": [],
        }
    }


def build_progress_case() -> dict:
    return {
        "user_profile": {
            "goal": "strength",
            "experience_level": "intermediate",
            "equipment": ["штанга", "гантели", "скамья"],
            "restrictions": [],
        },
        "current_session": {
            "date": "2026-03-31",
            "exercises": [
                {
                    "name": "Жим лёжа",
                    "sets_completed": 3,
                    "sets_planned": 3,
                    "reps": "8,8,8",
                    "weight_kg": 70,
                    "rpe": 7,
                },
                {
                    "name": "Тяга в наклоне",
                    "sets_completed": 3,
                    "sets_planned": 3,
                    "reps": "10,10,10",
                    "weight_kg": 50,
                    "rpe": 6,
                },
            ],
            "sleep_hours": 8,
            "fatigue_level": 3,
            "notes": "тренировка прошла уверенно",
        },
    }


def build_medical_risk_case() -> dict:
    return {
        "user_profile": {
            "goal": "general",
            "experience_level": "beginner",
            "equipment": ["гантели"],
            "restrictions": ["травма поясницы"],
        },
        "current_session": {
            "date": "2026-03-31",
            "exercises": [
                {
                    "name": "Румынская тяга",
                    "sets_completed": 1,
                    "sets_planned": 3,
                    "reps": "10",
                    "weight_kg": 30,
                    "rpe": 8,
                }
            ],
            "sleep_hours": 6,
            "fatigue_level": 7,
            "notes": "острая боль в пояснице во время движения",
        },
    }


def assert_sgr_shape(data: dict) -> None:
    required_fields = [
        "mode",
        "input_summary",
        "progress_assessment",
        "overload_assessment",
        "medical_risk_assessment",
        "restriction_assessment",
        "decision_trace",
        "final_recommendation",
    ]

    for field in required_fields:
        if field not in data:
            raise AssertionError(f"В SGR-ответе отсутствует поле: {field}")


def assert_api_shape(data: dict) -> None:
    required_fields = [
        "mode",
        "session_assessment",
        "next_session",
        "long_term_recommendation",
        "safety_warnings",
        "refused",
        "refuse_reason",
    ]

    for field in required_fields:
        if field not in data:
            raise AssertionError(f"В API-ответе отсутствует поле: {field}")


async def check_sgr_initial_plan() -> None:
    print_header("SGR: INITIAL PLAN")

    request_data = build_initial_plan_case()
    result = await get_sgr_response(request_data)

    assert_true(
        isinstance(result, CoachSGRResponse),
        "get_sgr_response возвращает CoachSGRResponse",
    )

    data = to_plain_dict(result)
    print(json.dumps(data, ensure_ascii=False, indent=2))

    assert_sgr_shape(data)
    assert_true(data["mode"] == "initial_plan", 'SGR mode = "initial_plan"')
    assert_true(
        data["input_summary"]["has_history"] is False,
        "has_history корректен для initial_plan",
    )
    assert_true(
        data["input_summary"]["has_current_session"] is False,
        "has_current_session корректен для initial_plan",
    )


async def check_sgr_progress_case() -> None:
    print_header("SGR: ПРОГРЕСС")

    request_data = build_progress_case()
    result = await get_sgr_response(request_data)

    assert_true(
        isinstance(result, CoachSGRResponse),
        "get_sgr_response возвращает CoachSGRResponse",
    )

    data = to_plain_dict(result)
    print(json.dumps(data, ensure_ascii=False, indent=2))

    assert_sgr_shape(data)
    assert_true(data["mode"] == "adaptation", 'SGR mode = "adaptation"')
    assert_true(
        data["medical_risk_assessment"]["medical_risk_detected"] is False,
        "medical risk отсутствует в progress-сценарии",
    )


async def check_sgr_medical_risk_case() -> None:
    print_header("SGR: МЕДИЦИНСКИЙ РИСК")

    request_data = build_medical_risk_case()
    result = await get_sgr_response(request_data)

    assert_true(
        isinstance(result, CoachSGRResponse),
        "get_sgr_response возвращает CoachSGRResponse",
    )

    data = to_plain_dict(result)
    print(json.dumps(data, ensure_ascii=False, indent=2))

    assert_sgr_shape(data)
    assert_true(
        data["medical_risk_assessment"]["medical_risk_detected"] is True,
        "medical_risk_detected=True в risky-сценарии",
    )
    assert_true(
        data["medical_risk_assessment"]["refusal_required"] is True,
        "refusal_required=True в risky-сценарии",
    )
    assert_true(
        data["decision_trace"]["final_action"] == "refuse",
        'final_action="refuse" в risky-сценарии',
    )
    assert_true(
        data["final_recommendation"]["refused"] is True,
        "final_recommendation.refused=True в risky-сценарии",
    )
    assert_true(
        data["final_recommendation"]["refuse_reason"] is not None,
        "refuse_reason заполнен в risky-сценарии",
    )
    assert_true(
        data["final_recommendation"]["exercise_changes"] == [],
        "exercise_changes пустой при отказе",
    )


async def check_api_wrapper_case() -> None:
    print_header("API WRAPPER: COACH_RESPONSE ИЗ SGR")

    request_data = build_progress_case()
    result = await get_coach_response(request_data)

    assert_true(
        isinstance(result, CoachResponse),
        "get_coach_response возвращает CoachResponse",
    )

    data = to_plain_dict(result)
    print(json.dumps(data, ensure_ascii=False, indent=2))

    assert_api_shape(data)
    assert_true(data["mode"] == "adaptation", 'CoachResponse mode = "adaptation"')
    assert_true(
        "decision" in data["next_session"],
        "В next_session присутствует decision",
    )
    assert_true(
        "reasoning" in data["next_session"],
        "В next_session присутствует reasoning",
    )


async def main() -> None:
    await check_sgr_initial_plan()
    await check_sgr_progress_case()
    await check_sgr_medical_risk_case()
    await check_api_wrapper_case()

    print_header("ИТОГ")
    print("Все SGR- и API-проверки успешно пройдены.")


if __name__ == "__main__":
    asyncio.run(main())