import asyncio
import json
from dataclasses import asdict, is_dataclass
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from coach.llm import get_coach_response


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

    raise TypeError(
        f"Не удалось преобразовать результат типа {type(value)} к словарю"
    )


def assert_required_response_shape(data: dict) -> None:
    required_top_level = [
        "mode",
        "session_assessment",
        "next_session",
        "long_term_recommendation",
        "safety_warnings",
        "refused",
        "refuse_reason",
    ]

    for field in required_top_level:
        if field not in data:
            raise AssertionError(f"В ответе отсутствует обязательное поле: {field}")

    if data["mode"] not in {"initial_plan", "adaptation"}:
        raise AssertionError('Поле "mode" должно быть "initial_plan" или "adaptation"')

    if not isinstance(data["next_session"], dict):
        raise AssertionError('Поле "next_session" должно быть объектом')

    next_session_required = ["decision", "exercise_changes", "reasoning"]
    for field in next_session_required:
        if field not in data["next_session"]:
            raise AssertionError(
                f'В "next_session" отсутствует обязательное поле: {field}'
            )

    if not isinstance(data["next_session"]["exercise_changes"], list):
        raise AssertionError('Поле "exercise_changes" должно быть массивом')

    if not isinstance(data["safety_warnings"], list):
        raise AssertionError('Поле "safety_warnings" должно быть массивом')

    if not isinstance(data["refused"], bool):
        raise AssertionError('Поле "refused" должно быть bool')

    if data["refused"] is False and data["refuse_reason"] is not None:
        raise AssertionError(
            'Если "refused" = false, то "refuse_reason" должен быть null'
        )

    if data["refused"] is True and not data["refuse_reason"]:
        raise AssertionError(
            'Если "refused" = true, то "refuse_reason" должен быть заполнен'
        )


async def run_case(case_name: str, request_data: dict) -> None:
    print_header(case_name)
    print("Входные данные:")
    print(json.dumps(request_data, ensure_ascii=False, indent=2))

    result = await get_coach_response(request_data)
    result_dict = to_plain_dict(result)

    print("\nОтвет модели:")
    print(json.dumps(result_dict, ensure_ascii=False, indent=2))

    assert_required_response_shape(result_dict)
    print("\n[OK] Ответ имеет корректную базовую структуру")


def build_cases() -> list[tuple[str, dict]]:
    initial_plan_case = (
        "СЦЕНАРИЙ 1: INITIAL PLAN",
        {
            "user_profile": {
                "goal": "набрать мышечную массу",
                "experience_level": "beginner",
                "equipment": ["гантели", "турник"],
                "restrictions": [],
            }
        },
    )

    progress_case = (
        "СЦЕНАРИЙ 2: ПРОГРЕСС И ВОЗМОЖНАЯ ПРОГРЕССИЯ",
        {
            "user_profile": {
                "goal": "увеличить силовые показатели",
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
                        "rpe": 6.5,
                    },
                ],
                "sleep_hours": 8,
                "fatigue_level": 3,
                "notes": "тренировка прошла уверенно",
            },
        },
    )

    overload_case = (
        "СЦЕНАРИЙ 3: ПЕРЕГРУЗКА",
        {
            "user_profile": {
                "goal": "сила",
                "experience_level": "intermediate",
                "equipment": ["штанга", "стойка", "скамья"],
                "restrictions": [],
            },
            "current_session": {
                "date": "2026-03-31",
                "exercises": [
                    {
                        "name": "Присед",
                        "sets_completed": 2,
                        "sets_planned": 4,
                        "reps": "8,7",
                        "weight_kg": 100,
                        "rpe": 9.5,
                    },
                    {
                        "name": "Жим лёжа",
                        "sets_completed": 2,
                        "sets_planned": 3,
                        "reps": "8,6",
                        "weight_kg": 80,
                        "rpe": 9.2,
                    },
                ],
                "sleep_hours": 5,
                "fatigue_level": 9,
                "notes": "сильная усталость после прошлой тренировки",
            },
        },
    )

    medical_refusal_case = (
        "СЦЕНАРИЙ 4: МЕДИЦИНСКИЙ РИСК И ОТКАЗ",
        {
            "user_profile": {
                "goal": "вернуться к тренировкам",
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
                        "rpe": 8.5,
                    }
                ],
                "sleep_hours": 6,
                "fatigue_level": 7,
                "notes": "острая боль в пояснице во время движения",
            },
        },
    )

    return [
        initial_plan_case,
        progress_case,
        overload_case,
        medical_refusal_case,
    ]


async def main() -> None:
    for case_name, request_data in build_cases():
        try:
            await run_case(case_name, request_data)
        except Exception as exc:
            print(f"\n[ERROR] Проверка сценария завершилась ошибкой: {exc}")
            raise

    print_header("ИТОГ")
    print("Все сценарии llm.py отработали без ошибок структуры ответа.")


if __name__ == "__main__":
    asyncio.run(main())