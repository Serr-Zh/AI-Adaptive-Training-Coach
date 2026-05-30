import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from coach.llm import get_coach_response_with_trace, get_sgr_response_with_trace


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


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


def build_confirmation_case() -> dict:
    return {
        "user_profile": {
            "goal": "strength",
            "experience_level": "intermediate",
            "equipment": ["штанга", "гантели", "скамья"],
            "restrictions": ["дискомфорт в колене"],
        },
        "current_session": {
            "date": "2026-03-31",
            "exercises": [
                {
                    "name": "Присед",
                    "sets_completed": 3,
                    "sets_planned": 3,
                    "reps": "6,6,6",
                    "weight_kg": 90,
                    "rpe": 7,
                }
            ],
            "sleep_hours": 8,
            "fatigue_level": 4,
            "notes": "есть небольшой дискомфорт в колене, но без резкой боли",
        },
    }


def assert_trace_contains(trace: dict, tool_name: str) -> None:
    names = [item["tool_name"] for item in trace["tool_calls"]]
    assert_true(tool_name in names, f"В trace есть вызов инструмента {tool_name}")


async def run_case(title: str, request_data: dict) -> None:
    print_header(title)
    sgr_response, trace = await get_sgr_response_with_trace(request_data)
    coach_response, coach_trace = await get_coach_response_with_trace(request_data)

    sgr_data = sgr_response.model_dump()
    trace_data = trace.model_dump()
    coach_data = coach_response.model_dump()
    coach_trace_data = coach_trace.model_dump()

    print("\nTOOL TRACE:")
    print(json.dumps(trace_data, ensure_ascii=False, indent=2))
    print("\nSGR RESPONSE:")
    print(json.dumps(sgr_data, ensure_ascii=False, indent=2))
    print("\nCOACH RESPONSE:")
    print(json.dumps(coach_data, ensure_ascii=False, indent=2))

    for required_tool in [
        "build_training_context",
        "retrieve_training_knowledge",
        "assess_restrictions",
        "assess_training_load",
        "assess_medical_risk",
        "request_confirmation",
    ]:
        assert_trace_contains(trace_data, required_tool)
        assert_trace_contains(coach_trace_data, required_tool)

    assert_true("mode" in sgr_data, "SGR-ответ содержит mode")
    assert_true("final_recommendation" in sgr_data, "SGR-ответ содержит final_recommendation")
    assert_true("next_session" in coach_data, "CoachResponse содержит next_session")

    if "острая боль" in json.dumps(request_data, ensure_ascii=False).lower() or "травма" in json.dumps(request_data, ensure_ascii=False).lower():
        assert_true(
            sgr_data["medical_risk_assessment"]["medical_risk_detected"] is True,
            "Опасный кейс помечается как medical risk",
        )
        assert_true(
            sgr_data["final_recommendation"]["refused"] is True,
            "В опасном кейсе происходит отказ",
        )


async def main() -> None:
    await run_case("TOOL CALLING: INITIAL PLAN", build_initial_plan_case())
    await run_case("TOOL CALLING: PROGRESS", build_progress_case())
    await run_case("TOOL CALLING: MEDICAL RISK", build_medical_risk_case())
    await run_case("TOOL CALLING: CONFIRMATION", build_confirmation_case())

    print_header("ИТОГ")
    print("Все проверки tool calling и trace успешно пройдены.")


if __name__ == "__main__":
    asyncio.run(main())
