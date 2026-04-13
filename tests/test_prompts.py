import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from prompts import (
    FINAL_SYSTEM_PROMPT,
    TOOL_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_final_user_prompt,
    build_structured_input,
    build_tool_user_prompt,
    build_user_prompt,
)


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[OK] {message}")


def build_initial_plan_request() -> dict:
    return {
        "user_profile": {
            "goal": "hypertrophy",
            "experience_level": "beginner",
            "equipment": ["гантели", "турник"],
            "restrictions": [],
        }
    }


def build_adaptation_request() -> dict:
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
                    "name": "Жим лёжа",
                    "sets_completed": 3,
                    "sets_planned": 3,
                    "reps": "8,8,8",
                    "weight_kg": 70,
                    "rpe": 7,
                }
            ],
            "sleep_hours": 8,
            "fatigue_level": 3,
            "notes": "тренировка прошла уверенно",
        },
    }


def build_long_history_request() -> dict:
    history = []
    for i in range(7):
        history.append(
            {
                "date": f"2026-03-{10 + i:02d}",
                "exercises": [
                    {
                        "name": f"Упражнение {i}",
                        "sets_completed": 3,
                        "sets_planned": 3,
                        "reps": "10,10,10",
                        "weight_kg": 20 + i,
                        "rpe": 6 + (i % 3),
                    }
                ],
                "sleep_hours": 7,
                "fatigue_level": 4,
                "notes": f"session {i}",
            }
        )

    return {
        "user_profile": {
            "goal": "general",
            "experience_level": "advanced",
            "equipment": ["велотренажер", "гантели"],
            "restrictions": [],
        },
        "session_history": history,
    }


def check_system_prompts() -> None:
    print_header("ПРОВЕРКА SYSTEM PROMPTS")

    assert_true(
        "build_training_context" in TOOL_SYSTEM_PROMPT,
        "TOOL_SYSTEM_PROMPT содержит указание на build_training_context",
    )
    assert_true(
        "request_confirmation" in TOOL_SYSTEM_PROMPT,
        "TOOL_SYSTEM_PROMPT содержит confirmation policy",
    )
    assert_true(
        "TOOL_PHASE_DONE" in TOOL_SYSTEM_PROMPT,
        "TOOL_SYSTEM_PROMPT содержит правило завершения tool phase",
    )

    assert_true(
        "<response_format>" in FINAL_SYSTEM_PROMPT,
        "FINAL_SYSTEM_PROMPT содержит блок response_format",
    )
    assert_true(
        "<critical_schema_rule>" in FINAL_SYSTEM_PROMPT,
        "FINAL_SYSTEM_PROMPT содержит блок critical_schema_rule",
    )
    assert_true(
        "<medical_safety_rules>" in FINAL_SYSTEM_PROMPT,
        "FINAL_SYSTEM_PROMPT содержит блок medical_safety_rules",
    )
    assert_true(
        "<sgr_steps>" in FINAL_SYSTEM_PROMPT,
        "FINAL_SYSTEM_PROMPT содержит блок sgr_steps",
    )
    assert_true(
        'decision_trace.final_action' in FINAL_SYSTEM_PROMPT,
        "FINAL_SYSTEM_PROMPT фиксирует ограничения на final_action",
    )
    assert_true(
        "medical_risk" in FINAL_SYSTEM_PROMPT,
        "FINAL_SYSTEM_PROMPT содержит правила medical risk",
    )

    assert_true(
        isinstance(SYSTEM_PROMPT, str) and len(SYSTEM_PROMPT) > 0,
        "SYSTEM_PROMPT сохранён для обратной совместимости",
    )


def check_structured_input() -> None:
    print_header("ПРОВЕРКА build_structured_input")

    mode, payload = build_structured_input(build_long_history_request())

    assert_true(
        mode == "adaptation",
        'mode корректно определяется как "adaptation"',
    )
    assert_true(
        len(payload["session_history"]) == 5,
        "История обрезается до последних 5 сессий",
    )
    assert_true(
        payload["session_history"][0]["date"] == "2026-03-12",
        "Обрезка истории начинается с корректной даты",
    )
    assert_true(
        payload["session_history"][-1]["date"] == "2026-03-16",
        "Обрезка истории заканчивается корректной датой",
    )


def check_tool_prompt() -> None:
    print_header("ПРОВЕРКА TOOL PROMPT")

    prompt_text = build_tool_user_prompt(build_initial_plan_request())

    assert_true("<input_data>" in prompt_text, "В tool prompt есть блок input_data")
    assert_true("TOOL_PHASE_DONE" in prompt_text, "В tool prompt есть инструкция завершения tool phase")
    assert_true('"mode": "initial_plan"' in prompt_text, "В tool prompt сериализуется mode=initial_plan")
    assert_true("Не возвращай здесь финальный SGR JSON" in prompt_text, "В tool prompt есть запрет на финальный SGR JSON")


def check_final_prompt() -> None:
    print_header("ПРОВЕРКА FINAL PROMPT")

    tool_outputs = {
        "build_training_context": {
            "mode": "adaptation",
            "brief_goal": "strength",
            "experience_level": "intermediate",
            "equipment_summary": "штанга, гантели, скамья",
            "restrictions_summary": "дискомфорт в колене",
            "has_history": False,
            "has_current_session": True,
            "latest_session_excerpt": "demo",
            "history_size": 0,
        }
    }

    prompt_text = build_final_user_prompt(build_adaptation_request(), tool_outputs)

    assert_true("<tool_outputs>" in prompt_text, "В final prompt есть блок tool_outputs")
    assert_true("build_training_context" in prompt_text, "В final prompt попадают результаты инструментов")
    assert_true('"mode": "adaptation"' in prompt_text, "В final prompt сериализуется adaptation-кейс")
    assert_true(
        "Заполни reasoning-схему строго по этапам" in prompt_text,
        "В final prompt есть инструкция по строгому заполнению SGR",
    )
    assert_true(
        "medical risk" in prompt_text.lower(),
        "В final prompt есть safety-инструкция по medical risk",
    )


def check_legacy_prompt_compatibility() -> None:
    print_header("ПРОВЕРКА ОБРАТНОЙ СОВМЕСТИМОСТИ")

    prompt_text = build_user_prompt(build_adaptation_request())

    assert_true("<input_data>" in prompt_text, "build_user_prompt сохраняет блок input_data")
    assert_true("<retrieved_knowledge>" in prompt_text, "build_user_prompt сохраняет retrieval-блок")
    assert_true("<mode_instruction>" in prompt_text, "build_user_prompt сохраняет mode_instruction")
    assert_true("<output_instruction>" in prompt_text, "build_user_prompt сохраняет output_instruction")
    assert_true(
        'Используй mode = "adaptation".' in prompt_text,
        'build_user_prompt корректно фиксирует mode = "adaptation"',
    )


def main() -> None:
    check_system_prompts()
    check_structured_input()
    check_tool_prompt()
    check_final_prompt()
    check_legacy_prompt_compatibility()

    print_header("ИТОГ")
    print("Все проверки prompts.py для tool-calling версии пройдены успешно.")


if __name__ == "__main__":
    main()