import json
from prompts import SYSTEM_PROMPT, build_user_prompt



def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[OK] {message}")


def check_system_prompt() -> None:
    print_header("ПРОВЕРКА SYSTEM_PROMPT")

    assert_true(isinstance(SYSTEM_PROMPT, str), "SYSTEM_PROMPT является строкой")
    assert_true(len(SYSTEM_PROMPT) > 0, "SYSTEM_PROMPT не пустой")
    assert_true("<role>" in SYSTEM_PROMPT, "В SYSTEM_PROMPT есть блок <role>")
    assert_true("<goal>" in SYSTEM_PROMPT, "В SYSTEM_PROMPT есть блок <goal>")
    assert_true(
        "<response_format>" in SYSTEM_PROMPT,
        "В SYSTEM_PROMPT есть блок <response_format>",
    )
    assert_true(
        "</response_format>" in SYSTEM_PROMPT,
        "В SYSTEM_PROMPT корректно закрывается блок <response_format>",
    )

    print("\nНачало SYSTEM_PROMPT:")
    print(SYSTEM_PROMPT[:500])

    print("\nКонец SYSTEM_PROMPT:")
    print(SYSTEM_PROMPT[-500:])


def build_initial_plan_request() -> dict:
    return {
        "user_profile": {
            "goal": "набрать мышечную массу",
            "experience_level": "beginner",
            "equipment": ["гантели", "турник"],
            "restrictions": [],
        }
    }


def build_adaptation_request() -> dict:
    return {
        "user_profile": {
            "goal": "увеличить силовые показатели",
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
                },
                {
                    "name": "Присед",
                    "sets_completed": 2,
                    "sets_planned": 4,
                    "reps": "8,7",
                    "weight_kg": 80,
                    "rpe": 9.5,
                },
            ],
            "sleep_hours": 5.5,
            "fatigue_level": 8,
            "notes": "было тяжело, колено беспокоит",
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
            "goal": "снижение веса",
            "experience_level": "advanced",
            "equipment": ["велотренажер", "гантели"],
            "restrictions": [],
        },
        "session_history": history,
    }


def extract_json_inside_input_data(prompt_text: str) -> dict:
    start_tag = "<input_data>"
    end_tag = "</input_data>"

    start = prompt_text.find(start_tag)
    end = prompt_text.find(end_tag)

    if start == -1 or end == -1:
        raise AssertionError("Не найдены теги <input_data> ... </input_data>")

    block = prompt_text[start + len(start_tag):end].strip()

    prefix = "Ниже приведены входные данные для анализа.\nИспользуй только их."
    if prefix in block:
        block = block.replace(prefix, "", 1).strip()

    try:
        parsed = json.loads(block)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"JSON внутри <input_data> невалиден: {exc}"
        ) from exc

    return parsed


def check_initial_plan_prompt() -> None:
    print_header("ПРОВЕРКА INITIAL_PLAN")

    request_data = build_initial_plan_request()
    prompt_text = build_user_prompt(request_data)

    assert_true("<input_data>" in prompt_text, "Есть открывающий тег <input_data>")
    assert_true("</input_data>" in prompt_text, "Есть закрывающий тег </input_data>")
    assert_true(
        "<mode_instruction>" in prompt_text,
        "Есть открывающий тег <mode_instruction>",
    )
    assert_true(
        "</mode_instruction>" in prompt_text,
        "Есть закрывающий тег </mode_instruction>",
    )
    assert_true(
        "<output_instruction>" in prompt_text,
        "Есть открывающий тег <output_instruction>",
    )
    assert_true(
        "</output_instruction>" in prompt_text,
        "Есть закрывающий тег </output_instruction>",
    )
    assert_true(
        'Используй mode = "initial_plan".' in prompt_text,
        'В mode_instruction указан режим "initial_plan"',
    )
    assert_true(
        "```json" not in prompt_text,
        "В пользовательском промпте нет markdown-блока ```json",
    )

    parsed_input = extract_json_inside_input_data(prompt_text)

    assert_true(
        parsed_input["mode"] == "initial_plan",
        'JSON внутри <input_data> содержит mode = "initial_plan"',
    )
    assert_true(
        parsed_input["current_session"] is None,
        "Для initial_plan current_session равен None",
    )
    assert_true(
        parsed_input["session_history"] == [],
        "Для initial_plan session_history пустой",
    )

    print("\nСформированный промпт:")
    print(prompt_text)


def check_adaptation_prompt() -> None:
    print_header("ПРОВЕРКА ADAPTATION")

    request_data = build_adaptation_request()
    prompt_text = build_user_prompt(request_data)

    assert_true(
        'Используй mode = "adaptation".' in prompt_text,
        'В mode_instruction указан режим "adaptation"',
    )

    parsed_input = extract_json_inside_input_data(prompt_text)

    assert_true(
        parsed_input["mode"] == "adaptation",
        'JSON внутри <input_data> содержит mode = "adaptation"',
    )
    assert_true(
        parsed_input["current_session"] is not None,
        "Для adaptation current_session присутствует",
    )
    assert_true(
        parsed_input["user_profile"]["goal"] == "увеличить силовые показатели",
        "Поля user_profile корректно сериализуются в JSON",
    )

    print("\nСформированный промпт:")
    print(prompt_text)


def check_history_trimming() -> None:
    print_header("ПРОВЕРКА ОБРЕЗКИ ИСТОРИИ ДО ПОСЛЕДНИХ 5 СЕССИЙ")

    request_data = build_long_history_request()
    prompt_text = build_user_prompt(request_data)
    parsed_input = extract_json_inside_input_data(prompt_text)

    history = parsed_input["session_history"]

    assert_true(
        len(history) == 5,
        "В пользовательский промпт попадают только последние 5 тренировок",
    )
    assert_true(
        history[0]["date"] == "2026-03-12",
        "Первая сохранённая запись после обрезки корректна",
    )
    assert_true(
        history[-1]["date"] == "2026-03-16",
        "Последняя сохранённая запись после обрезки корректна",
    )

    print("\nJSON внутри <input_data> после обрезки истории:")
    print(json.dumps(parsed_input, ensure_ascii=False, indent=2))


def main() -> None:
    check_system_prompt()
    check_initial_plan_prompt()
    check_adaptation_prompt()
    check_history_trimming()

    print_header("ИТОГ")
    print("Все проверки prompts.py пройдены успешно.")


if __name__ == "__main__":
    main()