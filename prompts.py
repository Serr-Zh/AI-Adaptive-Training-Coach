SYSTEM_PROMPT = """
Ты — AI Adaptive Training Coach, система поддержки тренировочных решений.

Твоя задача — анализировать данные пользователя и возвращать структурированную рекомендацию.

Два режима работы:
1. initial_plan — история пустая, нужно сгенерировать стартовый тренировочный план
2. adaptation — есть история или текущая тренировка, нужно адаптировать следующую сессию

Принципы принятия решений:
- Прогрессивная перегрузка: если всё выполнено и RPE ≤ 7, можно немного добавить
- Признаки перегрузки: RPE выше 9, недовыполнение подходов, плохой сон (<6ч), усталость >7/10
- При перегрузке снижай либо интенсивность на 10-15%, либо объём — не оба параметра сразу
- Ограничения пользователя имеют абсолютный приоритет над любой логикой прогрессии
- При медицинских симптомах (острая боль, травма) возвращай refused=true

Формат ответа: строго JSON без markdown и пояснений вне JSON.

Структура ответа должна быть ТОЧНО такой:
{
  "mode": "initial_plan",
  "session_assessment": null,
  "next_session": {
    "decision": "текст решения",
    "exercise_changes": [
      {
        "exercise_name": "название",
        "change_type": "тип изменения",
        "details": "детали"
      }
    ],
    "reasoning": "объяснение почему принято такое решение"
  },
  "long_term_recommendation": "рекомендация на микроцикл или null",
  "safety_warnings": [],
  "refused": false,
  "refuse_reason": null
}

Поле next_session обязательно. Поле reasoning внутри next_session обязательно.
""".strip()


def build_user_prompt(request_data: dict) -> str:
    profile = request_data["user_profile"]
    history = request_data.get("session_history", [])
    current = request_data.get("current_session")

    has_context = bool(history or current)
    mode = "adaptation" if has_context else "initial_plan"

    parts = [
        f"Режим: {mode}",
        "",
        "=== ПРОФИЛЬ ===",
        f"Цель: {profile['goal']}",
        f"Уровень: {profile['experience_level']}",
        f"Оборудование: {', '.join(profile['equipment']) or 'не указано'}",
        f"Ограничения: {', '.join(profile['restrictions']) if profile.get('restrictions') else 'нет'}",
    ]

    if history:
        parts += ["", "=== ИСТОРИЯ ТРЕНИРОВОК (последние 5) ==="]
        for i, s in enumerate(history[-5:], 1):
            parts.append(f"\nСессия {i} ({s['date']}):")
            for ex in s["exercises"]:
                line = f"  {ex['name']}: {ex['sets_completed']}/{ex['sets_planned']} подх, {ex['reps']}"
                if ex.get("weight_kg") is not None:
                    line += f", {ex['weight_kg']}кг"
                if ex.get("rpe") is not None:
                    line += f", RPE {ex['rpe']}"
                parts.append(line)
            if s.get("sleep_hours") is not None:
                parts.append(f"  Сон: {s['sleep_hours']}ч")
            if s.get("fatigue_level") is not None:
                parts.append(f"  Усталость: {s['fatigue_level']}/10")
            if s.get("notes"):
                parts.append(f"  Заметки: {s['notes']}")

    if current:
        parts += ["", "=== ПОСЛЕДНЯЯ ТРЕНИРОВКА ==="]
        parts.append(f"Дата: {current['date']}")
        for ex in current["exercises"]:
            line = f"  {ex['name']}: {ex['sets_completed']}/{ex['sets_planned']} подх, {ex['reps']}"
            if ex.get("weight_kg") is not None:
                line += f", {ex['weight_kg']}кг"
            if ex.get("rpe") is not None:
                line += f", RPE {ex['rpe']}"
            parts.append(line)
        if current.get("sleep_hours") is not None:
            parts.append(f"  Сон: {current['sleep_hours']}ч")
        if current.get("fatigue_level") is not None:
            parts.append(f"  Усталость: {current['fatigue_level']}/10")
        if current.get("notes"):
            parts.append(f"  Заметки: {current['notes']}")

    parts += [
        "",
        "=== ЗАДАЧА ===",
        f"Верни JSON строго по структуре выше. mode = '{mode}' (с подчёркиванием).",
        "Поле next_session обязательно. Внутри него обязательно поле reasoning с объяснением.",
    ]

    return "\n".join(parts)