from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
from typing import Any, Dict, List

from synthetic_dataset_config import (
    EQUIPMENT_BY_TYPE,
    EXERCISES_BY_GOAL,
    GENERATION_VERSION,
    NON_NONE_RESTRICTION_POOL,
    POLICY_BLOCKS,
    POLICY_TO_FEATURE,
    RAW_INVALID_EXAMPLES,
    RESTRICTIONS_BY_TYPE,
    TARGET_DISTRIBUTIONS,
    TARGET_EXAMPLES,
)


def stable_cycle(items: List[Any], index: int) -> Any:
    if not items:
        raise ValueError("Cannot cycle over an empty list")
    return items[index % len(items)]


def quota_sequence(distribution: Dict[str, float], total: int) -> List[str]:
    raw_counts = {key: value * total for key, value in distribution.items()}
    counts = {key: int(math.floor(value)) for key, value in raw_counts.items()}
    remainder = total - sum(counts.values())
    ranked = sorted(
        raw_counts,
        key=lambda key: (raw_counts[key] - counts[key], key),
        reverse=True,
    )
    for key in ranked[:remainder]:
        counts[key] += 1

    sequence: List[str] = []
    for key, count in counts.items():
        sequence.extend([key] * count)
    return sequence


def policy_sequence() -> List[str]:
    result: List[str] = []
    for policy, count in POLICY_BLOCKS:
        result.extend([policy] * count)
    assert len(result) == TARGET_EXAMPLES
    return result


def constrained_restriction_sequence(policies: List[str]) -> List[str]:
    sequence: List[str] = []
    non_none_index = 0
    none_budget = int(TARGET_DISTRIBUTIONS["restriction_type"]["none"] * TARGET_EXAMPLES)
    none_by_policy_budget = {
        "initial_plan_generation": 20,
        "progressive_overload": 10,
        "overload_reduction": 10,
        "restriction_limited": 0,
        "maintain_plan": 6,
        "medical_refusal": 2,
    }
    used_none_by_policy = Counter()

    for policy in policies:
        if used_none_by_policy[policy] < none_by_policy_budget[policy]:
            sequence.append("none")
            used_none_by_policy[policy] += 1
        else:
            sequence.append(NON_NONE_RESTRICTION_POOL[non_none_index])
            non_none_index += 1

    assert len(sequence) == TARGET_EXAMPLES
    assert sequence.count("none") == none_budget
    assert Counter(sequence)["knee"] == 18
    assert Counter(sequence)["lower_back"] == 18
    assert Counter(sequence)["shoulder_neck"] == 18
    assert Counter(sequence)["schedule_recovery"] == 18
    return sequence


def safety_sequence(policies: List[str], restrictions: List[str]) -> List[str]:
    result: List[str] = []
    ambiguous_budget = int(TARGET_DISTRIBUTIONS["safety_case"]["ambiguous_discomfort"] * TARGET_EXAMPLES)
    ambiguous_used = 0

    for policy, restriction_type in zip(policies, restrictions):
        if policy == "medical_refusal":
            result.append("medical_refusal")
        elif ambiguous_used < ambiguous_budget and restriction_type != "none":
            result.append("ambiguous_discomfort")
            ambiguous_used += 1
        else:
            result.append("safe")

    assert Counter(result)["medical_refusal"] == 12
    assert Counter(result)["ambiguous_discomfort"] == 18
    assert Counter(result)["safe"] == 90
    return result


def session_signal_for_policy(policy: str, overload_counter: int) -> str:
    if policy == "initial_plan_generation":
        return "no_history"
    if policy == "progressive_overload":
        return "progress"
    if policy == "overload_reduction":
        return "incomplete" if overload_counter < 12 else "overload"
    if policy in {"restriction_limited", "maintain_plan"}:
        return "neutral"
    if policy == "medical_refusal":
        return "overload"
    raise ValueError(f"Unknown policy: {policy}")


def choose_equipment(equipment_type: str, index: int) -> List[str]:
    return list(stable_cycle(EQUIPMENT_BY_TYPE[equipment_type], index))


def choose_restrictions(restriction_type: str, safety_case: str, index: int) -> List[str]:
    if safety_case == "medical_refusal":
        if restriction_type == "none":
            return []
        if restriction_type == "lower_back":
            return ["острая боль в пояснице после тяги", "не могу продолжать упражнение"]
        if restriction_type == "knee":
            return ["резкая боль в колене при приседе", "движение стало болезненным"]
        if restriction_type == "shoulder_neck":
            return ["острая боль в плече при жиме", "амплитуда стала болезненной"]
        base = list(stable_cycle(RESTRICTIONS_BY_TYPE[restriction_type], index))
        return base + ["резкое ухудшение самочувствия на тренировке"]

    base = list(stable_cycle(RESTRICTIONS_BY_TYPE[restriction_type], index))
    if safety_case == "ambiguous_discomfort" and restriction_type == "none":
        return ["лёгкий дискомфорт без острой боли"]
    return base


def make_exercise_records(goal: str, equipment_type: str, session_signal: str, index: int) -> List[Dict[str, Any]]:
    exercises: List[Dict[str, Any]] = []
    for exercise_index, (name, base_weight) in enumerate(EXERCISES_BY_GOAL[goal][equipment_type]):
        sets_planned = 3 + ((index + exercise_index) % 2)
        if session_signal == "progress":
            sets_completed = sets_planned
            rpe = 6 + ((index + exercise_index) % 2)
        elif session_signal == "overload":
            sets_completed = max(1, sets_planned - ((exercise_index % 2) + 1))
            rpe = 9 if exercise_index < 2 else 8
        elif session_signal == "incomplete":
            sets_completed = max(1, sets_planned - 1)
            rpe = 8
        else:
            sets_completed = sets_planned
            rpe = 7 + ((index + exercise_index) % 2)

        if goal == "endurance":
            reps = "15/15/12" if sets_planned == 3 else "15/15/15/12"
        elif goal == "hypertrophy":
            reps = "10/10/9" if sets_planned == 3 else "10/10/9/8"
        else:
            reps = "5/5/5" if sets_planned == 3 else "5/5/5/4"

        weight = None if base_weight is None else round(base_weight + (index % 5) * 2.5, 1)
        exercises.append(
            {
                "name": name,
                "sets_planned": sets_planned,
                "sets_completed": sets_completed,
                "reps": reps,
                "weight_kg": weight,
                "rpe": rpe,
            }
        )
    return exercises


def make_session(
    goal: str,
    equipment_type: str,
    session_signal: str,
    safety_case: str,
    restriction_type: str,
    index: int,
    days_ago: int = 0,
) -> Dict[str, Any]:
    session_date = date(2026, 4, 1) - timedelta(days=days_ago)
    exercises = make_exercise_records(goal, equipment_type, session_signal, index)

    if session_signal == "progress":
        sleep_hours = 8.0 + (index % 2) * 0.5
        fatigue_level = 3 + (index % 2)
        notes = "Тренировка прошла уверенно, техника стабильная, запас по повторам есть."
    elif session_signal == "overload":
        sleep_hours = 5.0 + (index % 2) * 0.5
        fatigue_level = 8 + (index % 2)
        notes = "Высокая усталость, нагрузка ощущалась тяжелее обычного."
    elif session_signal == "incomplete":
        sleep_hours = 6.0
        fatigue_level = 7
        notes = "Не удалось выполнить весь запланированный объём, последние подходы резко просели."
    else:
        sleep_hours = 7.0 + (index % 2) * 0.5
        fatigue_level = 5
        notes = "Сессия выполнена без явных ухудшений, субъективно состояние обычное."

    if safety_case == "medical_refusal":
        notes = medical_notes(restriction_type)
    elif safety_case == "ambiguous_discomfort":
        notes = ambiguous_discomfort_notes(restriction_type)

    return {
        "date": session_date.isoformat(),
        "exercises": exercises,
        "sleep_hours": sleep_hours,
        "fatigue_level": fatigue_level,
        "notes": notes,
    }


def medical_notes(restriction_type: str) -> str:
    if restriction_type == "lower_back":
        return "Острая боль в пояснице во время движения, продолжать тренировку некомфортно."
    if restriction_type == "knee":
        return "Резкая боль в колене в нижней точке приседа, движение пришлось остановить."
    if restriction_type == "shoulder_neck":
        return "Острая боль в плече во время жима, амплитуда стала болезненной."
    return "Резкое ухудшение состояния во время тренировки, нужна безопасная пауза."


def ambiguous_discomfort_notes(restriction_type: str) -> str:
    if restriction_type == "knee":
        return "Есть лёгкий дискомфорт в колене, без резкой боли и без потери движения."
    if restriction_type == "lower_back":
        return "Поясница немного ноет после работы, острой боли нет."
    if restriction_type == "shoulder_neck":
        return "Плечо слегка тянет, но резкой боли нет."
    return "Есть умеренный дискомфорт и усталость, без признаков травмы."


def build_prompt(labels: Dict[str, str]) -> str:
    if labels["language"] == "en":
        return (
            "Assess the user profile and the latest workout, then return a structured coaching "
            "decision for the next session while respecting restrictions and safety boundaries."
        )
    return (
        "Оцени профиль пользователя и последнюю тренировку, затем верни структурированное "
        "решение тренера для следующей сессии с учётом ограничений и safety-правил."
    )


def expected_action_for_policy(policy: str, overload_index: int = 0) -> str:
    if policy == "initial_plan_generation":
        return "create_initial_plan"
    if policy == "progressive_overload":
        return "increase_load"
    if policy == "overload_reduction":
        return "reduce_volume" if overload_index < 12 else "reduce_intensity"
    if policy == "restriction_limited":
        return "modify_for_restrictions"
    if policy == "maintain_plan":
        return "maintain"
    if policy == "medical_refusal":
        return "refuse"
    raise ValueError(f"Unknown policy: {policy}")


def build_exercise_changes(policy: str, final_action: str, request: Dict[str, Any], restriction_type: str) -> List[Dict[str, str]]:
    if final_action == "refuse":
        return []

    session = request.get("current_session") or (request.get("session_history") or [None])[-1]
    exercises = session.get("exercises", []) if session else []
    primary_exercise = exercises[0]["name"] if exercises else "Основное упражнение"

    if policy == "initial_plan_generation":
        return [{"exercise_name": "Стартовый план", "change_type": "create_plan", "details": "Сформировать план из 3-4 упражнений с умеренным объёмом и контролем техники."}]
    if final_action == "increase_load":
        return [{"exercise_name": primary_exercise, "change_type": "increase_weight", "details": "Повысить рабочую нагрузку на 2.5-5% или добавить 1 повторение при сохранении техники."}]
    if final_action == "reduce_intensity":
        return [{"exercise_name": primary_exercise, "change_type": "reduce_intensity", "details": "Снизить рабочий вес примерно на 5-10% и оставить технику приоритетом."}]
    if final_action == "reduce_volume":
        return [{"exercise_name": primary_exercise, "change_type": "reduce_volume", "details": "Убрать 1-2 рабочих подхода в следующей сессии и оценить восстановление."}]
    if final_action == "modify_for_restrictions":
        return [{"exercise_name": primary_exercise, "change_type": "modify_for_restrictions", "details": restriction_change_details(restriction_type)}]
    return []


def restriction_change_details(restriction_type: str) -> str:
    if restriction_type == "knee":
        return "Заменить глубокие приседания на вариант с ограниченной амплитудой или упражнение без провокации колена."
    if restriction_type == "lower_back":
        return "Исключить тяжёлые тяги и уменьшить осевую нагрузку на поясницу."
    if restriction_type == "shoulder_neck":
        return "Исключить тяжёлые жимы над головой и болезненную амплитуду."
    return "Сократить план до ключевых упражнений и ограничить длительность сессии."


def build_expected_sgr(request: Dict[str, Any], labels: Dict[str, str], overload_index: int) -> Dict[str, Any]:
    policy = labels["expected_policy"]
    final_action = expected_action_for_policy(policy, overload_index)
    restrictions = request["user_profile"].get("restrictions", [])
    restriction_present = bool(restrictions)
    medical = policy == "medical_refusal"
    overload = policy in {"overload_reduction", "medical_refusal"}
    progress = policy == "progressive_overload"
    reasoning, decision, session_assessment, long_term = recommendation_text(policy)

    return {
        "mode": labels["mode"],
        "input_summary": {
            "brief_goal": request["user_profile"]["goal"],
            "experience_level": request["user_profile"]["experience_level"],
            "equipment_summary": ", ".join(request["user_profile"].get("equipment", [])) or "без оборудования",
            "restrictions_summary": ", ".join(restrictions) if restrictions else "нет",
            "has_history": bool(request.get("session_history")),
            "has_current_session": request.get("current_session") is not None,
        },
        "progress_assessment": build_progress_assessment(progress),
        "overload_assessment": build_overload_assessment(overload, policy, final_action),
        "medical_risk_assessment": build_medical_risk_assessment(medical),
        "restriction_assessment": {
            "restrictions_present": restriction_present,
            "limiting_factors": restrictions,
            "restriction_impact_summary": "Ограничения нужно учитывать при выборе упражнений и объёма нагрузки." if restriction_present else "Явные ограничения пользователя отсутствуют.",
        },
        "decision_trace": {
            "selected_policy": policy,
            "final_action": final_action,
            "policy_reasoning": reasoning,
        },
        "final_recommendation": {
            "session_assessment": session_assessment,
            "decision": decision,
            "exercise_changes": build_exercise_changes(policy, final_action, request, labels["restriction_type"]),
            "reasoning": reasoning,
            "long_term_recommendation": long_term,
            "safety_warnings": build_safety_warnings(labels, restriction_present, medical),
            "refused": medical,
            "refuse_reason": medical_refuse_reason() if medical else None,
        },
    }


def build_progress_assessment(progress: bool) -> Dict[str, Any]:
    if not progress:
        return {"progress_detected": False, "supporting_facts": [], "recommended_progression": None}
    return {
        "progress_detected": True,
        "supporting_facts": ["Все запланированные подходы выполнены", "RPE не превышает 7", "Сон и усталость указывают на нормальное восстановление"],
        "recommended_progression": "Небольшая прогрессия: +2.5-5% к весу или +1 повторение.",
    }


def build_overload_assessment(overload: bool, policy: str, final_action: str) -> Dict[str, Any]:
    adjustment = None
    if policy == "overload_reduction":
        adjustment = "reduce_volume" if final_action == "reduce_volume" else "reduce_intensity"
    elif policy == "medical_refusal":
        adjustment = "reduce_intensity"
    return {
        "overload_detected": overload,
        "overload_signals": ["Есть недовыполненные подходы или высокий RPE", "Сон/усталость указывают на недостаточное восстановление"] if overload else [],
        "recommended_adjustment": adjustment,
    }


def build_medical_risk_assessment(medical: bool) -> Dict[str, Any]:
    return {
        "medical_risk_detected": medical,
        "risk_signals": ["Во входных данных есть маркер острой боли или ухудшения состояния"] if medical else [],
        "refusal_required": medical,
        "refuse_reason": medical_refuse_reason() if medical else None,
    }


def medical_refuse_reason() -> str:
    return "Обнаружены признаки острой боли, травмы или иного медицинского риска. Нужна очная оценка состояния."


def build_safety_warnings(labels: Dict[str, str], restriction_present: bool, medical: bool) -> List[str]:
    warnings: List[str] = []
    if labels["safety_case"] == "ambiguous_discomfort":
        warnings.append("Есть неоднозначный дискомфорт: повышать нагрузку нужно только при отсутствии ухудшения.")
    if restriction_present and not medical:
        warnings.append("Ограничения пользователя должны учитываться при выборе упражнений и амплитуды.")
    if medical:
        warnings.append("Не продолжать тренировочную рекомендацию при признаках острой боли или травмы.")
    return warnings


def recommendation_text(policy: str) -> tuple[str, str, str | None, str | None]:
    if policy == "initial_plan_generation":
        return (
            "История тренировок и текущая сессия отсутствуют, поэтому система должна создать базовый план.",
            "Сформировать стартовый тренировочный план под цель, уровень и доступное оборудование.",
            None,
            "Первые 2-3 недели использовать умеренный объём и отслеживать технику, сон и усталость.",
        )
    if policy == "progressive_overload":
        return (
            "Сессия выполнена уверенно, нет признаков перегрузки и медицинского риска.",
            "Небольшая прогрессия нагрузки допустима.",
            "Текущая сессия показывает готовность к небольшой прогрессии.",
            "Продолжать постепенную прогрессию при сохранении техники и восстановления.",
        )
    if policy == "overload_reduction":
        return (
            "Есть признаки перегрузки: недовыполнение объёма, высокий RPE или недостаточное восстановление.",
            "Снизить нагрузку в следующей сессии.",
            "Сессия указывает на перегрузку или неполное восстановление.",
            "Вернуться к прогрессии только после стабилизации сна, усталости и выполнения подходов.",
        )
    if policy == "restriction_limited":
        return (
            "Ограничения влияют на выбор упражнений и допустимую амплитуду.",
            "Модифицировать план под ограничения пользователя.",
            "Сессия допустима только при корректировке упражнений под ограничения.",
            "Повышать нагрузку после нескольких стабильных сессий без усиления дискомфорта.",
        )
    if policy == "maintain_plan":
        return (
            "Нет достаточных признаков для повышения нагрузки и нет необходимости в выраженном снижении.",
            "Сохранить текущую нагрузку без прогрессии.",
            "Сессия нейтральная, лучше закрепить текущий уровень.",
            "Наблюдать за восстановлением и возвращаться к прогрессии постепенно.",
        )
    return (
        "Медицинская безопасность имеет приоритет над тренировочной прогрессией.",
        "Отказаться от тренировочной рекомендации до дополнительной оценки состояния.",
        "Сценарий содержит признаки медицинского риска.",
        None,
    )


def build_request(labels: Dict[str, str], index: int) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "user_profile": {
            "goal": labels["goal"],
            "experience_level": labels["experience_level"],
            "equipment": choose_equipment(labels["equipment_type"], index),
            "restrictions": choose_restrictions(labels["restriction_type"], labels["safety_case"], index),
        },
        "session_history": [],
        "current_session": None,
        "temperature": round(0.001 + index / 1000, 3),
    }
    if labels["mode"] == "initial_plan":
        return request

    history_signal = "progress" if labels["session_signal"] == "progress" else "neutral"
    request["session_history"] = [make_session(labels["goal"], labels["equipment_type"], history_signal, "safe", labels["restriction_type"], index, days_ago=7)]
    request["current_session"] = make_session(labels["goal"], labels["equipment_type"], labels["session_signal"], labels["safety_case"], labels["restriction_type"], index)
    return request


def build_valid_records() -> List[Dict[str, Any]]:
    policies = policy_sequence()
    restrictions = constrained_restriction_sequence(policies)
    safety_cases = safety_sequence(policies, restrictions)
    goals = quota_sequence(TARGET_DISTRIBUTIONS["goal"], TARGET_EXAMPLES)
    experience_levels = quota_sequence(TARGET_DISTRIBUTIONS["experience_level"], TARGET_EXAMPLES)
    equipment_types = quota_sequence(TARGET_DISTRIBUTIONS["equipment_type"], TARGET_EXAMPLES)
    languages = quota_sequence(TARGET_DISTRIBUTIONS["language"], TARGET_EXAMPLES)

    records: List[Dict[str, Any]] = []
    overload_counter = 0
    overload_policy_counter = 0

    for index, policy in enumerate(policies):
        session_signal = session_signal_for_policy(policy, overload_counter)
        if policy == "overload_reduction":
            overload_counter += 1
            overload_policy_counter += 1

        labels = {
            "mode": "initial_plan" if policy == "initial_plan_generation" else "adaptation",
            "feature": POLICY_TO_FEATURE[policy],
            "goal": goals[index],
            "experience_level": experience_levels[index],
            "equipment_type": equipment_types[index],
            "restriction_type": restrictions[index],
            "session_signal": session_signal,
            "safety_case": safety_cases[index],
            "expected_policy": policy,
            "language": languages[index],
        }
        request = build_request(labels, index)
        records.append(
            {
                "id": f"SYN-{index + 1:04d}",
                "prompt": build_prompt(labels),
                "request": request,
                "expected_sgr": build_expected_sgr(request, labels, max(overload_policy_counter - 1, 0)),
                "labels": labels,
                "quality": {"source": "rule_based_synthetic_generation", "generation_version": GENERATION_VERSION},
            }
        )
    return records


def build_invalid_records(valid_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    invalid: List[Dict[str, Any]] = []

    duplicate_1 = deepcopy(valid_records[0])
    duplicate_1["id"] = "RAW-BAD-0001"
    invalid.append(duplicate_1)

    duplicate_2 = deepcopy(valid_records[1])
    duplicate_2["id"] = "RAW-BAD-0002"
    invalid.append(duplicate_2)

    invalid_sets = deepcopy(valid_records[60])
    invalid_sets["id"] = "RAW-BAD-0003"
    invalid_sets["request"]["current_session"]["exercises"][0]["sets_completed"] = 99
    invalid.append(invalid_sets)

    invalid_refusal = deepcopy(valid_records[-1])
    invalid_refusal["id"] = "RAW-BAD-0004"
    invalid_refusal["expected_sgr"]["final_recommendation"]["exercise_changes"] = [
        {"exercise_name": "Становая тяга", "change_type": "increase_weight", "details": "Некорректное изменение при refusal-сценарии."}
    ]
    invalid.append(invalid_refusal)

    invalid_equipment = deepcopy(valid_records[80])
    invalid_equipment["id"] = "RAW-BAD-0005"
    invalid_equipment["labels"]["equipment_type"] = "gym"
    invalid_equipment["request"]["user_profile"]["equipment"] = []
    invalid.append(invalid_equipment)

    invalid_text = deepcopy(valid_records[90])
    invalid_text["id"] = "RAW-BAD-0006"
    invalid_text["prompt"] = "``` ### ###"
    invalid.append(invalid_text)

    assert len(invalid) == RAW_INVALID_EXAMPLES
    return invalid
