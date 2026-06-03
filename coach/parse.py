import re
import logging

logger = logging.getLogger(__name__)


def _safe_get_dict(data: dict, key: str) -> dict:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def extract_json_from_model_answer(raw_answer: str) -> str:
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        raise ValueError(
            "Ожидалась непустая строка с JSON-ответом модели, "
            f"получено: {raw_answer!r}"
        )

    raw_answer = re.sub(r"```(?:json)?\s*", "", raw_answer).strip()
    raw_answer = raw_answer.replace("```", "").strip()

    json_start = raw_answer.find("{")
    json_end = raw_answer.rfind("}")

    if json_start == -1 or json_end == -1 or json_end < json_start:
        raise ValueError(f"JSON не найден в ответе модели: {raw_answer[:300]}")

    return raw_answer[json_start : json_end + 1]


def _as_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "да"}

    if value is None:
        return default

    return bool(value)


def _as_str(value, default=""):
    if value is None:
        return default

    if isinstance(value, str):
        return value.strip()

    return str(value)


def _normalize_sgr_exercise_changes(value):
    if not isinstance(value, list):
        return []

    normalized = []

    for item in value:
        if not isinstance(item, dict):
            normalized.append(
                {
                    "exercise_name": "Не указано",
                    "change_type": "modify",
                    "details": str(item),
                }
            )
            continue

        exercise_name = (
            item.get("exercise_name")
            or item.get("exercise")
            or item.get("name")
            or "Не указано"
        )

        change_type = item.get("change_type") or item.get("type")
        details = item.get("details") or item.get("description")

        if not change_type:
            if "new_weight_kg" in item:
                change_type = "increase_weight"
            elif "remove" in item:
                change_type = "remove_exercise"
            elif "replace_with" in item:
                change_type = "replace_exercise"
            else:
                change_type = "modify"

        if not details:
            if "new_weight_kg" in item:
                details = f"Новый рабочий вес: {item['new_weight_kg']} кг"
            elif "replace_with" in item:
                details = f"Заменить на {item['replace_with']}"
            elif "remove" in item:
                details = "Убрать упражнение"
            else:
                extra_parts = []

                for key, val in item.items():
                    if key not in {
                        "exercise_name",
                        "exercise",
                        "name",
                        "change_type",
                        "type",
                    }:
                        extra_parts.append(f"{key}: {val}")

                details = "; ".join(extra_parts) if extra_parts else "Без деталей"

        normalized.append(
            {
                "exercise_name": str(exercise_name),
                "change_type": str(change_type),
                "details": str(details),
            }
        )

    return normalized


def normalize_sgr_response_shape(response_data: dict) -> dict:
    if not isinstance(response_data, dict):
        return response_data

    mode = _as_str(response_data.get("mode"), "initial_plan").lower()

    if mode not in {"initial_plan", "adaptation"}:
        mode = "initial_plan"

    response_data["mode"] = mode

    input_summary = _safe_get_dict(response_data, "input_summary")

    response_data["input_summary"] = {
        "brief_goal": _as_str(
            input_summary.get("brief_goal")
            or input_summary.get("goal")
            or input_summary.get("goal_summary"),
            "Не указано",
        ),
        "experience_level": _as_str(input_summary.get("experience_level"), "beginner"),
        "equipment_summary": _as_str(
            input_summary.get("equipment_summary") or input_summary.get("equipment"),
            "Не указано",
        ),
        "restrictions_summary": _as_str(
            input_summary.get("restrictions_summary")
            or input_summary.get("restrictions"),
            "нет",
        ),
        "has_history": _as_bool(
            input_summary.get("has_history")
            if "has_history" in input_summary
            else input_summary.get("history_exists"),
            False,
        ),
        "has_current_session": _as_bool(
            input_summary.get("has_current_session")
            if "has_current_session" in input_summary
            else input_summary.get("current_session"),
            False,
        ),
    }

    progress = _safe_get_dict(response_data, "progress_assessment")

    response_data["progress_assessment"] = {
        "progress_detected": _as_bool(
            progress.get("progress_detected")
            if "progress_detected" in progress
            else progress.get("progress_signs_exist"),
            False,
        ),
        "supporting_facts": _as_list(
            progress.get("supporting_facts") or progress.get("progress_signs") or []
        ),
        "recommended_progression": (
            _as_str(
                progress.get("recommended_progression")
                or progress.get("progression")
                or progress.get("suggested_progression"),
                "",
            )
            or None
        ),
    }

    overload = _safe_get_dict(response_data, "overload_assessment")

    recommended_adjustment = (
        overload.get("recommended_adjustment")
        or overload.get("adjustment_type")
        or overload.get("suggested_adjustment")
    )
    recommended_adjustment = _as_str(recommended_adjustment, "")

    if recommended_adjustment not in {"reduce_intensity", "reduce_volume"}:
        recommended_adjustment = None

    response_data["overload_assessment"] = {
        "overload_detected": _as_bool(
            overload.get("overload_detected")
            if "overload_detected" in overload
            else overload.get("overload_signs_exist"),
            False,
        ),
        "overload_signals": _as_list(
            overload.get("overload_signals") or overload.get("signals") or []
        ),
        "recommended_adjustment": recommended_adjustment,
    }

    medical = _safe_get_dict(response_data, "medical_risk_assessment")

    medical_risk_detected = _as_bool(
        medical.get("medical_risk_detected")
        if "medical_risk_detected" in medical
        else medical.get("risk_detected"),
        False,
    )
    refusal_required = _as_bool(medical.get("refusal_required"), medical_risk_detected)

    response_data["medical_risk_assessment"] = {
        "medical_risk_detected": medical_risk_detected,
        "risk_signals": _as_list(
            medical.get("risk_signals") or medical.get("medical_signals") or []
        ),
        "refusal_required": refusal_required,
        "refuse_reason": (_as_str(medical.get("refuse_reason"), "") or None),
    }

    restriction = _safe_get_dict(response_data, "restriction_assessment")

    response_data["restriction_assessment"] = {
        "restrictions_present": _as_bool(
            restriction.get("restrictions_present")
            if "restrictions_present" in restriction
            else restriction.get("restrictions_exist"),
            False,
        ),
        "limiting_factors": _as_list(
            restriction.get("limiting_factors")
            or restriction.get("restriction_factors")
            or []
        ),
        "restriction_impact_summary": _as_str(
            restriction.get("restriction_impact_summary")
            or restriction.get("impact_summary"),
            "Ограничения не влияют на решение",
        ),
    }

    trace = _safe_get_dict(response_data, "decision_trace")

    selected_policy = _as_str(trace.get("selected_policy") or trace.get("main_rule"), "")

    _POLICY_ALIASES = {
        "medical safety": "medical_refusal",
        "medical_refusal_policy": "medical_refusal",
        "progression": "progressive_overload",
        "progressive overload": "progressive_overload",
        "overload": "overload_reduction",
        "initial plan": "initial_plan_generation",
    }

    VALID_POLICIES = {
        "medical_refusal",
        "restriction_limited",
        "overload_reduction",
        "progressive_overload",
        "maintain_plan",
        "initial_plan_generation",
    }

    selected_policy = _POLICY_ALIASES.get(selected_policy.lower(), selected_policy)

    if selected_policy not in VALID_POLICIES:
        if response_data["medical_risk_assessment"]["medical_risk_detected"]:
            selected_policy = "medical_refusal"
        elif response_data["mode"] == "initial_plan":
            selected_policy = "initial_plan_generation"
        elif response_data["overload_assessment"]["overload_detected"]:
            selected_policy = "overload_reduction"
        elif response_data["restriction_assessment"]["restrictions_present"]:
            selected_policy = "restriction_limited"
        elif response_data["progress_assessment"]["progress_detected"]:
            selected_policy = "progressive_overload"
        else:
            selected_policy = "maintain_plan"

    final_action = _as_str(trace.get("final_action"), "")

    _ACTION_ALIASES = {
        "proceed": "create_initial_plan" if response_data["mode"] == "initial_plan" else "maintain",
        "continue": "maintain",
        "adapt": "maintain",
        "increase": "increase_load",
        "increase_weight": "increase_load",
        "reduce_load": "reduce_intensity",
    }

    VALID_ACTIONS = {
        "refuse",
        "create_initial_plan",
        "increase_load",
        "reduce_intensity",
        "reduce_volume",
        "maintain",
        "modify_for_restrictions",
    }

    final_action = _ACTION_ALIASES.get(final_action, final_action)

    if final_action not in VALID_ACTIONS:
        if response_data["medical_risk_assessment"]["medical_risk_detected"]:
            final_action = "refuse"
        elif response_data["mode"] == "initial_plan":
            final_action = "create_initial_plan"
        elif response_data["overload_assessment"]["recommended_adjustment"] == "reduce_volume":
            final_action = "reduce_volume"
        elif response_data["overload_assessment"]["overload_detected"]:
            final_action = "reduce_intensity"
        elif response_data["restriction_assessment"]["restrictions_present"]:
            final_action = "modify_for_restrictions"
        elif response_data["progress_assessment"]["progress_detected"]:
            final_action = "increase_load"
        else:
            final_action = "maintain"

    response_data["decision_trace"] = {
        "selected_policy": selected_policy,
        "final_action": final_action,
        "policy_reasoning": _as_str(
            trace.get("policy_reasoning")
            or trace.get("rule_reasoning")
            or trace.get("main_rule")
            or "Решение выбрано на основе анализа входных данных",
            "Решение выбрано на основе анализа входных данных",
        ),
    }

    final = _safe_get_dict(response_data, "final_recommendation")

    exercise_changes = _normalize_sgr_exercise_changes(
        final.get("exercise_changes") or final.get("changes") or []
    )

    final_decision = _as_str(
        final.get("decision")
        or final.get("final_decision")
        or final.get("recommendation")
        or trace.get("final_decision"),
        "",
    )

    response_data["final_recommendation"] = {
        "session_assessment": (
            _as_str(final.get("session_assessment") or final.get("session_evaluation"), "")
            or None
        ),
        "decision": final_decision or "Решение сформировано на основе анализа",
        "exercise_changes": exercise_changes,
        "reasoning": _as_str(
            final.get("reasoning")
            or final.get("explanation")
            or response_data["decision_trace"]["policy_reasoning"],
            response_data["decision_trace"]["policy_reasoning"],
        ),
        "long_term_recommendation": (
            _as_str(final.get("long_term_recommendation"), "") or None
        ),
        "safety_warnings": _as_list(final.get("safety_warnings") or []),
        "refused": _as_bool(
            final.get("refused"),
            response_data["medical_risk_assessment"]["refusal_required"],
        ),
        "refuse_reason": (
            _as_str(
                final.get("refuse_reason")
                or response_data["medical_risk_assessment"]["refuse_reason"],
                "",
            )
            or None
        ),
    }

    if response_data["medical_risk_assessment"]["medical_risk_detected"]:
        response_data["decision_trace"]["selected_policy"] = "medical_refusal"
        response_data["decision_trace"]["final_action"] = "refuse"
        response_data["final_recommendation"]["refused"] = True
        response_data["final_recommendation"]["exercise_changes"] = []

        if not response_data["final_recommendation"]["refuse_reason"]:
            response_data["final_recommendation"]["refuse_reason"] = "Обнаружен медицинский риск"

        if not response_data["final_recommendation"]["decision"]:
            response_data["final_recommendation"]["decision"] = (
                "Отказ от тренировочной рекомендации из-за медицинского риска"
            )

    return response_data
