import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models import (
    CoachResponse,
    CoachSGRResponse,
    sgr_to_coach_response,
)


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[OK] {message}")


def build_valid_sgr_payload() -> dict:
    return {
        "mode": "adaptation",
        "input_summary": {
            "brief_goal": "Увеличение силовых показателей",
            "experience_level": "intermediate",
            "equipment_summary": "штанга, гантели, скамья",
            "restrictions_summary": "ограничений нет",
            "has_history": True,
            "has_current_session": True,
        },
        "progress_assessment": {
            "progress_detected": True,
            "supporting_facts": [
                "Все подходы выполнены",
                "RPE не превышает 7",
            ],
            "recommended_progression": "Небольшое увеличение рабочего веса",
        },
        "overload_assessment": {
            "overload_detected": False,
            "overload_signals": [],
            "recommended_adjustment": None,
        },
        "medical_risk_assessment": {
            "medical_risk_detected": False,
            "risk_signals": [],
            "refusal_required": False,
            "refuse_reason": None,
        },
        "restriction_assessment": {
            "restrictions_present": False,
            "limiting_factors": [],
            "restriction_impact_summary": "Ограничения не влияют на решение",
        },
        "decision_trace": {
            "selected_policy": "progressive_overload",
            "final_action": "increase_load",
            "policy_reasoning": "Признаки прогресса позволяют немного увеличить нагрузку",
        },
        "final_recommendation": {
            "session_assessment": "Тренировка выполнена уверенно",
            "decision": "Немного увеличить рабочие веса в следующей сессии",
            "exercise_changes": [
                {
                    "exercise_name": "Жим лёжа",
                    "change_type": "increase_weight",
                    "details": "70 -> 72.5 кг",
                }
            ],
            "reasoning": "Пользователь выполнил все подходы с умеренным RPE",
            "long_term_recommendation": "Продолжать прогрессию 2-3 недели",
            "safety_warnings": [],
            "refused": False,
            "refuse_reason": None,
        },
    }


def build_refusal_sgr_payload() -> dict:
    return {
        "mode": "adaptation",
        "input_summary": {
            "brief_goal": "Возврат к тренировкам",
            "experience_level": "beginner",
            "equipment_summary": "гантели",
            "restrictions_summary": "травма поясницы",
            "has_history": False,
            "has_current_session": True,
        },
        "progress_assessment": {
            "progress_detected": False,
            "supporting_facts": [],
            "recommended_progression": None,
        },
        "overload_assessment": {
            "overload_detected": True,
            "overload_signals": ["Высокая усталость"],
            "recommended_adjustment": "reduce_intensity",
        },
        "medical_risk_assessment": {
            "medical_risk_detected": True,
            "risk_signals": ["Острая боль в пояснице во время упражнения"],
            "refusal_required": True,
            "refuse_reason": "Обнаружен медицинский риск: острая боль в пояснице",
        },
        "restriction_assessment": {
            "restrictions_present": True,
            "limiting_factors": ["травма поясницы"],
            "restriction_impact_summary": "Силовая нагрузка на поясницу небезопасна",
        },
        "decision_trace": {
            "selected_policy": "medical_refusal",
            "final_action": "refuse",
            "policy_reasoning": "Медицинская безопасность имеет максимальный приоритет",
        },
        "final_recommendation": {
            "session_assessment": "Сессия небезопасна из-за боли в пояснице",
            "decision": "Отказ от тренировочной рекомендации до дополнительной оценки состояния",
            "exercise_changes": [],
            "reasoning": "Из-за острой боли в пояснице нельзя безопасно рекомендовать продолжение тренировки",
            "long_term_recommendation": None,
            "safety_warnings": [
                "Нужна консультация специалиста перед продолжением тренировок"
            ],
            "refused": True,
            "refuse_reason": "Обнаружен медицинский риск: острая боль в пояснице",
        },
    }


def check_valid_sgr_model() -> None:
    print_header("ПРОВЕРКА ВАЛИДНОЙ SGR-СХЕМЫ")

    payload = build_valid_sgr_payload()
    sgr = CoachSGRResponse(**payload)

    assert_true(sgr.mode.value == "adaptation", 'mode корректно валидируется')
    assert_true(
        sgr.progress_assessment.progress_detected is True,
        "progress_assessment корректно заполнен",
    )
    assert_true(
        sgr.decision_trace.final_action.value == "increase_load",
        "decision_trace.final_action корректен",
    )
    assert_true(
        sgr.final_recommendation.refused is False,
        "final_recommendation.refused корректен",
    )


def check_sgr_to_api_mapping() -> None:
    print_header("ПРОВЕРКА ПРЕОБРАЗОВАНИЯ SGR -> COACH_RESPONSE")

    payload = build_valid_sgr_payload()
    sgr = CoachSGRResponse(**payload)
    api_response = sgr_to_coach_response(sgr)

    assert_true(
        isinstance(api_response, CoachResponse),
        "sgr_to_coach_response возвращает CoachResponse",
    )
    assert_true(
        api_response.mode == "adaptation",
        "mode корректно перенесён в CoachResponse",
    )
    assert_true(
        api_response.next_session.decision == payload["final_recommendation"]["decision"],
        "decision корректно перенесён",
    )
    assert_true(
        len(api_response.next_session.exercise_changes) == 1,
        "exercise_changes корректно перенесены",
    )
    assert_true(
        api_response.refused is False,
        "refused корректно перенесён",
    )


def check_refusal_sgr_model() -> None:
    print_header("ПРОВЕРКА REFUSAL-СЦЕНАРИЯ В SGR")

    payload = build_refusal_sgr_payload()
    sgr = CoachSGRResponse(**payload)

    assert_true(
        sgr.medical_risk_assessment.medical_risk_detected is True,
        "medical_risk_detected=True в risky-сценарии",
    )
    assert_true(
        sgr.medical_risk_assessment.refusal_required is True,
        "refusal_required=True в risky-сценарии",
    )
    assert_true(
        sgr.decision_trace.final_action.value == "refuse",
        'final_action="refuse" при medical risk',
    )
    assert_true(
        sgr.final_recommendation.refused is True,
        "final_recommendation.refused=True при medical risk",
    )
    assert_true(
        sgr.final_recommendation.refuse_reason is not None,
        "refuse_reason заполнен при отказе",
    )
    assert_true(
        sgr.final_recommendation.exercise_changes == [],
        "exercise_changes пустой при отказе",
    )


def check_invalid_inconsistent_payload() -> None:
    print_header("ПРОВЕРКА ВАЛИДАЦИИ НЕСОГЛАСОВАННОГО SGR")

    invalid_payload = build_refusal_sgr_payload()
    invalid_payload["decision_trace"]["final_action"] = "increase_load"

    try:
        CoachSGRResponse(**invalid_payload)
    except Exception:
        print("[OK] Несогласованный payload корректно отклонён валидацией")
        return

    raise AssertionError("Несогласованный payload не был отклонён")


def main() -> None:
    check_valid_sgr_model()
    check_sgr_to_api_mapping()
    check_refusal_sgr_model()
    check_invalid_inconsistent_payload()

    print_header("ИТОГ")
    print("Все проверки SGR-моделей пройдены успешно.")


if __name__ == "__main__":
    main()