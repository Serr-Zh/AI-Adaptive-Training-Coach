from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from synthetic_dataset_config import (
    EXPERIENCE_LEVELS,
    FINAL_ACTIONS,
    GENERATION_VERSION,
    GOALS,
    MAX_ALLOWED_DEVIATION_PP,
    MAX_ALLOWED_SAFETY_DEVIATION_PP,
    POLICIES,
    TARGET_DISTRIBUTIONS,
    TARGET_EXAMPLES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - depends on original project environment
    from models import CoachRequest, CoachSGRResponse  # type: ignore
except Exception:  # pragma: no cover - standalone fallback
    CoachRequest = None  # type: ignore
    CoachSGRResponse = None  # type: ignore


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_for_hash(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_hash(request: Dict[str, Any]) -> str:
    return hashlib.sha256(normalize_for_hash(request).encode("utf-8")).hexdigest()


def fallback_validate_training_session(session: Dict[str, Any]) -> Optional[str]:
    if not isinstance(session, dict):
        return "training_session_not_dict"
    if not isinstance(session.get("date"), str) or not session.get("date"):
        return "invalid_session_date"

    exercises = session.get("exercises", [])
    if not isinstance(exercises, list):
        return "exercises_not_list"

    for exercise in exercises:
        if not isinstance(exercise, dict):
            return "exercise_not_dict"
        if not exercise.get("name"):
            return "empty_exercise_name"
        reason = validate_sets_and_rpe(exercise)
        if reason:
            return reason

    sleep_hours = session.get("sleep_hours")
    if sleep_hours is not None and (not isinstance(sleep_hours, (int, float)) or sleep_hours < 0 or sleep_hours > 24):
        return "invalid_sleep_hours"

    fatigue = session.get("fatigue_level")
    if fatigue is not None and (not isinstance(fatigue, int) or fatigue < 1 or fatigue > 10):
        return "invalid_fatigue_level"

    return None


def validate_sets_and_rpe(exercise: Dict[str, Any]) -> Optional[str]:
    sets_planned = exercise.get("sets_planned")
    sets_completed = exercise.get("sets_completed")
    if not isinstance(sets_planned, int) or sets_planned < 1:
        return "invalid_sets_planned"
    if not isinstance(sets_completed, int) or sets_completed < 0:
        return "invalid_sets_completed"
    if sets_completed > sets_planned:
        return "sets_completed_gt_planned"
    rpe = exercise.get("rpe")
    if rpe is not None and (not isinstance(rpe, int) or rpe < 1 or rpe > 10):
        return "invalid_rpe"
    return None


def fallback_validate_request(request: Dict[str, Any]) -> Optional[str]:
    if not isinstance(request, dict):
        return "request_not_dict"

    profile = request.get("user_profile")
    if not isinstance(profile, dict):
        return "missing_user_profile"
    if profile.get("goal") not in GOALS:
        return "invalid_goal"
    if profile.get("experience_level") not in EXPERIENCE_LEVELS:
        return "invalid_experience_level"
    if not isinstance(profile.get("equipment"), list):
        return "equipment_not_list"
    if not isinstance(profile.get("restrictions", []), list):
        return "restrictions_not_list"

    session_history = request.get("session_history", [])
    if not isinstance(session_history, list):
        return "session_history_not_list"
    for session in session_history:
        reason = fallback_validate_training_session(session)
        if reason:
            return reason

    current_session = request.get("current_session")
    if current_session is not None:
        reason = fallback_validate_training_session(current_session)
        if reason:
            return reason

    temperature = request.get("temperature", 0.3)
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 1:
        return "invalid_temperature"

    return None


def fallback_validate_expected_sgr(expected_sgr: Dict[str, Any]) -> Optional[str]:
    if not isinstance(expected_sgr, dict):
        return "expected_sgr_not_dict"
    if expected_sgr.get("mode") not in {"initial_plan", "adaptation"}:
        return "invalid_sgr_mode"

    trace = expected_sgr.get("decision_trace", {})
    final = expected_sgr.get("final_recommendation", {})
    medical = expected_sgr.get("medical_risk_assessment", {})

    if trace.get("selected_policy") not in POLICIES:
        return "invalid_policy"
    if trace.get("final_action") not in FINAL_ACTIONS:
        return "invalid_final_action"
    if medical.get("refusal_required") and not final.get("refused"):
        return "medical_refusal_not_refused"
    if final.get("refused") and not final.get("refuse_reason"):
        return "refused_without_reason"
    if final.get("refused") and final.get("exercise_changes"):
        return "refused_with_exercise_changes"
    if medical.get("medical_risk_detected") and trace.get("final_action") != "refuse":
        return "medical_risk_without_refuse_action"

    return None


def validate_with_project_models(record: Dict[str, Any]) -> Optional[str]:
    if CoachRequest is None or CoachSGRResponse is None:
        return None
    try:
        CoachRequest(**record["request"])
    except Exception as exc:  # pragma: no cover
        return f"project_request_validation_error:{exc.__class__.__name__}"
    try:
        CoachSGRResponse(**record["expected_sgr"])
    except Exception as exc:  # pragma: no cover
        return f"project_sgr_validation_error:{exc.__class__.__name__}"
    return None


def quality_filter_reason(record: Dict[str, Any], seen_hashes: set[str]) -> Optional[str]:
    prompt = normalize_text(str(record.get("prompt", "")))
    if len(prompt) < 30 or re.fullmatch(r"[`#\s]+", prompt):
        return "low_quality_prompt"

    labels = record.get("labels", {})
    for dimension in TARGET_DISTRIBUTIONS:
        if dimension not in labels:
            return f"missing_label:{dimension}"

    request = record.get("request", {})
    expected = record.get("expected_sgr", {})

    for validator in (validate_with_project_models, fallback_validate_request, fallback_validate_expected_sgr):
        reason = validator(record) if validator is validate_with_project_models else validator(request if validator is fallback_validate_request else expected)
        if reason:
            return reason

    reason = validate_cross_field_consistency(record)
    if reason:
        return reason

    hsh = request_hash(request)
    if hsh in seen_hashes:
        return "duplicate_request"
    seen_hashes.add(hsh)
    return None


def validate_cross_field_consistency(record: Dict[str, Any]) -> Optional[str]:
    labels = record["labels"]
    request = record["request"]
    expected = record["expected_sgr"]

    if labels.get("mode") == "initial_plan" and request.get("current_session") is not None:
        return "initial_plan_with_current_session"
    if labels.get("mode") == "adaptation" and request.get("current_session") is None and not request.get("session_history"):
        return "adaptation_without_session_data"
    if labels.get("equipment_type") == "gym" and not request.get("user_profile", {}).get("equipment"):
        return "gym_without_equipment"
    if labels.get("expected_policy") == "medical_refusal":
        if expected.get("decision_trace", {}).get("final_action") != "refuse":
            return "medical_policy_without_refuse"
        if not expected.get("final_recommendation", {}).get("refused"):
            return "medical_policy_not_refused"
    if labels.get("expected_policy") == "restriction_limited" and labels.get("restriction_type") == "none":
        return "restriction_policy_without_restriction"
    return None


def process_records(raw_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for raw in raw_records:
        record = deepcopy(raw)
        record["prompt"] = normalize_text(str(record.get("prompt", "")))
        reason = quality_filter_reason(record, seen_hashes)
        record.setdefault("quality", {})
        if reason:
            record["quality"].update(
                {
                    "is_valid_schema": False,
                    "is_duplicate": reason == "duplicate_request",
                    "is_low_quality": reason != "duplicate_request",
                    "filter_reason": reason,
                }
            )
            rejected.append(record)
        else:
            record["quality"].update(
                {
                    "is_valid_schema": True,
                    "is_duplicate": False,
                    "is_low_quality": False,
                    "filter_reason": None,
                }
            )
            accepted.append(record)

    return accepted, rejected


def compare_distributions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    result: Dict[str, Any] = {}
    for dimension, targets in TARGET_DISTRIBUTIONS.items():
        counts = Counter(record["labels"][dimension] for record in records)
        result[dimension] = [distribution_row(dimension, value, target_share, counts.get(value, 0), total) for value, target_share in targets.items()]
    return result


def distribution_row(dimension: str, value: str, target_share: float, count: int, total: int) -> Dict[str, Any]:
    actual_share = count / total if total else 0.0
    deviation_pp = round((actual_share - target_share) * 100, 2)
    allowed_deviation = MAX_ALLOWED_SAFETY_DEVIATION_PP if dimension in {"safety_case", "expected_policy"} and value in {"medical_refusal", "medical_safety_refusal"} else MAX_ALLOWED_DEVIATION_PP
    return {
        "value": value,
        "target_share": round(target_share, 4),
        "actual_share": round(actual_share, 4),
        "target_percent": round(target_share * 100, 2),
        "actual_percent": round(actual_share * 100, 2),
        "count": count,
        "deviation_pp": deviation_pp,
        "within_tolerance": abs(deviation_pp) <= allowed_deviation,
    }


def filter_stats(rejected: List[Dict[str, Any]]) -> Dict[str, int]:
    reasons = Counter(record["quality"]["filter_reason"] for record in rejected)
    return dict(sorted(reasons.items()))


def build_stats(raw_records: List[Dict[str, Any]], accepted: List[Dict[str, Any]], rejected: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "generation_version": GENERATION_VERSION,
        "raw_examples": len(raw_records),
        "accepted_examples": len(accepted),
        "rejected_examples": len(rejected),
        "filtered_share": round(len(rejected) / len(raw_records), 6) if raw_records else 0.0,
        "target_examples": TARGET_EXAMPLES,
        "target_distributions": TARGET_DISTRIBUTIONS,
        "coverage": compare_distributions(accepted),
        "filter_reasons": filter_stats(rejected),
    }
