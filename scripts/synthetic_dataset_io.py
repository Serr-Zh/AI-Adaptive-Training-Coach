from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def flatten_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    labels = record["labels"]
    expected = record["expected_sgr"]
    request = record["request"]
    current_session = request.get("current_session")
    return {
        "id": record["id"],
        "mode": labels["mode"],
        "feature": labels["feature"],
        "goal": labels["goal"],
        "experience_level": labels["experience_level"],
        "equipment_type": labels["equipment_type"],
        "restriction_type": labels["restriction_type"],
        "session_signal": labels["session_signal"],
        "safety_case": labels["safety_case"],
        "expected_policy": labels["expected_policy"],
        "language": labels["language"],
        "final_action": expected["decision_trace"]["final_action"],
        "refused": expected["final_recommendation"]["refused"],
        "exercise_changes_count": len(expected["final_recommendation"]["exercise_changes"]),
        "has_history": bool(request.get("session_history")),
        "has_current_session": current_session is not None,
        "prompt": record["prompt"],
    }


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_for_csv(record) for record in records]
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
