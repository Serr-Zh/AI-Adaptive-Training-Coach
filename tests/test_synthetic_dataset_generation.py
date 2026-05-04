from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_synthetic_dataset.py"
DATASET = ROOT / "data" / "synthetic_eval_dataset.jsonl"
REJECTED = ROOT / "data" / "synthetic_eval_rejected.jsonl"
STATS = ROOT / "data" / "synthetic_generation_stats.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_synthetic_generation_pipeline_runs() -> None:
    subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True)
    assert DATASET.exists()
    assert REJECTED.exists()
    assert STATS.exists()


def test_dataset_has_at_least_100_valid_examples() -> None:
    records = read_jsonl(DATASET)
    assert len(records) >= 100
    assert all(record["quality"]["is_valid_schema"] for record in records)
    assert all(not record["quality"]["is_duplicate"] for record in records)
    assert all(not record["quality"]["is_low_quality"] for record in records)


def test_medical_refusal_consistency() -> None:
    records = read_jsonl(DATASET)
    medical = [record for record in records if record["labels"]["expected_policy"] == "medical_refusal"]
    assert medical
    for record in medical:
        expected = record["expected_sgr"]
        assert expected["decision_trace"]["final_action"] == "refuse"
        assert expected["final_recommendation"]["refused"] is True
        assert expected["final_recommendation"]["refuse_reason"]
        assert expected["final_recommendation"]["exercise_changes"] == []


def test_rejected_records_are_reported() -> None:
    rejected = read_jsonl(REJECTED)
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    assert len(rejected) == stats["rejected_examples"]
    assert stats["filtered_share"] > 0
    assert {record["quality"]["filter_reason"] for record in rejected}


def test_coverage_within_tolerance() -> None:
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    for rows in stats["coverage"].values():
        for row in rows:
            assert row["within_tolerance"], row
