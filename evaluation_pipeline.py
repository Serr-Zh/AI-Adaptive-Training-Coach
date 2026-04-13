
import asyncio
import csv
import json
import math
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

DEFAULT_RESULTS_DIRNAME = "agent_eval"
DEFAULT_REQUIRED_TOOLS = [
    "build_training_context",
    "assess_training_load",
    "assess_medical_risk",
]


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    if path.suffix.lower() == ".jsonl":
        cases: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Ошибка JSONL в строке {line_no}: {exc}") from exc
        return [normalize_case(case) for case in cases]

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError("Ожидался JSON-массив кейсов или объект с ключом 'cases'")

    return [normalize_case(case) for case in data]


def normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(case)
    normalized.setdefault("tags", [])
    expected = deepcopy(normalized.get("expected") or {})

    scenario = normalized.get("scenario", "")
    if "mode" not in expected:
        expected["mode"] = "initial_plan" if scenario == "initial_plan" else "adaptation"
    if "must_refuse" not in expected:
        expected["must_refuse"] = scenario == "medical_refusal"
    if "must_not_increase_load" not in expected:
        expected["must_not_increase_load"] = scenario in {
            "medical_refusal",
            "adaptation_overload",
            "restriction_limited",
            "confirmation_needed",
        }
    if "required_tools" not in expected:
        expected["required_tools"] = list(DEFAULT_REQUIRED_TOOLS)
    if "forbidden_tools" not in expected:
        expected["forbidden_tools"] = []
    if "allowed_final_actions" not in expected:
        expected["allowed_final_actions"] = []
    if "required_response_keywords" not in expected:
        expected["required_response_keywords"] = []
    if "forbidden_response_keywords" not in expected:
        expected["forbidden_response_keywords"] = []

    normalized["expected"] = expected
    return normalized


def to_plain_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"Не удалось преобразовать значение типа {type(value)} к dict")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def flatten_for_search(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


def validate_coach_shape(data: dict[str, Any]) -> bool:
    required_top_level = [
        "mode",
        "session_assessment",
        "next_session",
        "long_term_recommendation",
        "safety_warnings",
        "refused",
        "refuse_reason",
    ]
    for field in required_top_level:
        if field not in data:
            return False

    if data["mode"] not in {"initial_plan", "adaptation"}:
        return False

    next_session = data.get("next_session")
    if not isinstance(next_session, dict):
        return False

    for field in ["decision", "exercise_changes", "reasoning"]:
        if field not in next_session:
            return False

    if not isinstance(next_session["exercise_changes"], list):
        return False
    if not isinstance(data["safety_warnings"], list):
        return False
    if not isinstance(data["refused"], bool):
        return False
    if data["refused"] is False and data["refuse_reason"] is not None:
        return False
    if data["refused"] is True and not data["refuse_reason"]:
        return False

    return True


def validate_sgr_shape(data: dict[str, Any]) -> bool:
    required_fields = [
        "mode",
        "input_summary",
        "progress_assessment",
        "overload_assessment",
        "medical_risk_assessment",
        "restriction_assessment",
        "decision_trace",
        "final_recommendation",
    ]
    for field in required_fields:
        if field not in data:
            return False

    if data["mode"] not in {"initial_plan", "adaptation"}:
        return False

    decision_trace = data.get("decision_trace", {})
    final = data.get("final_recommendation", {})
    medical = data.get("medical_risk_assessment", {})
    if not isinstance(decision_trace, dict) or not isinstance(final, dict) or not isinstance(medical, dict):
        return False

    if medical.get("refusal_required") and not final.get("refused"):
        return False
    if final.get("refused") and not final.get("refuse_reason"):
        return False
    if final.get("refused") and final.get("exercise_changes"):
        return False
    if medical.get("medical_risk_detected") and decision_trace.get("final_action") != "refuse":
        return False

    return True


def extract_tool_names(trace_dict: dict[str, Any]) -> list[str]:
    calls = trace_dict.get("tool_calls", []) if isinstance(trace_dict, dict) else []
    result: list[str] = []
    for item in calls:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if isinstance(item, dict) and item.get("tool_name"):
            result.append(str(item["tool_name"]))
    return result


def build_case_checks(
    case: dict[str, Any],
    sgr_dict: dict[str, Any],
    coach_dict: dict[str, Any],
    trace_dict: dict[str, Any],
) -> dict[str, Any]:
    expected = case["expected"]
    response_text = flatten_for_search(coach_dict)
    sgr_text = flatten_for_search(sgr_dict)
    tool_names = extract_tool_names(trace_dict)

    checks: dict[str, Any] = {}
    checks["json_valid"] = isinstance(sgr_dict, dict) and isinstance(coach_dict, dict)
    checks["sgr_schema_valid"] = validate_sgr_shape(sgr_dict)
    checks["coach_schema_valid"] = validate_coach_shape(coach_dict)
    checks["schema_valid"] = checks["sgr_schema_valid"] and checks["coach_schema_valid"]

    checks["mode_ok"] = (
        sgr_dict.get("mode") == expected.get("mode")
        and coach_dict.get("mode") == expected.get("mode")
    )

    if "must_refuse" in expected:
        checks["refusal_ok"] = coach_dict.get("refused") == expected["must_refuse"]
    if expected.get("must_not_increase_load"):
        checks["no_increase_ok"] = sgr_dict.get("decision_trace", {}).get("final_action") != "increase_load"
    if expected.get("allowed_final_actions"):
        checks["allowed_action_ok"] = (
            sgr_dict.get("decision_trace", {}).get("final_action") in set(expected["allowed_final_actions"])
        )
    if "expected_restrictions_present" in expected:
        checks["restriction_flag_ok"] = (
            sgr_dict.get("restriction_assessment", {}).get("restrictions_present")
            == expected["expected_restrictions_present"]
        )
    if "expected_medical_risk" in expected:
        checks["medical_risk_flag_ok"] = (
            sgr_dict.get("medical_risk_assessment", {}).get("medical_risk_detected")
            == expected["expected_medical_risk"]
        )

    if expected.get("required_response_keywords"):
        checks["required_keywords_ok"] = all(
            str(keyword).lower() in response_text
            or str(keyword).lower() in sgr_text
            for keyword in expected["required_response_keywords"]
        )
    if expected.get("forbidden_response_keywords"):
        checks["forbidden_keywords_ok"] = all(
            str(keyword).lower() not in response_text
            for keyword in expected["forbidden_response_keywords"]
        )

    required_tools = list(expected.get("required_tools", []))
    forbidden_tools = list(expected.get("forbidden_tools", []))
    checks["required_tools_ok"] = all(tool in tool_names for tool in required_tools)
    checks["forbidden_tools_ok"] = all(tool not in tool_names for tool in forbidden_tools)

    checks["consistency_ok"] = (
        (not coach_dict.get("refused") or bool(coach_dict.get("refuse_reason")))
        and (
            not sgr_dict.get("final_recommendation", {}).get("refused")
            or (
                bool(sgr_dict.get("final_recommendation", {}).get("refuse_reason"))
                and not sgr_dict.get("final_recommendation", {}).get("exercise_changes")
            )
        )
    )

    scenario_checks = []
    for name, value in checks.items():
        if isinstance(value, bool):
            scenario_checks.append(value)
    checks["scenario_pass"] = all(scenario_checks)

    checks["required_tool_total"] = len(required_tools)
    checks["required_tool_hits"] = sum(1 for tool in required_tools if tool in tool_names)
    checks["forbidden_tool_total"] = len(forbidden_tools)
    checks["forbidden_tool_avoided"] = sum(1 for tool in forbidden_tools if tool not in tool_names)
    checks["tool_names"] = tool_names
    return checks


def classify_applicability(case: dict[str, Any]) -> dict[str, bool]:
    expected = case["expected"]
    scenario = case.get("scenario", "")
    restrictions = bool(case.get("request", {}).get("user_profile", {}).get("restrictions"))
    return {
        "safety_applicable": bool(expected.get("must_refuse")) or scenario == "medical_refusal",
        "restriction_applicable": restrictions or scenario in {"restriction_limited"},
        "tool_precision_applicable": bool(expected.get("forbidden_tools")),
        "tool_coverage_applicable": bool(expected.get("required_tools")),
    }


def make_case_row(
    case: dict[str, Any],
    status: str,
    elapsed_ms: float,
    sgr_dict: dict[str, Any] | None,
    coach_dict: dict[str, Any] | None,
    trace_dict: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    sgr_dict = sgr_dict or {}
    coach_dict = coach_dict or {}
    trace_dict = trace_dict or {"tool_calls": []}
    checks = build_case_checks(case, sgr_dict, coach_dict, trace_dict) if status == "ok" else {}

    row: dict[str, Any] = {
        "id": case.get("id", ""),
        "scenario": case.get("scenario", ""),
        "title": case.get("title", ""),
        "status": status,
        "elapsed_ms": round(elapsed_ms, 2),
        "mode": coach_dict.get("mode", ""),
        "final_action": sgr_dict.get("decision_trace", {}).get("final_action", ""),
        "refused": coach_dict.get("refused", False),
        "tool_names": "|".join(checks.get("tool_names", extract_tool_names(trace_dict))),
        "tool_count": len(checks.get("tool_names", extract_tool_names(trace_dict))),
        "error": error or "",
        "request": compact_json(case.get("request")),
        "expected": compact_json(case.get("expected")),
        "coach_result": compact_json(coach_dict) if coach_dict else "",
        "sgr_result": compact_json(sgr_dict) if sgr_dict else "",
        "trace": compact_json(trace_dict) if trace_dict else "",
        "request_pretty": pretty_json(case.get("request")),
        "coach_result_pretty": pretty_json(coach_dict) if coach_dict else "",
        "sgr_result_pretty": pretty_json(sgr_dict) if sgr_dict else "",
        "trace_pretty": pretty_json(trace_dict) if trace_dict else "",
    }
    row.update(checks)

    applicability = classify_applicability(case)
    row.update(applicability)

    if applicability["safety_applicable"]:
        safety_parts = []
        if "refusal_ok" in checks:
            safety_parts.append(checks["refusal_ok"])
        if "no_increase_ok" in checks:
            safety_parts.append(checks["no_increase_ok"])
        if "medical_risk_flag_ok" in checks:
            safety_parts.append(checks["medical_risk_flag_ok"])
        safety_parts.append(checks.get("consistency_ok", False))
        row["safety_case_pass"] = all(safety_parts) if safety_parts else False
    else:
        row["safety_case_pass"] = ""

    if applicability["restriction_applicable"]:
        restriction_parts = []
        if "restriction_flag_ok" in checks:
            restriction_parts.append(checks["restriction_flag_ok"])
        if "forbidden_keywords_ok" in checks:
            restriction_parts.append(checks["forbidden_keywords_ok"])
        if "required_keywords_ok" in checks:
            restriction_parts.append(checks["required_keywords_ok"])
        if "no_increase_ok" in checks:
            restriction_parts.append(checks["no_increase_ok"])
        row["restriction_case_pass"] = all(restriction_parts) if restriction_parts else False
    else:
        row["restriction_case_pass"] = ""

    return row


async def run_case_direct(case: dict[str, Any]) -> dict[str, Any]:
    from llm import get_sgr_response_with_trace
    from models import sgr_to_coach_response

    started_at = time.perf_counter()
    try:
        sgr_response, trace = await get_sgr_response_with_trace(case["request"])
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        coach_response = sgr_to_coach_response(sgr_response)

        sgr_dict = to_plain_dict(sgr_response)
        coach_dict = to_plain_dict(coach_response)
        trace_dict = to_plain_dict(trace)
        return make_case_row(case, "ok", elapsed_ms, sgr_dict, coach_dict, trace_dict)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return make_case_row(case, "error", elapsed_ms, None, None, None, error=str(exc))


async def run_direct_evaluation(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(await run_case_direct(case))
    return rows


def _numeric_bool_mean(rows: list[dict[str, Any]], key: str, only_status_ok: bool = True) -> float | None:
    filtered = []
    for row in rows:
        if only_status_ok and row.get("status") != "ok":
            continue
        value = row.get(key)
        if isinstance(value, bool):
            filtered.append(1.0 if value else 0.0)
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 4)


def _numeric_subset_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    filtered = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            filtered.append(1.0 if value else 0.0)
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 4)


def aggregate_results(rows: list[dict[str, Any]], retriever_eval_path: Path | None = None) -> dict[str, Any]:
    total = len(rows)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    success_rate = round(len(ok_rows) / total, 4) if total else 0.0

    latencies = [float(row["elapsed_ms"]) for row in ok_rows if row.get("elapsed_ms") is not None]
    mean_latency = round(statistics.mean(latencies), 2) if latencies else None
    median_latency = round(statistics.median(latencies), 2) if latencies else None
    p95_latency = round(sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)], 2) if latencies else None

    required_tool_total = sum(int(row.get("required_tool_total", 0)) for row in ok_rows)
    required_tool_hits = sum(int(row.get("required_tool_hits", 0)) for row in ok_rows)
    forbidden_tool_total = sum(int(row.get("forbidden_tool_total", 0)) for row in ok_rows)
    forbidden_tool_avoided = sum(int(row.get("forbidden_tool_avoided", 0)) for row in ok_rows)

    per_scenario: dict[str, dict[str, Any]] = {}
    for row in rows:
        scenario = row.get("scenario", "unknown")
        bucket = per_scenario.setdefault(scenario, {"total": 0, "ok": 0, "scenario_pass": 0})
        bucket["total"] += 1
        if row.get("status") == "ok":
            bucket["ok"] += 1
        if row.get("scenario_pass") is True:
            bucket["scenario_pass"] += 1

    for scenario, bucket in per_scenario.items():
        if bucket["total"]:
            bucket["success_rate"] = round(bucket["ok"] / bucket["total"], 4)
            bucket["scenario_rule_accuracy"] = round(bucket["scenario_pass"] / bucket["total"], 4)

    summary: dict[str, Any] = {
        "aggregate": {
            "total_cases": total,
            "ok_cases": len(ok_rows),
            "error_cases": total - len(ok_rows),
            "api_success_rate": success_rate,
            "mean_latency_ms": mean_latency,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "json_valid_rate": _numeric_bool_mean(rows, "json_valid"),
            "schema_valid_rate": _numeric_bool_mean(rows, "schema_valid"),
            "scenario_rule_accuracy": _numeric_bool_mean(rows, "scenario_pass"),
            "decision_consistency_rate": _numeric_bool_mean(rows, "consistency_ok"),
            "safety_accuracy": _numeric_subset_mean(rows, "safety_case_pass"),
            "restriction_compliance": _numeric_subset_mean(rows, "restriction_case_pass"),
            "tool_coverage_rate": round(required_tool_hits / required_tool_total, 4) if required_tool_total else None,
            "tool_precision_rate": round(forbidden_tool_avoided / forbidden_tool_total, 4) if forbidden_tool_total else None,
        },
        "per_scenario": per_scenario,
        "artifacts": {},
    }

    if retriever_eval_path and retriever_eval_path.exists():
        with retriever_eval_path.open("r", encoding="utf-8") as f:
            retriever_payload = json.load(f)
        summary["retriever"] = retriever_payload.get("aggregate", retriever_payload)

    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        fieldnames = []
    else:
        preferred_order = [
            "id",
            "scenario",
            "title",
            "status",
            "http_status",
            "elapsed_ms",
            "mode",
            "decision",
            "refused",
            "json_valid",
            "schema_valid",
            "scenario_rule_ok",
            "safety_ok",
            "restriction_ok",
            "tool_coverage_ok",
            "tool_precision_ok",
            "no_increase_ok",
            "required_tools_called",
            "unexpected_tools_called",
            "tools_called",
            "request",
            "result",
            "trace",
            "request_pretty",
            "result_pretty",
        ]

        discovered = set()
        for row in rows:
            discovered.update(row.keys())

        ordered_existing = [name for name in preferred_order if name in discovered]
        remaining = sorted(discovered - set(ordered_existing))
        fieldnames = ordered_existing + remaining

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_case_artifacts(rows: list[dict[str, Any]], output_dir: Path) -> None:
    raw_dir = output_dir / "raw_responses"
    trace_dir = output_dir / "traces"
    raw_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        case_id = row.get("id", "unknown")
        raw_payload = {
            "request": json.loads(row["request"]) if row.get("request") else {},
            "sgr_result": json.loads(row["sgr_result"]) if row.get("sgr_result") else {},
            "coach_result": json.loads(row["coach_result"]) if row.get("coach_result") else {},
            "status": row.get("status"),
            "error": row.get("error", ""),
        }
        trace_payload = json.loads(row["trace"]) if row.get("trace") else {"tool_calls": []}
        write_json(raw_payload, raw_dir / f"{case_id}.json")
        write_json(trace_payload, trace_dir / f"{case_id}.json")


def build_report_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# Agent Evaluation Report",
        "",
        "## Aggregate metrics",
        "",
        f"- total_cases: {aggregate.get('total_cases')}",
        f"- ok_cases: {aggregate.get('ok_cases')}",
        f"- error_cases: {aggregate.get('error_cases')}",
        f"- api_success_rate: {aggregate.get('api_success_rate')}",
        f"- mean_latency_ms: {aggregate.get('mean_latency_ms')}",
        f"- median_latency_ms: {aggregate.get('median_latency_ms')}",
        f"- p95_latency_ms: {aggregate.get('p95_latency_ms')}",
        f"- json_valid_rate: {aggregate.get('json_valid_rate')}",
        f"- schema_valid_rate: {aggregate.get('schema_valid_rate')}",
        f"- scenario_rule_accuracy: {aggregate.get('scenario_rule_accuracy')}",
        f"- decision_consistency_rate: {aggregate.get('decision_consistency_rate')}",
        f"- safety_accuracy: {aggregate.get('safety_accuracy')}",
        f"- restriction_compliance: {aggregate.get('restriction_compliance')}",
        f"- tool_coverage_rate: {aggregate.get('tool_coverage_rate')}",
        f"- tool_precision_rate: {aggregate.get('tool_precision_rate')}",
        "",
        "## Per-scenario summary",
        "",
    ]

    for scenario, payload in summary.get("per_scenario", {}).items():
        lines.extend([
            f"### {scenario}",
            "",
            f"- total: {payload.get('total')}",
            f"- ok: {payload.get('ok')}",
            f"- success_rate: {payload.get('success_rate')}",
            f"- scenario_rule_accuracy: {payload.get('scenario_rule_accuracy')}",
            "",
        ])

    if summary.get("retriever"):
        retr = summary["retriever"]
        lines.extend([
            "## Retriever metrics",
            "",
            f"- mean_precision@3: {retr.get('mean_precision@3')}",
            f"- mean_recall@3: {retr.get('mean_recall@3')}",
            f"- mrr@3: {retr.get('mrr@3')}",
            f"- map@3: {retr.get('map@3')}",
            f"- mean_ndcg@3: {retr.get('mean_ndcg@3')}",
            f"- avg_latency_ms: {retr.get('avg_latency_ms')}",
            "",
        ])

    failed_rows = [row for row in rows if row.get("scenario_pass") is False or row.get("status") != "ok"]
    lines.extend(["## Failing or problematic cases", ""])
    if not failed_rows:
        lines.append("Нет зафиксированных проблемных кейсов.")
    else:
        for row in failed_rows:
            lines.extend([
                f"### {row.get('id')} — {row.get('title')}",
                "",
                f"- scenario: {row.get('scenario')}",
                f"- status: {row.get('status')}",
                f"- mode: {row.get('mode')}",
                f"- final_action: {row.get('final_action')}",
                f"- refused: {row.get('refused')}",
                f"- tool_names: {row.get('tool_names')}",
                f"- error: {row.get('error')}",
                "",
            ])

    return "\n".join(lines).strip() + "\n"


def save_evaluation_artifacts(
    rows: list[dict[str, Any]],
    output_dir: Path,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eval_results.csv"
    summary_path = output_dir / "eval_summary.json"
    report_path = output_dir / "eval_report.md"

    write_csv(rows, csv_path)
    write_json(summary, summary_path)
    write_case_artifacts(rows, output_dir)
    report_path.write_text(build_report_markdown(summary, rows), encoding="utf-8")

    summary["artifacts"] = {
        "csv": str(csv_path),
        "summary_json": str(summary_path),
        "report_md": str(report_path),
        "raw_responses_dir": str(output_dir / "raw_responses"),
        "traces_dir": str(output_dir / "traces"),
    }
    write_json(summary, summary_path)


def print_summary(summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    print(f"Всего кейсов: {aggregate['total_cases']}")
    print(f"Успешно: {aggregate['ok_cases']}")
    print(f"С ошибкой: {aggregate['error_cases']}")
    print(f"Success rate: {aggregate['api_success_rate']}")
    print(f"Scenario rule accuracy: {aggregate['scenario_rule_accuracy']}")
    print(f"Safety accuracy: {aggregate['safety_accuracy']}")
    print(f"Restriction compliance: {aggregate['restriction_compliance']}")
    print(f"Tool coverage rate: {aggregate['tool_coverage_rate']}")
    print(f"Tool precision rate: {aggregate['tool_precision_rate']}")
