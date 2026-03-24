import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx


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
                except json.JSONDecodeError as e:
                    raise ValueError(f"Ошибка JSONL в строке {line_no}: {e}") from e
        return cases

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return data["cases"]

    raise ValueError("Ожидался JSON-массив кейсов или объект с ключом 'cases'")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def run_case(client: httpx.Client, url: str, case: dict[str, Any]) -> dict[str, Any]:
    request_payload = case["request"]
    started_at = time.perf_counter()

    try:
        response = client.post(url, json=request_payload)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)

        try:
            response_body = response.json()
        except Exception:
            response_body = response.text

        return {
            "id": case.get("id", ""),
            "scenario": case.get("scenario", ""),
            "title": case.get("title", ""),
            "status": "ok" if response.is_success else "http_error",
            "http_status": response.status_code,
            "elapsed_ms": elapsed_ms,
            "request": compact_json(request_payload),
            "result": compact_json(response_body),
            "request_pretty": pretty_json(request_payload),
            "result_pretty": pretty_json(response_body) if not isinstance(response_body, str) else response_body,
        }
    except httpx.RequestError as e:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return {
            "id": case.get("id", ""),
            "scenario": case.get("scenario", ""),
            "title": case.get("title", ""),
            "status": "request_error",
            "http_status": "",
            "elapsed_ms": elapsed_ms,
            "request": compact_json(request_payload),
            "result": compact_json({"error": str(e)}),
            "request_pretty": pretty_json(request_payload),
            "result_pretty": pretty_json({"error": str(e)}),
        }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "scenario",
        "title",
        "status",
        "http_status",
        "elapsed_ms",
        "request",
        "result",
        "request_pretty",
        "result_pretty",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    errors = total - ok
    avg_ms = round(sum(float(r["elapsed_ms"]) for r in rows) / total, 2) if total else 0.0

    print(f"Всего кейсов: {total}")
    print(f"Успешно: {ok}")
    print(f"С ошибкой: {errors}")
    print(f"Среднее время ответа: {avg_ms} мс")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Прогон проверочного набора через API AI Adaptive Training Coach"
    )
    parser.add_argument(
        "--input",
        default="data/validation_cases.json",
        help="Путь к JSON/JSONL файлу с кейсами",
    )
    parser.add_argument(
        "--output",
        default="results/eval_results.csv",
        help="Путь к итоговому CSV файлу",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Базовый URL API",
    )
    parser.add_argument(
        "--endpoint",
        default="/coach",
        help="Endpoint для прогона",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Таймаут запроса в секундах",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    url = args.base_url.rstrip("/") + "/" + args.endpoint.lstrip("/")

    try:
        cases = load_cases(input_path)
    except Exception as e:
        print(f"Не удалось загрузить набор данных: {e}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout) as client:
        for case in cases:
            rows.append(run_case(client, url, case))

    write_csv(rows, output_path)
    print_summary(rows)
    print(f"CSV сохранён: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
