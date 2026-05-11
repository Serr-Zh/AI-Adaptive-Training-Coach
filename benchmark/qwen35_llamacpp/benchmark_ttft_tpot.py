import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request, error


def load_prompts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Файл prompts должен содержать JSON-массив.")

    result = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Каждый prompt должен быть объектом.")
        if "id" not in item or "prompt" not in item:
            raise ValueError("Каждый prompt должен содержать поля id и prompt.")
        result.append({"id": str(item["id"]), "prompt": str(item["prompt"])})

    return result


def post_json_stream(url: str, payload: Dict[str, Any], timeout: int = 300):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    return request.urlopen(req, timeout=timeout)


def parse_sse_line(raw_line: bytes) -> Optional[Dict[str, Any]]:
    line = raw_line.decode("utf-8", errors="replace").strip()

    if not line:
        return None

    if not line.startswith("data:"):
        return None

    data = line[len("data:"):].strip()

    if data == "[DONE]":
        return {"done": True}

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def extract_delta_text(event: Dict[str, Any]) -> str:
    choices = event.get("choices")
    if not choices:
        return ""

    choice = choices[0]
    delta = choice.get("delta", {})

    parts = []

    content = delta.get("content")
    if isinstance(content, str) and content:
        parts.append(content)

    reasoning_content = delta.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        parts.append(reasoning_content)

    return "".join(parts)


def extract_usage_completion_tokens(event: Dict[str, Any]) -> Optional[int]:
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return None

    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, int):
        return completion_tokens

    return None


def run_one_request(
    server_url: str,
    model_name: str,
    prompt_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    config_name: str,
    timeout: int,
) -> Dict[str, Any]:
    url = server_url.rstrip("/") + "/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    }

    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None

    chunk_count = 0
    generated_text_parts: List[str] = []
    completion_tokens_from_usage: Optional[int] = None

    try:
        with post_json_stream(url, payload, timeout=timeout) as resp:
            for raw_line in resp:
                now = time.perf_counter()
                event = parse_sse_line(raw_line)

                if event is None:
                    continue

                if event.get("done") is True:
                    break

                usage_tokens = extract_usage_completion_tokens(event)
                if usage_tokens is not None:
                    completion_tokens_from_usage = usage_tokens

                text_piece = extract_delta_text(event)
                if not text_piece:
                    continue

                if first_token_time is None:
                    first_token_time = now

                last_token_time = now
                chunk_count += 1
                generated_text_parts.append(text_piece)

    except error.URLError as exc:
        raise RuntimeError(
            f"Не удалось подключиться к серверу {url}. "
            f"Проверь, что llama-server запущен на 127.0.0.1:8080. Ошибка: {exc}"
        ) from exc

    end_time = time.perf_counter()

    if first_token_time is None:
        ttft_s = None
        tpot_s = None
    else:
        ttft_s = first_token_time - start_time

        if completion_tokens_from_usage is not None and completion_tokens_from_usage > 1:
            output_tokens = completion_tokens_from_usage
            token_count_method = "usage.completion_tokens"
        else:
            output_tokens = chunk_count
            token_count_method = "stream_chunk_count"

        if output_tokens > 1 and last_token_time is not None:
            tpot_s = (last_token_time - first_token_time) / (output_tokens - 1)
        else:
            tpot_s = None

    generated_text = "".join(generated_text_parts)

    if completion_tokens_from_usage is not None:
        output_tokens_final = completion_tokens_from_usage
        token_count_method_final = "usage.completion_tokens"
    else:
        output_tokens_final = chunk_count
        token_count_method_final = "stream_chunk_count"

    return {
        "config_name": config_name,
        "prompt_id": prompt_id,
        "prompt_chars": len(prompt),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ttft_s": ttft_s,
        "tpot_s": tpot_s,
        "total_time_s": end_time - start_time,
        "output_tokens": output_tokens_final,
        "stream_chunks": chunk_count,
        "token_count_method": token_count_method_final,
        "generated_text_preview": generated_text[:300].replace("\n", "\\n"),
    }


def mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.mean(clean)


def median(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.median(clean)


def percentile(values: List[Optional[float]], p: float) -> Optional[float]:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None

    if len(clean) == 1:
        return clean[0]

    k = (len(clean) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(clean) - 1)

    if f == c:
        return clean[f]

    return clean[f] + (clean[c] - clean[f]) * (k - f)


def fmt_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "config_name",
        "run_index",
        "prompt_id",
        "prompt_chars",
        "max_tokens",
        "temperature",
        "ttft_s",
        "tpot_s",
        "total_time_s",
        "output_tokens",
        "stream_chunks",
        "token_count_method",
        "generated_text_preview",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            clean_row = dict(row)
            clean_row["ttft_s"] = fmt_float(clean_row.get("ttft_s"))
            clean_row["tpot_s"] = fmt_float(clean_row.get("tpot_s"))
            clean_row["total_time_s"] = fmt_float(clean_row.get("total_time_s"))
            writer.writerow(clean_row)


def write_summary_csv(path: Path, config_name: str, rows: List[Dict[str, Any]]) -> None:
    ttft_values = [row.get("ttft_s") for row in rows]
    tpot_values = [row.get("tpot_s") for row in rows]
    total_values = [row.get("total_time_s") for row in rows]

    summary = {
        "config_name": config_name,
        "requests": len(rows),
        "mean_ttft_s": mean(ttft_values),
        "median_ttft_s": median(ttft_values),
        "p95_ttft_s": percentile(ttft_values, 95),
        "mean_tpot_s": mean(tpot_values),
        "median_tpot_s": median(tpot_values),
        "p95_tpot_s": percentile(tpot_values, 95),
        "mean_total_time_s": mean(total_values),
        "median_total_time_s": median(total_values),
    }

    fieldnames = list(summary.keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        clean_summary = {}
        for key, value in summary.items():
            if isinstance(value, float):
                clean_summary[key] = fmt_float(value)
            else:
                clean_summary[key] = value

        writer.writerow(clean_summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TTFT and TPOT for llama.cpp OpenAI-compatible server."
    )

    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8080",
        help="Base URL of llama-server.",
    )
    parser.add_argument(
        "--model",
        default="qwen35",
        help="Model name for OpenAI-compatible request.",
    )
    parser.add_argument(
        "--prompts",
        default="prompts_qwen35.json",
        help="Path to JSON file with prompts.",
    )
    parser.add_argument(
        "--config-name",
        default="baseline",
        help="Name of current benchmark configuration.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per prompt.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum generated tokens per request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--output",
        default="results_baseline.csv",
        help="Output CSV file for raw measurements.",
    )
    parser.add_argument(
        "--summary-output",
        default="summary_baseline.csv",
        help="Output CSV file for aggregated measurements.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP request timeout in seconds.",
    )

    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompts))
    rows: List[Dict[str, Any]] = []

    print(f"Server URL: {args.server_url}")
    print(f"Config: {args.config_name}")
    print(f"Prompts: {len(prompts)}")
    print(f"Runs per prompt: {args.runs}")
    print()

    for run_index in range(1, args.runs + 1):
        for prompt_item in prompts:
            prompt_id = prompt_item["id"]
            prompt = prompt_item["prompt"]

            print(f"Running {args.config_name}: run={run_index}, prompt={prompt_id}")

            row = run_one_request(
                server_url=args.server_url,
                model_name=args.model,
                prompt_id=prompt_id,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                config_name=args.config_name,
                timeout=args.timeout,
            )

            row["run_index"] = run_index
            rows.append(row)

            ttft = row["ttft_s"]
            tpot = row["tpot_s"]
            total = row["total_time_s"]
            output_tokens = row["output_tokens"]

            print(
                f"  TTFT={fmt_float(ttft, 4)} s, "
                f"TPOT={fmt_float(tpot, 4)} s/token, "
                f"total={fmt_float(total, 4)} s, "
                f"output_tokens={output_tokens}"
            )

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    write_csv(output_path, rows)
    write_summary_csv(summary_path, args.config_name, rows)

    print()
    print(f"Raw results saved to: {output_path.resolve()}")
    print(f"Summary saved to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()