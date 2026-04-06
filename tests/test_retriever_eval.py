import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from scripts.evaluate_retriever import evaluate


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[OK] {message}")


def main() -> None:
    result = evaluate(top_k=3)
    aggregate = result["aggregate"]

    assert_true(aggregate["query_count"] >= 20, "Оценка запускается на наборе >= 20 запросов")
    assert_true(0.0 <= aggregate["mean_precision@3"] <= 1.0, "mean_precision@3 лежит в [0,1]")
    assert_true(0.0 <= aggregate["mean_recall@3"] <= 1.0, "mean_recall@3 лежит в [0,1]")
    assert_true(0.0 <= aggregate["mrr@3"] <= 1.0, "mrr@3 лежит в [0,1]")
    assert_true(0.0 <= aggregate["map@3"] <= 1.0, "map@3 лежит в [0,1]")
    assert_true(0.0 <= aggregate["mean_ndcg@3"] <= 1.0, "mean_ndcg@3 лежит в [0,1]")
    assert_true(aggregate["avg_latency_ms"] >= 0.0, "Средняя latency неотрицательна")

    print("\nАгрегированные результаты:")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
