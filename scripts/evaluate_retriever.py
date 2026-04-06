import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from retriever import retrieve_documents  # noqa: E402

BENCHMARK_DIR = ROOT / "benchmark"
QUERIES_PATH = BENCHMARK_DIR / "queries.jsonl"
QRELS_PATH = BENCHMARK_DIR / "qrels" / "test.tsv"
RUNS_DIR = BENCHMARK_DIR / "runs"
RESULTS_DIR = ROOT / "results"
RUN_PATH = RUNS_DIR / "current_lexical.tsv"
RESULTS_PATH = RESULTS_DIR / "retriever_eval.json"


def load_queries(path: Path) -> list[dict]:
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query_id"]
            did = row["doc_id"]
            rel = int(row["relevance"])
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    tp = sum(1 for did in top if did in relevant_ids)
    return tp / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    tp = sum(1 for did in top if did in relevant_ids)
    return tp / len(relevant_ids)


def reciprocal_rank_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, did in enumerate(retrieved_ids[:k], start=1):
        if did in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, did in enumerate(retrieved_ids[:k], start=1):
        if did in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevance_map: dict[str, int], k: int) -> float:
    dcg = 0.0
    for rank, did in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_map.get(did, 0)
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    ideal_rels = sorted((rel for rel in relevance_map.values() if rel > 0), reverse=True)[:k]
    if not ideal_rels:
        return 0.0

    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        idcg += rel / math.log2(rank + 1)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate(top_k: int = 3) -> dict:
    queries = load_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_query = []
    run_rows = []
    latencies_ms = []

    for query in queries:
        qid = query["query_id"]
        text = query["text"]
        relevant_map = qrels.get(qid, {})
        relevant_ids = {did for did, rel in relevant_map.items() if rel > 0}

        started = time.perf_counter()
        docs = retrieve_documents(text, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)

        retrieved_ids = [doc["id"] for doc in docs]

        for rank, doc in enumerate(docs, start=1):
            run_rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc["id"],
                    "score": doc.get("score", 0.0),
                    "rank": rank,
                }
            )

        metrics = {
            "query_id": qid,
            "query": text,
            "relevant_doc_ids": sorted(relevant_ids),
            "retrieved_doc_ids": retrieved_ids,
            f"precision@{top_k}": precision_at_k(retrieved_ids, relevant_ids, top_k),
            f"recall@{top_k}": recall_at_k(retrieved_ids, relevant_ids, top_k),
            f"mrr@{top_k}": reciprocal_rank_at_k(retrieved_ids, relevant_ids, top_k),
            f"map@{top_k}": average_precision_at_k(retrieved_ids, relevant_ids, top_k),
            f"ndcg@{top_k}": ndcg_at_k(retrieved_ids, relevant_map, top_k),
            "latency_ms": round(elapsed_ms, 3),
        }
        per_query.append(metrics)

    with RUN_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "doc_id", "score", "rank"], delimiter="\t")
        writer.writeheader()
        writer.writerows(run_rows)

    aggregate = {
        "top_k": top_k,
        "query_count": len(per_query),
        f"mean_precision@{top_k}": round(statistics.mean(item[f"precision@{top_k}"] for item in per_query), 4),
        f"mean_recall@{top_k}": round(statistics.mean(item[f"recall@{top_k}"] for item in per_query), 4),
        f"mrr@{top_k}": round(statistics.mean(item[f"mrr@{top_k}"] for item in per_query), 4),
        f"map@{top_k}": round(statistics.mean(item[f"map@{top_k}"] for item in per_query), 4),
        f"mean_ndcg@{top_k}": round(statistics.mean(item[f"ndcg@{top_k}"] for item in per_query), 4),
        "avg_latency_ms": round(statistics.mean(latencies_ms), 3),
        "p95_latency_ms": round(sorted(latencies_ms)[max(0, math.ceil(0.95 * len(latencies_ms)) - 1)], 3),
        "retriever": "lexical_overlap_v1",
    }

    payload = {
        "aggregate": aggregate,
        "per_query": per_query,
        "artifacts": {
            "queries": str(QUERIES_PATH.relative_to(ROOT)),
            "qrels": str(QRELS_PATH.relative_to(ROOT)),
            "run": str(RUN_PATH.relative_to(ROOT)),
        },
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


if __name__ == "__main__":
    result = evaluate(top_k=3)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
