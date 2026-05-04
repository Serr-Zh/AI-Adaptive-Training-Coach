from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from synthetic_dataset_builders import build_invalid_records, build_valid_records
from synthetic_dataset_io import write_csv, write_json, write_jsonl
from synthetic_dataset_quality import build_stats, process_records

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation dataset for AI Adaptive Training Coach.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory for generated dataset files.")
    parser.add_argument("--jsonl-name", default="synthetic_eval_dataset.jsonl", help="Accepted dataset JSONL file name.")
    parser.add_argument("--csv-name", default="synthetic_eval_dataset.csv", help="Accepted dataset CSV file name.")
    parser.add_argument("--rejected-name", default="synthetic_eval_rejected.jsonl", help="Rejected records JSONL file name.")
    parser.add_argument("--stats-name", default="synthetic_generation_stats.json", help="Generation statistics JSON file name.")
    return parser.parse_args()


def generate_outputs(data_dir: Path, jsonl_name: str, csv_name: str, rejected_name: str, stats_name: str) -> Dict[str, Any]:
    valid_records = build_valid_records()
    raw_records = valid_records + build_invalid_records(valid_records)
    accepted, rejected = process_records(raw_records)
    stats = build_stats(raw_records, accepted, rejected)

    dataset_jsonl = data_dir / jsonl_name
    dataset_csv = data_dir / csv_name
    rejected_jsonl = data_dir / rejected_name
    stats_json = data_dir / stats_name

    write_jsonl(dataset_jsonl, accepted)
    write_csv(dataset_csv, accepted)
    write_jsonl(rejected_jsonl, rejected)
    write_json(stats_json, stats)

    return {
        "raw_examples": stats["raw_examples"],
        "accepted_examples": stats["accepted_examples"],
        "rejected_examples": stats["rejected_examples"],
        "filtered_share": stats["filtered_share"],
        "outputs": {
            "dataset_jsonl": str(dataset_jsonl),
            "dataset_csv": str(dataset_csv),
            "rejected_jsonl": str(rejected_jsonl),
            "stats_json": str(stats_json),
        },
    }


def main() -> None:
    args = parse_args()
    result = generate_outputs(
        data_dir=args.data_dir,
        jsonl_name=args.jsonl_name,
        csv_name=args.csv_name,
        rejected_name=args.rejected_name,
        stats_name=args.stats_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
