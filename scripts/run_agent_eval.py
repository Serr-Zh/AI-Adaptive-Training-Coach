
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from coach.evaluation import (
    ROOT,
    aggregate_results,
    load_cases,
    print_summary,
    run_direct_evaluation,
    save_evaluation_artifacts,
)


async def _amain() -> int:
    parser = argparse.ArgumentParser(
        description="Полный eval-пайплайн для AI Adaptive Training Coach"
    )
    parser.add_argument(
        "--input",
        default="data/eval_cases_v2.json",
        help="Путь к JSON/JSONL набору оценочных кейсов",
    )
    parser.add_argument(
        "--output-dir",
        default="results/agent_eval",
        help="Директория для результатов оценки",
    )
    parser.add_argument(
        "--retriever-eval",
        default="results/retriever_eval.json",
        help="Путь к JSON с метриками retriever",
    )

    args = parser.parse_args()

    cases = load_cases(Path(args.input))
    rows = await run_direct_evaluation(cases)
    summary = aggregate_results(rows, Path(args.retriever_eval))
    save_evaluation_artifacts(rows, Path(args.output_dir), summary)
    print_summary(summary)

    print(f"CSV сохранён: {Path(args.output_dir) / 'eval_results.csv'}")
    print(f"Summary JSON сохранён: {Path(args.output_dir) / 'eval_summary.json'}")
    print(f"Markdown report сохранён: {Path(args.output_dir) / 'eval_report.md'}")
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
