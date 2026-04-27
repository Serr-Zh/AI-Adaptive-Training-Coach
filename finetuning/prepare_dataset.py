import argparse
import json
from pathlib import Path
from typing import Any, Dict

from datasets import DatasetDict, load_dataset


DATASET_NAME = "its-myrto/fitness-question-answers"

SYSTEM_PROMPT = (
    "You are an AI fitness coach. "
    "Answer the user's fitness question clearly, safely, and practically. "
    "Do not provide medical diagnosis. "
    "If the question contains pain, injury, or medical symptoms, recommend consulting a qualified professional."
)


def build_prompt(question: str) -> str:
    question = question.strip()

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"User question:\n"
        f"{question}\n\n"
        f"Coach answer:"
    )


def normalize_example(example: Dict[str, Any]) -> Dict[str, str]:
    question = str(example.get("Question", "")).strip()
    answer = str(example.get("Answer", "")).strip()

    prompt = build_prompt(question)

    return {
        "prompt": prompt,
        "completion": answer,
        "text": f"{prompt} {answer}",
    }


def is_valid_example(example: Dict[str, Any]) -> bool:
    question = str(example.get("Question", "")).strip()
    answer = str(example.get("Answer", "")).strip()

    return len(question) > 0 and len(answer) > 0


def save_jsonl(dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for item in dataset:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare fitness QA dataset for LLM fine-tuning."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="finetuning/data/processed",
        help="Directory where train/validation JSONL files will be saved.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="Validation split size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    raw_dataset = load_dataset(DATASET_NAME)

    if "train" not in raw_dataset:
        raise ValueError("Expected dataset to contain a 'train' split.")

    dataset = raw_dataset["train"]

    dataset = dataset.filter(is_valid_example)

    original_columns = dataset.column_names

    dataset = dataset.map(
        normalize_example,
        remove_columns=original_columns,
    )

    split_dataset = dataset.train_test_split(
        test_size=args.validation_size,
        seed=args.seed,
        shuffle=True,
    )

    dataset_dict = DatasetDict(
        {
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        }
    )

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    metadata_path = output_dir / "dataset_metadata.json"

    save_jsonl(dataset_dict["train"], train_path)
    save_jsonl(dataset_dict["validation"], validation_path)

    metadata = {
        "dataset_name": DATASET_NAME,
        "task": "Question Answering / Abstractive QA / Supervised Fine-Tuning",
        "source_columns": ["Question", "Answer"],
        "output_columns": ["prompt", "completion", "text"],
        "train_size": len(dataset_dict["train"]),
        "validation_size": len(dataset_dict["validation"]),
        "validation_ratio": args.validation_size,
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Dataset preparation completed.")
    print(f"Dataset name: {DATASET_NAME}")
    print(f"Train examples: {len(dataset_dict['train'])}")
    print(f"Validation examples: {len(dataset_dict['validation'])}")
    print(f"Train file: {train_path}")
    print(f"Validation file: {validation_path}")
    print(f"Metadata file: {metadata_path}")


if __name__ == "__main__":
    main()