import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_torch_dtype(dtype_name: str):
    dtype_name = str(dtype_name).lower()

    if dtype_name == "float16":
        return torch.float16

    if dtype_name == "bfloat16":
        return torch.bfloat16

    if dtype_name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def ensure_gpu_available_for_qlora(config: Dict[str, Any]) -> None:
    quantization_config = config.get("quantization", {})
    use_4bit = bool(quantization_config.get("use_4bit", False))

    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training requires CUDA-compatible NVIDIA GPU. "
            "This machine does not have CUDA available. "
            "Prepare dataset and configs locally, then run training in Google Colab or Kaggle with GPU enabled."
        )


def build_quantization_config(config: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    quantization_config = config.get("quantization", {})
    use_4bit = bool(quantization_config.get("use_4bit", False))

    if not use_4bit:
        return None

    compute_dtype = get_torch_dtype(
        quantization_config.get("bnb_4bit_compute_dtype", "float16")
    )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(
            quantization_config.get("bnb_4bit_use_double_quant", True)
        ),
    )


def build_lora_config(config: Dict[str, Any]) -> LoraConfig:
    lora_config = config["lora"]

    return LoraConfig(
        r=int(lora_config["r"]),
        lora_alpha=int(lora_config["lora_alpha"]),
        lora_dropout=float(lora_config["lora_dropout"]),
        target_modules=list(lora_config["target_modules"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_name": None,
            "total_memory_gb": None,
        }

    device_index = 0
    properties = torch.cuda.get_device_properties(device_index)

    return {
        "cuda_available": True,
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(device_index),
        "total_memory_gb": round(properties.total_memory / 1024**3, 2),
    }


def get_environment_info() -> Dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "gpu": get_gpu_info(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LoRA/QLoRA adapter for fitness question answering."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment config.",
    )

    args = parser.parse_args()

    config = load_yaml_config(args.config)

    ensure_gpu_available_for_qlora(config)

    experiment_name = config["experiment_name"]
    model_name = config["model_name"]

    dataset_config = config["dataset"]
    training_config = config["training"]

    output_dir = Path(training_config["output_dir"])
    metrics_dir = Path(training_config["metrics_dir"])
    metrics_path = metrics_dir / f"{experiment_name}_metrics.json"

    started_at = time.time()

    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Output dir: {output_dir}")
    print(f"Metrics path: {metrics_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("=" * 80)

    dataset = load_dataset(
        "json",
        data_files={
            "train": dataset_config["train_file"],
            "validation": dataset_config["validation_file"],
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    quantization_config = build_quantization_config(config)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    if bool(training_config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    lora_config = build_lora_config(config)

    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training_config["num_train_epochs"]),
        max_steps=int(training_config.get("max_steps", -1)),
        per_device_train_batch_size=int(training_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        learning_rate=float(training_config["learning_rate"]),
        logging_steps=int(training_config["logging_steps"]),
        eval_steps=int(training_config["eval_steps"]),
        save_steps=int(training_config["save_steps"]),
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        fp16=bool(training_config.get("fp16", True)),
        bf16=bool(training_config.get("bf16", False)),
        report_to="none",
        seed=int(training_config["seed"]),
        remove_unused_columns=True,
        optim="paged_adamw_8bit" if quantization_config is not None else "adamw_torch",
        gradient_checkpointing=bool(training_config.get("gradient_checkpointing", True)),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        dataset_text_field=dataset_config["text_field"],
        max_seq_length=int(training_config["max_seq_length"]),
        args=training_arguments,
        packing=False,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    finished_at = time.time()
    training_time_seconds = finished_at - started_at

    eval_loss = float(eval_result.get("eval_loss", 0.0))
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")

    metrics = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "task": "Question Answering / Abstractive QA / Supervised Fine-Tuning",
        "method": "QLoRA" if quantization_config is not None else "LoRA",
        "train_size": len(dataset["train"]),
        "validation_size": len(dataset["validation"]),
        "training_time_seconds": training_time_seconds,
        "training_time_minutes": training_time_seconds / 60,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "lora": config["lora"],
        "training_config": training_config,
        "quantization": config.get("quantization", {}),
        "adapter_output_dir": str(output_dir),
        "environment": get_environment_info(),
    }

    save_json(metrics, metrics_path)

    print("=" * 80)
    print("Training completed.")
    print(f"Eval loss: {eval_loss}")
    print(f"Perplexity: {perplexity}")
    print(f"Training time, min: {training_time_seconds / 60:.2f}")
    print(f"Adapter saved to: {output_dir}")
    print(f"Metrics saved to: {metrics_path}")
    print("=" * 80)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()