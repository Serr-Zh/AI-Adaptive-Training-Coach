import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS_DIR = Path("finetuning/outputs/metrics")
PLOTS_DIR = Path("finetuning/outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_metric(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    metric_files = sorted(METRICS_DIR.glob("*_metrics.json"))

    if not metric_files:
        raise FileNotFoundError(f"No metrics files found in {METRICS_DIR}")

    rows = []
    for path in metric_files:
        data = load_metric(path)
        rows.append(
            {
                "experiment_name": data["experiment_name"],
                "model_name": data["model_name"],
                "method": data["method"],
                "training_time_minutes": data["training_time_minutes"],
                "train_loss": data["train_metrics"]["train_loss"],
                "eval_loss": data["eval_loss"],
                "perplexity": data["perplexity"],
                "lora_r": data["lora"]["r"],
                "lora_alpha": data["lora"]["lora_alpha"],
                "learning_rate": data["training_config"]["learning_rate"],
                "max_steps": data["training_config"]["max_steps"],
            }
        )

    df = pd.DataFrame(rows)
    df["quality_score"] = 1 / df["perplexity"]

    summary_path = PLOTS_DIR / "finetuning_results_summary.csv"
    df.to_csv(summary_path, index=False)

    plt.figure(figsize=(9, 6))
    plt.scatter(df["training_time_minutes"], df["quality_score"], s=110)

    for _, row in df.iterrows():
        plt.annotate(
            row["experiment_name"],
            (row["training_time_minutes"], row["quality_score"]),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=8,
        )

    plt.xlabel("Цена: время обучения, минуты")
    plt.ylabel("Качество: 1 / perplexity")
    plt.title("Кривая цена/качество для LoRA fine-tuning")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "price_quality_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(df["experiment_name"], df["eval_loss"])
    plt.xlabel("Эксперимент")
    plt.ylabel("Validation loss")
    plt.title("Validation loss по экспериментам")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eval_loss_by_experiment.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(df["experiment_name"], df["perplexity"])
    plt.xlabel("Эксперимент")
    plt.ylabel("Perplexity")
    plt.title("Perplexity по экспериментам")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "perplexity_by_experiment.png", dpi=180)
    plt.close()

    print(f"Saved summary to {summary_path}")
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
