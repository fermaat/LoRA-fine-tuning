import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from cfg import config_logger, logger, settings
from src.utils import get_token_lengths_sft

config_logger(log_level=settings.log_level)


def print_examples(
    dataset,
    predictions: List[str],
    references: List[str],
    num_examples: int = 5,
    experiment_name: str = "",
):
    """
    Print sample predictions vs references for inspection.

    Args:
        dataset: The evaluation dataset
        predictions (List[str]): Model predictions
        references (List[str]): Reference texts
        num_examples (int): Number of examples to print
    """
    with open(f"{settings.results_path}/{experiment_name}_examples.txt", "w") as f:

        def write_line(line):
            f.write(line)
            logger.info(line)

        write_line("--- Examples (Prediction vs. Reference) ---\n")
        for i in range(min(num_examples, len(predictions))):
            write_line(f"\n--- Example {i + 1} ---\n")
            write_line(f"Prompt: {dataset[i]['prompt']}\n")
            write_line(f"Reference (Chosen): {references[i]}\n")
            write_line(f"Model Prediction: {predictions[i]}\n")
            write_line("-" * 30 + "\n")


def plot_metric_scores(metrics_dict, out_path="metrics.png", experiment_name: str = ""):
    logger.info("Plotting evaluation metrics...")
    labels = list(metrics_dict.keys())
    values = [
        metrics_dict[k] if isinstance(metrics_dict[k], (int, float)) else 0
        for k in labels
    ]
    plt.bar(labels, values)
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.savefig(f"{settings.results_path}/{experiment_name}_{out_path}")


def plot_lengths(dataset, tokenizer, dataset_type="sft"):
    if dataset_type == "sft":
        lengths = dataset.map(
            lambda example: get_token_lengths_sft(example, tokenizer),
            remove_columns=dataset.column_names,
        )

    all_lengths = lengths["length"]
    logger.info(f"max: {max(all_lengths)}")
    logger.info(f"Percentil 95: {np.percentile(all_lengths, 95)}")
    logger.info(f"avg: {np.mean(all_lengths)}")

    # Histogram
    plt.hist(all_lengths, bins=50)
    plt.title("tokenized length distribution")
    plt.xlabel("Tokens")
    plt.ylabel("freq")
    plt.show()


def plot_training_logs_old(jsonl_filepath, output_dir):
    epochs = []
    loss = []
    grad_norm = []
    learning_rate = []
    rewards_accuracies = []

    with open(jsonl_filepath, "r") as f:
        for line in f:
            entry = json.loads(line)
            epochs.append(entry.get("epoch", None))
            loss.append(entry.get("loss", None))
            grad_norm.append(entry.get("grad_norm", None))
            learning_rate.append(entry.get("learning_rate", None))
            rewards_accuracies.append(entry.get("rewards/accuracies", None))

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(epochs, loss, label="Loss")
    axs[0, 0].set_title("Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")

    axs[0, 1].plot(epochs, learning_rate, label="Learning Rate", color="orange")
    axs[0, 1].set_title("Learning Rate")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("LR")

    axs[1, 0].plot(epochs, grad_norm, label="Grad Norm", color="green")
    axs[1, 0].set_title("Gradient Norm")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Grad Norm")

    axs[1, 1].plot(epochs, rewards_accuracies, label="Rewards Accuracies", color="red")
    axs[1, 1].set_title("Rewards Accuracies")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Accuracy")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "curve.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[✅] Loss plot saved to: {save_path}")


def plot_training_logs(
    jsonl_filepath: str, output_dir: str, metrics_to_plot: Optional[list] = None
):
    """
    Plot training metrics from JSON log file

    Args:
        jsonl_filepath: Path to the JSONL log file
        output_dir: Directory to save plots
        metrics_to_plot: List of metrics to plot. If None, plots common metrics.
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["loss", "learning_rate", "grad_norm", "rewards/accuracies"]

    # Read and parse log file
    data = {"epochs": [], "steps": []}

    with open(jsonl_filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)

                # Skip summary entries for plotting
                if entry.get("type") == "summary":
                    continue

                data["epochs"].append(entry.get("epoch", 0))
                data["steps"].append(entry.get("step", 0))

                # Collect all metrics
                for key, value in entry.items():
                    if key not in ["type", "step", "epoch"] and value is not None:
                        if key not in data:
                            data[key] = []
                        data[key].append(value)

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
                continue

    if not data["epochs"]:
        print("No training data found in log file")
        return

    # Filter metrics that exist in data
    available_metrics = [m for m in metrics_to_plot if m in data and data[m]]

    if not available_metrics:
        print(f"None of the requested metrics {metrics_to_plot} found in log file")
        print(
            f"Available metrics: {[k for k in data.keys() if k not in ['epochs', 'steps', 'type']]}"
        )
        return

    # Create subplots
    n_metrics = len(available_metrics)
    cols = 2
    rows = (n_metrics + 1) // 2

    fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if n_metrics == 1:
        axs = [axs]
    elif rows == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]

    for i, metric in enumerate(available_metrics):
        if i >= len(axs):
            break

        ax = axs[i]
        color = colors[i % len(colors)]

        # Plot metric vs epochs
        ax.plot(data["epochs"], data[metric], label=metric, color=color, linewidth=2)
        ax.set_title(metric.replace("/", " ").replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("/", " ").replace("_", " ").title())
        ax.grid(True, alpha=0.3)

        # Add some styling
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide empty subplots
    for i in range(len(available_metrics), len(axs)):
        axs[i].set_visible(False)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[✅] Training curves saved to: {save_path}")

    # Print summary statistics
    print("\n📊 Training Summary:")
    print(f"Total epochs: {max(data['epochs']):.2f}")
    print(f"Total steps: {max(data['steps'])}")
    if "loss" in data:
        print(f"Final loss: {data['loss'][-1]:.4f}")
        print(f"Best loss: {min(data['loss']):.4f}")
