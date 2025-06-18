from typing import List

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
    print(f"max: {max(all_lengths)}")
    print(f"Percentil 95: {np.percentile(all_lengths, 95)}")
    print(f"avg: {np.mean(all_lengths)}")

    # Histogram
    plt.hist(all_lengths, bins=50)
    plt.title("tokenized length distribution")
    plt.xlabel("Tokens")
    plt.ylabel("freq")
    plt.show()
