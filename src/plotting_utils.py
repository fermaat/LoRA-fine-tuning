from typing import List

import matplotlib.pyplot as plt

from cfg import config_logger, logger, settings

config_logger(log_level=settings.log_level)


def print_examples(
    dataset, predictions: List[str], references: List[str], num_examples: int = 5
):
    """
    Print sample predictions vs references for inspection.

    Args:
        dataset: The evaluation dataset
        predictions (List[str]): Model predictions
        references (List[str]): Reference texts
        num_examples (int): Number of examples to print
    """
    logger.info("\n--- Examples (Prediction vs. Reference) ---")

    for i in range(min(num_examples, len(predictions))):
        logger.info(f"\n--- Example {i + 1} ---")
        logger.info(f"Prompt: {dataset[i]['prompt']}")
        logger.info(f"Reference (Chosen): {references[i]}")
        logger.info(f"Model Prediction: {predictions[i]}")
        logger.info("-" * 30)


def plot_metric_scores(metrics_dict, out_path="metrics.png"):
    labels = list(metrics_dict.keys())
    values = [
        metrics_dict[k] if isinstance(metrics_dict[k], (int, float)) else 0
        for k in labels
    ]
    plt.bar(labels, values)
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.savefig(out_path)
