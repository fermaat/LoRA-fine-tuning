import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

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
    jsonl_filepath: str,
    output_dir: str,
    metrics_to_plot: Optional[list] = None,
    smoothing_window: Optional[int] = None,
    downsample_factor: Optional[int] = None,
    max_points: Optional[int] = 1000,
    smoothing_method: str = "moving_average",
):
    """
    Plot training metrics from JSON log file with optional smoothing and downsampling.
    Handles cases with/without evaluation data gracefully.
    """
    import json
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    if metrics_to_plot is None:
        metrics_to_plot = ["loss", "learning_rate", "grad_norm", "rewards/accuracies"]

    # Read and parse log file
    train_data = {"epochs": [], "steps": []}
    eval_data = {"epochs": [], "steps": []}

    with open(jsonl_filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)

                if entry.get("type") == "summary":
                    continue
                elif entry.get("type") == "eval":
                    # Evaluation metrics
                    eval_data["epochs"].append(entry.get("epoch", 0))
                    eval_data["steps"].append(entry.get("step", 0))
                    for key, value in entry.items():
                        if key not in ["type", "step", "epoch"] and value is not None:
                            if key not in eval_data:
                                eval_data[key] = []
                            eval_data[key].append(value)
                else:
                    # Training metrics
                    train_data["epochs"].append(entry.get("epoch", 0))
                    train_data["steps"].append(entry.get("step", 0))
                    for key, value in entry.items():
                        if key not in ["type", "step", "epoch"] and value is not None:
                            if key not in train_data:
                                train_data[key] = []
                            train_data[key].append(value)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
                continue

    if not train_data["epochs"]:
        print("No training data found in log file")
        return

    # Convert to numpy arrays
    for key in train_data:
        train_data[key] = np.array(train_data[key])

    # Handle eval data only if it exists
    has_eval_data = len(eval_data["epochs"]) > 0
    if has_eval_data:
        for key in eval_data:
            eval_data[key] = np.array(eval_data[key])
        print(f"Found {len(eval_data['epochs'])} evaluation data points")
    else:
        print("No evaluation data found in logs")

    # Calculate min and final loss from training data
    min_data_loss = min(train_data["loss"]) if "loss" in train_data else None
    final_data_loss = train_data["loss"][-1] if "loss" in train_data else None

    # Apply downsampling to training data
    train_total_points = len(train_data["epochs"])
    print(f"Original training data points: {train_total_points}")

    if downsample_factor:
        train_indices = np.arange(0, train_total_points, downsample_factor)
        print(f"Downsampling training data by factor {downsample_factor}")
    elif max_points and train_total_points > max_points:
        downsample_factor = max(1, train_total_points // max_points)
        train_indices = np.arange(0, train_total_points, downsample_factor)
        print(f"Auto-downsampling training data by factor {downsample_factor}")
    else:
        train_indices = np.arange(train_total_points)

    # Apply downsampling to training data
    for key in train_data:
        train_data[key] = train_data[key][train_indices]

    # Apply downsampling to eval data if it exists
    if has_eval_data:
        eval_total_points = len(eval_data["epochs"])
        print(f"Original eval data points: {eval_total_points}")

        # Less aggressive downsampling for eval data
        if downsample_factor:
            eval_downsample = max(1, downsample_factor // 2)
            eval_indices = np.arange(0, eval_total_points, eval_downsample)
            print(f"Downsampling eval data by factor {eval_downsample}")
        elif max_points and eval_total_points > max_points // 4:
            eval_downsample = max(1, eval_total_points // (max_points // 4))
            eval_indices = np.arange(0, eval_total_points, eval_downsample)
            print(f"Auto-downsampling eval data by factor {eval_downsample}")
        else:
            eval_indices = np.arange(eval_total_points)

        for key in eval_data:
            eval_data[key] = eval_data[key][eval_indices]

    print(f"Final training points: {len(train_data['epochs'])}")
    if has_eval_data:
        print(f"Final eval points: {len(eval_data['epochs'])}")

    # Filter metrics that exist in data
    available_metrics = []
    for m in metrics_to_plot:
        # Check training metrics
        if m in train_data and len(train_data[m]) > 0:
            available_metrics.append((m, "train"))

        # Check eval metrics if eval data exists
        if has_eval_data:
            # Handle both 'eval_metric' and 'metric' formats
            eval_m = m if m.startswith("eval_") else f"eval_{m}"
            if eval_m in eval_data and len(eval_data[eval_m]) > 0:
                available_metrics.append((eval_m, "eval"))
            elif m in eval_data and len(eval_data[m]) > 0:
                available_metrics.append((m, "eval"))

    if not available_metrics:
        print(f"None of the requested metrics {metrics_to_plot} found in log file")
        train_metrics = [
            k for k in train_data.keys() if k not in ["epochs", "steps", "type"]
        ]
        print(f"Available training metrics: {train_metrics}")
        if has_eval_data:
            eval_metrics = [
                k for k in eval_data.keys() if k not in ["epochs", "steps", "type"]
            ]
            print(f"Available eval metrics: {eval_metrics}")
        return

    def smooth_data(y, method="moving_average", window=None):
        """Apply smoothing to data"""
        if window is None or window <= 1 or len(y) <= window:
            return y

        if method == "moving_average":
            # Simple moving average
            return np.convolve(y, np.ones(window) / window, mode="valid")
        elif method == "exponential":
            # Exponential moving average
            alpha = 2.0 / (window + 1)
            smoothed = np.zeros_like(y)
            smoothed[0] = y[0]
            for i in range(1, len(y)):
                smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i - 1]
            return smoothed
        elif method == "savgol":
            # Savitzky-Golay filter
            window = min(window, len(y))
            if window % 2 == 0:
                window -= 1  # Must be odd
            if window < 3:
                return y
            try:
                return savgol_filter(y, window, 2)  # 2nd order polynomial
            except:
                return y  # Fallback if savgol fails
        else:
            return y

    # Create subplots
    n_metrics = len(available_metrics)
    cols = 2
    rows = (n_metrics + 1) // 2
    fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))

    if n_metrics == 1:
        axs = [axs]
    elif rows == 1:
        axs = axs.flatten()
    else:
        axs = axs.flatten()

    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]

    for i, (metric, metric_type) in enumerate(available_metrics):
        if i >= len(axs):
            break

        ax = axs[i]
        color = colors[i % len(colors)]

        # Get the correct data source
        if metric_type == "train":
            data_source = train_data
            metric_key = metric
            label_prefix = ""
        else:
            data_source = eval_data
            # Use the exact metric key as stored in eval_data
            metric_key = metric
            label_prefix = "eval_" if not metric.startswith("eval_") else ""

        x_data = data_source["epochs"]
        y_data = data_source[metric_key]  # Now using the exact key from eval_data

        # Apply smoothing if requested (only for training data)
        if smoothing_window and smoothing_window > 1 and metric_type == "train":
            y_smoothed = smooth_data(y_data, smoothing_method, smoothing_window)

            # Adjust x_data for moving average (which shortens the array)
            if smoothing_method == "moving_average" and len(y_smoothed) < len(x_data):
                x_smoothed = x_data[smoothing_window - 1 :]
            else:
                x_smoothed = x_data

            # Plot both original (faded) and smoothed data
            ax.plot(
                x_data,
                y_data,
                color=color,
                alpha=0.3,
                linewidth=1,
                label=f"{label_prefix}{metric_key} (raw)",
            )
            ax.plot(
                x_smoothed,
                y_smoothed,
                color=color,
                linewidth=2,
                label=f"{label_prefix}{metric_key} (smoothed)",
            )
            ax.legend()
        else:
            # Plot original data (for eval or unsmoothed training)
            ax.plot(
                x_data,
                y_data,
                color=color,
                linewidth=2,
                label=f"{label_prefix}{metric_key}",
            )
            if metric_type == "eval":
                # Mark eval points with dots for clarity
                ax.scatter(x_data, y_data, color=color, s=30, zorder=3)

        ax.set_title(
            f"{label_prefix}{metric_key}".replace("/", " ").replace("_", " ").title()
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(
            f"{label_prefix}{metric_key}".replace("/", " ").replace("_", " ").title()
        )
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
    print(f"Total epochs: {max(train_data['epochs']):.2f}")
    print(f"Total steps: {max(train_data['steps'])}")
    if "loss" in train_data:
        print(f"Final training loss: {final_data_loss:.4f}")
        print(f"Best training loss: {min_data_loss:.4f}")
    if "loss" in eval_data:
        print(f"Final eval loss: {eval_data['loss'][-1]:.4f}")
        print(f"Best eval loss: {min(eval_data['loss']):.4f}")
