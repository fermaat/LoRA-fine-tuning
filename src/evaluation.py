"""
Model Evaluation Module

This module provides functionality to evaluate language models using various metrics
including BLEU, ROUGE, and BERTScore on given datasets.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cfg import config_logger, logger, settings
from src.data_loading import get_references, load_evaluation_dataset
from src.model import load_model_and_tokenizer
from src.plotting_utils import plot_metric_scores, print_examples
from src.utils import setup_device


def load_evaluation_metrics() -> Tuple[Any, Any, Optional[Any]]:
    """
    Load evaluation metrics (BLEU, ROUGE, BERTScore).

    Returns:
        Tuple[Any, Any, Optional[Any]]: Loaded BLEU, ROUGE, and BERTScore metrics
    """
    logger.info("Loading evaluation metrics (BLEU, ROUGE, BERTScore)...")

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    try:
        bertscore_metric = evaluate.load("bertscore")
    except Exception as e:
        logger.warning(
            f"Warning: Could not load BERTScore. Ensure 'sentence-transformers' is installed. Error: {e}"
        )
        bertscore_metric = None

    return bleu_metric, rouge_metric, bertscore_metric


def generate_response(
    prompts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_new_tokens: int = 100,
) -> List[str]:
    """
    Generate responses for a batch of prompts using the model.

    Args:
        prompts (List[str]): List of input prompts
        model (AutoModelForCausalLM): The language model
        tokenizer (AutoTokenizer): The tokenizer
        device (torch.device): The device to run inference on
        max_new_tokens (int): Maximum number of new tokens to generate

    Returns:
        List[str]: Generated responses for each prompt
    """
    # Ensure pad_token is set for batch inference
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # A common fallback for Gemma

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,  # For faster, deterministic generation
            do_sample=False,  # Disable sampling
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded_responses = []
    for i, output in enumerate(outputs):
        # Decode only the newly generated tokens
        generated_text = tokenizer.decode(
            output[inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        decoded_responses.append(generated_text)

    return decoded_responses


def perform_inference(
    dataset,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 4,
    max_new_tokens: int = 100,
) -> Tuple[List[str], List[str]]:
    """
    Perform inference on the entire dataset and collect predictions and references.

    Args:
        dataset: The evaluation dataset
        model (AutoModelForCausalLM): The language model
        tokenizer (AutoTokenizer): The tokenizer
        device (torch.device): The device to run inference on
        batch_size (int): Batch size for inference
        max_new_tokens (int): Maximum number of new tokens to generate

    Returns:
        Tuple[List[str], List[str]]: Generated predictions and reference texts
    """
    logger.info("Performing inference and collecting responses...")

    predictions = []
    references = []

    # Iterate over the dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating responses"):
        batch_indices = list(range(i, min(i + batch_size, len(dataset))))
        batch_data = dataset.select(batch_indices)

        batch_prompts = batch_data["prompt"]

        batch_references = get_references(batch_data)

        generated_texts = generate_response(
            batch_prompts, model, tokenizer, device, max_new_tokens
        )

        predictions.extend(generated_texts)
        references.extend(batch_references)
        if i > batch_size:
            break

    logger.info("Inference complete.")
    logger.info(f"Total generated responses: {len(predictions)}")
    logger.info(f"Total references collected: {len(references)}")

    return predictions, references


def calculate_metrics(
    predictions: List[str],
    references: List[str],
    bleu_metric,
    rouge_metric,
    bertscore_metric: Optional[Any] = None,
    experiment_name: str = "evaluation_results",
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for predictions vs references.

    Args:
        predictions (List[str]): Model predictions
        references (List[str]): Reference texts
        bleu_metric: BLEU metric evaluator
        rouge_metric: ROUGE metric evaluator
        bertscore_metric (Optional[Any]): BERTScore metric evaluator

    Returns:
        Dict[str, Any]: Dictionary containing all calculated metrics
    """
    logger.info("\nCalculating evaluation metrics...")

    results = {}

    # BLEU
    bleu_results = bleu_metric.compute(
        predictions=predictions, references=[[ref] for ref in references]
    )
    results["bleu"] = bleu_results
    logger.info(f"BLEU: {bleu_results}")

    # ROUGE
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    results["rouge"] = rouge_results
    logger.info(f"ROUGE: {rouge_results}")

    results_short = {"bleu": bleu_results["bleu"], "rougeL": rouge_results["rougeL"]}

    # BERTScore
    if bertscore_metric:
        logger.info("Calculating BERTScore (this might take some time)...")
        bertscore_results = bertscore_metric.compute(
            predictions=predictions, references=references, lang="en"
        )
        avg_bertscore_f1 = sum(bertscore_results["f1"]) / len(bertscore_results["f1"])
        results["bertscore"] = {
            "full_results": bertscore_results,
            "average_f1": avg_bertscore_f1,
        }
        logger.info(f"BERTScore (Average F1): {avg_bertscore_f1}")
        results_short["bertscore"] = avg_bertscore_f1

        # "bertscore": results.get("bertscore", {}).get("average_f1", 0.0)}
    plot_metric_scores(results_short, experiment_name=experiment_name)

    return results


def evaluate_model(
    model_id: str = None,
    dataset_name: str = None,
    dataset_split: str = None,
    batch_size: int = None,
    max_new_tokens: int = None,
    num_examples: int = 5,
    experiment_name: str = "",
) -> Dict[str, Any]:
    """
    Main function to evaluate a language model on a given dataset.

    Args:
        model_id (str, optional): HuggingFace model identifier. Defaults to settings.default_model_id
        dataset_name (str, optional): Dataset name. Defaults to settings.default_dataset_name
        dataset_split (str, optional): Dataset split. Defaults to settings.default_evaluation_dataset_split
        batch_size (int, optional): Batch size for inference. Defaults to settings.evaluation_batch_size
        max_new_tokens (int, optional): Max tokens to generate. Defaults to settings.max_new_tokens_evaluation
        num_examples (int): Number of examples to print for inspection

    Returns:
        Dict[str, Any]: Dictionary containing all evaluation results
    """
    # Use settings defaults if not provided
    model_id = model_id or settings.default_model_id
    dataset_name = dataset_name or settings.default_dataset_name
    dataset_split = dataset_split or settings.default_evaluation_dataset_split
    batch_size = batch_size or settings.evaluation_batch_size
    max_new_tokens = max_new_tokens or settings.max_new_tokens_evaluation
    peft_config = settings.peft_config if hasattr(settings, "peft_config") else None
    use_peft = peft_config is not None

    # Setup
    config_logger(log_level=settings.log_level)
    device = setup_device()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_id, device, use_peft=use_peft, peft_config=peft_config
    )

    # Load dataset
    dataset = load_evaluation_dataset(dataset_name, dataset_split)

    # Load metrics
    bleu_metric, rouge_metric, bertscore_metric = load_evaluation_metrics()

    # Perform inference
    predictions, references = perform_inference(
        dataset, model, tokenizer, device, batch_size, max_new_tokens
    )

    # Calculate metrics
    results = calculate_metrics(
        predictions,
        references,
        bleu_metric,
        rouge_metric,
        bertscore_metric,
        experiment_name,
    )

    # Print examples
    print_examples(dataset, predictions, references, num_examples, experiment_name)

    logger.info("Evaluation complete!")
    output_dict = {
        "metrics": results,
        "predictions": predictions,
        "references": references,
        "dataset_info": {
            "name": dataset_name,
            "split": dataset_split,
            "size": len(dataset),
        },
        "model_info": {"id": model_id, "device": str(device)},
    }
    with open(f"{settings.results_path}/{experiment_name}_results.json", "w") as f:
        json.dump(output_dict, f, indent=2)

    return output_dict


# For backward compatibility and direct script execution
def main():
    """Main function for direct script execution."""
    return evaluate_model(experiment_name="raw_test_eval")


if __name__ == "__main__":
    results = main()
