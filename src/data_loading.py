import json

from datasets import load_dataset

from cfg import logger, settings


def extract_best_responses(split):
    logger.info(
        f"Loading dataset best response for '{settings.default_dataset_name}'..."
    )
    dataset = load_dataset(settings.default_dataset_name)[split]

    sft_data = []
    for example in dataset:
        sft_data.append(
            {"prompt": example["prompt"], "response": example["chosen"][1]["content"]}
        )
    logger.info(f"Best responses extracted for {len(sft_data)} examples.")

    with open(f"data/ultrafeedback_{split}.json", "w") as f:
        for item in sft_data:
            f.write(
                json.dumps(
                    {"text": f"USER: {item['prompt']}\nASSISTANT: {item['response']}"}
                )
                + "\n"
            )


def load_evaluation_dataset(dataset_name: str, split: str):
    """
    Load the evaluation dataset.

    Args:
        dataset_name (str): Name of the dataset to load
        split (str): Dataset split to use (e.g., 'test', 'validation')

    Returns:
        Dataset: The loaded dataset
    """
    logger.info(f"Loading dataset: {dataset_name}, split: {split}...")

    dataset = load_dataset(dataset_name, split=split)
    logger.info(f"Dataset loaded. Number of examples: {len(dataset)}")

    return dataset


def get_references(dataset):
    """
    Extract reference responses from the dataset.

    Args:
        dataset: The evaluation dataset

    Returns:
        List[str]: List of reference responses
    """
    logger.info("Extracting references f")
    # second response is the actual response. TODO: make sure it comes from the right field
    references = [item[1]["content"] for item in dataset["chosen"]]

    logger.info(f"Extracted {len(references)} reference responses.")
    return references


if __name__ == "__main__":
    extract_best_responses(split="train_sft")
