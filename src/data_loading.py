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


def sft_formatting_func(example):
    messages = example["chosen"]
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"### User:\n{content}\n"
        elif role == "assistant":
            formatted += f"### Assistant:\n{content}\n"
    return formatted


def is_short_enough(example, tokenizer, max_tokens):
    text = ""
    for msg in example["chosen"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            text += f"### User:\n{content}\n"
        elif role == "assistant":
            text += f"### Assistant:\n{content}\n"
    length = len(tokenizer(text, truncation=False)["input_ids"])
    return length <= max_tokens


def load_and_prepare_sft_dataset(
    dataset_name: str, split: str, tokenizer, max_length, max_samples=None
):
    dataset = load_dataset(dataset_name, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"sampling dataset length to {max_samples}")
    logger.info(f"Filtering dataset length to {max_length}")
    dataset = dataset.filter(
        lambda x: is_short_enough(x, tokenizer=tokenizer, max_tokens=max_length)
    )
    return dataset.map(
        lambda x: {"text": sft_formatting_func(x)}, remove_columns=dataset.column_names
    )


def load_and_prepare_alignment_dataset(
    dataset_name, split, tokenizer, max_length, max_samples=None
):
    dataset = load_dataset(dataset_name, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"sampling dataset length to {max_samples}")

    def preprocess(example):
        prompt = example["chosen"][0]["content"]
        chosen = example["chosen"][1]["content"]
        rejected = example["rejected"][1]["content"]

        chosen_full = prompt + chosen
        rejected_full = prompt + rejected

        chosen_tok = tokenizer(
            chosen_full, truncation=True, max_length=max_length, padding="max_length"
        )
        rejected_tok = tokenizer(
            rejected_full, truncation=True, max_length=max_length, padding="max_length"
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_input_ids": chosen_tok["input_ids"],
            "rejected_input_ids": rejected_tok["input_ids"],
            "chosen_attention_mask": chosen_tok["attention_mask"],
            "rejected_attention_mask": rejected_tok["attention_mask"],
        }

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    return dataset


def load_and_prepare_orpo_alignment_dataset(
    dataset_name, split, tokenizer, max_length, max_samples=None
):
    dataset = load_dataset(dataset_name, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"sampling dataset length to {max_samples}")

    def preprocess(example):
        prompt = example["chosen"][0]["content"]
        chosen = example["chosen"][1]["content"]
        rejected = example["rejected"][1]["content"]

        # Formato simple que funciona bien con ORPO/DPO
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    # Filtrar ejemplos problemáticos
    def filter_valid_examples(example):
        # Verificar que los textos no estén vacíos
        if (
            not example["prompt"].strip()
            or not example["chosen"].strip()
            or not example["rejected"].strip()
        ):
            return False

        # Verificar que chosen y rejected sean diferentes
        if example["chosen"].strip() == example["rejected"].strip():
            return False

        # Verificar longitudes razonables
        prompt_len = len(tokenizer.encode(example["prompt"]))
        chosen_len = len(tokenizer.encode(example["chosen"]))
        rejected_len = len(tokenizer.encode(example["rejected"]))

        # Filtrar ejemplos demasiado largos
        if (
            prompt_len + chosen_len > max_length * 0.9
            or prompt_len + rejected_len > max_length * 0.9
        ):
            return False

        # Filtrar ejemplos demasiado cortos
        if chosen_len < 5 or rejected_len < 5:
            return False

        return True

    original_size = len(dataset)
    dataset = dataset.filter(filter_valid_examples)
    filtered_size = len(dataset)

    if filtered_size < original_size:
        logger.info(f"Filtered dataset: {original_size} -> {filtered_size} examples")
        logger.info(f"Removed {original_size - filtered_size} problematic examples")

    # Log algunos ejemplos para verificar
    if len(dataset) > 0:
        sample = dataset[0]
        logger.info(f"Sample prompt length: {len(tokenizer.encode(sample['prompt']))}")
        logger.info(f"Sample chosen length: {len(tokenizer.encode(sample['chosen']))}")
        logger.info(
            f"Sample rejected length: {len(tokenizer.encode(sample['rejected']))}"
        )

    return dataset


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
