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


if __name__ == "__main__":
    extract_best_responses(split="train_sft")
